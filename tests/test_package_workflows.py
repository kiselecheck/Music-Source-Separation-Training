import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml

import msst
import msst.training as training_module
from msst.experiments import EXPERIMENT_DIRECTORIES, tracked_run


def _write_dataset_config(path: Path) -> None:
    path.write_text(
        """
audio:
  sample_rate: 44100
  chunk_size: 64
  min_mean_abs: 0
training:
  instruments: [vocals]
  batch_size: 1
  num_steps: 1
  read_metadata_procs: 1
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_init_experiment_creates_portable_layout(tmp_path):
    experiment = msst.init_experiment(tmp_path / "dereverb-v5")

    assert experiment.name == "dereverb-v5"
    assert experiment.manifest_path.is_file()
    assert all((experiment.path / name).is_dir() for name in EXPERIMENT_DIRECTORIES)
    assert list((experiment.path / "jobs").iterdir()) == []
    manifest = yaml.safe_load(experiment.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {"schema_version", "name", "created_at", "paths"}

    with pytest.raises(FileExistsError, match="already exists"):
        msst.init_experiment(experiment.path)


def test_experiment_tracking_records_success_and_failure(tmp_path):
    experiment = msst.init_experiment(tmp_path / "experiment")
    config_path = tmp_path / "config.yaml"
    config_path.write_text("training: {}\n", encoding="utf-8")

    with tracked_run(
        experiment,
        task="train",
        parameters={"config_path": config_path, "epochs": 1},
        run_name="smoke",
    ) as run:
        assert run is not None
        run.add_artifacts(tmp_path / "results")
        run.set_result({"loss": 0.5})

    completed = json.loads(run.record_path.read_text(encoding="utf-8"))
    assert completed["status"] == "completed"
    assert completed["inputs"]["config_path"]["snapshot"] == "config.yaml"
    assert completed["result"] == {"loss": 0.5}

    with pytest.raises(RuntimeError, match="boom"):
        with tracked_run(
            experiment,
            task="valid",
            parameters={},
        ) as failed_run:
            raise RuntimeError("boom")
    failed = json.loads(failed_run.record_path.read_text(encoding="utf-8"))
    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"


def test_build_metadata_uses_real_audio_and_writes_manifest(tmp_path):
    config_path = tmp_path / "config.yaml"
    _write_dataset_config(config_path)
    track = tmp_path / "dataset" / "track-1"
    track.mkdir(parents=True)
    sf.write(track / "vocals.wav", np.zeros((128, 2), dtype=np.float32), 44100)

    result = msst.build_metadata(
        config_path=config_path,
        data_path=track.parent,
        results_path=tmp_path / "results",
        dataset_type=1,
        num_workers=1,
        verbose=False,
    )

    assert result.track_count == 1
    assert result.path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == result.fingerprint
    assert manifest["config_sha256"]
    assert manifest["metadata_signature"]
    assert manifest["dataset_type"] == 1


def test_safe_check_reports_missing_inputs_without_loading_runtime(tmp_path):
    report = msst.check(
        config_path=tmp_path / "missing.yaml",
        data_path=tmp_path / "missing-data",
        results_path=tmp_path / "results",
        mode="safe",
    )

    assert report.ok is False
    assert {item.name for item in report.errors} == {
        "config_path",
        "data_path",
        "valid_path",
    }
    assert report.peak_allocated_gib is None


def test_compute_training_loss_is_shared_real_backward_path():
    from msst._training import compute_training_loss

    class TinySeparator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Conv1d(2, 4, kernel_size=1)

        def forward(self, mixture):
            values = self.projection(mixture)
            return values.reshape(values.shape[0], 2, 2, values.shape[-1])

    model = TinySeparator()
    config = SimpleNamespace(training=SimpleNamespace(normalize=False))
    args = SimpleNamespace(model_type="mdx23c", use_standard_loss=False)
    targets = torch.ones((1, 2, 2, 16))
    mixtures = torch.ones((1, 2, 16))

    def loss_function(predictions, expected, mixture, active_stem_mask):
        return torch.nn.functional.mse_loss(predictions, expected)

    loss, mask = compute_training_loss(
        model,
        config,
        args,
        (targets, mixtures),
        torch.device("cpu"),
        [0],
        loss_function,
        False,
    )
    loss.backward()

    assert mask is None
    assert all(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "launcher,module_name,function_name",
    [
        ("standard", "msst._training", "train_model"),
        ("ddp", "msst._train_ddp", "train_model_ddp"),
        ("accelerate", "msst._train_accelerate", "train_model"),
    ],
)
def test_training_launcher_dispatches_to_requested_runtime(
    monkeypatch, launcher, module_name, function_name
):
    calls = []
    fake = SimpleNamespace(**{function_name: lambda args: calls.append(args)})
    monkeypatch.setitem(sys.modules, module_name, fake)

    training_module.train_model({"model_type": "mdx23c"}, launcher=launcher)

    assert calls == [{"model_type": "mdx23c"}]


def test_training_rejects_unknown_launcher():
    with pytest.raises(ValueError, match="launcher"):
        training_module.train_model({}, launcher="unknown")


def test_public_api_exports_new_workflows():
    assert callable(msst.build_metadata)
    assert callable(msst.check)
    assert callable(msst.init_experiment)
    assert "MetadataResult" in msst.__all__
    assert "CheckReport" in msst.__all__
