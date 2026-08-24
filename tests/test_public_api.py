import argparse
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import msst
import msst.training as training_module
import msst.validation as validation_module
from msst import (
    OptionalDependencyError,
    OutputFormatError,
    Separator,
    ValidationConfig,
    ValidationResult,
)
from msst.inference import InferenceConfig, inference, proc_folder
from msst.training import train
from msst.validation import valid


def _install_fake_inference(monkeypatch, *, load_error=None):
    calls = SimpleNamespace(load=[], separate=[], folder=[], proc=[])
    config = SimpleNamespace(
        audio=SimpleNamespace(sample_rate=44100, num_channels=2),
        training=SimpleNamespace(instruments=["vocals", "drums"]),
    )

    def parse_args(payload):
        defaults = {
            "draw_spectro": 0,
            "flac_file": False,
            "pcm_type": "FLOAT",
            "filename_template": "{file_name}/{instr}",
            "input_folder": "",
            "store_dir": "",
        }
        defaults.update(payload or {})
        return argparse.Namespace(**defaults)

    settings = SimpleNamespace(
        parse_args_inference=parse_args,
        validate_sndfile_subtype=lambda args: args.pcm_type,
    )

    def load_separator(args):
        calls.load.append(args)
        if load_error is not None:
            raise load_error
        return object(), config, "cpu"

    def separate_audio(model, args, loaded_config, device, waveform):
        calls.separate.append(np.asarray(waveform))
        length = np.asarray(waveform).shape[-1]
        return {
            "vocals": np.ones((2, length), dtype=np.float32),
            "drums": np.zeros((2, length), dtype=np.float32),
        }

    def run_folder(model, args, loaded_config, device, verbose):
        calls.folder.append((args, verbose))
        return [str(Path(args.store_dir) / "song" / "vocals.wav")]

    legacy = SimpleNamespace(
        load_separator=load_separator,
        separate_audio=separate_audio,
        run_folder=run_folder,
        proc_folder=lambda args: calls.proc.append(args) or ["legacy.wav"],
    )
    monkeypatch.setitem(sys.modules, "msst.utils.settings", settings)
    monkeypatch.setitem(sys.modules, "msst._inference", legacy)
    return calls


def test_package_has_release_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[a-z]+\d+)?", msst.__version__)
    assert "Separator" in msst.__all__
    assert callable(msst.inference)
    assert callable(msst.train)
    assert callable(msst.valid)
    assert "ValidationResult" in msst.__all__


def test_programmatic_parsers_accept_namespace_and_reject_unknown_keys():
    from msst.utils.settings import (
        parse_args_inference,
        parse_args_train,
        parse_args_valid,
    )

    inference_args = parse_args_inference(
        argparse.Namespace(config_path="model.yaml", force_cpu=True)
    )
    assert inference_args.config_path == "model.yaml"
    assert inference_args.force_cpu is True

    train_args = parse_args_train(argparse.Namespace(model_type="bs_roformer"))
    assert train_args.model_type == "bs_roformer"

    valid_args = parse_args_valid(
        argparse.Namespace(force_cpu=True, verbose=True)
    )
    assert valid_args.force_cpu is True
    assert valid_args.verbose is True

    with pytest.raises(TypeError, match="Unknown argument.*typo"):
        parse_args_inference({"typo": 1})


def test_inference_loads_model_once_and_returns_output_paths(monkeypatch, tmp_path):
    calls = _install_fake_inference(monkeypatch)

    result = inference(
        model_type="bs_roformer",
        config_path=tmp_path / "model.yaml",
        checkpoint_path=tmp_path / "model.ckpt",
        input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
        device_ids=(0, 1),
    )

    assert result == [str(tmp_path / "output" / "song" / "vocals.wav")]
    assert len(calls.load) == 1
    assert calls.load[0].device_ids == [0, 1]
    assert calls.folder[0][0].input_folder == str(tmp_path / "input")
    assert calls.folder[0][0].store_dir == str(tmp_path / "output")


def test_inference_accepts_legacy_string_device_ids(monkeypatch, tmp_path):
    calls = _install_fake_inference(monkeypatch)

    inference(
        config_path="model.yaml",
        checkpoint_path="model.ckpt",
        input_folder=tmp_path / "input",
        output_folder=tmp_path / "output",
        device_ids="0, 1",
    )

    assert calls.load[0].device_ids == [0, 1]


def test_separator_reuses_loaded_model_for_multiple_arrays(monkeypatch):
    calls = _install_fake_inference(monkeypatch)
    separator = Separator(
        model_type="bs_roformer",
        config_path="model.yaml",
        checkpoint_path="model.ckpt",
        force_cpu=True,
    )

    first = separator.separate(np.zeros(32, dtype=np.float32))
    second = separator.separate(
        np.zeros((32, 2), dtype=np.float32),
        channels_first=False,
    )

    assert len(calls.load) == 1
    assert len(calls.separate) == 2
    assert calls.separate[1].shape == (2, 32)
    assert tuple(first) == ("vocals", "drums")
    assert tuple(second) == ("vocals", "drums")
    assert separator.sample_rate == 44100
    assert separator.instruments == ("vocals", "drums")
    assert separator.device == "cpu"

    separator.close()
    separator.close()
    with pytest.raises(RuntimeError, match="closed"):
        separator.separate(np.zeros(4))


def test_separator_resamples_input(monkeypatch):
    calls = _install_fake_inference(monkeypatch)
    resample_calls = []
    monkeypatch.setitem(
        sys.modules,
        "librosa",
        SimpleNamespace(
            resample=lambda waveform, **kwargs: (
                resample_calls.append(kwargs) or waveform[..., ::2]
            )
        ),
    )
    separator = Separator(config_path="model.yaml", checkpoint_path="model.ckpt")

    separator.separate(np.zeros((2, 20)), sample_rate=88200)

    assert resample_calls == [{"orig_sr": 88200, "target_sr": 44100, "axis": -1}]
    assert calls.separate[0].shape == (2, 10)


def test_separator_validates_constructor_and_file_inputs(monkeypatch):
    _install_fake_inference(monkeypatch)
    with pytest.raises(ValueError, match="bigshifts"):
        Separator(
            config_path="model.yaml",
            checkpoint_path="model.ckpt",
            bigshifts=0,
        )
    with pytest.raises(ValueError, match="device_ids"):
        Separator(
            config_path="model.yaml",
            checkpoint_path="model.ckpt",
            device_ids=[],
        )

    separator = Separator(config_path="model.yaml", checkpoint_path="model.ckpt")
    with pytest.raises(FileNotFoundError, match="Audio file"):
        separator.separate_file("missing.wav")
    with pytest.raises(ValueError, match="positive"):
        separator.separate(np.zeros(4), sample_rate=0)


def test_missing_model_dependency_has_install_hint(monkeypatch):
    error = ModuleNotFoundError("No module named 'einops'", name="einops")
    _install_fake_inference(monkeypatch, load_error=error)

    with pytest.raises(OptionalDependencyError) as caught:
        Separator(
            model_type="bs_roformer",
            config_path="model.yaml",
            checkpoint_path="model.ckpt",
        )

    assert caught.value.dependency == "einops"
    assert 'pip install "msst[bs-roformer]"' in str(caught.value)


def test_mamba_dependency_error_explains_platform_limit(monkeypatch):
    error = ModuleNotFoundError("No module named 'mamba_ssm'", name="mamba_ssm")
    _install_fake_inference(monkeypatch, load_error=error)

    with pytest.raises(OptionalDependencyError) as caught:
        Separator(
            model_type="bs_mamba2",
            config_path="model.yaml",
            checkpoint_path="model.ckpt",
        )

    if sys.platform.startswith("linux"):
        assert 'pip install "msst[bs-mamba2]"' in str(caught.value)
    else:
        assert "not supported" in str(caught.value)
        assert "Linux x86_64" in str(caught.value)


def test_model_factory_reports_extra_from_effective_config(monkeypatch, tmp_path):
    from msst.utils import settings

    config_path = tmp_path / "model.yaml"
    config_path.write_text(
        "training:\n  model_type: scnet_tran\n  instruments: [stem]\n"
        "model: {}\n",
        encoding="utf-8",
    )
    missing = ModuleNotFoundError("No module named 'einops'", name="einops")
    monkeypatch.setattr(
        settings,
        "build_model_from_config",
        lambda *args, **kwargs: (_ for _ in ()).throw(missing),
    )

    with pytest.raises(OptionalDependencyError) as caught:
        settings.get_model_from_config("mdx23c", str(config_path))

    assert 'pip install "msst[scnet-tran]"' in str(caught.value)


def test_proc_folder_keeps_legacy_mapping_api(monkeypatch):
    calls = _install_fake_inference(monkeypatch)

    result = proc_folder({"config_path": "model.yaml"}, force_cpu=True)

    assert result == ["legacy.wav"]
    assert calls.proc == [{"config_path": "model.yaml", "force_cpu": True}]


def test_proc_folder_accepts_public_aliases(monkeypatch):
    calls = _install_fake_inference(monkeypatch)

    proc_folder(checkpoint_path=Path("model.ckpt"), output_folder=Path("out"))

    assert calls.proc == [
        {"start_check_point": "model.ckpt", "store_dir": "out"}
    ]


def test_proc_folder_rejects_duplicate_aliases():
    with pytest.raises(TypeError, match="not both"):
        proc_folder(
            checkpoint_path="new-name.ckpt",
            start_check_point="old-name.ckpt",
        )


def test_inference_config_cannot_be_mixed_with_overrides():
    config = InferenceConfig("config.yaml", "model.ckpt", "input", "output")

    with pytest.raises(TypeError, match="cannot be combined"):
        proc_folder(config, force_cpu=True)


def test_inference_config_validates_combinations():
    with pytest.raises(ValueError, match="bigshifts"):
        InferenceConfig("config.yaml", "model.ckpt", "input", "output", bigshifts=0)
    with pytest.raises(OutputFormatError, match="FLAC does not support FLOAT"):
        InferenceConfig(
            "config.yaml",
            "model.ckpt",
            "input",
            "output",
            flac_file=True,
            pcm_type="FLOAT",
        )
    with pytest.raises(ValueError, match="draw_spectro"):
        InferenceConfig(
            "config.yaml",
            "model.ckpt",
            "input",
            "output",
            draw_spectro=-1,
        )


def test_training_exposes_conventional_keyword_api(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "msst._training",
        SimpleNamespace(train_model=lambda args: calls.append(args) or "trained"),
    )

    result = train(
        model_type="bs_roformer",
        config_path=tmp_path / "model.yaml",
        checkpoint_path=tmp_path / "model.ckpt",
        results_path=tmp_path / "results",
        data_path=[tmp_path / "dataset-a", tmp_path / "dataset-b"],
        valid_path=tmp_path / "validation",
        device_ids=1,
        num_workers=4,
    )

    assert result is None
    assert calls == [
        {
            "config_path": str(tmp_path / "model.yaml"),
            "results_path": str(tmp_path / "results"),
            "data_path": [
                str(tmp_path / "dataset-a"),
                str(tmp_path / "dataset-b"),
            ],
            "valid_path": [str(tmp_path / "validation")],
            "model_type": "bs_roformer",
            "device_ids": [1],
            "num_workers": 4,
            "start_check_point": str(tmp_path / "model.ckpt"),
        }
    ]


def test_training_missing_dependency_has_install_hint(monkeypatch):
    error = ModuleNotFoundError("No module named 'wandb'", name="wandb")

    def fail_import(name, package=None):
        raise error

    monkeypatch.setattr(training_module, "import_module", fail_import)

    with pytest.raises(OptionalDependencyError) as caught:
        training_module.train_model({"model_type": "bs_roformer"})

    assert caught.value.dependency == "wandb"
    assert 'pip install "msst[train,bs-roformer]"' in str(caught.value)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"data_path": []}, "data_path"),
        ({"device_ids": []}, "device_ids"),
        ({"device_ids": [-1]}, "negative"),
        ({"device_ids": [0, 0]}, "unique"),
    ],
)
def test_training_rejects_invalid_public_arguments(kwargs, message):
    values = {
        "config_path": "model.yaml",
        "results_path": "results",
        "data_path": ["dataset"],
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        train(**values)


def test_validation_exposes_typed_keyword_api(monkeypatch, tmp_path):
    calls = []
    legacy_result = (
        {"sdr": 8.5},
        {"sdr": {"vocals": [8.0, np.float32(9.0)]}},
    )
    monkeypatch.setitem(
        sys.modules,
        "msst._validation",
        SimpleNamespace(
            check_validation=lambda args: calls.append(args) or legacy_result
        ),
    )

    result = valid(
        config_path=tmp_path / "model.yaml",
        checkpoint_path=tmp_path / "model.ckpt",
        valid_path=[tmp_path / "valid-a", tmp_path / "valid-b"],
        model_type="bs_roformer",
        device_ids=(0, 1),
        metrics=("sdr", "k_sdr", "sdr"),
        output_folder=tmp_path / "output",
        extension=".flac",
        force_cpu=True,
        verbose=True,
    )

    assert result == ValidationResult(
        averages={"sdr": 8.5},
        per_track={"sdr": {"vocals": [8.0, 9.0]}},
    )
    assert calls == [
        {
            "config_path": str(tmp_path / "model.yaml"),
            "start_check_point": str(tmp_path / "model.ckpt"),
            "valid_path": [
                str(tmp_path / "valid-a"),
                str(tmp_path / "valid-b"),
            ],
            "model_type": "bs_roformer",
            "device_ids": [0, 1],
            "force_cpu": True,
            "metrics": ["sdr", "k_sdr"],
            "store_dir": str(tmp_path / "output"),
            "extension": "flac",
            "use_tta": False,
            "draw_spectro": 0,
            "lora_checkpoint_peft": "",
            "lora_checkpoint_loralib": "",
            "verbose": True,
        }
    ]


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"valid_path": []}, "valid_path"),
        ({"device_ids": []}, "device_ids"),
        ({"device_ids": [-1]}, "negative"),
        ({"device_ids": [0, 0]}, "unique"),
        ({"metrics": []}, "metrics"),
        ({"metrics": ["not-a-metric"]}, "Unsupported"),
        ({"draw_spectro": -1}, "draw_spectro"),
        ({"extension": ""}, "extension"),
    ],
)
def test_validation_rejects_invalid_public_arguments(kwargs, message):
    values = {
        "config_path": "model.yaml",
        "checkpoint_path": "model.ckpt",
        "valid_path": "validation",
    }
    values.update(kwargs)
    with pytest.raises((TypeError, ValueError), match=message):
        valid(**values)


def test_validation_config_translates_defaults():
    config = ValidationConfig("model.yaml", "model.ckpt", "validation")

    assert config.to_legacy_args() == {
        "config_path": "model.yaml",
        "start_check_point": "model.ckpt",
        "valid_path": ["validation"],
        "model_type": "mdx23c",
        "device_ids": [0],
        "force_cpu": False,
        "metrics": ["sdr"],
        "store_dir": "",
        "extension": "wav",
        "use_tta": False,
        "draw_spectro": 0,
        "lora_checkpoint_peft": "",
        "lora_checkpoint_loralib": "",
        "verbose": False,
    }


def test_validation_missing_dependency_has_install_hint(monkeypatch):
    error = ModuleNotFoundError("No module named 'auraloss'", name="auraloss")

    def fail_import(name, package=None):
        raise error

    monkeypatch.setattr(validation_module, "import_module", fail_import)

    with pytest.raises(OptionalDependencyError) as caught:
        valid(
            model_type="bs_roformer",
            config_path="model.yaml",
            checkpoint_path="model.ckpt",
            valid_path="validation",
        )

    assert caught.value.dependency == "auraloss"
    assert 'pip install "msst[validation,bs-roformer]"' in str(caught.value)


def test_validation_rejects_invalid_runtime_result(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "msst._validation",
        SimpleNamespace(check_validation=lambda args: None),
    )

    with pytest.raises(RuntimeError, match="invalid result"):
        valid(
            config_path="model.yaml",
            checkpoint_path="model.ckpt",
            valid_path="validation",
        )
