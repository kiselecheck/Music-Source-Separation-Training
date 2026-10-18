from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch
import wandb
import yaml
from ml_collections import ConfigDict

import msst
from examples.custom_model import TinySeparator
from msst._training import train_one_epoch
from msst._validation import valid
from msst.utils.settings import build_model_from_config


MODEL_NAME = "test_tiny_separator"


def tiny_config() -> ConfigDict:
    return ConfigDict(
        {
            "audio": {
                "num_channels": 2,
                "chunk_size": 64,
                "sample_rate": 8000,
            },
            "training": {
                "instruments": ["vocals", "other"],
                "target_instrument": None,
                "grad_clip": 0.0,
            },
            "inference": {
                "chunk_size": 64,
                "batch_size": 2,
                "num_overlap": 2,
                "normalize": False,
            },
        }
    )


@pytest.fixture
def registered_tiny_model():
    msst.unregister_model(MODEL_NAME)
    msst.register_model(MODEL_NAME, TinySeparator)
    yield
    msst.unregister_model(MODEL_NAME)


def test_custom_model_trains_one_epoch_and_runs_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    registered_tiny_model,
):
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    channels, samples = 2, 256
    vocals = (rng.standard_normal((samples, channels)) * 0.03).astype(np.float32)
    other = (rng.standard_normal((samples, channels)) * 0.03).astype(np.float32)
    mixture = vocals + other

    valid_path = tmp_path / "valid" / "toy_track"
    valid_path.mkdir(parents=True)
    sf.write(valid_path / "mixture.wav", mixture, 8000, subtype="FLOAT")
    sf.write(valid_path / "vocals.wav", vocals, 8000, subtype="FLOAT")
    sf.write(valid_path / "other.wav", other, 8000, subtype="FLOAT")

    config = tiny_config()
    model = build_model_from_config(MODEL_NAME, config)
    assert isinstance(model, TinySeparator)
    before = next(model.parameters()).detach().clone()

    target = torch.from_numpy(np.stack((vocals.T, other.T), axis=0))[None]
    mix_batch = torch.from_numpy(mixture.T)[None]
    args = SimpleNamespace(model_type=MODEL_NAME, use_standard_loss=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    monkeypatch.setattr(wandb, "log", lambda *args, **kwargs: None)
    all_losses = {}

    train_one_epoch(
        model=model,
        config=config,
        args=args,
        optimizer=optimizer,
        device=torch.device("cpu"),
        device_ids=[],
        epoch=0,
        use_amp=False,
        scaler=torch.amp.GradScaler("cuda", enabled=False),
        scheduler=SimpleNamespace(name="none"),
        gradient_accumulation_steps=1,
        train_loader=[(target, mix_batch)],
        multi_loss=lambda prediction, expected, _mixture, _mask: (
            prediction - expected
        ).abs().mean(),
        all_losses=all_losses,
    )

    assert len(all_losses["epoch_0"]) == 1
    assert np.isfinite(all_losses["epoch_0"][0])
    assert not torch.equal(before, next(model.parameters()).detach())

    manifest_path = tmp_path / "validation_tracks.csv"
    manifest_path.write_text(
        "key,mixture_path,group\n"
        "toy/toy_track,valid/toy_track/mixture.wav,smoke\n",
        encoding="utf-8",
    )
    track_csv_path = tmp_path / "track_metrics.csv"

    validation_args = SimpleNamespace(
        valid_path=[str(tmp_path / "valid")],
        validation_manifest=str(manifest_path),
        track_csv_path=str(track_csv_path),
        metrics=["sdr"],
        model_type=MODEL_NAME,
        use_tta=False,
        store_dir="",
        extension="wav",
        draw_spectro=0,
    )
    metric_avg, all_metrics = valid(
        model, validation_args, config, torch.device("cpu"), verbose=False
    )

    assert set(all_metrics["sdr"]) == {"vocals", "other"}
    assert all(len(all_metrics["sdr"][stem]) == 1 for stem in ("vocals", "other"))
    assert np.isfinite(metric_avg["sdr"])
    with track_csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert {row["instrument"] for row in rows} == {"vocals", "other"}
    assert {row["track_key"] for row in rows} == {"toy/toy_track"}
    assert {row["validation_group"] for row in rows} == {"smoke"}
    assert all(float(row["elapsed_seconds"]) > 0 for row in rows)

    config_path = tmp_path / "tiny_config.yaml"
    config_path.write_text(yaml.safe_dump(config.to_dict()), encoding="utf-8")
    checkpoint_path = tmp_path / "tiny_checkpoint.ckpt"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
    public_csv_path = tmp_path / "public_track_metrics.csv"

    result = msst.valid(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        valid_path=tmp_path / "valid",
        model_type=MODEL_NAME,
        force_cpu=True,
        metrics=("sdr",),
        validation_manifest=manifest_path,
        track_csv_path=public_csv_path,
    )

    assert np.isfinite(result.averages["sdr"])
    assert set(result.per_track["sdr"]) == {"vocals", "other"}
    with public_csv_path.open(newline="", encoding="utf-8") as csv_file:
        public_rows = list(csv.DictReader(csv_file))
    assert len(public_rows) == 2
    assert {row["checkpoint_path"] for row in public_rows} == {
        str(checkpoint_path)
    }
    assert {row["track_key"] for row in public_rows} == {"toy/toy_track"}
