from pathlib import Path

import numpy as np
import soundfile as sf
from ml_collections import ConfigDict

from msst.utils.dataset import MSSDataset


def _write_audio(path: Path, value: float, frames: int = 4096) -> None:
    audio = np.full((frames, 2), value, dtype=np.float32)
    sf.write(path, audio, 44100, subtype="FLOAT")


def test_single_target_from_explicit_mixture_filters_sparse_tracks(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    silent_target = tmp_path / "silent_target"
    with_target = tmp_path / "with_target"
    without_target = tmp_path / "without_target"
    silent_target.mkdir()
    with_target.mkdir()
    without_target.mkdir()

    _write_audio(silent_target / "synth_bass.wav", 0.0)
    _write_audio(silent_target / "mixture.wav", 0.2)
    _write_audio(with_target / "synth_bass.wav", 0.1)
    _write_audio(with_target / "mixture.wav", 0.3)
    _write_audio(without_target / "synth_keys.wav", 0.2)
    _write_audio(without_target / "mixture.wav", 0.4)

    config = ConfigDict(
        {
            "audio": {"chunk_size": 2048, "min_mean_abs": 0.05, "sample_rate": 44100},
            "training": {
                "batch_size": 1,
                "num_steps": 1,
                "instruments": ["synth_bass", "other"],
                "target_instrument": "synth_bass",
                "single_target_from_mixture": True,
                "precompute_chunks": False,
                "read_metadata_procs": 1,
            },
            "augmentations": {"enable": False},
        }
    )

    dataset = MSSDataset(
        config,
        str(tmp_path),
        metadata_path=str(tmp_path / "metadata.pkl"),
        dataset_type=6,
        batch_size=1,
        verbose=False,
    )

    assert [Path(track).name for track, _ in dataset.metadata] == [
        "silent_target",
        "with_target",
    ]

    choices = iter(dataset.metadata)
    monkeypatch.setattr("msst.utils.dataset.random.choice", lambda _: next(choices))

    target, mixture = dataset[0]
    assert target.shape == (1, 2, 2048)
    assert mixture.shape == (2, 2048)
    np.testing.assert_allclose(target.numpy(), 0.1, atol=1e-6)
    np.testing.assert_allclose(mixture.numpy(), 0.3, atol=1e-6)
    assert "Attempts max!" not in capsys.readouterr().out

    dataset.chunks_metadata = [(str(with_target), 0)]
    dataset.do_chunks = True
    precomputed_target, precomputed_mixture = dataset[0]
    np.testing.assert_allclose(precomputed_target.numpy(), 0.1, atol=1e-6)
    np.testing.assert_allclose(precomputed_mixture.numpy(), 0.3, atol=1e-6)
