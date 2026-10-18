import os
from argparse import Namespace
from pathlib import Path

import numpy as np
import soundfile as sf
from ml_collections import ConfigDict

from msst._validation import _read_composed_mixture, get_mixture_paths
from scripts.prepare_musdb_bass_hardneg_views import (
    discover_tracks,
    make_views,
    normalized_track_name,
    validate_views,
)


def _write_audio(path: Path, value: float, frames: int = 4096) -> None:
    sf.write(path, np.full((frames, 2), value, dtype=np.float32), 44100, subtype="FLOAT")


def _write_musdb_track(track: Path) -> None:
    track.mkdir(parents=True)
    for instrument, value in {"bass": 0.1, "drums": 0.2, "other": 0.3, "vocals": 0.4}.items():
        _write_audio(track / f"{instrument}.wav", value)


def _config() -> ConfigDict:
    return ConfigDict(
        {
            "training": {
                "instruments": ["synth_bass", "other"],
                "target_instrument": "synth_bass",
                "single_target_from_mixture": True,
                "mixture_source_instruments": ["bass", "drums", "other", "vocals"],
            },
            "inference": {"mixture_names": ["mixture"]},
        }
    )


def test_validation_composes_a_mixture_for_component_only_tracks(tmp_path: Path) -> None:
    track = tmp_path / "ordinary_bass_track"
    _write_musdb_track(track)
    paths = get_mixture_paths(Namespace(valid_path=[str(tmp_path)]), False, _config(), "wav")

    assert paths == [str(track)]
    mixture, sample_rate = _read_composed_mixture(paths[0], _config())
    assert sample_rate == 44100
    assert mixture.shape == (2, 4096)
    np.testing.assert_allclose(mixture, 1.0, atol=1e-6)


def test_validation_follows_directory_symlink_views(tmp_path: Path) -> None:
    source_track = tmp_path / "source" / "ordinary_bass_track"
    _write_musdb_track(source_track)
    view_root = tmp_path / "views"
    view_root.mkdir()
    view = view_root / "00000_track"
    try:
        os.symlink(source_track, view, target_is_directory=True)
    except OSError as exc:
        import pytest

        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    paths = get_mixture_paths(Namespace(valid_path=[str(view_root)]), False, _config(), "wav")
    assert paths == [str(view)]


def test_musdb_views_are_symlinks_and_keep_official_splits_disjoint(tmp_path: Path) -> None:
    musdb_root = tmp_path / "musdb"
    train_root = musdb_root / "train"
    test_root = musdb_root / "test"
    train_track = train_root / "Example Artist - Bass Song"
    test_track = test_root / "Other Artist - Test Song"
    for track in (train_track, test_track):
        track.mkdir(parents=True)
        for instrument in ("bass", "drums", "other", "vocals"):
            (track / f"{instrument}.wav").touch()

    train_tracks = discover_tracks(train_root)
    test_tracks = discover_tracks(test_root)
    view_root = tmp_path / "views"
    view_root.mkdir()
    train_manifest = make_views(train_tracks, view_root / "train", "train", 3)
    test_manifest = make_views(test_tracks, view_root / "valid", "test", 1)
    validate_views(view_root, train_manifest, test_manifest)

    assert len(train_manifest) == 3
    assert all((view_root / "train" / row["view"]).is_symlink() for row in train_manifest)
    assert normalized_track_name(
        "synth_10stems_v1_1/clear_v2_dataset_karaoke_version_small_mu_too_bright_ab12cd34ef"
    ) == normalized_track_name("Mu - Too Bright")
