import csv
from argparse import Namespace

import pytest
from ml_collections import ConfigDict

from msst._validation import get_mixture_paths
from msst._validation_manifest import load_validation_manifest


def _write_manifest(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["key", "mixture_path", "group"])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_selects_tracks_by_key_in_csv_order(tmp_path):
    valid_root = tmp_path / "valid"
    for name in ("song_a", "song_b"):
        folder = valid_root / name
        folder.mkdir(parents=True)
        (folder / "mixture.wav").write_bytes(b"audio")
    manifest = tmp_path / "selection.csv"
    _write_manifest(manifest, [
        {"key": "second", "mixture_path": "valid/song_b/mixture.wav", "group": "hard"},
        {"key": "first", "mixture_path": "valid/song_a/mixture.wav", "group": "regular"},
    ])

    tracks = load_validation_manifest(manifest, [valid_root])
    assert [track.key for track in tracks.values()] == ["second", "first"]
    assert [track.group for track in tracks.values()] == ["hard", "regular"]

    args = Namespace(valid_path=[str(valid_root)], validation_manifest=str(manifest))
    config = ConfigDict({"inference": {"batch_size": 1, "num_overlap": 1}})
    paths = get_mixture_paths(args, False, config, "wav")
    assert [str(path) for path in paths] == list(tracks)


def test_manifest_can_select_a_track_directory_for_a_composed_mixture(tmp_path):
    valid_root = tmp_path / "valid"
    track = valid_root / "song"
    track.mkdir(parents=True)
    manifest = tmp_path / "selection.csv"
    _write_manifest(manifest, [
        {"key": "components/song", "mixture_path": "valid/song", "group": "holdout"},
    ])

    tracks = load_validation_manifest(manifest, [valid_root])

    assert list(tracks) == [str(track)]
    assert next(iter(tracks.values())).group == "holdout"


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"key": "same", "mixture_path": "valid/song_a/mixture.wav"},
          {"key": "same", "mixture_path": "valid/song_b/mixture.wav"}], "duplicate key"),
        ([{"key": "bad", "mixture_path": "outside/mixture.wav"}], "outside valid_path"),
        ([{"key": "missing", "mixture_path": "valid/absent/mixture.wav"}], "mixture not found"),
    ],
)
def test_manifest_rejects_ambiguous_or_invalid_paths(tmp_path, rows, message):
    valid_root = tmp_path / "valid"
    valid_root.mkdir()
    for name in ("song_a", "song_b"):
        folder = valid_root / name
        folder.mkdir()
        (folder / "mixture.wav").write_bytes(b"audio")
    manifest = tmp_path / "selection.csv"
    _write_manifest(manifest, rows)

    with pytest.raises(ValueError, match=message):
        load_validation_manifest(manifest, [valid_root])
