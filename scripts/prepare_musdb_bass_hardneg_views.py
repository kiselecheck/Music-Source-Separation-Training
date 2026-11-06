#!/usr/bin/env python3
"""Create symlink-only ordinary-bass negatives from the MUSDB train/test split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path


MIXTURE_SOURCES = ("bass", "drums", "other", "vocals")
TARGET_NAMES = ("synth_bass.wav", "synth_bass.flac")
PLAN_BUCKETS = (
    "positive_train",
    "hard_positive_train",
    "hard_negative_train",
    "hard_positive_valid",
    "hard_negative_valid",
)


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()[:10]


def normalized_track_name(value: str) -> str:
    value = value.rsplit("/", 1)[-1].lower()
    value = re.sub(r"_[0-9a-f]{10}$", "", value)
    prefixes = (
        r"^clear_v\d+_dataset_karaoke_version_small_",
        r"^clear_v\d+_dataset_extreememusic_sorted_",
        r"^clear_v\d+_dataset_alibi_extracted_reduced_",
        r"^clear_v\d+_isolated_tracks_dataset_\d+_",
    )
    for prefix in prefixes:
        value = re.sub(prefix, "", value)
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def synth_plan_track_names(plan_path: Path) -> set[str]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    names = set()
    for item in plan["stems"].values():
        for bucket in PLAN_BUCKETS:
            for row in item.get(bucket, []):
                names.add(normalized_track_name(row["mixture_key"]))
    return names


def discover_tracks(split_root: Path) -> list[Path]:
    tracks = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not tracks:
        raise ValueError(f"No MUSDB tracks found in {split_root}")
    for track in tracks:
        names = {entry.name for entry in track.iterdir() if entry.is_file()}
        missing = [f"{stem}.wav" for stem in MIXTURE_SOURCES if f"{stem}.wav" not in names]
        if missing:
            raise FileNotFoundError(f"{track} is missing required MUSDB stems: {missing}")
        if any(name in names for name in TARGET_NAMES):
            raise ValueError(f"MUSDB negative unexpectedly has synth_bass target: {track}")
    return tracks


def make_views(tracks: list[Path], output: Path, split: str, repeats: int) -> list[dict[str, str]]:
    output.mkdir(parents=True)
    manifest = []
    for track_index, track in enumerate(tracks):
        source_group = f"musdb/{split}/{track.name}"
        for repeat in range(repeats):
            link_name = f"{track_index:04d}_r{repeat:02d}_{digest(source_group)}"
            link = output / link_name
            link.symlink_to(track, target_is_directory=True)
            manifest.append(
                {
                    "view": link_name,
                    "source_group": source_group,
                    "track_dir": str(track),
                }
            )
    return manifest


def validate_views(view_root: Path, train_manifest: list[dict[str, str]], valid_manifest: list[dict[str, str]]) -> None:
    for split, manifest in (("train", train_manifest), ("valid", valid_manifest)):
        split_root = view_root / split
        actual_names = {path.name for path in split_root.iterdir()}
        expected_names = {row["view"] for row in manifest}
        if actual_names != expected_names:
            raise ValueError(f"{split}: expected {len(expected_names)} links, found {len(actual_names)} entries")
        for row in manifest:
            link = split_root / row["view"]
            if link.resolve(strict=True) != Path(row["track_dir"]).resolve(strict=True):
                raise ValueError(f"Invalid {split} symlink: {link}")

    train_groups = {row["source_group"] for row in train_manifest}
    valid_groups = {row["source_group"] for row in valid_manifest}
    if train_groups & valid_groups:
        raise ValueError("MUSDB train/test source groups overlap")
    if len(valid_groups) != len(valid_manifest):
        raise ValueError("MUSDB validation contains duplicate source groups")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musdb-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--synth-plan", type=Path, required=True)
    parser.add_argument("--train-repeats", type=int, default=20)
    args = parser.parse_args()
    if args.train_repeats < 1:
        parser.error("--train-repeats must be at least 1")
    if args.output_root.exists():
        raise FileExistsError(f"Output already exists: {args.output_root}")

    train_tracks = discover_tracks(args.musdb_root / "train")
    valid_tracks = discover_tracks(args.musdb_root / "test")
    if {p.name for p in train_tracks} & {p.name for p in valid_tracks}:
        raise ValueError("MUSDB train/test track names overlap")
    synth_names = synth_plan_track_names(args.synth_plan)
    overlap = sorted(
        {
            track.name
            for track in train_tracks + valid_tracks
            if normalized_track_name(track.name) in synth_names
        }
    )
    if overlap:
        train_tracks = [
            track for track in train_tracks
            if normalized_track_name(track.name) not in synth_names
        ]
        valid_tracks = [
            track for track in valid_tracks
            if normalized_track_name(track.name) not in synth_names
        ]
    if not train_tracks or not valid_tracks:
        raise ValueError("No MUSDB train/test tracks remain after cross-dataset overlap filtering")

    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{args.output_root.name}.staging-",
            dir=args.output_root.parent,
        )
    )
    try:
        train_manifest = make_views(
            train_tracks, staging / "train", "train", args.train_repeats
        )
        valid_manifest = make_views(valid_tracks, staging / "valid", "test", 1)
        validate_views(staging, train_manifest, valid_manifest)
        report = {
            "status": "passed",
            "source": "MUSDB official train/test split",
            "target": "synth_bass absent; ordinary bass.wav present",
            "mixture_sources": list(MIXTURE_SOURCES),
            "storage": "directory symlinks only; no audio copied",
            "train_unique_tracks": len(train_tracks),
            "train_sampling_weight": len(train_manifest),
            "train_repeats_per_track": args.train_repeats,
            "valid_unique_tracks": len(valid_tracks),
            "train_valid_source_group_overlap": 0,
            "normalized_title_overlap_with_synth_plan_excluded": len(overlap),
            "excluded_overlap_examples": overlap[:20],
        }
        (staging / "manifest.json").write_text(
            json.dumps(
                {"train": train_manifest, "valid": valid_manifest},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (staging / "validation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        os.replace(staging, args.output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Created ordinary-bass symlink views at {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
