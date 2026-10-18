"""Read an explicit, keyed selection of full validation tracks."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .errors import ConfigurationError


@dataclass(frozen=True, slots=True)
class ValidationTrack:
    key: str
    mixture_path: str
    group: str = ""


def load_validation_manifest(
    manifest_path: str | os.PathLike[str],
    valid_roots: Sequence[str | os.PathLike[str]],
) -> dict[str, ValidationTrack]:
    """Return tracks keyed by mixture path, preserving CSV order.

    Relative mixture paths are resolved against the CSV file. Every listed
    mixture must live beneath one of ``valid_roots`` so a manifest cannot
    silently evaluate a different dataset.
    """

    source = Path(manifest_path).absolute()
    if not source.is_file():
        raise ConfigurationError(f"Validation manifest not found: {source}")
    roots = [Path(os.path.abspath(root)) for root in valid_roots]
    if not roots:
        raise ConfigurationError("Validation manifest requires at least one valid_path root")

    tracks: dict[str, ValidationTrack] = {}
    seen_keys: set[str] = set()
    with source.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames is None or not {"key", "mixture_path"}.issubset(reader.fieldnames):
            raise ConfigurationError(
                f"Validation manifest {source} needs 'key' and 'mixture_path' columns"
            )
        for line_number, row in enumerate(reader, start=2):
            key = (row.get("key") or "").strip()
            raw_path = (row.get("mixture_path") or "").strip()
            if not key or not raw_path:
                raise ConfigurationError(
                    f"Validation manifest {source}, line {line_number}: "
                    "key and mixture_path must be non-empty"
                )
            if key in seen_keys:
                raise ConfigurationError(
                    f"Validation manifest {source}, line {line_number}: duplicate key {key!r}"
                )
            candidate = Path(raw_path)
            if not candidate.is_absolute():
                candidate = source.parent / candidate
            candidate = Path(os.path.abspath(candidate))
            if not any(candidate.is_relative_to(root) for root in roots):
                raise ConfigurationError(
                    f"Validation manifest {source}, line {line_number}: "
                    f"{candidate} is outside valid_path roots"
                )
            if not candidate.is_file() and not candidate.is_dir():
                raise ConfigurationError(
                    f"Validation manifest {source}, line {line_number}: "
                    f"mixture not found (expected an audio file or track directory): {candidate}"
                )
            path = str(candidate)
            if path in tracks:
                raise ConfigurationError(
                    f"Validation manifest {source}, line {line_number}: "
                    f"mixture path occurs more than once: {candidate}"
                )
            seen_keys.add(key)
            tracks[path] = ValidationTrack(
                key=key,
                mixture_path=path,
                group=(row.get("group") or "").strip(),
            )
    if not tracks:
        raise ConfigurationError(f"Validation manifest {source} contains no tracks")
    return tracks
