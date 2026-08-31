"""Public dataset-metadata API used by both scripts and training."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any

from ._version import __version__

PathValue = str | PathLike[str]


@dataclass(frozen=True)
class MetadataResult:
    """Description of a generated MSST metadata cache."""

    path: Path
    manifest_path: Path
    dataset_type: int
    track_count: int
    fingerprint: str
    cache_was_present: bool


def _metadata_entries(metadata: Any) -> list[tuple[str, int]]:
    if isinstance(metadata, dict):
        entries = [item for values in metadata.values() for item in values]
    else:
        entries = list(metadata)
    return sorted((str(path), int(length)) for path, length in entries)


def _fingerprint(metadata: Any) -> str:
    payload = json.dumps(
        _metadata_entries(metadata),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_metadata(
    *,
    config_path: PathValue,
    data_path: PathValue | list[PathValue] | tuple[PathValue, ...],
    results_path: PathValue | None = None,
    metadata_path: PathValue | None = None,
    model_type: str = "mdx23c",
    dataset_type: int = 1,
    num_workers: int | None = None,
    refresh: bool = False,
    verbose: bool = True,
) -> MetadataResult:
    """Scan a training dataset and write the metadata cache used by training.

    Args:
        config_path: Model/training YAML configuration.
        data_path: One dataset root or a sequence of roots.
        results_path: Directory receiving ``metadata_<type>.pkl`` when
            ``metadata_path`` is omitted.
        metadata_path: Explicit destination for the pickle cache.
        model_type: Model family used to select the config loader.
        dataset_type: MSST dataset layout, from 1 through 8.
        num_workers: Number of metadata-reading workers. ``None`` uses the
            configuration or the runtime default.
        refresh: Ignore an existing cache and read every track again.
        verbose: Print scan progress.

    Returns:
        A typed result containing paths, track count, and a content hash.

    Raises:
        FileNotFoundError: A config or dataset root does not exist.
        ValueError: Arguments or dataset contents are invalid.
    """

    config_file = Path(config_path).expanduser()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {config_file}")
    if dataset_type not in range(1, 9):
        raise ValueError("dataset_type must be between 1 and 8")
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be at least 1")

    raw_paths = [data_path] if isinstance(data_path, (str, PathLike)) else list(data_path)
    if not raw_paths:
        raise ValueError("data_path must contain at least one dataset path")
    dataset_roots = [Path(path).expanduser() for path in raw_paths]
    missing_roots = [path for path in dataset_roots if not path.exists()]
    if missing_roots:
        raise FileNotFoundError(
            "Dataset path does not exist: " + ", ".join(map(str, missing_roots))
        )

    if metadata_path is None:
        if results_path is None:
            raise ValueError("results_path or metadata_path is required")
        output_path = Path(results_path).expanduser() / f"metadata_{dataset_type}.pkl"
    else:
        output_path = Path(metadata_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_was_present = output_path.is_file() and not refresh

    from .utils.dataset import MSSDataset
    from .utils.settings import load_config

    config = load_config(model_type, str(config_file))
    if num_workers is not None:
        config.training.read_metadata_procs = num_workers
    metadata = MSSDataset.collect_metadata(
        config,
        [str(path) for path in dataset_roots],
        metadata_path=str(output_path),
        dataset_type=dataset_type,
        verbose=verbose,
        reuse=not refresh,
    )
    collector_manifest_path = output_path.with_suffix(".json")
    collector_manifest = json.loads(
        collector_manifest_path.read_text(encoding="utf-8")
    )
    entries = _metadata_entries(metadata)
    if not entries:
        raise ValueError(
            "No usable tracks were found. Check data_path, dataset_type, and "
            "training.instruments in the config."
        )

    fingerprint = _fingerprint(metadata)
    manifest_path = output_path.with_suffix(".json")
    manifest = {
        "schema_version": 1,
        "msst_version": __version__,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_file.resolve()),
        "config_sha256": hashlib.sha256(config_file.read_bytes()).hexdigest(),
        "data_paths": [str(path.resolve()) for path in dataset_roots],
        "dataset_type": dataset_type,
        "track_count": len(entries),
        "fingerprint": fingerprint,
        "metadata_signature": collector_manifest["metadata_signature"],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return MetadataResult(
        path=output_path.resolve(),
        manifest_path=manifest_path.resolve(),
        dataset_type=dataset_type,
        track_count=len(entries),
        fingerprint=fingerprint,
        cache_was_present=cache_was_present,
    )


__all__ = ["MetadataResult", "build_metadata"]
