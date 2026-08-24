"""Checkpoint-format metadata shared by training and inference."""

from __future__ import annotations

from typing import Any

from ._version import __version__


CHECKPOINT_FORMAT_VERSION = 1


def build_checkpoint_metadata(
    *,
    model_type: str,
    instruments: list[str] | tuple[str, ...],
    target_instrument: str | None = None,
) -> dict[str, Any]:
    """Return portable metadata for a newly written MSST checkpoint."""

    metadata: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "msst_version": __version__,
        "model_type": str(model_type),
        "target_stem_count": 1 if target_instrument else len(instruments),
    }
    if target_instrument:
        metadata["target_instrument"] = str(target_instrument)
    return metadata


__all__ = ["CHECKPOINT_FORMAT_VERSION", "build_checkpoint_metadata"]
