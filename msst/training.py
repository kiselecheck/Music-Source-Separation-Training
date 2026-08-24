"""Library-oriented facade for the existing training implementation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from importlib import import_module
from os import PathLike
from typing import Any

from .errors import optional_dependency_error

PathValue = str | PathLike[str]


def _paths(value: PathValue | Sequence[PathValue] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, PathLike)):
        return [str(value)]
    return [str(path) for path in value]


def train(
    *,
    config_path: PathValue,
    results_path: PathValue,
    data_path: PathValue | Sequence[PathValue],
    valid_path: PathValue | Sequence[PathValue] | None = None,
    model_type: str = "mdx23c",
    checkpoint_path: PathValue | None = None,
    device_ids: int | Sequence[int] = (0,),
    **options: Any,
) -> Any:
    """Train an MSST model using conventional keyword arguments.

    Less common training flags can be supplied through ``options`` using the
    same names as the command-line arguments.
    """

    if "start_check_point" in options and checkpoint_path is not None:
        raise TypeError(
            "use either 'checkpoint_path' or 'start_check_point', not both"
        )

    if not str(config_path).strip():
        raise ValueError("config_path is required")
    if not str(results_path).strip():
        raise ValueError("results_path is required")

    if isinstance(device_ids, int):
        normalized_device_ids = [device_ids]
    else:
        normalized_device_ids = list(device_ids)
    if not normalized_device_ids:
        raise ValueError("device_ids must contain at least one device")
    if any(
        not isinstance(device_id, int) or isinstance(device_id, bool)
        for device_id in normalized_device_ids
    ):
        raise TypeError("device_ids must contain integer CUDA indices")
    if any(device_id < 0 for device_id in normalized_device_ids):
        raise ValueError("device_ids cannot contain negative CUDA indices")
    if len(set(normalized_device_ids)) != len(normalized_device_ids):
        raise ValueError("device_ids must contain unique CUDA indices")

    normalized_data_path = _paths(data_path)
    if not normalized_data_path:
        raise ValueError("data_path must contain at least one dataset path")
    normalized_valid_path = _paths(valid_path)

    args: dict[str, Any] = {
        "config_path": str(config_path),
        "results_path": str(results_path),
        "data_path": normalized_data_path,
        "valid_path": normalized_valid_path,
        "model_type": model_type,
        "device_ids": normalized_device_ids,
        **options,
    }
    if checkpoint_path is not None:
        args["start_check_point"] = str(checkpoint_path)
    return train_model(args)


def train_model(
    args: argparse.Namespace | Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> Any:
    """Compatibility entry point matching the existing ``train_model`` API."""

    if args is None:
        payload: argparse.Namespace | dict[str, Any] | None = (
            dict(kwargs) if kwargs else None
        )
    elif isinstance(args, argparse.Namespace):
        if kwargs:
            raise TypeError("keyword overrides cannot be combined with Namespace")
        payload = args
    elif isinstance(args, Mapping):
        payload = dict(args)
        payload.update(kwargs)
    else:
        raise TypeError("args must be an argparse.Namespace, a mapping, or None")

    try:
        legacy_train_model = import_module("._training", __package__).train_model
        return legacy_train_model(payload)
    except ModuleNotFoundError as error:
        model_type = "mdx23c"
        if isinstance(payload, argparse.Namespace):
            model_type = getattr(payload, "model_type", model_type)
        elif isinstance(payload, dict):
            model_type = payload.get("model_type", model_type)
        extra = f"train,{str(model_type).replace('_', '-')}"
        raise optional_dependency_error(
            error,
            extra=extra,
            feature="Model training",
        ) from error


__all__ = ["train", "train_model"]
