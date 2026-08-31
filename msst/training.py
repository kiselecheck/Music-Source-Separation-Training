"""Library-oriented facade for the existing training implementation."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from importlib import import_module
from os import PathLike
from typing import Any, Literal

from .errors import optional_dependency_error
from .experiments import Experiment, tracked_run

PathValue = str | PathLike[str]
TrainingOptionValue = bool | int | float | str | Sequence[str] | None
Launcher = Literal["standard", "ddp", "accelerate"]


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
    valid_path: PathValue | Sequence[PathValue],
    model_type: str = "mdx23c",
    checkpoint_path: PathValue | None = None,
    device_ids: int | Sequence[int] = (0,),
    launcher: Launcher = "standard",
    experiment: PathValue | Experiment | None = None,
    run_name: str | None = None,
    **options: TrainingOptionValue,
) -> None:
    """Train an MSST model using conventional keyword arguments.

    Args:
        config_path: YAML configuration used to construct and train the model.
        results_path: Directory for checkpoints, history, and model metadata.
        data_path: One training dataset root or a sequence of roots.
        valid_path: Validation dataset root or roots. The training runtime
            evaluates after every epoch, so this argument is required.
        model_type: Model family key, for example ``"bs_roformer"``.
        checkpoint_path: Optional checkpoint used to initialize the model.
        device_ids: CUDA index or indices used for training.
        launcher: ``"standard"`` for CPU/single-GPU/DataParallel,
            ``"ddp"`` for DistributedDataParallel, or ``"accelerate"`` for
            the historical Hugging Face Accelerate runtime.
        experiment: Optional directory created by :func:`msst.init_experiment`.
            When set, MSST writes a self-contained run record below ``runs/``.
        run_name: Optional label included in the tracked run directory.
        **options: Less common training flags using the same names and value
            types as ``msst-train --help``.

    Returns:
        None. Training artifacts are written below ``results_path``.

    Raises:
        TypeError: An option name or value type is invalid.
        ValueError: A required path or device list is empty or invalid.
        MSSTError: Model construction, checkpoint loading, or training fails.
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
    if not normalized_valid_path:
        raise ValueError("valid_path must contain at least one validation path")

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
    tracking_parameters = dict(args)
    tracking_parameters.pop("start_check_point", None)
    tracking_parameters.update(
        {
            "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
            "launcher": launcher,
        }
    )
    with tracked_run(
        experiment,
        task="train",
        parameters=tracking_parameters,
        run_name=run_name,
    ) as run:
        train_model(args, launcher=launcher)
        if run is not None:
            run.add_artifacts(str(results_path))


def train_model(
    args: argparse.Namespace | Mapping[str, Any] | None = None,
    /,
    *,
    launcher: Launcher = "standard",
    **kwargs: Any,
) -> None:
    """Compatibility entry point with a selectable training launcher."""

    if launcher not in ("standard", "ddp", "accelerate"):
        raise ValueError(
            "launcher must be 'standard', 'ddp', or 'accelerate'"
        )

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
        module_name, function_name = {
            "standard": ("._training", "train_model"),
            "ddp": ("._train_ddp", "train_model_ddp"),
            "accelerate": ("._train_accelerate", "train_model"),
        }[launcher]
        legacy_train_model = getattr(
            import_module(module_name, __package__), function_name
        )
        legacy_train_model(payload)
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


__all__ = ["Launcher", "TrainingOptionValue", "train", "train_model"]
