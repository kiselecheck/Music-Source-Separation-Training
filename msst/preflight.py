"""Training preflight checks with safe and real-step modes."""

from __future__ import annotations

import argparse
import os
import tempfile
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Sequence

PathValue = str | PathLike[str]
CheckMode = Literal["safe", "full"]
CheckStatus = Literal["pass", "warning", "error"]


@dataclass(frozen=True)
class CheckItem:
    """One preflight finding."""

    name: str
    status: CheckStatus
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class CheckReport:
    """Result of :func:`check`."""

    mode: CheckMode
    items: tuple[CheckItem, ...]
    peak_allocated_gib: float | None = None
    peak_reserved_gib: float | None = None

    @property
    def ok(self) -> bool:
        return all(item.status != "error" for item in self.items)

    @property
    def errors(self) -> tuple[CheckItem, ...]:
        return tuple(item for item in self.items if item.status == "error")

    @property
    def warnings(self) -> tuple[CheckItem, ...]:
        return tuple(item for item in self.items if item.status == "warning")

    def raise_for_errors(self) -> None:
        if self.errors:
            messages = "; ".join(
                f"{item.name}: {item.message}" for item in self.errors
            )
            raise RuntimeError(f"MSST preflight failed: {messages}")


def _paths(value: PathValue | Sequence[PathValue] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, PathLike)):
        return [str(value)]
    return [str(path) for path in value]


def _writable_parent(path: Path) -> Path | None:
    candidate = path.expanduser().resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    if candidate.exists() and candidate.is_dir() and os.access(candidate, os.W_OK):
        return candidate
    return None


def check(
    *,
    config_path: PathValue,
    data_path: PathValue | Sequence[PathValue],
    results_path: PathValue,
    valid_path: PathValue | Sequence[PathValue] | None = None,
    model_type: str = "mdx23c",
    checkpoint_path: PathValue | None = None,
    device_ids: int | Sequence[int] = (0,),
    dataset_type: int = 1,
    mode: CheckMode = "safe",
    memory_headroom: float = 0.15,
    **options: Any,
) -> CheckReport:
    """Check whether a training run is ready to start.

    ``mode="safe"`` validates paths, config, dataset audio, optional model
    dependencies, checkpoint compatibility, CUDA visibility, and a conservative
    lower-bound memory estimate. It does not execute forward or backward.

    ``mode="full"`` performs every safe check and then runs one real batch
    through the canonical training forward, loss, backward, and one in-memory
    optimizer step. It never changes or writes a checkpoint. Peak VRAM from
    that representative step is reported, but no finite check can guarantee
    memory usage for every random augmentation or later batch.

    Args:
        config_path: Training YAML configuration.
        data_path: Training dataset root or roots.
        results_path: Planned training output directory.
        valid_path: Optional validation dataset root or roots.
        model_type: Model family key.
        checkpoint_path: Optional initial checkpoint.
        device_ids: CUDA device index or indices.
        dataset_type: MSST dataset layout, from 1 through 8.
        mode: Exactly ``"safe"`` or ``"full"``.
        memory_headroom: Required free fraction after a full GPU train-step.
        **options: Additional options accepted by ``msst train``.

    Returns:
        A report whose ``ok`` property is false when any check failed.
    """

    if mode not in ("safe", "full"):
        raise ValueError("mode must be 'safe' or 'full'")
    if not 0 <= memory_headroom < 1:
        raise ValueError("memory_headroom must be in the range [0, 1)")
    normalized_devices = [device_ids] if isinstance(device_ids, int) else list(device_ids)
    if not normalized_devices:
        raise ValueError("device_ids must contain at least one device")
    if any(not isinstance(value, int) or isinstance(value, bool) for value in normalized_devices):
        raise TypeError("device_ids must contain integer CUDA indices")
    if any(value < 0 for value in normalized_devices):
        raise ValueError("device_ids cannot contain negative CUDA indices")
    if len(set(normalized_devices)) != len(normalized_devices):
        raise ValueError("device_ids must contain unique CUDA indices")
    if dataset_type not in range(1, 9):
        raise ValueError("dataset_type must be between 1 and 8")

    items: list[CheckItem] = []

    def record(
        name: str,
        status: CheckStatus,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        items.append(CheckItem(name, status, message, details))

    config_file = Path(config_path).expanduser()
    dataset_roots = [Path(path).expanduser() for path in _paths(data_path)]
    validation_roots = [Path(path).expanduser() for path in _paths(valid_path)]
    checkpoint_file = Path(checkpoint_path).expanduser() if checkpoint_path else None
    results_directory = Path(results_path).expanduser()

    if config_file.is_file():
        record("config_path", "pass", f"Found {config_file}")
    else:
        record("config_path", "error", f"File does not exist: {config_file}")
    missing_data = [str(path) for path in dataset_roots if not path.exists()]
    if dataset_roots and not missing_data:
        record("data_path", "pass", f"Found {len(dataset_roots)} dataset root(s)")
    else:
        message = "data_path is empty" if not dataset_roots else "Missing: " + ", ".join(missing_data)
        record("data_path", "error", message)
    missing_validation = [str(path) for path in validation_roots if not path.exists()]
    if missing_validation:
        record("valid_path", "error", "Missing: " + ", ".join(missing_validation))
    elif validation_roots:
        record("valid_path", "pass", f"Found {len(validation_roots)} validation root(s)")
    else:
        record(
            "valid_path",
            "error",
            "No validation path was provided; the training loop validates after every epoch",
        )
    if checkpoint_file is None:
        record("checkpoint", "warning", "No initial checkpoint was provided")
    elif checkpoint_file.is_file():
        record("checkpoint_path", "pass", f"Found {checkpoint_file}")
    else:
        record("checkpoint_path", "error", f"File does not exist: {checkpoint_file}")
    writable_parent = _writable_parent(results_directory)
    if writable_parent is None:
        record("results_path", "error", f"No writable parent for {results_directory}")
    else:
        record("results_path", "pass", f"Writable via {writable_parent}")

    if any(item.status == "error" for item in items):
        return CheckReport(mode, tuple(items))

    import torch

    runtime_args: argparse.Namespace | None = None
    model = None
    config = None
    try:
        from .utils.settings import parse_args_train

        payload = {
            "config_path": str(config_file),
            "data_path": [str(path) for path in dataset_roots],
            "valid_path": [str(path) for path in validation_roots] or None,
            "results_path": str(results_directory),
            "model_type": model_type,
            "start_check_point": str(checkpoint_file) if checkpoint_file else "",
            "device_ids": normalized_devices,
            "dataset_type": dataset_type,
            **options,
        }
        parsed_args = parse_args_train(payload)
        runtime_args = parsed_args
        record("arguments", "pass", "Training arguments use the public CLI contract")
        if parsed_args.persistent_workers and parsed_args.num_workers <= 0:
            record(
                "dataloader",
                "error",
                "persistent_workers requires num_workers greater than zero",
            )
        elif parsed_args.prefetch_factor is not None and parsed_args.num_workers <= 0:
            record(
                "dataloader",
                "error",
                "prefetch_factor requires num_workers greater than zero",
            )
        else:
            record("dataloader", "pass", "DataLoader worker options are compatible")
    except Exception as error:
        record("arguments", "error", str(error))

    if runtime_args is not None:
        try:
            from .utils.settings import get_model_from_config

            model, config = get_model_from_config(
                runtime_args.model_type,
                runtime_args.config_path,
                pretrained=not bool(runtime_args.start_check_point),
            )
            if 'model_type' in config.training:
                runtime_args.model_type = config.training.model_type
            record(
                "model",
                "pass",
                f"Constructed {type(model).__name__} on CPU",
                {"parameters": sum(parameter.numel() for parameter in model.parameters())},
            )
        except Exception as error:
            record("model", "error", f"{type(error).__name__}: {error}")

    if model is not None and runtime_args is not None and checkpoint_file is not None:
        try:
            from .utils.model_utils import load_start_checkpoint

            checkpoint = torch.load(
                checkpoint_file,
                weights_only=False,
                map_location="cpu",
            )
            load_start_checkpoint(runtime_args, model, checkpoint, type_="train")
            record("checkpoint_compatibility", "pass", "Checkpoint loaded into the model")
        except Exception as error:
            record(
                "checkpoint_compatibility",
                "error",
                f"{type(error).__name__}: {error}",
            )

    if config is not None:
        try:
            from .utils.dataset import MSSDataset

            with tempfile.TemporaryDirectory(prefix="msst-check-metadata-") as temporary:
                metadata = MSSDataset.collect_metadata(
                    config,
                    [str(path) for path in dataset_roots],
                    metadata_path=str(Path(temporary) / "metadata.pkl"),
                    dataset_type=dataset_type,
                    verbose=False,
                    reuse=False,
                )
            track_count = (
                sum(len(value) for value in metadata.values())
                if isinstance(metadata, dict)
                else len(metadata)
            )
            if track_count == 0:
                raise ValueError("No usable tracks found")
            record("dataset", "pass", f"Read metadata for {track_count} track entries")
        except Exception as error:
            record("dataset", "error", f"{type(error).__name__}: {error}")

        if validation_roots:
            try:
                import soundfile as sf

                inference_config = getattr(config, "inference", {})
                mixture_names: list[str] = []
                configured_names = getattr(
                    inference_config, "mixture_names", None
                )
                if configured_names:
                    if isinstance(configured_names, (list, tuple)):
                        mixture_names.extend(map(str, configured_names))
                    else:
                        mixture_names.append(str(configured_names))
                input_name = getattr(inference_config, "input_name", None)
                if input_name:
                    mixture_names.insert(0, str(input_name))
                if not mixture_names:
                    mixture_names = ["mixture"]
                mixture_files: list[Path] = []
                for root in validation_roots:
                    for mixture_name in dict.fromkeys(mixture_names):
                        mixture_files.extend(root.rglob(f"{mixture_name}.wav"))
                        mixture_files.extend(root.rglob(f"{mixture_name}.flac"))
                mixture_files = list(dict.fromkeys(mixture_files))
                if not mixture_files:
                    raise ValueError(
                        "No validation mixtures were found for names: "
                        + ", ".join(dict.fromkeys(mixture_names))
                    )
                instruments = list(config.training.instruments)
                other_fix = bool(getattr(config.training, "other_fix", False))
                for audio_path in mixture_files:
                    sf.info(audio_path)
                    for instrument in instruments:
                        reference_name = (
                            "vocals"
                            if instrument == "other" and other_fix
                            else str(instrument)
                        )
                        reference_candidates = [
                            audio_path.parent / f"{reference_name}.wav",
                            audio_path.parent / f"{reference_name}.flac",
                        ]
                        reference = next(
                            (path for path in reference_candidates if path.is_file()),
                            None,
                        )
                        if reference is None:
                            raise FileNotFoundError(
                                f"Reference '{reference_name}' is missing next to {audio_path}"
                            )
                        sf.info(reference)
                record(
                    "validation_dataset",
                    "pass",
                    f"Decoded metadata for {len(mixture_files)} validation mixture(s)",
                )
            except Exception as error:
                record(
                    "validation_dataset",
                    "error",
                    f"{type(error).__name__}: {error}",
                )

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        invalid = [value for value in normalized_devices if value >= device_count]
        if invalid:
            record(
                "cuda",
                "error",
                f"Unavailable device ids {invalid}; visible device count is {device_count}",
            )
        else:
            device_details = []
            for device_id in normalized_devices:
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
                device_details.append(
                    {
                        "id": device_id,
                        "name": torch.cuda.get_device_name(device_id),
                        "free_gib": free_bytes / 1024 ** 3,
                        "total_gib": total_bytes / 1024 ** 3,
                    }
                )
            record("cuda", "pass", f"Found {len(device_details)} requested GPU(s)", {"devices": device_details})
    else:
        record("cuda", "error" if mode == "full" else "warning", "CUDA is not available")

    if model is not None:
        parameter_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        )
        conservative_bytes = parameter_bytes * 4
        status: CheckStatus = "pass"
        message = (
            f"Parameters use {parameter_bytes / 1024 ** 3:.2f} GiB; "
            f"minimum train-state estimate is {conservative_bytes / 1024 ** 3:.2f} GiB"
        )
        if torch.cuda.is_available() and normalized_devices[0] < torch.cuda.device_count():
            free_bytes, _ = torch.cuda.mem_get_info(normalized_devices[0])
            if conservative_bytes > free_bytes * (1 - memory_headroom):
                status = "error"
                message += ", which exceeds the requested free-memory budget"
        record("memory_estimate", status, message)

    if mode == "safe" or any(item.status == "error" for item in items):
        return CheckReport(mode, tuple(items))

    peak_allocated = None
    peak_reserved = None
    try:
        assert model is not None and config is not None and runtime_args is not None
        from ._training import compute_training_loss
        from .utils.dataset import prepare_data
        from .utils.losses import choice_loss
        from .utils.model_utils import (
            get_lora,
            get_optimizer,
            initialize_model_and_device,
        )

        with tempfile.TemporaryDirectory(prefix="msst-check-full-") as temporary:
            runtime_args.results_path = temporary
            runtime_args.num_workers = 0
            runtime_args.persistent_workers = False
            runtime_args.prefetch_factor = None
            batch_size = int(config.training.batch_size) * len(normalized_devices)
            train_loader = prepare_data(config, runtime_args, batch_size)
            data = next(iter(train_loader))
            model = get_lora(runtime_args, config, model)
            if runtime_args.freeze_layers is not None:
                for name, parameter in model.named_parameters():
                    if any(
                        name.startswith(prefix)
                        for prefix in runtime_args.freeze_layers
                    ):
                        parameter.requires_grad = False
            device, model = initialize_model_and_device(model, normalized_devices)
            model.train()
            optimizer = get_optimizer(config, model)
            amp_enabled = bool(
                getattr(config.training, "use_amp", True)
                and device.type == "cuda"
            )
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            optimizer.zero_grad(set_to_none=True)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            loss, _ = compute_training_loss(
                model,
                config,
                runtime_args,
                data,
                device,
                normalized_devices,
                choice_loss(runtime_args, config),
                amp_enabled,
            )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_clip = getattr(config.training, "grad_clip", 0)
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            if device.type == "cuda":
                peak_allocated = torch.cuda.max_memory_allocated(device) / 1024 ** 3
                peak_reserved = torch.cuda.max_memory_reserved(device) / 1024 ** 3
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                remaining_fraction = free_bytes / total_bytes
                status = "pass" if remaining_fraction >= memory_headroom else "error"
                message = (
                    f"Forward, backward, and optimizer step completed; peak allocated "
                    f"{peak_allocated:.2f} GiB, peak reserved {peak_reserved:.2f} GiB, "
                    f"free after step {remaining_fraction:.1%}"
                )
                if status == "error":
                    message += f" (required headroom: {memory_headroom:.1%})"
            else:
                status = "pass"
                message = "Forward, backward, and optimizer step completed on CPU"
            record("train_step", status, message)
            optimizer.zero_grad(set_to_none=True)
    except torch.OutOfMemoryError as error:
        record("train_step", "error", f"CUDA out of memory: {error}")
    except Exception as error:
        record("train_step", "error", f"{type(error).__name__}: {error}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return CheckReport(mode, tuple(items), peak_allocated, peak_reserved)


__all__ = ["CheckItem", "CheckMode", "CheckReport", "check"]
