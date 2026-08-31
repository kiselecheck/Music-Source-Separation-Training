"""Stable, typed validation API for evaluating trained separators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal

from .errors import optional_dependency_error

if TYPE_CHECKING:
    from .experiments import Experiment

PathValue = str | PathLike[str]
DeviceIds = int | Sequence[int]
ValidationMetric = Literal[
    "k_sdr",
    "sdr",
    "l1_freq",
    "si_sdr",
    "neg_log_wmse",
    "aura_stft",
    "aura_mrstft",
    "bleedless",
    "fullness",
    "l1_snr",
    "bleedless_mr",
    "fullness_mr",
]

SUPPORTED_VALIDATION_METRICS: frozenset[ValidationMetric] = frozenset(
    {
        "k_sdr",
        "sdr",
        "l1_freq",
        "si_sdr",
        "neg_log_wmse",
        "aura_stft",
        "aura_mrstft",
        "bleedless",
        "fullness",
        "l1_snr",
        "bleedless_mr",
        "fullness_mr",
    }
)


def _paths(value: PathValue | Sequence[PathValue]) -> list[str]:
    if isinstance(value, (str, PathLike)):
        paths = [str(value)]
    else:
        paths = [str(path) for path in value]
    if not paths or any(not path.strip() for path in paths):
        raise ValueError("valid_path must contain at least one non-empty path")
    return paths


def _device_ids(value: DeviceIds) -> list[int]:
    normalized = [value] if isinstance(value, int) else list(value)
    if not normalized:
        raise ValueError("device_ids must contain at least one device")
    if any(
        not isinstance(device_id, int) or isinstance(device_id, bool)
        for device_id in normalized
    ):
        raise TypeError("device_ids must contain integer CUDA indices")
    if any(device_id < 0 for device_id in normalized):
        raise ValueError("device_ids cannot contain negative CUDA indices")
    if len(set(normalized)) != len(normalized):
        raise ValueError("device_ids must contain unique CUDA indices")
    return normalized


def _metrics(
    value: ValidationMetric | Sequence[ValidationMetric],
) -> list[ValidationMetric]:
    normalized = [value] if isinstance(value, str) else list(value)
    if not normalized:
        raise ValueError("metrics must contain at least one metric")
    unknown = sorted(set(normalized) - SUPPORTED_VALIDATION_METRICS)
    if unknown:
        supported = ", ".join(sorted(SUPPORTED_VALIDATION_METRICS))
        raise ValueError(
            f"Unsupported validation metric(s): {', '.join(unknown)}. "
            f"Supported metrics: {supported}"
        )
    return list(dict.fromkeys(normalized))


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for one validation run.

    Paths may be strings or :class:`os.PathLike` objects. ``valid_path`` accepts
    one validation root or a sequence of roots. The model configuration decides
    which mixture filename and reference stems are expected in each track.
    """

    config_path: PathValue
    checkpoint_path: PathValue
    valid_path: PathValue | Sequence[PathValue]
    model_type: str = "mdx23c"
    device_ids: DeviceIds = 0
    force_cpu: bool = False
    metrics: ValidationMetric | Sequence[ValidationMetric] = ("sdr",)
    output_folder: PathValue | None = None
    extension: str = "wav"
    use_tta: bool = False
    draw_spectro: float = 0
    lora_checkpoint_peft: PathValue | None = None
    lora_checkpoint_loralib: PathValue | None = None
    verbose: bool = False

    def __post_init__(self) -> None:
        if not str(self.config_path).strip():
            raise ValueError("config_path is required")
        if not str(self.checkpoint_path).strip():
            raise ValueError("checkpoint_path is required")
        _paths(self.valid_path)
        _device_ids(self.device_ids)
        _metrics(self.metrics)
        if not self.extension.lstrip(".").strip():
            raise ValueError("extension cannot be empty")
        if self.draw_spectro < 0:
            raise ValueError("draw_spectro cannot be negative")

    def to_legacy_args(self) -> dict[str, Any]:
        """Translate public parameter names to the packaged validation runtime."""

        return {
            "config_path": str(self.config_path),
            "start_check_point": str(self.checkpoint_path),
            "valid_path": _paths(self.valid_path),
            "model_type": self.model_type,
            "device_ids": _device_ids(self.device_ids),
            "force_cpu": self.force_cpu,
            "metrics": _metrics(self.metrics),
            "store_dir": (
                "" if self.output_folder is None else str(self.output_folder)
            ),
            "extension": self.extension.lstrip(".").lower(),
            "use_tta": self.use_tta,
            "draw_spectro": self.draw_spectro,
            "lora_checkpoint_peft": (
                ""
                if self.lora_checkpoint_peft is None
                else str(self.lora_checkpoint_peft)
            ),
            "lora_checkpoint_loralib": (
                ""
                if self.lora_checkpoint_loralib is None
                else str(self.lora_checkpoint_loralib)
            ),
            "verbose": self.verbose,
        }


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Metrics returned by :func:`valid`.

    ``averages`` maps each requested metric to its aggregate score.
    ``per_track`` has the shape ``metric -> instrument -> scores`` and preserves
    the individual values used to compute those averages.
    """

    averages: dict[str, float]
    per_track: dict[str, dict[str, list[float]]]

    @classmethod
    def from_legacy(cls, value: Any) -> "ValidationResult":
        """Normalize the historical ``(averages, all_metrics)`` return value."""

        if not isinstance(value, tuple) or len(value) != 2:
            raise RuntimeError(
                "Validation runtime returned an invalid result; expected "
                "(averages, per_track_metrics)"
            )
        averages, per_track = value
        if not isinstance(averages, Mapping) or not isinstance(per_track, Mapping):
            raise RuntimeError("Validation runtime returned non-mapping metrics")

        normalized_per_track: dict[str, dict[str, list[float]]] = {}
        for metric_name, instruments in per_track.items():
            if not isinstance(instruments, Mapping):
                raise RuntimeError(
                    f"Validation metric {metric_name!r} has invalid stem data"
                )
            normalized_per_track[str(metric_name)] = {}
            for instrument, scores in instruments.items():
                values = scores.values() if isinstance(scores, Mapping) else scores
                normalized_per_track[str(metric_name)][str(instrument)] = [
                    float(score) for score in values
                ]

        return cls(
            averages={str(name): float(score) for name, score in averages.items()},
            per_track=normalized_per_track,
        )


def valid(
    *,
    config_path: PathValue,
    checkpoint_path: PathValue,
    valid_path: PathValue | Sequence[PathValue],
    model_type: str = "mdx23c",
    device_ids: DeviceIds = 0,
    force_cpu: bool = False,
    metrics: ValidationMetric | Sequence[ValidationMetric] = ("sdr",),
    output_folder: PathValue | None = None,
    extension: str = "wav",
    use_tta: bool = False,
    draw_spectro: float = 0,
    lora_checkpoint_peft: PathValue | None = None,
    lora_checkpoint_loralib: PathValue | None = None,
    verbose: bool = False,
    experiment: PathValue | "Experiment" | None = None,
    run_name: str | None = None,
) -> ValidationResult:
    """Evaluate a checkpoint against one or more validation datasets.

    Args:
        config_path: YAML configuration used to construct the model.
        checkpoint_path: Checkpoint containing the weights to evaluate.
        valid_path: Validation root or roots containing mixtures and references.
        model_type: Model family key, for example ``"bs_roformer"``.
        device_ids: CUDA index or indices. Only the first is used for a
            single-device run; multiple indices enable parallel validation.
        force_cpu: Run on CPU even when CUDA is available.
        metrics: Metric name or names to calculate.
        output_folder: Optional directory for separated stems and result logs.
        extension: Validation audio extension, without or with a leading dot.
        use_tta: Apply validation-time channel/polarity augmentation.
        draw_spectro: Number of seconds to include in saved spectrograms;
            ``0`` disables them.
        lora_checkpoint_peft: Optional PEFT LoRA checkpoint.
        lora_checkpoint_loralib: Optional loralib checkpoint.
        verbose: Print validation paths, progress, and aggregate metrics.
        experiment: Optional MSST experiment directory for run tracking.
        run_name: Optional label included in the tracked run directory.

    Returns:
        A :class:`ValidationResult` containing aggregate and per-track scores.

    Raises:
        FileNotFoundError: A required config, checkpoint, or dataset is missing.
        ValueError: A public argument is empty, negative, or unsupported.
        MSSTError: Model, checkpoint, device, or audio processing fails.
    """

    config = ValidationConfig(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        valid_path=valid_path,
        model_type=model_type,
        device_ids=device_ids,
        force_cpu=force_cpu,
        metrics=metrics,
        output_folder=output_folder,
        extension=extension,
        use_tta=use_tta,
        draw_spectro=draw_spectro,
        lora_checkpoint_peft=lora_checkpoint_peft,
        lora_checkpoint_loralib=lora_checkpoint_loralib,
        verbose=verbose,
    )
    from .experiments import tracked_run

    parameters = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "valid_path": [str(path) for path in _paths(valid_path)],
        "output_folder": str(output_folder) if output_folder else None,
        "model_type": model_type,
        "device_ids": device_ids,
        "force_cpu": force_cpu,
        "metrics": metrics,
    }
    with tracked_run(
        experiment,
        task="valid",
        parameters=parameters,
        run_name=run_name,
    ) as run:
        try:
            legacy = import_module("._validation", __package__)
            result = legacy.check_validation(config.to_legacy_args())
        except ModuleNotFoundError as error:
            extra = f"validation,{model_type.replace('_', '-')}"
            raise optional_dependency_error(
                error,
                extra=extra,
                feature="Model validation",
            ) from error
        normalized = ValidationResult.from_legacy(result)
        if run is not None:
            if output_folder is not None:
                run.add_artifacts(str(output_folder))
            run.set_result({"averages": normalized.averages})
        return normalized


__all__ = [
    "SUPPORTED_VALIDATION_METRICS",
    "ValidationConfig",
    "ValidationMetric",
    "ValidationResult",
    "valid",
]
