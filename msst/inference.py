"""Stable, library-oriented source-separation API."""

from __future__ import annotations

import argparse
import threading
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from importlib import import_module
from os import PathLike
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Any, Literal

from .errors import AudioFileError, OutputFormatError, optional_dependency_error

if TYPE_CHECKING:
    from .experiments import Experiment

PathValue = str | PathLike[str]
DeviceIds = int | str | Sequence[int]
PcmType = Literal["PCM_16", "PCM_24", "FLOAT"]

# Formats decoded directly by the libsndfile bundled with current SoundFile
# wheels.  The second group is accepted, but needs a system decoder such as
# FFmpeg through librosa's optional backend.
NATIVE_AUDIO_EXTENSIONS = frozenset(
    {
        ".aif",
        ".aifc",
        ".aiff",
        ".au",
        ".caf",
        ".flac",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".rf64",
        ".snd",
        ".voc",
        ".w64",
        ".wav",
        ".wave",
    }
)
DECODER_DEPENDENT_AUDIO_EXTENSIONS = frozenset(
    {
        ".3gp",
        ".aac",
        ".ac3",
        ".amr",
        ".m4a",
        ".m4b",
        ".mka",
        ".mp4",
        ".webm",
        ".wma",
    }
)
SUPPORTED_AUDIO_EXTENSIONS = (
    NATIVE_AUDIO_EXTENSIONS | DECODER_DEPENDENT_AUDIO_EXTENSIONS
)


def _path(value: PathValue) -> str:
    return str(value)


def _device_ids(value: DeviceIds) -> int | list[int]:
    if isinstance(value, bool):
        raise TypeError("device_ids must contain integer CUDA indices")
    if isinstance(value, int):
        normalized = [value]
    elif isinstance(value, str):
        try:
            normalized = [int(part) for part in value.replace(",", " ").split()]
        except ValueError as error:
            raise TypeError(
                f"device_ids must contain integer CUDA indices, got {value!r}"
            ) from error
    else:
        normalized = list(value)
    if not normalized:
        raise ValueError("device_ids must contain at least one device")
    if any(
        not isinstance(device_id, int) or isinstance(device_id, bool)
        for device_id in normalized
    ):
        raise TypeError("device_ids must contain integer CUDA indices")
    if any(device_id < 0 for device_id in normalized):
        raise ValueError(f"device_ids cannot contain negative CUDA indices: {normalized}")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"device_ids must contain unique CUDA indices: {normalized}")
    return normalized


def _validate_output_options(
    *,
    draw_spectro: float,
    flac_file: bool,
    pcm_type: PcmType | None,
    filename_template: str,
) -> None:
    if draw_spectro < 0:
        raise ValueError("draw_spectro cannot be negative")
    if not filename_template or not filename_template.strip():
        raise ValueError("filename_template cannot be empty")
    if pcm_type == "FLOAT" and flac_file:
        raise OutputFormatError(
            "FLAC does not support FLOAT samples; use PCM_16 or PCM_24, "
            "or omit pcm_type to select PCM_24 automatically"
        )


def _audio_decode_error(path: Path, error: Exception) -> AudioFileError:
    detail = str(error).strip() or type(error).__name__
    suffix = path.suffix.lower()
    hint = ""
    if suffix in DECODER_DEPENDENT_AUDIO_EXTENSIONS:
        hint = (
            " This format requires a compatible system decoder (usually "
            "FFmpeg); install one or convert the file to WAV, FLAC, MP3, "
            "OGG, or AIFF."
        )
    return AudioFileError(f"Cannot decode audio file {path}: {detail}.{hint}")


def _legacy_inference() -> ModuleType:
    return import_module("._inference", __package__)


def _model_extra(model_type: str) -> str:
    return model_type.replace("_", "-")


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    """Configuration for a one-shot folder inference call."""

    config_path: PathValue
    checkpoint_path: PathValue
    input_folder: PathValue
    output_folder: PathValue
    model_type: str = "mdx23c"
    device_ids: DeviceIds = 0
    force_cpu: bool = False
    use_tta: bool = False
    bigshifts: int = 1
    extract_instrumental: bool = False
    disable_detailed_pbar: bool = False
    draw_spectro: float = 0
    flac_file: bool = False
    pcm_type: PcmType | None = None
    filename_template: str = "{relative_path}/{instr}"
    lora_checkpoint_peft: PathValue | None = None
    lora_checkpoint_loralib: PathValue | None = None
    skip_errors: bool = False

    def __post_init__(self) -> None:
        if self.bigshifts < 1:
            raise ValueError("bigshifts must be at least 1")
        _device_ids(self.device_ids)
        _validate_output_options(
            draw_spectro=self.draw_spectro,
            flac_file=self.flac_file,
            pcm_type=self.pcm_type,
            filename_template=self.filename_template,
        )

    def to_legacy_args(self) -> dict[str, Any]:
        """Translate public names to the historical CLI implementation."""

        values = asdict(self)
        values["config_path"] = _path(self.config_path)
        values["start_check_point"] = _path(self.checkpoint_path)
        values["input_folder"] = _path(self.input_folder)
        values["store_dir"] = _path(self.output_folder)
        values["device_ids"] = _device_ids(self.device_ids)
        values.pop("checkpoint_path")
        values.pop("output_folder")
        for key in ("lora_checkpoint_peft", "lora_checkpoint_loralib"):
            value = values[key]
            values[key] = "" if value is None else _path(value)
        return values


class Separator:
    """A reusable source separator that loads model weights only once.

    The instance serializes calls to the underlying model. This avoids unsafe
    concurrent mutation in model-specific inference implementations while still
    allowing one loaded model to process many arrays, files, or folders.
    """

    def __init__(
        self,
        *,
        config_path: PathValue,
        checkpoint_path: PathValue,
        model_type: str = "mdx23c",
        device_ids: DeviceIds = 0,
        force_cpu: bool = False,
        use_tta: bool = False,
        bigshifts: int = 1,
        extract_instrumental: bool = False,
        detailed_progress: bool = True,
        lora_checkpoint_peft: PathValue | None = None,
        lora_checkpoint_loralib: PathValue | None = None,
    ) -> None:
        if bigshifts < 1:
            raise ValueError("bigshifts must be at least 1")

        payload = {
            "config_path": _path(config_path),
            "start_check_point": _path(checkpoint_path),
            "model_type": model_type,
            "device_ids": _device_ids(device_ids),
            "force_cpu": force_cpu,
            "use_tta": use_tta,
            "bigshifts": bigshifts,
            "extract_instrumental": extract_instrumental,
            "disable_detailed_pbar": not detailed_progress,
            "lora_checkpoint_peft": (
                "" if lora_checkpoint_peft is None else _path(lora_checkpoint_peft)
            ),
            "lora_checkpoint_loralib": (
                "" if lora_checkpoint_loralib is None else _path(lora_checkpoint_loralib)
            ),
        }

        self._lock = threading.RLock()
        self._closed = False
        try:
            settings = import_module(".utils.settings", __package__)
            self._args = settings.parse_args_inference(payload)
            legacy = _legacy_inference()
            self._model, self._config, self._device = legacy.load_separator(
                self._args
            )
        except ModuleNotFoundError as error:
            raise optional_dependency_error(
                error,
                extra=_model_extra(model_type),
                feature=f"The {model_type!r} separator",
            ) from error

    @property
    def sample_rate(self) -> int:
        """Sample rate expected and returned by the loaded model."""

        return int(getattr(self._config.audio, "sample_rate", 44100))

    @property
    def instruments(self) -> tuple[str, ...]:
        """Stem names produced by the current configuration."""

        target = getattr(self._config.training, "target_instrument", None)
        names = [target] if target else list(self._config.training.instruments)
        if self._args.extract_instrumental and "instrumental" not in names:
            names.append("instrumental")
        return tuple(names)

    @property
    def device(self) -> str:
        return str(self._device)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Separator is closed")

    def separate(
        self,
        audio: Any,
        *,
        sample_rate: int | None = None,
        channels_first: bool = True,
    ) -> dict[str, Any]:
        """Separate an in-memory waveform and return channels-first stems.

        ``audio`` may have shape ``(samples,)``, ``(channels, samples)``, or
        ``(samples, channels)`` when ``channels_first=False``. Audio at a
        different sample rate is resampled to :attr:`sample_rate`.
        """

        import numpy as np

        self._ensure_open()
        waveform = np.asarray(audio)
        if waveform.ndim == 2 and not channels_first:
            waveform = waveform.T
        if sample_rate is not None:
            if sample_rate <= 0:
                raise ValueError("sample_rate must be a positive integer")
            if sample_rate != self.sample_rate:
                librosa = import_module("librosa")
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sample_rate,
                    target_sr=self.sample_rate,
                    axis=-1,
                )

        with self._lock:
            self._ensure_open()
            legacy = _legacy_inference()
            return legacy.separate_audio(
                self._model,
                self._args,
                self._config,
                self._device,
                waveform,
            )

    def separate_file(self, path: PathValue) -> dict[str, Any]:
        """Load and separate one audio file without writing output files."""

        self._ensure_open()
        input_path = Path(path).expanduser()
        if not input_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {input_path}")
        if input_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
            raise AudioFileError(
                f"Unsupported audio extension {input_path.suffix or '<none>'!r} "
                f"for {input_path}. Supported extensions: {supported}"
            )
        librosa = import_module("librosa")
        try:
            waveform, _ = librosa.load(
                input_path,
                sr=self.sample_rate,
                mono=False,
            )
        except Exception as error:
            raise _audio_decode_error(input_path, error) from error
        return self.separate(waveform)

    def separate_folder(
        self,
        input_folder: PathValue,
        output_folder: PathValue,
        *,
        draw_spectro: float = 0,
        flac_file: bool = False,
        pcm_type: PcmType | None = None,
        filename_template: str = "{relative_path}/{instr}",
        verbose: bool = True,
        skip_errors: bool = False,
    ) -> list[str]:
        """Separate a folder with this loaded model and return written paths."""

        self._ensure_open()
        _validate_output_options(
            draw_spectro=draw_spectro,
            flac_file=flac_file,
            pcm_type=pcm_type,
            filename_template=filename_template,
        )
        values = vars(self._args).copy()
        values.update(
            input_folder=_path(input_folder),
            store_dir=_path(output_folder),
            draw_spectro=draw_spectro,
            flac_file=flac_file,
            pcm_type=pcm_type,
            filename_template=filename_template,
            skip_errors=skip_errors,
        )
        args = argparse.Namespace(**values)
        settings = import_module(".utils.settings", __package__)
        args.pcm_type = settings.validate_sndfile_subtype(args)
        with self._lock:
            self._ensure_open()
            legacy = _legacy_inference()
            return legacy.run_folder(
                self._model,
                args,
                self._config,
                self._device,
                verbose=verbose,
            )

    def close(self) -> None:
        """Release references to model weights and clear the CUDA cache."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._model = None
            if str(self._device).startswith("cuda"):
                torch = import_module("torch")
                torch.cuda.empty_cache()

    def __enter__(self) -> "Separator":
        self._ensure_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def inference(
    *,
    config_path: PathValue,
    checkpoint_path: PathValue,
    input_folder: PathValue,
    output_folder: PathValue,
    model_type: str = "mdx23c",
    device_ids: DeviceIds = 0,
    force_cpu: bool = False,
    use_tta: bool = False,
    bigshifts: int = 1,
    extract_instrumental: bool = False,
    disable_detailed_pbar: bool = False,
    draw_spectro: float = 0,
    flac_file: bool = False,
    pcm_type: PcmType | None = None,
    filename_template: str = "{relative_path}/{instr}",
    lora_checkpoint_peft: PathValue | None = None,
    lora_checkpoint_loralib: PathValue | None = None,
    skip_errors: bool = False,
    experiment: PathValue | "Experiment" | None = None,
    run_name: str | None = None,
) -> list[str]:
    """Load a separator once, process a folder, and release the model.

    Set ``experiment`` to an MSST experiment directory to record inputs,
    environment, status, and generated files without changing where outputs
    are written.
    """

    from .experiments import tracked_run

    parameters = {
        "config_path": str(config_path),
        "checkpoint_path": str(checkpoint_path),
        "input_folder": str(input_folder),
        "output_folder": str(output_folder),
        "model_type": model_type,
        "device_ids": device_ids,
        "force_cpu": force_cpu,
        "use_tta": use_tta,
        "bigshifts": bigshifts,
    }
    with tracked_run(
        experiment,
        task="inference",
        parameters=parameters,
        run_name=run_name,
    ) as run:
        with Separator(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            model_type=model_type,
            device_ids=device_ids,
            force_cpu=force_cpu,
            use_tta=use_tta,
            bigshifts=bigshifts,
            extract_instrumental=extract_instrumental,
            detailed_progress=not disable_detailed_pbar,
            lora_checkpoint_peft=lora_checkpoint_peft,
            lora_checkpoint_loralib=lora_checkpoint_loralib,
        ) as separator:
            outputs = separator.separate_folder(
                input_folder,
                output_folder,
                draw_spectro=draw_spectro,
                flac_file=flac_file,
                pcm_type=pcm_type,
                filename_template=filename_template,
                skip_errors=skip_errors,
            )
        if run is not None:
            run.add_artifacts(outputs)
            run.set_result({"output_count": len(outputs)})
        return outputs


def proc_folder(
    args: InferenceConfig | Mapping[str, Any] | None = None,
    /,
    **kwargs: Any,
) -> list[str]:
    """Compatibility entry point for the historical dictionary-based API."""

    if isinstance(args, InferenceConfig):
        if kwargs:
            raise TypeError("keyword overrides cannot be combined with InferenceConfig")
        payload: dict[str, Any] | None = args.to_legacy_args()
    elif args is None:
        payload = dict(kwargs) if kwargs else None
    elif isinstance(args, Mapping):
        payload = dict(args)
        payload.update(kwargs)
    else:
        raise TypeError("args must be an InferenceConfig, a mapping, or None")

    if payload is not None:
        aliases = {
            "checkpoint_path": "start_check_point",
            "output_folder": "store_dir",
        }
        for public_name, legacy_name in aliases.items():
            if public_name not in payload:
                continue
            if legacy_name in payload:
                raise TypeError(
                    f"use either {public_name!r} or {legacy_name!r}, not both"
                )
            payload[legacy_name] = _path(payload.pop(public_name))

    try:
        return _legacy_inference().proc_folder(payload)
    except ModuleNotFoundError as error:
        model_type = "mdx23c" if payload is None else payload.get("model_type", "mdx23c")
        raise optional_dependency_error(
            error,
            extra=_model_extra(model_type),
            feature=f"The {model_type!r} separator",
        ) from error


__all__ = [
    "DECODER_DEPENDENT_AUDIO_EXTENSIONS",
    "InferenceConfig",
    "NATIVE_AUDIO_EXTENSIONS",
    "SUPPORTED_AUDIO_EXTENSIONS",
    "Separator",
    "inference",
    "proc_folder",
]
