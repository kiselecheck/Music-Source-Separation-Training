# coding: utf-8
"""Command-line and compatibility inference implementation.

The stable library API lives in :mod:`msst.inference`.  Functions in this
module remain importable for existing integrations and are also used by the
public facade so the CLI and Python API share one inference implementation.
"""

from __future__ import annotations

__author__ = "Roman Solovyev (ZFTurbo): https://github.com/ZFTurbo/"

import argparse
import string
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from msst.errors import (
    AudioFileError,
    CheckpointError,
    ConfigurationError,
    DeviceError,
    MSSTError,
)
from msst.checkpoints import CHECKPOINT_FORMAT_VERSION
from msst.inference import (
    DECODER_DEPENDENT_AUDIO_EXTENSIONS,
    SUPPORTED_AUDIO_EXTENSIONS,
)
from msst.utils.audio_utils import denormalize_audio, draw_spectrogram, normalize_audio
from msst.utils.model_utils import (
    apply_tta,
    bigshifts_wrapper,
    compile_model,
    load_start_checkpoint,
    prefer_target_instrument,
)
from msst.utils.settings import get_model_from_config, parse_args_inference


def resolve_device(args: argparse.Namespace) -> torch.device:
    """Select the requested inference device."""

    if args.force_cpu:
        return torch.device("cpu")

    device_ids = args.device_ids
    if isinstance(device_ids, str):
        try:
            device_ids = [int(part) for part in device_ids.replace(",", " ").split()]
        except ValueError as error:
            raise DeviceError(
                f"device_ids must contain integer CUDA indices, got {args.device_ids!r}"
            ) from error
    elif isinstance(device_ids, int):
        device_ids = [device_ids]
    else:
        device_ids = list(device_ids)
    if not device_ids:
        raise DeviceError("device_ids must contain at least one CUDA index")
    if any(
        not isinstance(device_id, int) or isinstance(device_id, bool)
        for device_id in device_ids
    ):
        raise DeviceError(
            f"device_ids must contain integer CUDA indices, got {device_ids!r}"
        )
    if any(device_id < 0 for device_id in device_ids):
        raise DeviceError(f"CUDA device indices cannot be negative: {device_ids}")
    if len(set(device_ids)) != len(device_ids):
        raise DeviceError(f"CUDA device indices must be unique: {device_ids}")
    args.device_ids = device_ids
    first_device = device_ids[0]
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        invalid = [device_id for device_id in device_ids if device_id >= device_count]
        if invalid:
            raise DeviceError(
                f"CUDA device index/indices {invalid} are unavailable; "
                f"detected {device_count} CUDA device(s) with valid indices "
                f"0..{device_count - 1}"
            )
        return torch.device(f"cuda:{first_device}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_checkpoint_metadata(
    checkpoint: Any,
    *,
    checkpoint_path: Path,
    model_type: str,
    config: Any,
) -> None:
    """Reject known checkpoint/config mismatches before loading tensor weights.

    Older checkpoints contain only a state dict and remain supported. Newer
    checkpoints may include a small ``meta`` mapping, which lets the public API
    report a precise model/stem mismatch instead of a long ``load_state_dict``
    traceback or a later output-shape failure.
    """

    if not isinstance(checkpoint, Mapping):
        return
    metadata = checkpoint.get("meta")
    if not isinstance(metadata, Mapping):
        return

    format_version = metadata.get("format_version")
    if format_version is not None:
        try:
            format_version = int(format_version)
        except (TypeError, ValueError) as error:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} has invalid meta.format_version="
                f"{format_version!r}; expected a positive integer."
            ) from error
        if format_version < 1:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} has invalid meta.format_version="
                f"{format_version}; expected a positive integer."
            )
        if format_version > CHECKPOINT_FORMAT_VERSION:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} uses format version {format_version}, "
                f"but this MSST release supports up to {CHECKPOINT_FORMAT_VERSION}. "
                "Upgrade msst before loading this checkpoint."
            )

    checkpoint_model_type = metadata.get("model_type")
    if checkpoint_model_type and str(checkpoint_model_type) != model_type:
        raise CheckpointError(
            f"Checkpoint {checkpoint_path} declares model_type="
            f"{checkpoint_model_type!r}, but the configuration selects "
            f"model_type={model_type!r}. Use the checkpoint's matching config."
        )

    instruments = list(config.training.instruments)
    configured_target = getattr(config.training, "target_instrument", None)
    checkpoint_target = metadata.get("target_instrument")
    if checkpoint_target:
        checkpoint_target = str(checkpoint_target)
        if checkpoint_target not in instruments:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} targets {checkpoint_target!r}, "
                "but that stem is absent from training.instruments in the config."
            )
        if configured_target is None and not (
            len(instruments) == 1 and instruments[0] == checkpoint_target
        ):
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} targets {checkpoint_target!r}, "
                "but the config has no matching training.target_instrument."
            )
        if configured_target is not None and configured_target != checkpoint_target:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} targets {checkpoint_target!r}, "
                f"but the config targets {configured_target!r}."
            )

    checkpoint_stem_count = metadata.get("target_stem_count")
    if checkpoint_stem_count is not None:
        try:
            checkpoint_stem_count = int(checkpoint_stem_count)
        except (TypeError, ValueError) as error:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} has invalid meta.target_stem_count="
                f"{checkpoint_stem_count!r}; expected a positive integer."
            ) from error
        configured_stem_count = 1 if configured_target else len(instruments)
        if checkpoint_stem_count < 1:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} has invalid meta.target_stem_count="
                f"{checkpoint_stem_count}; expected a positive integer."
            )
        if checkpoint_stem_count != configured_stem_count:
            raise CheckpointError(
                f"Checkpoint {checkpoint_path} declares {checkpoint_stem_count} "
                f"output stem(s), but the config selects {configured_stem_count}."
            )


def load_separator(
    args: argparse.Namespace,
) -> tuple[nn.Module, Any, torch.device]:
    """Load and prepare a separator model without processing any audio."""

    if not args.config_path:
        raise ValueError("config_path is required")
    config_path = Path(args.config_path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not args.start_check_point:
        raise ValueError("start_check_point/checkpoint_path is required")
    checkpoint_path = Path(args.start_check_point).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = resolve_device(args)
    print("Using device:", device)

    started = time.time()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    try:
        model, config = get_model_from_config(args.model_type, str(config_path))
    except (ModuleNotFoundError, MSSTError):
        raise
    except Exception as error:
        detail = str(error).strip() or type(error).__name__
        raise ConfigurationError(
            f"Cannot build model_type={args.model_type!r} from config "
            f"{config_path}: {type(error).__name__}: {detail}"
        ) from error
    if "model_type" in config.training:
        args.model_type = config.training.model_type

    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    except Exception as error:
        detail = str(error).strip() or type(error).__name__
        raise CheckpointError(
            f"Cannot read checkpoint {checkpoint_path}: "
            f"{type(error).__name__}: {detail}"
        ) from error
    validate_checkpoint_metadata(
        checkpoint,
        checkpoint_path=checkpoint_path,
        model_type=args.model_type,
        config=config,
    )
    try:
        load_start_checkpoint(args, model, checkpoint, type_="inference")
    except Exception as error:
        detail = str(error).strip() or type(error).__name__
        raise CheckpointError(
            f"Checkpoint {checkpoint_path} is incompatible with "
            f"model_type={args.model_type!r} and config {config_path}: "
            f"{type(error).__name__}: {detail}"
        ) from error
    if args.lora_checkpoint_peft:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.lora_checkpoint_peft)

    device_ids = args.device_ids
    if (
        device.type == "cuda"
        and isinstance(device_ids, (list, tuple))
        and len(device_ids) > 1
    ):
        model = nn.DataParallel(model, device_ids=list(device_ids))

    try:
        model = model.to(device)
    except Exception as error:
        detail = str(error).strip() or type(error).__name__
        raise DeviceError(
            f"Cannot move model_type={args.model_type!r} to {device}: "
            f"{type(error).__name__}: {detail}"
        ) from error
    model.eval()
    if args.compile:
        if device.type != "cuda" or isinstance(model, nn.DataParallel):
            print("Warning: --compile requires a single CUDA device; ignoring it")
        else:
            model = compile_model(model)
    print("Instruments:", config.training.instruments)
    print(f"Model load time: {time.time() - started:.2f} sec")
    return model, config, device


def _normalization_enabled(config: Any) -> bool:
    inference_config = getattr(config, "inference", None)
    if inference_config is None:
        return False
    if hasattr(inference_config, "get"):
        return bool(inference_config.get("normalize", False))
    return bool(getattr(inference_config, "normalize", False))


def _prepare_audio(mix: np.ndarray, config: Any) -> np.ndarray:
    raw_audio = np.asarray(mix)
    if raw_audio.dtype.kind not in "fiu":
        raise ValueError(
            f"Audio must have a real numeric dtype; received {raw_audio.dtype}"
        )
    if raw_audio.dtype.kind in "iu":
        limits = np.iinfo(raw_audio.dtype)
        scale = float(max(abs(limits.min), limits.max))
        audio = raw_audio.astype(np.float32) / scale
    else:
        audio = raw_audio.astype(np.float32, copy=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    if audio.ndim != 2:
        raise ValueError(
            "Audio must have shape (samples,) or (channels, samples); "
            f"received {audio.shape}"
        )
    if audio.shape[-1] == 0:
        raise ValueError("Audio must contain at least one sample")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinite values")

    expected_channels = getattr(config.audio, "num_channels", None)
    if audio.shape[0] == 1 and expected_channels == 2:
        audio = np.repeat(audio, 2, axis=0)
    elif expected_channels == 1 and audio.shape[0] > 1:
        audio = np.mean(audio, axis=0, keepdims=True, dtype=np.float32)
    elif expected_channels is not None and audio.shape[0] != expected_channels:
        raise ValueError(
            f"Model expects {expected_channels} audio channel(s), but received "
            f"{audio.shape[0]}"
        )
    return np.ascontiguousarray(audio)


def separate_audio(
    model: nn.Module,
    args: argparse.Namespace,
    config: Any,
    device: torch.device | str,
    mix: np.ndarray,
    *,
    detailed_progress: bool | None = None,
) -> dict[str, np.ndarray]:
    """Separate one channels-first waveform using an already loaded model."""

    prepared_mix = _prepare_audio(mix, config)
    original_mix = prepared_mix.copy()
    normalization = _normalization_enabled(config)

    if normalization:
        prepared_mix, normalization_params = normalize_audio(prepared_mix)

    if detailed_progress is None:
        detailed_progress = not args.disable_detailed_pbar

    with torch.inference_mode():
        waveforms = bigshifts_wrapper(
            config,
            model,
            prepared_mix,
            device,
            model_type=args.model_type,
            pbar=detailed_progress,
            bigshifts=args.bigshifts,
        )

        if args.use_tta:
            waveforms = apply_tta(
                config,
                model,
                prepared_mix,
                waveforms,
                device,
                args.model_type,
                bigshifts=args.bigshifts,
                pbar=detailed_progress,
            )

    instruments = list(prefer_target_instrument(config))
    if isinstance(waveforms, np.ndarray):
        if len(instruments) == 1 and waveforms.ndim == 2:
            separated = {instruments[0]: waveforms}
        elif waveforms.ndim >= 3 and waveforms.shape[0] == len(instruments):
            separated = dict(zip(instruments, waveforms))
        else:
            raise RuntimeError(
                "Model returned an array that cannot be matched to configured "
                f"instruments: shape={waveforms.shape}, instruments={instruments}"
            )
    else:
        separated = dict(waveforms)

    missing = [instrument for instrument in instruments if instrument not in separated]
    if missing:
        raise RuntimeError(f"Model output is missing configured stems: {missing}")

    if args.extract_instrumental:
        source = "vocals" if "vocals" in instruments else instruments[0]
        separated["instrumental"] = original_mix - separated[source]
        instruments.append("instrumental")

    result: dict[str, np.ndarray] = {}
    for instrument in instruments:
        estimates = np.asarray(separated[instrument])
        if normalization:
            estimates = denormalize_audio(estimates, normalization_params)
        if estimates.ndim not in (1, 2):
            raise RuntimeError(
                f"Stem {instrument!r} has invalid shape {estimates.shape}; "
                "expected (samples,) or (channels, samples)"
            )
        if not np.isfinite(estimates).all():
            raise RuntimeError(f"Stem {instrument!r} contains NaN or infinite values")
        result[instrument] = estimates
    return result


def format_filename(template: str, **kwargs: Any) -> tuple[list[str], str]:
    """Format and validate an output path derived from ``template``."""

    if not template or not template.strip():
        raise ValueError("filename_template cannot be empty")
    formatter = string.Formatter()
    try:
        fields = [
            field_name
            for _, field_name, _, _ in formatter.parse(template)
            if field_name is not None
        ]
    except ValueError as error:
        raise ValueError(f"Invalid filename_template {template!r}: {error}") from error
    unknown = sorted(set(fields) - set(kwargs))
    if unknown:
        raise ValueError(
            f"Unknown filename_template field(s): {', '.join(unknown)}. "
            f"Available fields: {', '.join(sorted(kwargs))}"
        )
    try:
        result = template.format(**kwargs)
    except (KeyError, ValueError, AttributeError, IndexError) as error:
        raise ValueError(f"Invalid filename_template {template!r}: {error}") from error
    result = result.replace("\\", "/")
    path = PurePosixPath(result)
    parts = [part for part in path.parts if part not in ("", ".")]
    if (
        not parts
        or path.is_absolute()
        or ".." in parts
        or parts[0].endswith(":")
    ):
        raise ValueError(f"filename_template produced an unsafe path: {result!r}")
    return parts[:-1], parts[-1]


def _write_stems(
    stems: dict[str, np.ndarray],
    *,
    args: argparse.Namespace,
    sample_rate: int,
    relative_path: Path,
    started_at: int,
    reserved_paths: set[Path] | None = None,
) -> list[str]:
    output_paths: list[str] = []
    file_name = relative_path.stem
    dir_name = relative_path.parent.as_posix()
    if dir_name == ".":
        dir_name = "."
    relative_without_suffix = relative_path.with_suffix("").as_posix()
    input_extension = relative_path.suffix.lower().lstrip(".")

    codec = "flac" if args.flac_file else "wav"
    model_name = Path(args.start_check_point).stem
    output_root = Path(args.store_dir).expanduser()

    for instrument, estimates in stems.items():
        directories, output_name = format_filename(
            args.filename_template,
            instr=instrument,
            start_time=started_at,
            file_name=file_name,
            dir_name=dir_name,
            relative_path=relative_without_suffix,
            extension=input_extension,
            model_type=args.model_type,
            model=model_name,
        )
        output_dir = output_root.joinpath(*directories)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{output_name}.{codec}"

        resolved_output_path = output_path.resolve()
        if reserved_paths is not None:
            if resolved_output_path in reserved_paths:
                raise ValueError(
                    f"filename_template maps multiple inputs/stems to the same "
                    f"output path: {output_path}. Include {{relative_path}}, "
                    "{extension}, or another distinguishing field in the template."
                )
            reserved_paths.add(resolved_output_path)

        sf.write(output_path, np.asarray(estimates).T, sample_rate, subtype=args.pcm_type)
        output_paths.append(str(output_path))

        if args.draw_spectro > 0:
            image_path = output_dir / f"{output_name}.jpg"
            waveform = np.asarray(estimates).T
            if waveform.ndim == 1:
                waveform = waveform[:, np.newaxis]
            draw_spectrogram(
                waveform,
                sample_rate,
                args.draw_spectro,
                str(image_path),
            )
            print("Wrote file:", image_path)

    return output_paths


def run_folder(
    model: nn.Module,
    args: argparse.Namespace,
    config: Any,
    device: torch.device | str,
    verbose: bool = False,
) -> list[str]:
    """Process a folder with an already loaded model and return written files."""

    started = time.time()
    input_root = Path(args.input_folder).expanduser()
    if not input_root.is_dir():
        raise NotADirectoryError(f"Input folder not found: {input_root}")
    if not args.store_dir:
        raise ValueError("store_dir/output_folder is required")

    output_root = Path(args.store_dir).expanduser()
    resolved_input = input_root.resolve()
    resolved_output = output_root.resolve()
    if resolved_output == resolved_input:
        raise ValueError(
            "input_folder and output_folder must be different directories; "
            "writing stems into the input directory can overwrite or reprocess inputs"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    output_is_nested = (
        resolved_output != resolved_input
        and resolved_output.is_relative_to(resolved_input)
    )

    all_files = sorted(path for path in input_root.rglob("*") if path.is_file())
    ignored_count = sum(
        path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS for path in all_files
    )
    mixture_paths = [
        path for path in all_files
        if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    if output_is_nested:
        mixture_paths = [
            path for path in mixture_paths if not path.resolve().is_relative_to(resolved_output)
        ]

    if not mixture_paths:
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise AudioFileError(
            f"No supported audio files found in {input_root}. "
            f"Supported extensions: {supported}"
        )

    sample_rate = int(getattr(config.audio, "sample_rate", 44100))
    print(f"Total files found: {len(mixture_paths)}. Using sample rate: {sample_rate}")
    if ignored_count:
        print(f"Ignored {ignored_count} file(s) with non-audio extensions.")

    paths = mixture_paths if verbose else tqdm(mixture_paths, desc="Total progress")
    output_paths: list[str] = []
    reserved_paths: set[Path] = set()
    started_at = int(started)
    for path in paths:
        try:
            mix, loaded_sample_rate = librosa.load(
                path,
                sr=sample_rate,
                mono=False,
            )
            stems = separate_audio(model, args, config, device, mix)
            relative_path = path.relative_to(input_root)
            output_paths.extend(
                _write_stems(
                    stems,
                    args=args,
                    sample_rate=int(loaded_sample_rate),
                    relative_path=relative_path,
                    started_at=started_at,
                    reserved_paths=reserved_paths,
                )
            )
        except Exception as error:
            if not getattr(args, "skip_errors", True):
                if isinstance(error, MSSTError):
                    raise
                detail = str(error).strip() or type(error).__name__
                hint = ""
                if path.suffix.lower() in DECODER_DEPENDENT_AUDIO_EXTENSIONS:
                    hint = (
                        " This format needs a compatible system decoder (usually "
                        "FFmpeg); alternatively convert it to WAV, FLAC, MP3, OGG, "
                        "or AIFF."
                    )
                raise AudioFileError(
                    f"Cannot process audio file {path}: "
                    f"{type(error).__name__}: {detail}.{hint}"
                ) from error
            print(f"Cannot process track: {path}")
            detail = str(error).strip() or type(error).__name__
            print(f"Error message: {type(error).__name__}: {detail}")

    print(f"Elapsed time: {time.time() - started:.2f} seconds.")
    return output_paths


def proc_folder(dict_args: dict[str, Any] | None) -> list[str]:
    """Historical dict/CLI entry point, now returning written output paths."""

    args = parse_args_inference(dict_args)
    model, config, device = load_separator(args)
    return run_folder(model, args, config, device, verbose=True)


if __name__ == "__main__":
    try:
        proc_folder(None)
    except (MSSTError, FileNotFoundError, NotADirectoryError, TypeError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
