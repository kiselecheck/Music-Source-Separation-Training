"""Run one end-to-end folder separation in either source tree."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


OUTPUT_TEMPLATE = "{dir_name}/{file_name}/{instr}"


def _run_package(args: argparse.Namespace) -> tuple[float, list[float], int, list[str]]:
    from msst.inference import Separator

    started = time.perf_counter()
    separator = Separator(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        device_ids=0,
        force_cpu=True,
        use_tta=args.use_tta,
        bigshifts=args.bigshifts,
        detailed_progress=False,
    )
    load_seconds = time.perf_counter() - started
    try:
        sample_rate = separator.sample_rate
        instruments = list(separator.instruments)
        processing_runs = []
        for _ in range(args.repeats):
            run_started = time.perf_counter()
            separator.separate_folder(
                args.input_folder,
                args.output_folder,
                flac_file=False,
                pcm_type="FLOAT",
                filename_template=OUTPUT_TEMPLATE,
                verbose=False,
            )
            processing_runs.append(time.perf_counter() - run_started)
        return load_seconds, processing_runs, sample_rate, instruments
    finally:
        separator.close()


def _run_upstream(args: argparse.Namespace) -> tuple[float, list[float], int, list[str]]:
    from inference import run_folder
    from utils.model_utils import load_start_checkpoint
    from utils.settings import get_model_from_config, parse_args_inference

    legacy_args = parse_args_inference(
        {
            "config_path": str(args.config),
            "start_check_point": str(args.checkpoint),
            "input_folder": str(args.input_folder),
            "store_dir": str(args.output_folder),
            "model_type": args.model_type,
            "force_cpu": True,
            "device_ids": [0],
            "use_tta": args.use_tta,
            "bigshifts": args.bigshifts,
            "extract_instrumental": False,
            "disable_detailed_pbar": True,
            "flac_file": False,
            "pcm_type": "FLOAT",
            "filename_template": OUTPUT_TEMPLATE,
        }
    )
    started = time.perf_counter()
    model, config = get_model_from_config(args.model_type, str(args.config))
    if "model_type" in config.training:
        legacy_args.model_type = config.training.model_type
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    load_start_checkpoint(legacy_args, model, checkpoint, type_="inference")
    device = torch.device("cpu")
    model = model.to(device).eval()
    load_seconds = time.perf_counter() - started

    processing_runs = []
    for _ in range(args.repeats):
        run_started = time.perf_counter()
        run_folder(model, legacy_args, config, device, verbose=True)
        processing_runs.append(time.perf_counter() - run_started)
    sample_rate = int(getattr(config.audio, "sample_rate", 44100))
    instruments = list(config.training.instruments)
    target = getattr(config.training, "target_instrument", None)
    if target:
        instruments = [target]
    return load_seconds, processing_runs, sample_rate, instruments


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("upstream", "package"), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bigshifts", type=int, default=1)
    parser.add_argument("--use-tta", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    sys.path.insert(0, str(source_root))
    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(20260921)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False

    run_started = time.perf_counter()
    if args.implementation == "upstream":
        load_seconds, processing_runs, sample_rate, instruments = _run_upstream(args)
    else:
        load_seconds, processing_runs, sample_rate, instruments = _run_package(args)
    output_files = sorted(
        path.relative_to(args.output_folder).as_posix()
        for path in args.output_folder.rglob("*")
        if path.is_file()
    )
    result = {
        "implementation": args.implementation,
        "model_type": args.model_type,
        "sample_rate": sample_rate,
        "instruments": instruments,
        "threads": args.threads,
        "model_load_seconds": load_seconds,
        "processing_runs_seconds": processing_runs,
        "processing_median_seconds": statistics.median(processing_runs),
        "output_files": output_files,
        "runner_seconds": time.perf_counter() - run_started,
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
