"""Console entry points for the public MSST package."""

from __future__ import annotations

import argparse
import sys
from typing import Callable

from .errors import MSSTError


def _exit_on_user_error(action: Callable[[], None]) -> None:
    try:
        action()
    except (
        MSSTError,
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"error: {error}") from error


def inference() -> None:
    """Run folder inference using command-line arguments."""

    def action() -> None:
        from .experiments import tracked_run
        from .inference import proc_folder
        from .utils.settings import parse_args_inference

        args = parse_args_inference(None)
        parameters = vars(args).copy()
        with tracked_run(
            args.experiment or None,
            task="inference",
            parameters=parameters,
            run_name=args.run_name or None,
        ) as run:
            outputs = proc_folder(parameters)
            if run is not None:
                run.add_artifacts(outputs)
                run.set_result({"output_count": len(outputs)})

    _exit_on_user_error(action)


def _train(launcher_override: str | None = None) -> None:
    from .experiments import tracked_run
    from .training import train_model
    from .utils.settings import parse_args_train

    args = parse_args_train(None)
    launcher = launcher_override or args.launcher
    parameters = vars(args).copy()
    with tracked_run(
        args.experiment or None,
        task="train",
        parameters=parameters,
        run_name=args.run_name or None,
    ) as run:
        train_model(args, launcher=launcher)
        if run is not None:
            run.add_artifacts(args.results_path)


def train() -> None:
    """Run standard, DDP, or Accelerate training."""

    _exit_on_user_error(_train)


def train_ddp() -> None:
    """Compatibility alias for DDP training."""

    _exit_on_user_error(lambda: _train("ddp"))


def train_accelerate() -> None:
    """Compatibility alias for Accelerate training."""

    _exit_on_user_error(lambda: _train("accelerate"))


def valid() -> None:
    """Run the historical validation CLI through the packaged runtime."""

    def action() -> None:
        from ._validation import check_validation
        from .experiments import tracked_run
        from .utils.settings import parse_args_valid

        args = parse_args_valid(None)
        parameters = vars(args).copy()
        with tracked_run(
            args.experiment or None,
            task="valid",
            parameters=parameters,
            run_name=args.run_name or None,
        ) as run:
            result = check_validation(args)
            if run is not None:
                if args.store_dir:
                    run.add_artifacts(args.store_dir)
                if isinstance(result, tuple) and result:
                    run.set_result({"averages": result[0]})

    _exit_on_user_error(action)


def metadata() -> None:
    """Generate training metadata without constructing a DataLoader."""

    parser = argparse.ArgumentParser(prog="msst metadata")
    parser.add_argument("--config_path", "--config-path", required=True)
    parser.add_argument("--data_path", "--data-path", nargs="+", required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--results_path", "--results-path")
    destination.add_argument("--metadata_path", "--metadata-path")
    parser.add_argument("--model_type", "--model-type", default="mdx23c")
    parser.add_argument("--dataset_type", "--dataset-type", type=int, default=1)
    parser.add_argument("--num_workers", "--num-workers", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    def action() -> None:
        from .metadata import build_metadata

        result = build_metadata(
            config_path=args.config_path,
            data_path=args.data_path,
            results_path=args.results_path,
            metadata_path=args.metadata_path,
            model_type=args.model_type,
            dataset_type=args.dataset_type,
            num_workers=args.num_workers,
            refresh=args.refresh,
            verbose=not args.quiet,
        )
        print(f"Metadata: {result.path}")
        print(f"Manifest: {result.manifest_path}")
        print(f"Track entries: {result.track_count}")
        print(f"Fingerprint: {result.fingerprint}")

    _exit_on_user_error(action)


def init() -> None:
    """Create a portable MSST experiment directory."""

    parser = argparse.ArgumentParser(prog="msst init")
    parser.add_argument("path")
    parser.add_argument("--name")
    args = parser.parse_args()

    def action() -> None:
        from .experiments import init_experiment

        experiment = init_experiment(args.path, name=args.name)
        print(f"Experiment: {experiment.path}")
        print(f"Manifest: {experiment.manifest_path}")

    _exit_on_user_error(action)


def check() -> None:
    """Run safe checks or one full representative training step."""

    pre_parser = argparse.ArgumentParser(prog="msst check", add_help=False)
    pre_parser.add_argument("--mode", choices=["safe", "full"], default="safe")
    pre_parser.add_argument(
        "--memory_headroom",
        "--memory-headroom",
        type=float,
        default=0.15,
    )
    pre_args, training_argv = pre_parser.parse_known_args()
    original_argv = sys.argv
    try:
        sys.argv = ["msst check", *training_argv]
        from .utils.settings import parse_args_train

        args = parse_args_train(None)
    finally:
        sys.argv = original_argv

    def action() -> None:
        from .preflight import check as run_check

        values = vars(args).copy()
        config_path = values.pop("config_path")
        data_path = values.pop("data_path")
        results_path = values.pop("results_path")
        valid_path = values.pop("valid_path")
        model_type = values.pop("model_type")
        checkpoint_path = values.pop("start_check_point") or None
        device_ids = values.pop("device_ids")
        dataset_type = values.pop("dataset_type")
        values.pop("launcher", None)
        values.pop("experiment", None)
        values.pop("run_name", None)
        if not config_path or not data_path or not results_path:
            raise ValueError("--config_path, --data_path, and --results_path are required")
        report = run_check(
            config_path=config_path,
            data_path=data_path,
            results_path=results_path,
            valid_path=valid_path,
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            device_ids=device_ids,
            dataset_type=dataset_type,
            mode=pre_args.mode,
            memory_headroom=pre_args.memory_headroom,
            **values,
        )
        for item in report.items:
            print(f"[{item.status.upper()}] {item.name}: {item.message}")
        if report.peak_allocated_gib is not None:
            print(f"Peak allocated: {report.peak_allocated_gib:.2f} GiB")
            print(f"Peak reserved: {report.peak_reserved_gib:.2f} GiB")
        if not report.ok:
            raise RuntimeError(
                f"preflight failed with {len(report.errors)} error(s)"
            )

    _exit_on_user_error(action)


def main() -> None:
    """Dispatch ``msst <command>``."""

    commands: dict[str, Callable[[], None]] = {
        "check": check,
        "inference": inference,
        "init": init,
        "metadata": metadata,
        "train": train,
        "valid": valid,
    }
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("usage: msst {init,metadata,check,train,valid,inference} ...")
        print("Run 'msst <command> --help' for command-specific options.")
        return
    command = sys.argv[1]
    handler = commands.get(command)
    if handler is None:
        raise SystemExit(
            f"error: unknown command {command!r}; choose from {', '.join(commands)}"
        )
    sys.argv = [f"msst {command}", *sys.argv[2:]]
    handler()


__all__ = [
    "check",
    "inference",
    "init",
    "main",
    "metadata",
    "train",
    "train_accelerate",
    "train_ddp",
    "valid",
]
