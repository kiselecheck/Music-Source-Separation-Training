"""Compare complete model-checkpoint-audio-folder inference and record timings."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parents[1]
RUNNER = SCRIPT_DIR / "real_inference_runner.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _git_status(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_test_audio(path: Path, seconds: float = 8.0, sample_rate: int = 44100) -> None:
    sample_count = round(seconds * sample_rate)
    time_axis = np.arange(sample_count, dtype=np.float64) / sample_rate
    envelope = np.minimum(1.0, time_axis * 8.0) * np.minimum(1.0, (seconds - time_axis) * 8.0)
    left = (
        0.21 * np.sin(2 * np.pi * 110.0 * time_axis)
        + 0.14 * np.sin(2 * np.pi * (330.0 * time_axis + 3.0 * time_axis**2))
        + 0.04 * np.random.default_rng(8102).standard_normal(sample_count)
    )
    right = (
        0.18 * np.sin(2 * np.pi * 164.8 * time_axis + 0.2)
        + 0.12 * np.sin(2 * np.pi * 493.9 * time_axis)
        + 0.04 * np.random.default_rng(8103).standard_normal(sample_count)
    )
    signal = (np.stack((left, right), axis=1) * envelope[:, None]).astype(np.float32)
    sf.write(path, signal, sample_rate, subtype="PCM_24")


def _run_one(
    *,
    implementation: str,
    source_root: Path,
    model_type: str,
    config: Path,
    checkpoint: Path,
    input_folder: Path,
    output_folder: Path,
    result_path: Path,
    repeats: int,
    threads: int,
    bigshifts: int,
    use_tta: bool,
) -> tuple[dict[str, Any], float, str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--implementation",
        implementation,
        "--source-root",
        str(source_root),
        "--model-type",
        model_type,
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--input-folder",
        str(input_folder),
        "--output-folder",
        str(output_folder),
        "--repeats",
        str(repeats),
        "--threads",
        str(threads),
        "--bigshifts",
        str(bigshifts),
        "--report",
        str(result_path),
    ]
    if use_tta:
        command.append("--use-tta")
    started = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=source_root,
        capture_output=True,
        text=True,
    )
    process_wall = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(
            f"{implementation} real-model runner failed ({result.returncode}).\n"
            f"stdout:\n{result.stdout[-6000:]}\n"
            f"stderr:\n{result.stderr[-6000:]}"
        )
    return json.loads(result_path.read_text(encoding="utf-8")), process_wall, result.stdout[-2000:]


def _read_wave(path: Path) -> tuple[np.ndarray, sf._SoundFileInfo]:
    info = sf.info(path)
    data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if sample_rate != info.samplerate:
        raise AssertionError(f"SoundFile sample-rate disagreement for {path}")
    if not np.isfinite(data).all():
        raise AssertionError(f"Non-finite samples in output {path}")
    return data, info


def compare_model(
    *,
    model_type: str,
    config: Path,
    checkpoint: Path,
    input_audio: Path | None,
    upstream_root: Path,
    report_path: Path,
    repeats: int = 2,
    threads: int = 1,
    bigshifts: int = 1,
    use_tta: bool = False,
    limit_seconds: float = 2.0,
) -> dict[str, Any]:
    config = config.resolve()
    checkpoint = checkpoint.resolve()
    upstream_root = upstream_root.resolve()
    if not config.is_file():
        raise FileNotFoundError(config)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (upstream_root / "models").is_dir():
        raise ValueError(f"Not an upstream checkout: {upstream_root}")
    if repeats < 1 or threads < 1 or bigshifts < 1 or limit_seconds < 0:
        raise ValueError("repeats, threads, bigshifts must be positive and limit_seconds non-negative")

    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="msst-real-parity-") as temp_name:
        work_root = Path(temp_name)
        input_folder = work_root / "input" / "test-track"
        input_folder.mkdir(parents=True)
        fixture = input_folder / "audio.wav"
        if input_audio is None:
            if limit_seconds == 0:
                raise ValueError("limit_seconds=0 requires --audio so the input duration is defined")
            _make_test_audio(fixture, seconds=limit_seconds)
            fixture_kind = "deterministic synthetic stereo audio"
        else:
            input_audio = input_audio.resolve()
            if not input_audio.is_file():
                raise FileNotFoundError(input_audio)
            source_audio, source_rate = sf.read(
                input_audio, dtype="float32", always_2d=True
            )
            max_frames = round(limit_seconds * source_rate)
            if max_frames > 0:
                source_audio = source_audio[:max_frames]
            sf.write(fixture, source_audio, source_rate, subtype="PCM_24")
            fixture_kind = "user-selected local audio file"
        # Both implementations must ignore an unrelated sidecar and emit the same audio stems.
        (input_folder / "readme.txt").write_text("non-audio sidecar\n", encoding="utf-8")
        audio_info = sf.info(fixture)
        outputs: dict[str, dict[str, Any]] = {}
        process_wall: dict[str, float] = {}
        for implementation, source_root in (
            ("upstream", upstream_root),
            ("package", PACKAGE_ROOT),
        ):
            output_folder = work_root / f"output-{implementation}"
            worker_report = work_root / f"{implementation}.json"
            outputs[implementation], process_wall[implementation], _ = _run_one(
                implementation=implementation,
                source_root=source_root,
                model_type=model_type,
                config=config,
                checkpoint=checkpoint,
                input_folder=input_folder,
                output_folder=output_folder,
                result_path=worker_report,
                repeats=repeats,
                threads=threads,
                bigshifts=bigshifts,
                use_tta=use_tta,
            )

        upstream_files = outputs["upstream"]["output_files"]
        package_files = outputs["package"]["output_files"]
        comparisons = []
        passed = upstream_files == package_files and bool(upstream_files)
        for relative_name in sorted(set(upstream_files) & set(package_files)):
            upstream_path = work_root / "output-upstream" / Path(relative_name)
            package_path = work_root / "output-package" / Path(relative_name)
            upstream_audio, upstream_info = _read_wave(upstream_path)
            package_audio, package_info = _read_wave(package_path)
            same_metadata = (
                upstream_info.samplerate == package_info.samplerate
                and upstream_info.channels == package_info.channels
                and upstream_info.frames == package_info.frames
                and upstream_info.format == package_info.format
                and upstream_info.subtype == package_info.subtype
            )
            same_shape = upstream_audio.shape == package_audio.shape
            metrics: dict[str, Any] = {
                "file": relative_name,
                "same_metadata": same_metadata,
                "upstream_format": upstream_info.format,
                "upstream_subtype": upstream_info.subtype,
                "package_format": package_info.format,
                "package_subtype": package_info.subtype,
                "upstream_shape": list(upstream_audio.shape),
                "package_shape": list(package_audio.shape),
            }
            if same_shape:
                delta = package_audio.astype(np.float64) - upstream_audio.astype(np.float64)
                reference_rms = float(np.sqrt(np.mean(upstream_audio.astype(np.float64) ** 2)))
                rmse = float(np.sqrt(np.mean(delta**2)))
                metrics.update(
                    max_abs_error=float(np.max(np.abs(delta))),
                    rmse=rmse,
                    relative_rmse=rmse / max(reference_rms, 1e-12),
                    exact_sample_match=bool(np.array_equal(package_audio, upstream_audio)),
                )
                same_samples = bool(np.allclose(package_audio, upstream_audio, rtol=1e-5, atol=1e-6))
            else:
                same_samples = False
            metrics["passed"] = same_metadata and same_shape and same_samples
            passed = passed and metrics["passed"]
            comparisons.append(metrics)

        upstream_median = statistics.median(outputs["upstream"]["processing_runs_seconds"])
        package_median = statistics.median(outputs["package"]["processing_runs_seconds"])
        audio_seconds = audio_info.duration
        report: dict[str, Any] = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "passed": passed,
            "comparison_scope": "folder-level separation from checkpoint load through float-WAV output",
            "options": {
                "model_type": model_type,
                "config_file": config.name,
                "config_sha256": _sha256(config),
                "checkpoint_file": checkpoint.name,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "checkpoint_sha256": _sha256(checkpoint),
                "audio_fixture": fixture_kind,
                "audio_sha256": _sha256(fixture),
                "audio_seconds": audio_seconds,
                "audio_sample_rate": audio_info.samplerate,
                "audio_channels": audio_info.channels,
                "audio_limit_seconds": limit_seconds,
                "repeats": repeats,
                "threads": threads,
                "bigshifts": bigshifts,
                "use_tta": use_tta,
                "output_subtype": "FLOAT",
            },
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "processor": platform.processor(),
                "logical_cpu_count": os.cpu_count(),
                "upstream_revision": _git_revision(upstream_root),
                "upstream_worktree_status": _git_status(upstream_root),
                "package_revision": _git_revision(PACKAGE_ROOT),
                "package_worktree_status": _git_status(PACKAGE_ROOT),
            },
            "implementations": outputs,
            "process_wall_seconds": process_wall,
            "timing_comparison": {
                "upstream_processing_median_seconds": upstream_median,
                "package_processing_median_seconds": package_median,
                "package_over_upstream_time_ratio": package_median / upstream_median,
                "upstream_realtime_factor": audio_seconds / upstream_median,
                "package_realtime_factor": audio_seconds / package_median,
            },
            "stem_comparisons": comparisons,
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path(os.environ["MSST_UPSTREAM_ROOT"])
        if os.environ.get("MSST_UPSTREAM_ROOT")
        else None,
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--limit-seconds",
        type=float,
        default=2.0,
        help="Use only this leading audio fragment; pass 0 to use the entire file.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--bigshifts", type=int, default=1)
    parser.add_argument("--use-tta", action="store_true")
    args = parser.parse_args()
    if args.upstream_root is None:
        parser.error("pass --upstream-root or set MSST_UPSTREAM_ROOT")
    report = compare_model(
        model_type=args.model_type,
        config=args.config,
        checkpoint=args.checkpoint,
        input_audio=args.audio,
        upstream_root=args.upstream_root,
        report_path=args.report,
        repeats=args.repeats,
        limit_seconds=args.limit_seconds,
        threads=args.threads,
        bigshifts=args.bigshifts,
        use_tta=args.use_tta,
    )
    markdown_path = args.report.with_suffix(".md")
    comparison_rows = [
        f"| `{stem['file']}` | {stem['max_abs_error']:.3g} | {stem['rmse']:.3g} | {stem['passed']} |"
        for stem in report["stem_comparisons"]
    ]
    markdown = "\n".join(
        [
            f"# Real-model parity benchmark: {args.model_type}",
            "",
            f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
            f"- Upstream revision: `{report['environment']['upstream_revision']}`",
            f"- Package revision: `{report['environment']['package_revision']}`",
            f"- Audio duration: {report['options']['audio_seconds']:.3f} s; input rate {report['options']['audio_sample_rate']} Hz",
            f"- Checkpoint SHA-256: `{report['options']['checkpoint_sha256']}`",
            "",
            "| Stem | Maximum absolute error | RMSE | Pass |",
            "|---|---:|---:|:---:|",
            *comparison_rows,
            "",
            "| Implementation | Load (s) | Processing median (s) | Real-time factor |",
            "|---|---:|---:|---:|",
            f"| Upstream | {report['implementations']['upstream']['model_load_seconds']:.3f} | {report['timing_comparison']['upstream_processing_median_seconds']:.3f} | {report['timing_comparison']['upstream_realtime_factor']:.3f}x |",
            f"| MSST | {report['implementations']['package']['model_load_seconds']:.3f} | {report['timing_comparison']['package_processing_median_seconds']:.3f} | {report['timing_comparison']['package_realtime_factor']:.3f}x |",
            f"",
            f"MSST/upstream processing-time ratio: {report['timing_comparison']['package_over_upstream_time_ratio']:.3f}x; lower is faster. Timings are descriptive and are not pass/fail thresholds.",
        ]
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Benchmark report: {markdown_path}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
