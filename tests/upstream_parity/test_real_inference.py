"""Opt-in integration tests using locally available release checkpoints."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = Path(__file__).with_name("benchmark_real_inference.py")
UPSTREAM_ENV = "MSST_UPSTREAM_ROOT"


@pytest.mark.parametrize(
    ("model_type", "config_env", "checkpoint_env"),
    (
        ("mdx23c", "MSST_PARITY_MDX23C_CONFIG", "MSST_PARITY_MDX23C_CHECKPOINT"),
        (
            "scnet_masked",
            "MSST_PARITY_SCNET_MASKED_CONFIG",
            "MSST_PARITY_SCNET_MASKED_CHECKPOINT",
        ),
        (
            "bs_roformer",
            "MSST_PARITY_BS_ROFORMER_CONFIG",
            "MSST_PARITY_BS_ROFORMER_CHECKPOINT",
        ),
    ),
)
def test_release_checkpoint_folder_inference_matches_upstream(
    model_type: str,
    config_env: str,
    checkpoint_env: str,
    tmp_path: Path,
    record_property,
) -> None:
    upstream_value = os.environ.get(UPSTREAM_ENV)
    config_value = os.environ.get(config_env)
    checkpoint_value = os.environ.get(checkpoint_env)
    if not upstream_value or not config_value or not checkpoint_value:
        pytest.skip(
            f"Set {UPSTREAM_ENV}, {config_env}, and {checkpoint_env} to run "
            "the real-checkpoint folder-inference comparison"
        )

    report_dir = Path(os.environ.get("MSST_PARITY_REPORT_DIR", tmp_path)).resolve()
    report_path = report_dir / f"{model_type}-real-inference-parity.json"
    command = [
        sys.executable,
        str(BENCHMARK),
        "--model-type",
        model_type,
        "--config",
        config_value,
        "--checkpoint",
        checkpoint_value,
        "--upstream-root",
        upstream_value,
        "--report",
        str(report_path),
        "--repeats",
        os.environ.get("MSST_PARITY_REPEATS", "2"),
        "--limit-seconds",
        os.environ.get("MSST_PARITY_AUDIO_SECONDS", "2"),
        "--threads",
        os.environ.get("MSST_PARITY_THREADS", "1"),
        "--bigshifts",
        os.environ.get("MSST_PARITY_BIGSHIFTS", "1"),
    ]
    audio_value = os.environ.get("MSST_PARITY_AUDIO")
    if audio_value:
        command.extend(("--audio", audio_value))
    if os.environ.get("MSST_PARITY_USE_TTA", "0").lower() in {"1", "true", "yes"}:
        command.append("--use-tta")

    result = subprocess.run(
        command,
        cwd=PACKAGE_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode:
        pytest.fail(
            f"Real-model {model_type} parity benchmark failed.\n"
            f"stdout:\n{result.stdout[-6000:]}\n"
            f"stderr:\n{result.stderr[-6000:]}"
        )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["passed"]
    record_property("report", str(report_path))
    record_property(
        "package_over_upstream_time_ratio",
        report["timing_comparison"]["package_over_upstream_time_ratio"],
    )
