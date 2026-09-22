"""Numerical parity checks for shared data, loss, and training helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("utility_runner.py")


def _run_utility(
    *, implementation: str, source_root: Path, audio_path: Path, output_path: Path
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--implementation",
            implementation,
            "--source-root",
            str(source_root),
            "--audio",
            str(audio_path),
            "--output",
            str(output_path),
        ],
        check=True,
        cwd=source_root,
    )


def test_shared_data_losses_and_optimizer_match_upstream(tmp_path: Path, record_property) -> None:
    upstream_value = os.environ.get("MSST_UPSTREAM_ROOT")
    if not upstream_value:
        pytest.skip("Set MSST_UPSTREAM_ROOT to run shared-utility parity tests")
    upstream_root = Path(upstream_value).resolve()
    if not (upstream_root / "utils").is_dir():
        pytest.fail(f"MSST_UPSTREAM_ROOT is not a source checkout: {upstream_root}")

    generator = np.random.default_rng(918)
    audio = (generator.standard_normal((12000, 2)) * 0.12).astype(np.float32)
    audio[:, 0] += 0.21
    audio[:, 1] -= 0.16
    audio_path = tmp_path / "audio-fixture.wav"
    sf.write(audio_path, audio, 44100, subtype="FLOAT")

    upstream_path = tmp_path / "shared-upstream.pt"
    package_path = tmp_path / "shared-package.pt"
    _run_utility(
        implementation="upstream",
        source_root=upstream_root,
        audio_path=audio_path,
        output_path=upstream_path,
    )
    _run_utility(
        implementation="package",
        source_root=PACKAGE_ROOT,
        audio_path=audio_path,
        output_path=package_path,
    )

    upstream = torch.load(upstream_path, map_location="cpu", weights_only=False)
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    assert package["dataset_load_chunk_available"] == upstream["dataset_load_chunk_available"]
    assert package["arrays"].keys() == upstream["arrays"].keys()
    for name in upstream["arrays"]:
        torch.testing.assert_close(
            package["arrays"][name], upstream["arrays"][name], rtol=1e-6, atol=1e-7
        )
    assert package["timings"].keys() == upstream["timings"].keys()
    for operation in upstream["timings"]:
        record_property(f"upstream_{operation}", upstream["timings"][operation])
        record_property(f"package_{operation}", package["timings"][operation])
        record_property(
            f"package_over_upstream_{operation}",
            package["timings"][operation] / max(upstream["timings"][operation], 1e-12),
        )
