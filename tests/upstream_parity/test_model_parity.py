"""Numerical model checks against a separate ZFTurbo checkout.

Run with MSST_UPSTREAM_ROOT pointing to a clean checkout of
ZFTurbo/Music-Source-Separation-Training. The tests stay outside the installed
``msst`` namespace and compare identical weights and input tensors.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("model_runner.py")
UPSTREAM_ENV = "MSST_UPSTREAM_ROOT"


def upstream_root() -> Path:
    value = os.environ.get(UPSTREAM_ENV)
    if not value:
        pytest.skip(f"Set {UPSTREAM_ENV} to a ZFTurbo checkout to run parity tests")
    root = Path(value).resolve()
    if not (root / "models").is_dir() or not (root / "utils").is_dir():
        pytest.fail(f"{UPSTREAM_ENV} is not a ZFTurbo checkout: {root}")
    return root


def run_model(
    *,
    implementation: str,
    source_root: Path,
    model_case: str,
    output: Path,
    reference: Path | None = None,
    pipeline: str = "forward",
    bigshifts: int = 2,
    use_tta: bool = False,
) -> None:
    command = [
        sys.executable,
        str(RUNNER),
        "--implementation",
        implementation,
        "--source-root",
        str(source_root),
        "--case",
        model_case,
        "--pipeline",
        pipeline,
        "--bigshifts",
        str(bigshifts),
        "--output",
        str(output),
    ]
    if use_tta:
        command.append("--use-tta")
    if reference is not None:
        command.extend(("--reference", str(reference)))
    subprocess.run(command, check=True, cwd=source_root)


@pytest.mark.parametrize("model_case", ("dttnet", "mdx23c", "scnet"))
def test_forward_output_matches_upstream(
    model_case: str, tmp_path: Path, record_property
) -> None:
    reference_path = tmp_path / f"{model_case}-upstream.pt"
    package_path = tmp_path / f"{model_case}-package.pt"

    run_model(
        implementation="upstream",
        source_root=upstream_root(),
        model_case=model_case,
        output=reference_path,
    )
    run_model(
        implementation="package",
        source_root=PACKAGE_ROOT,
        model_case=model_case,
        output=package_path,
        reference=reference_path,
    )

    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    package = torch.load(package_path, map_location="cpu", weights_only=False)

    assert package["output"].shape == reference["output"].shape
    assert package["state_dict"].keys() == reference["state_dict"].keys()
    torch.testing.assert_close(package["output"], reference["output"], rtol=1e-5, atol=1e-6)
    record_property(
        "upstream_model_init_seconds", reference["timings"]["model_init_seconds"]
    )
    record_property(
        "package_model_init_seconds", package["timings"]["model_init_seconds"]
    )
    record_property("upstream_forward_seconds", reference["timings"]["inference_seconds"])
    record_property("package_forward_seconds", package["timings"]["inference_seconds"])
    record_property(
        "package_to_upstream_forward_time_ratio",
        package["timings"]["inference_seconds"]
        / reference["timings"]["inference_seconds"],
    )


@pytest.mark.parametrize("model_case", ("dttnet", "mdx23c", "scnet"))
@pytest.mark.parametrize(
    ("bigshifts", "use_tta"),
    ((1, False), (2, True)),
    ids=("overlap-and-padding", "bigshifts-plus-tta"),
)
def test_chunked_inference_matches_upstream(
    model_case: str,
    bigshifts: int,
    use_tta: bool,
    tmp_path: Path,
    record_property,
) -> None:
    reference_path = tmp_path / f"{model_case}-inference-upstream.pt"
    package_path = tmp_path / f"{model_case}-inference-package.pt"

    run_model(
        implementation="upstream",
        source_root=upstream_root(),
        model_case=model_case,
        output=reference_path,
        pipeline="inference",
        bigshifts=bigshifts,
        use_tta=use_tta,
    )
    run_model(
        implementation="package",
        source_root=PACKAGE_ROOT,
        model_case=model_case,
        output=package_path,
        reference=reference_path,
        pipeline="inference",
        bigshifts=bigshifts,
        use_tta=use_tta,
    )

    reference = torch.load(reference_path, map_location="cpu", weights_only=False)
    package = torch.load(package_path, map_location="cpu", weights_only=False)
    assert package["output"].keys() == reference["output"].keys()
    for stem in reference["output"]:
        assert package["output"][stem].shape == reference["output"][stem].shape
        torch.testing.assert_close(
            package["output"][stem], reference["output"][stem], rtol=1e-5, atol=1e-6
        )

    record_property("upstream_inference_seconds", reference["timings"]["inference_seconds"])
    record_property("package_inference_seconds", package["timings"]["inference_seconds"])
    record_property(
        "package_to_upstream_speed_ratio",
        package["timings"]["inference_seconds"]
        / reference["timings"]["inference_seconds"],
    )
