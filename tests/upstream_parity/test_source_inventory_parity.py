"""Guard the set of upstream model families and shipped configs."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def upstream_root() -> Path:
    value = os.environ.get("MSST_UPSTREAM_ROOT")
    if not value:
        pytest.skip("Set MSST_UPSTREAM_ROOT to run source-inventory parity checks")
    root = Path(value).resolve()
    if not (root / "utils" / "settings.py").is_file():
        pytest.fail(f"MSST_UPSTREAM_ROOT is not a source checkout: {root}")
    return root


def _model_types(settings_path: Path) -> set[str]:
    source = settings_path.read_text(encoding="utf-8")
    function_name = (
        "build_model_from_config"
        if "def build_model_from_config(" in source
        else "get_model_from_config"
    )
    start = source.index(f"def {function_name}(")
    next_function = source.find("\ndef ", start + 1)
    factory = source[start:] if next_function < 0 else source[start:next_function]
    return set(
        re.findall(r"(?:if|elif)\s+model_type\s*==\s*['\"]([^'\"]+)['\"]", factory)
    )


def test_package_registers_the_same_model_families_as_upstream() -> None:
    upstream = _model_types(upstream_root() / "utils" / "settings.py")
    package = _model_types(PACKAGE_ROOT / "msst" / "utils" / "settings.py")
    assert package == upstream


def test_all_configs_shared_with_upstream_are_byte_identical() -> None:
    upstream_configs = upstream_root() / "configs"
    package_configs = PACKAGE_ROOT / "configs"
    files = sorted(
        path.relative_to(upstream_configs)
        for path in upstream_configs.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    )
    assert files, "No YAML config files found in upstream checkout"
    for relative_path in files:
        package_path = package_configs / relative_path
        assert package_path.is_file(), f"Missing upstream config: {relative_path}"
        package_text = package_path.read_text(encoding="utf-8").replace("\r\n", "\n").rstrip("\n")
        upstream_text = (
            (upstream_configs / relative_path)
            .read_text(encoding="utf-8")
            .replace("\r\n", "\n")
            .rstrip("\n")
        )
        assert package_text == upstream_text, (
            f"Config differs from upstream: {relative_path}"
        )

    # DTTNet is registered in both source trees; there should be no one-sided config.
    package_only = {
        path.relative_to(package_configs)
        for path in package_configs.rglob("*")
        if path.is_file() and path.suffix.lower() in {".yaml", ".yml"}
    } - set(files)
    assert package_only == set()
