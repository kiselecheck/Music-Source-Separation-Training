"""Validate the contents and metadata of an MSST wheel before publication."""

from __future__ import annotations

import argparse
import email
import zipfile
from pathlib import Path


REQUIRED_FILES = {
    "msst/__init__.py",
    "msst/_inference.py",
    "msst/_training.py",
    "msst/_validation.py",
    "msst/_version.py",
    "msst/errors.py",
    "msst/inference.py",
    "msst/py.typed",
    "msst/validation.py",
    "msst/models/bs_roformer/bs_roformer.py",
    "msst/training.py",
    "msst/utils/dataset.py",
    "msst/utils/settings.py",
}
FORBIDDEN_PREFIXES = (
    "configs/",
    "gui/",
    "models/",
    "scripts/",
    "tests/",
    "utils/",
)
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", ".idea", "wandb"}
FORBIDDEN_SUFFIXES = {".ckpt", ".ipynb", ".pyc"}


def check_wheel(path: Path) -> None:
    if path.suffix != ".whl" or not path.is_file():
        raise ValueError(f"Expected a wheel path, received: {path}")

    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        missing = sorted(REQUIRED_FILES - names)
        if missing:
            raise RuntimeError(f"Wheel is missing runtime files: {missing}")

        unexpected = sorted(
            name
            for name in names
            if name.startswith(FORBIDDEN_PREFIXES)
            or any(part in FORBIDDEN_PARTS for part in Path(name).parts)
            or Path(name).suffix in FORBIDDEN_SUFFIXES
        )
        if unexpected:
            raise RuntimeError(
                f"Wheel contains non-runtime project files: {unexpected[:10]}"
            )

        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Expected one METADATA file, found {metadata_names}")
        metadata = email.message_from_bytes(archive.read(metadata_names[0]))
        if metadata["Name"] != "msst":
            raise RuntimeError(f"Unexpected distribution name: {metadata['Name']!r}")
        if not metadata["Version"]:
            raise RuntimeError("Distribution version is empty")

        entry_points = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(entry_points) != 1:
            raise RuntimeError("Wheel must contain exactly one entry_points.txt")
        scripts = archive.read(entry_points[0]).decode("utf-8")
        for command in ("msst-inference", "msst-train", "msst-valid"):
            if command not in scripts:
                raise RuntimeError(f"Wheel does not define {command!r}")

    print(f"Validated {path.name}: msst {metadata['Version']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    check_wheel(args.wheel)


if __name__ == "__main__":
    main()
