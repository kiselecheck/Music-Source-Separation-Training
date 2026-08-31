"""Portable experiment layout and optional local run tracking."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import sys
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml  # type: ignore[import-untyped]

from ._version import __version__

PathValue = str | PathLike[str]
EXPERIMENT_DIRECTORIES = (
    "configs",
    "checkpoints",
    "jobs",
    "logs",
    "results",
    "runs",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Experiment:
    """Resolved MSST experiment directory."""

    path: Path
    name: str
    manifest_path: Path

    def directory(self, name: str) -> Path:
        if name not in EXPERIMENT_DIRECTORIES:
            raise KeyError(f"Unknown experiment directory: {name}")
        return self.path / name


def init_experiment(path: PathValue, *, name: str | None = None) -> Experiment:
    """Create a cluster-neutral experiment layout.

    No scheduler-specific scripts are generated. The empty ``jobs`` directory
    can hold local scripts, Slurm jobs, or another scheduler's files.
    """

    root = Path(path).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "experiment.yaml"
    if manifest_path.exists():
        raise FileExistsError(f"Experiment already exists: {manifest_path}")
    experiment_name = (name or root.name).strip()
    if not experiment_name:
        raise ValueError("Experiment name cannot be empty")

    for directory in EXPERIMENT_DIRECTORIES:
        (root / directory).mkdir(exist_ok=True)
    manifest = {
        "schema_version": 1,
        "name": experiment_name,
        "created_at": _now(),
        "paths": {directory: directory for directory in EXPERIMENT_DIRECTORIES},
    }
    manifest_path.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return Experiment(root, experiment_name, manifest_path)


def load_experiment(path: PathValue | Experiment) -> Experiment:
    """Load and validate an existing experiment directory."""

    if isinstance(path, Experiment):
        return path
    root = Path(path).expanduser().resolve()
    manifest_path = root / "experiment.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"MSST experiment manifest does not exist: {manifest_path}. "
            "Create it with 'msst init <directory>'."
        )
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError(f"Unsupported experiment manifest: {manifest_path}")
    name = str(manifest.get("name", "")).strip()
    if not name:
        raise ValueError(f"Experiment name is missing in {manifest_path}")
    for directory in EXPERIMENT_DIRECTORIES:
        (root / directory).mkdir(exist_ok=True)
    return Experiment(root, name, manifest_path)


class ExperimentRun:
    """A run record that updates ``run.json`` on success or failure."""

    def __init__(
        self,
        experiment: Experiment,
        task: str,
        parameters: Mapping[str, Any],
        run_name: str | None = None,
    ) -> None:
        self.experiment = experiment
        self.task = task
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        suffix = uuid.uuid4().hex[:8]
        label = re.sub(r"[^A-Za-z0-9._-]+", "-", (run_name or task).strip())
        label = label.strip("._-") or task
        self.path = experiment.directory("runs") / f"{timestamp}-{label}-{suffix}"
        self.path.mkdir(parents=False)
        self.record_path = self.path / "run.json"
        self._record: dict[str, Any] = {
            "schema_version": 1,
            "run_id": self.path.name,
            "experiment": experiment.name,
            "task": task,
            "status": "running",
            "started_at": _now(),
            "parameters": _json_value(parameters),
            "environment": self._environment(),
            "artifacts": [],
        }
        self._snapshot_inputs(parameters)
        self._write()

    @staticmethod
    def _environment() -> dict[str, Any]:
        environment: dict[str, Any] = {
            "msst": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "hostname": socket.gethostname(),
            "executable": sys.executable,
        }
        slurm = {
            key: value
            for key, value in os.environ.items()
            if key.startswith("SLURM_")
        }
        if slurm:
            environment["slurm"] = slurm
        try:
            import torch

            environment["torch"] = torch.__version__
            environment["cuda_build"] = torch.version.cuda
            environment["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                environment["gpus"] = [
                    torch.cuda.get_device_name(index)
                    for index in range(torch.cuda.device_count())
                ]
        except Exception as error:
            environment["torch_error"] = str(error)
        return environment

    def _snapshot_inputs(self, parameters: Mapping[str, Any]) -> None:
        inputs: dict[str, Any] = {}
        for key in ("config_path", "checkpoint_path", "start_check_point"):
            value = parameters.get(key)
            if not value:
                continue
            source = Path(str(value)).expanduser()
            if not source.is_file():
                continue
            inputs[key] = {
                "path": str(source.resolve()),
                "sha256": _sha256(source),
                "size": source.stat().st_size,
            }
            if key == "config_path":
                snapshot = self.path / "config.yaml"
                shutil.copy2(source, snapshot)
                inputs[key]["snapshot"] = snapshot.name
        if inputs:
            self._record["inputs"] = inputs

    def add_artifacts(self, paths: Any) -> None:
        """Add one or more output paths to the run record."""

        if paths is None:
            return
        values = paths if isinstance(paths, (list, tuple, set)) else [paths]
        known = set(self._record["artifacts"])
        for value in values:
            path = str(value)
            if path not in known:
                self._record["artifacts"].append(path)
                known.add(path)
        self._write()

    def set_result(self, result: Any) -> None:
        self._record["result"] = _json_value(result)
        self._write()

    def finish(self, error: BaseException | None = None) -> None:
        self._record["finished_at"] = _now()
        if error is None:
            self._record["status"] = "completed"
        else:
            self._record["status"] = "failed"
            self._record["error"] = {
                "type": type(error).__name__,
                "message": str(error),
            }
        self._write()

    def _write(self) -> None:
        temporary = self.record_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.record_path)

    def __enter__(self) -> "ExperimentRun":
        return self

    def __exit__(self, error_type, error, traceback) -> Literal[False]:
        self.finish(error)
        return False


def track_run(
    experiment: PathValue | Experiment | None,
    *,
    task: str,
    parameters: Mapping[str, Any],
    run_name: str | None = None,
) -> ExperimentRun | None:
    """Create a run tracker, or return ``None`` when tracking is disabled."""

    if experiment is None:
        return None
    return ExperimentRun(load_experiment(experiment), task, parameters, run_name)


@contextmanager
def tracked_run(
    experiment: PathValue | Experiment | None,
    *,
    task: str,
    parameters: Mapping[str, Any],
    run_name: str | None = None,
):
    """Context manager form of :func:`track_run`."""

    run = track_run(
        experiment,
        task=task,
        parameters=parameters,
        run_name=run_name,
    )
    if run is None:
        yield None
    else:
        with run:
            yield run


__all__ = [
    "EXPERIMENT_DIRECTORIES",
    "Experiment",
    "ExperimentRun",
    "init_experiment",
    "load_experiment",
    "track_run",
    "tracked_run",
]
