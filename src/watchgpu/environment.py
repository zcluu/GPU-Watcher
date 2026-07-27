from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

VALIDATION_SCRIPT = r"""
import json
import platform

result = {
    "python": platform.python_version(),
    "torch": None,
    "cuda": None,
    "cuda_available": False,
    "nvml_available": False,
}
errors = []
try:
    import torch
    result["torch"] = torch.__version__
    result["cuda"] = torch.version.cuda
    result["cuda_available"] = bool(torch.cuda.is_available())
    if not result["cuda_available"]:
        errors.append("PyTorch cannot access CUDA")
except Exception as exc:
    errors.append(f"PyTorch validation failed: {exc}")
try:
    import pynvml
    pynvml.nvmlInit()
    try:
        pynvml.nvmlDeviceGetCount()
        result["nvml_available"] = True
    finally:
        pynvml.nvmlShutdown()
except Exception as exc:
    errors.append(f"NVML validation failed: {exc}")
result["errors"] = errors
print(json.dumps(result))
raise SystemExit(0 if not errors else 1)
""".strip()


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    def __call__(self, command: tuple[str, ...]) -> CommandResult: ...


class PythonSource(StrEnum):
    WATCHGPU_PYTHON = "WATCHGPU_PYTHON"
    CURRENT = "current"
    NAMED_ENVIRONMENT = "named-environment"


@dataclass(frozen=True, slots=True)
class PythonValidation:
    executable: Path
    valid: bool
    python_version: str | None = None
    torch_version: str | None = None
    cuda_version: str | None = None
    cuda_available: bool = False
    nvml_available: bool = False
    error: str | None = None


@dataclass(frozen=True, slots=True)
class PythonSelection:
    executable: Path
    source: PythonSource
    validation: PythonValidation
    checked: tuple[PythonValidation, ...]


class EnvironmentDiscoveryError(RuntimeError):
    def __init__(self, message: str, *, checked: tuple[PythonValidation, ...]) -> None:
        super().__init__(message)
        self.checked = checked


def run_command(command: tuple[str, ...]) -> CommandResult:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CommandResult(returncode=127, stderr=str(exc))
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def validate_python(
    executable: str | os.PathLike[str], *, runner: CommandRunner = run_command
) -> PythonValidation:
    path = Path(os.path.abspath(Path(executable).expanduser()))
    result = runner((str(path), "-c", VALIDATION_SCRIPT))
    payload: object
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        payload = None
    if not isinstance(payload, dict):
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        return PythonValidation(
            executable=path,
            valid=False,
            error=f"interpreter validation failed: {detail}",
        )

    errors = payload.get("errors")
    error_text = "; ".join(str(value) for value in errors) if isinstance(errors, list) else None
    cuda_available = payload.get("cuda_available") is True
    nvml_available = payload.get("nvml_available") is True
    valid = result.returncode == 0 and cuda_available and nvml_available
    if not valid and not error_text:
        error_text = result.stderr.strip() or "PyTorch, CUDA, or NVML is unavailable"
    return PythonValidation(
        executable=path,
        valid=valid,
        python_version=_optional_string(payload.get("python")),
        torch_version=_optional_string(payload.get("torch")),
        cuda_version=_optional_string(payload.get("cuda")),
        cuda_available=cuda_available,
        nvml_available=nvml_available,
        error=error_text,
    )


def validate_current_python(
    *,
    current_executable: str = sys.executable,
    runner: CommandRunner = run_command,
) -> PythonValidation:
    return validate_python(current_executable, runner=runner)


def discover_python(
    *,
    environ: Mapping[str, str] = os.environ,
    current_executable: str = sys.executable,
    runner: CommandRunner = run_command,
    find_executable: Callable[[str], str | None] = shutil.which,
) -> PythonSelection:
    requested = environ.get("WATCHGPU_PYTHON")
    if requested:
        validation = validate_python(requested, runner=runner)
        if not validation.valid:
            raise EnvironmentDiscoveryError(
                f"WATCHGPU_PYTHON is not usable: {validation.error}",
                checked=(validation,),
            )
        return PythonSelection(
            executable=validation.executable,
            source=PythonSource.WATCHGPU_PYTHON,
            validation=validation,
            checked=(validation,),
        )

    current_validation = validate_current_python(
        current_executable=current_executable,
        runner=runner,
    )
    if current_validation.valid:
        return PythonSelection(
            executable=current_validation.executable,
            source=PythonSource.CURRENT,
            validation=current_validation,
            checked=(current_validation,),
        )

    checked = [current_validation]
    seen = {current_validation.executable}
    environment_name = environ.get("WATCHGPU_ENV_NAME", "").strip()
    if environment_name:
        for candidate in _named_environment_candidates(
            environment_name,
            runner=runner,
            find_executable=find_executable,
        ):
            if candidate in seen:
                continue
            seen.add(candidate)
            validation = validate_python(candidate, runner=runner)
            checked.append(validation)
            if validation.valid:
                return PythonSelection(
                    executable=validation.executable,
                    source=PythonSource.NAMED_ENVIRONMENT,
                    validation=validation,
                    checked=tuple(checked),
                )
    raise EnvironmentDiscoveryError(
        "no usable WatchGPU Python interpreter was found; activate an environment "
        "with CUDA-enabled PyTorch, set WATCHGPU_PYTHON=/absolute/path/to/python, "
        "or set WATCHGPU_ENV_NAME to a conda/mamba environment name",
        checked=tuple(checked),
    )


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _named_environment_candidates(
    environment_name: str,
    *,
    runner: CommandRunner,
    find_executable: Callable[[str], str | None],
) -> tuple[Path, ...]:
    candidates: list[Path] = []
    for manager_name in ("mamba", "conda", "micromamba"):
        manager = find_executable(manager_name)
        if manager is None:
            continue
        result = runner((manager, "env", "list", "--json"))
        if result.returncode != 0:
            continue
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("envs"), list):
            continue
        for raw_path in payload["envs"]:
            if not isinstance(raw_path, str):
                continue
            environment_path = Path(raw_path).expanduser()
            if environment_path.name == environment_name:
                candidates.append(environment_path / "bin" / "python")
    return tuple(candidates)
