from __future__ import annotations

import json
from pathlib import Path

import pytest

from watchgpu.environment import (
    CommandResult,
    EnvironmentDiscoveryError,
    PythonSource,
    discover_python,
)
from watchgpu.paths import WatchGPUPaths


class FakeRunner:
    def __init__(self, results: dict[tuple[str, ...], CommandResult]) -> None:
        self.results = results
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> CommandResult:
        self.commands.append(command)
        return self.results.get(command, CommandResult(returncode=127, stderr="not found"))


def _validation_command(python: str) -> tuple[str, ...]:
    return (python, "-c", pytest.importorskip("watchgpu.environment").VALIDATION_SCRIPT)


def _valid() -> CommandResult:
    return CommandResult(
        returncode=0,
        stdout=json.dumps(
            {
                "python": "3.12.1",
                "torch": "2.7.0",
                "cuda": "12.8",
                "cuda_available": True,
                "nvml_available": True,
            }
        ),
    )


def _invalid(reason: str = "CUDA unavailable") -> CommandResult:
    return CommandResult(
        returncode=1,
        stdout=json.dumps(
            {
                "python": "3.12.1",
                "torch": "2.7.0",
                "cuda": "12.8",
                "cuda_available": False,
                "nvml_available": True,
                "errors": [reason],
            }
        ),
    )


def test_watchgpu_python_is_validated_and_takes_priority() -> None:
    requested = "/srv/envs/watchgpu/bin/python"
    runner = FakeRunner({_validation_command(requested): _valid()})

    selection = discover_python(
        environ={"WATCHGPU_PYTHON": requested},
        current_executable="/usr/bin/python3",
        runner=runner,
        find_executable=lambda _name: None,
    )

    assert selection.executable == Path(requested)
    assert selection.source is PythonSource.WATCHGPU_PYTHON
    assert runner.commands == [_validation_command(requested)]


def test_invalid_watchgpu_python_does_not_silently_fall_back() -> None:
    requested = "/broken/python"
    runner = FakeRunner(
        {
            _validation_command(requested): CommandResult(
                returncode=1, stderr="PyTorch is unavailable"
            )
        }
    )

    with pytest.raises(EnvironmentDiscoveryError, match="WATCHGPU_PYTHON"):
        discover_python(
            environ={"WATCHGPU_PYTHON": requested},
            current_executable="/good/current/python",
            runner=runner,
            find_executable=lambda _name: None,
        )

    assert runner.commands == [_validation_command(requested)]


def test_current_interpreter_is_used_when_it_passes_validation() -> None:
    current = "/opt/current/bin/python"
    runner = FakeRunner({_validation_command(current): _valid()})

    selection = discover_python(
        environ={},
        current_executable=current,
        runner=runner,
        find_executable=lambda _name: None,
    )

    assert selection.executable == Path(current)
    assert selection.source is PythonSource.CURRENT
    assert selection.validation.cuda_available is True
    assert selection.validation.nvml_available is True


def test_named_environment_is_discovered_from_environment_manager_json() -> None:
    current = "/usr/bin/python3"
    manager = "/usr/local/bin/mamba"
    candidate = "/srv/users/alice/envs/gpu-tools/bin/python"
    runner = FakeRunner(
        {
            _validation_command(current): _invalid(),
            (manager, "env", "list", "--json"): CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        "envs": [
                            "/srv/users/alice/envs/other",
                            "/srv/users/alice/envs/gpu-tools",
                        ]
                    }
                ),
            ),
            _validation_command(candidate): _valid(),
        }
    )

    selection = discover_python(
        environ={"WATCHGPU_ENV_NAME": "gpu-tools"},
        current_executable=current,
        runner=runner,
        find_executable=lambda name: manager if name == "mamba" else None,
    )

    assert selection.executable == Path(candidate)
    assert selection.source is PythonSource.NAMED_ENVIRONMENT
    assert [check.executable for check in selection.checked] == [
        Path(current),
        Path(candidate),
    ]


def test_environment_manager_scan_is_opt_in() -> None:
    current = "/usr/bin/python3"
    manager = "/usr/local/bin/conda"
    runner = FakeRunner({_validation_command(current): _invalid()})

    with pytest.raises(EnvironmentDiscoveryError, match="WATCHGPU_ENV_NAME"):
        discover_python(
            environ={},
            current_executable=current,
            runner=runner,
            find_executable=lambda name: manager if name == "conda" else None,
        )

    assert (manager, "env", "list", "--json") not in runner.commands


def test_paths_follow_per_host_xdg_directories() -> None:
    paths = WatchGPUPaths.discover(
        environ={
            "XDG_RUNTIME_DIR": "/run/user/1234",
            "XDG_CONFIG_HOME": "/example/users/demo/.xdg-config",
            "XDG_STATE_HOME": "/example/users/demo/.xdg-state",
        },
        home=Path("/example/users/demo"),
        uid=1234,
    )

    assert paths.socket_path == Path("/run/user/1234/watchgpu/watchgpu.sock")
    assert paths.pid_path == Path("/run/user/1234/watchgpu/supervisor.pid")
    assert paths.config_path == Path(
        "/example/users/demo/.xdg-config/watchgpu/config.toml"
    )
    assert paths.log_path == Path("/example/users/demo/.xdg-state/watchgpu/watchgpu.log")
    assert paths.leases_path == Path(
        "/example/users/demo/.xdg-state/watchgpu/leases.json"
    )
    assert paths.restart_state_path == Path(
        "/example/users/demo/.xdg-state/watchgpu/restart.json"
    )
    assert paths.systemd_unit_path == Path(
        "/example/users/demo/.xdg-config/systemd/user/watchgpu.service"
    )


def test_relative_xdg_paths_are_ignored() -> None:
    paths = WatchGPUPaths.discover(
        environ={
            "XDG_RUNTIME_DIR": "relative/run",
            "XDG_CONFIG_HOME": "relative/config",
            "XDG_STATE_HOME": "relative/state",
        },
        home=Path("/example/users/demo"),
        uid=1234,
    )

    assert paths.config_dir == Path("/example/users/demo/.config/watchgpu")
    assert paths.state_dir == Path("/example/users/demo/.local/state/watchgpu")
    assert paths.runtime_dir in {
        Path("/run/user/1234/watchgpu"),
        Path("/tmp/watchgpu-runtime-1234"),
    }


def test_explicit_runtime_directory_preserves_systemd_path() -> None:
    paths = WatchGPUPaths.discover(
        environ={"WATCHGPU_RUNTIME_DIR": "/tmp/private-watchgpu-runtime"},
        home=Path("/example/users/demo"),
        uid=1234,
    )

    assert paths.runtime_dir == Path("/tmp/private-watchgpu-runtime")
