from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import watchgpu.cli as cli_module
from watchgpu.cli import app
from watchgpu.config import GPUConfig, WatchGPUConfig, load_config, save_config
from watchgpu.models import GPUSnapshot


def test_status_json_reports_stopped_without_a_daemon(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["status", "--json"],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout)["daemon"] == "STOPPED"


def test_root_help_exposes_main_operator_workflows() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    for command in ("doctor", "start", "status", "console", "stop", "request"):
        assert command in result.stdout


def test_start_dry_run_resolves_gpu_without_writing_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeObserver:
        def __enter__(self) -> FakeObserver:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def snapshots(self) -> tuple[GPUSnapshot, ...]:
            return (GPUSnapshot(0, "GPU-host", "A40", 46_068, 20_000, 0),)

    monkeypatch.setattr(cli_module, "NVMLGPUObserver", FakeObserver)
    config_home = tmp_path / "config"
    result = CliRunner().invoke(
        app,
        ["start", "--gpus", "0", "--leave-free", "2.5", "--dry-run"],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(config_home),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0
    assert "GPU-host" in result.stdout
    assert "2560 MiB" in result.stdout
    assert not (config_home / "watchgpu" / "config.toml").exists()


def test_config_set_submits_full_candidate_with_expected_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []

    def client_call(
        _socket_path: Path,
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        calls.append((method, params))
        if method == "status.get":
            return {
                "policy": {
                    "revision": 7,
                    "config": WatchGPUConfig(
                        gpus=[GPUConfig(selector="GPU-0")]
                    ).model_dump(mode="json"),
                }
            }
        return {"status": "APPLIED", "revision": 8}

    monkeypatch.setattr(cli_module, "_client_call", client_call)
    result = CliRunner().invoke(
        app,
        [
            "config",
            "set",
            "--gpu",
            "GPU-0",
            "--leave-free",
            "3GiB",
            "--runtime-only",
        ],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0
    _method, params = calls[1]
    assert params is not None
    assert params["expected_revision"] == 7
    assert params["save"] is False
    config = params["config"]
    assert isinstance(config, dict)
    assert config["gpus"][0]["leave_free_mib"] == 3072


def test_config_set_works_offline_and_persists_gpu_uuid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeObserver:
        def __enter__(self) -> FakeObserver:
            return self

        def __exit__(self, *_args: object) -> None:
            pass

        def snapshots(self) -> tuple[GPUSnapshot, ...]:
            return (GPUSnapshot(0, "GPU-host", "A40", 46_068, 20_000, 0),)

    monkeypatch.setattr(cli_module, "NVMLGPUObserver", FakeObserver)
    env = {
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    result = CliRunner().invoke(
        app,
        [
            "config",
            "set",
            "--gpus",
            "0",
            "--gpu",
            "0",
            "--reserve-limit",
            "16MiB",
        ],
        env=env,
    )

    assert result.exit_code == 0, result.output
    saved = load_config(tmp_path / "config" / "watchgpu" / "config.toml")
    assert saved.gpus == [
        GPUConfig(selector="GPU-host", reserve_limit_mib=16)
    ]


def test_restart_schedule_can_be_set_shown_and_disabled_while_stopped(
    tmp_path: Path,
) -> None:
    env = {
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }
    config_path = tmp_path / "config" / "watchgpu" / "config.toml"
    save_config(WatchGPUConfig(gpus=[GPUConfig(selector="GPU-0")]), config_path)
    runner = CliRunner()

    configured = runner.invoke(
        app,
        [
            "restart",
            "schedule",
            "set",
            "--at",
            "04:00",
            "--jitter",
            "20m",
            "--defer-while-leased",
        ],
        env=env,
    )
    shown = runner.invoke(app, ["restart", "schedule", "show", "--json"], env=env)
    disabled = runner.invoke(app, ["restart", "schedule", "disable"], env=env)

    assert configured.exit_code == 0, configured.output
    assert shown.exit_code == 0, shown.output
    shown_payload = json.loads(shown.stdout)
    assert shown_payload["daemon"] == "STOPPED"
    assert shown_payload["config"] == {
        "at": "04:00",
        "defer_while_leased": True,
        "enabled": True,
        "jitter_seconds": 1200,
    }
    assert disabled.exit_code == 0, disabled.output
    persisted = load_config(config_path).maintenance_restart
    assert persisted.enabled is False
    assert persisted.at == "04:00"
    assert persisted.jitter_seconds == 1200


def test_restart_schedule_set_hot_applies_with_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict[str, object] | None]] = []
    current = WatchGPUConfig(gpus=[GPUConfig(selector="GPU-0")])

    def client_call(
        _socket_path: Path,
        method: str,
        params: dict[str, object] | None = None,
    ) -> dict[str, object]:
        calls.append((method, params))
        if method == "status.get":
            return {
                "policy": {
                    "revision": 4,
                    "config": current.model_dump(mode="json"),
                },
                "maintenance_restart": {"state": "DISABLED"},
            }
        return {"status": "APPLIED", "revision": 5}

    monkeypatch.setattr(cli_module, "_client_call", client_call)
    result = CliRunner().invoke(
        app,
        ["restart", "schedule", "set", "--at", "05:30", "--jitter", "90s"],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0, result.output
    assert [method for method, _params in calls] == ["status.get", "policy.apply"]
    params = calls[1][1]
    assert params is not None
    assert params["expected_revision"] == 4
    assert params["save"] is True
    candidate = params["config"]
    assert isinstance(candidate, dict)
    assert candidate["maintenance_restart"] == {
        "enabled": True,
        "at": "05:30",
        "jitter_seconds": 90,
        "defer_while_leased": True,
    }
