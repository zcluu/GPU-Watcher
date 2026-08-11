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
    assert "Release WatchGPU-owned memory" in result.stdout


def test_config_help_uses_short_names_and_explains_cpu_controls() -> None:
    result = CliRunner().invoke(app, ["config", "set", "--help"])

    assert result.exit_code == 0
    for option in ("--free", "--limit", "--duty", "--cpu-limit", "--cpu-target"):
        assert option in result.stdout
    assert "health-work" in result.stdout
    assert "Hard CPU" in result.stdout
    assert "--maintenance-cpu" not in result.stdout


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


def test_config_set_short_cpu_alias_hot_applies_target(
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
                    "revision": 2,
                    "config": WatchGPUConfig(
                        gpus=[GPUConfig(selector="GPU-0")]
                    ).model_dump(mode="json"),
                }
            }
        return {"status": "APPLIED", "revision": 3}

    monkeypatch.setattr(cli_module, "_client_call", client_call)
    result = CliRunner().invoke(
        app,
        ["config", "set", "-c", "75", "--runtime-only"],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0, result.output
    params = calls[1][1]
    assert params is not None
    config = params["config"]
    assert isinstance(config, dict)
    assert config["maintenance_cpu_target_percent"] == 75


@pytest.mark.parametrize(
    ("key", "value", "expected_key", "expected_value"),
    [
        (
            "maintenance_cpu_target_percent",
            "60",
            "maintenance_cpu_target_percent",
            60,
        ),
        ("cpu_budget_percent", "80", "cpu_budget_percent", 80),
        (
            "maintenance_compute_enabled",
            "false",
            "maintenance_compute_enabled",
            False,
        ),
        ("leave_free_mib", "2.5", "leave_free_mib", 2560),
    ],
)
def test_config_set_accepts_config_show_key_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    key: str,
    value: str,
    expected_key: str,
    expected_value: object,
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
                    "revision": 3,
                    "config": WatchGPUConfig(
                        gpus=[GPUConfig(selector="GPU-0")]
                    ).model_dump(mode="json"),
                }
            }
        return {"status": "APPLIED", "revision": 4}

    monkeypatch.setattr(cli_module, "_client_call", client_call)
    result = CliRunner().invoke(
        app,
        ["config", "set", key, value, "--runtime-only"],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0, result.output
    params = calls[1][1]
    assert params is not None
    config = params["config"]
    assert isinstance(config, dict)
    assert config[expected_key] == expected_value


def test_config_set_direct_key_reports_unknown_or_invalid_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = WatchGPUConfig(gpus=[GPUConfig(selector="GPU-0")])
    monkeypatch.setattr(
        cli_module,
        "_client_call",
        lambda _path, method, _params=None: (
            {"policy": {"revision": 1, "config": current.model_dump(mode="json")}}
            if method == "status.get"
            else {"status": "APPLIED"}
        ),
    )
    env = {
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }

    unknown = CliRunner().invoke(app, ["config", "set", "not_a_key", "1"], env=env)
    invalid = CliRunner().invoke(
        app, ["config", "set", "cpu_budget_percent", "101"], env=env
    )

    assert unknown.exit_code != 0
    assert "unknown config key" in unknown.output
    assert invalid.exit_code != 0
    assert "less than or equal to 100" in invalid.output


def test_config_set_accepts_key_equals_value_form(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = WatchGPUConfig(gpus=[GPUConfig(selector="GPU-0")])
    calls: list[tuple[str, dict[str, object] | None]] = []

    def client_call(
        _path: Path, method: str, params: dict[str, object] | None = None
    ) -> dict[str, object]:
        calls.append((method, params))
        if method == "status.get":
            return {
                "policy": {"revision": 1, "config": current.model_dump(mode="json")}
            }
        return {"status": "APPLIED", "revision": 2}

    monkeypatch.setattr(cli_module, "_client_call", client_call)
    result = CliRunner().invoke(
        app,
        ["config", "set", "maintenance_cpu_target_percent=40"],
        env={
            "XDG_RUNTIME_DIR": str(tmp_path / "run"),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_STATE_HOME": str(tmp_path / "state"),
        },
    )

    assert result.exit_code == 0, result.output
    params = calls[1][1]
    assert params is not None
    config = params["config"]
    assert isinstance(config, dict)
    assert config["maintenance_cpu_target_percent"] == 40


def test_config_set_direct_key_persists_while_daemon_is_stopped(tmp_path: Path) -> None:
    env = {
        "XDG_RUNTIME_DIR": str(tmp_path / "run"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
    }

    result = CliRunner().invoke(
        app,
        ["config", "set", "maintenance_cpu_target_percent", "35"],
        env=env,
    )

    assert result.exit_code == 0, result.output
    saved = load_config(tmp_path / "config" / "watchgpu" / "config.toml")
    assert saved.maintenance_cpu_target_percent == 35


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
