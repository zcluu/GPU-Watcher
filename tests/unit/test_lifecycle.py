from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

from watchgpu.environment import CommandResult
from watchgpu.lifecycle import (
    BackgroundMode,
    DetachedPidStore,
    DetachedProcessState,
    LeaseStateStore,
    LingerState,
    PersistedLease,
    PersistedLeaseState,
    RestartSchedule,
    RestartScheduler,
    RestartScheduleState,
    RestartStateStore,
    ShutdownResult,
    ShutdownResultStore,
    choose_background_mode,
    probe_user_systemd,
    render_systemd_user_unit,
)


class FakeRunner:
    def __init__(self, results: dict[tuple[str, ...], CommandResult]) -> None:
        self.results = results
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...]) -> CommandResult:
        self.commands.append(command)
        return self.results.get(command, CommandResult(returncode=127, stderr="not found"))


def test_auto_mode_prefers_available_user_systemd_and_reports_linger() -> None:
    runner = FakeRunner(
        {
            ("systemctl", "--user", "show-environment"): CommandResult(returncode=0),
            ("loginctl", "show-user", "alice", "-p", "Linger", "--value"): CommandResult(
                returncode=0, stdout="yes\n"
            ),
        }
    )

    probe = probe_user_systemd(username="alice", runner=runner)

    assert probe.available is True
    assert probe.linger is LingerState.YES
    assert probe.session_bound is False
    assert choose_background_mode(BackgroundMode.AUTO, probe) is BackgroundMode.SYSTEMD_USER


def test_auto_mode_falls_back_to_detached_without_user_systemd() -> None:
    runner = FakeRunner(
        {
            ("systemctl", "--user", "show-environment"): CommandResult(
                returncode=1, stderr="Failed to connect to bus"
            )
        }
    )

    probe = probe_user_systemd(username="alice", runner=runner)

    assert probe.available is False
    assert probe.linger is LingerState.UNKNOWN
    assert probe.session_bound is True
    assert choose_background_mode("auto", probe) is BackgroundMode.DETACHED


def test_user_unit_uses_discovered_python_without_shell_activation() -> None:
    unit = render_systemd_user_unit(
        Path("/srv/alice/envs/gpu-tools/bin/python"),
        daemon_args=("--config", "/srv/alice/config files/watchgpu.toml"),
        cpu_quota_percent=80,
        environment={
            "XDG_CONFIG_HOME": "/srv/alice/config files",
            "WATCHGPU_RUNTIME_DIR": "/run/user/1234/watchgpu",
            "XDG_STATE_HOME": "/srv/alice/state",
        },
    )

    assert (
        'ExecStart="/srv/alice/envs/gpu-tools/bin/python" -m watchgpu.cli daemon '
        '--foreground --config "/srv/alice/config files/watchgpu.toml"'
    ) in unit
    assert "CPUQuota=80%" in unit
    assert "Environment=OMP_NUM_THREADS=1" in unit
    assert 'Environment="XDG_CONFIG_HOME=/srv/alice/config files"' in unit
    assert 'Environment="WATCHGPU_RUNTIME_DIR=/run/user/1234/watchgpu"' in unit
    assert 'Environment="XDG_STATE_HOME=/srv/alice/state"' in unit
    assert "source " not in unit
    assert "/etc/systemd" not in unit


def test_detached_pid_record_rejects_a_reused_or_stale_pid(tmp_path: Path) -> None:
    starts = {4321: 987_654}
    store = DetachedPidStore(
        tmp_path / "runtime" / "supervisor.pid",
        start_time_reader=lambda pid: starts.get(pid),
    )

    record = store.write_for_process(4321)

    assert record.pid == 4321
    assert record.start_time_ticks == 987_654
    assert store.inspect().state is DetachedProcessState.RUNNING
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert list(store.path.parent.glob("*.tmp")) == []

    starts[4321] = 987_655
    status = store.inspect()

    assert status.state is DetachedProcessState.STALE
    assert status.record == record
    assert status.observed_start_time_ticks == 987_655


def test_scheduled_restart_uses_fixed_time_jitter_and_defers_for_leases() -> None:
    scheduler = RestartScheduler(
        RestartSchedule(
            enabled=True,
            at="04:00",
            jitter_seconds=1200,
            defer_while_leased=True,
        ),
        jitter_source=lambda maximum: 600 if maximum == 1200 else -1,
    )
    before = datetime(2026, 7, 14, 3, 0, tzinfo=UTC)

    scheduled = scheduler.evaluate(before, leases_active=False)
    pending = scheduler.evaluate(
        datetime(2026, 7, 14, 4, 10, tzinfo=UTC), leases_active=True
    )
    due = scheduler.evaluate(
        datetime(2026, 7, 14, 4, 11, tzinfo=UTC), leases_active=False
    )

    expected = datetime(2026, 7, 14, 4, 10, tzinfo=UTC)
    assert scheduled.state is RestartScheduleState.SCHEDULED
    assert scheduled.scheduled_for == expected
    assert pending.state is RestartScheduleState.PENDING
    assert pending.scheduled_for == expected
    assert due.state is RestartScheduleState.DUE
    assert due.scheduled_for == expected


def test_completed_maintenance_date_survives_scheduler_reconstruction(
    tmp_path: Path,
) -> None:
    state_store = RestartStateStore(tmp_path / "state" / "restart.json")
    schedule = RestartSchedule(enabled=True, at="04:00", jitter_seconds=1200)
    first = RestartScheduler(schedule, jitter_source=lambda _maximum: 300)

    due = first.evaluate(
        datetime(2026, 7, 18, 4, 5, tzinfo=UTC), leases_active=False
    )
    assert due.state is RestartScheduleState.DUE
    state_store.save(first.mark_executed(datetime(2026, 7, 18, 4, 5, tzinfo=UTC)))

    reconstructed = RestartScheduler(
        schedule,
        last_executed_local_date=state_store.load().last_executed_local_date,
        jitter_source=lambda _maximum: 900,
    )
    status = reconstructed.evaluate(
        datetime(2026, 7, 18, 4, 5, 1, tzinfo=UTC), leases_active=False
    )

    assert status.state is RestartScheduleState.SCHEDULED
    assert status.scheduled_for == datetime(2026, 7, 19, 4, 15, tzinfo=UTC)
    assert reconstructed.last_executed_local_date == date(2026, 7, 18)
    assert state_store.path.stat().st_mode & 0o777 == 0o600


def test_active_leases_round_trip_through_atomic_json_state(tmp_path: Path) -> None:
    store = LeaseStateStore(tmp_path / "state" / "leases.json")
    leases = (
        PersistedLease(
            lease_id="lease-7",
            state=PersistedLeaseState.ORPHANED,
            task_name="resnet-run",
            client_pid=7654,
            client_start_time=1_721_234_567.25,
            gpu_uuids=("GPU-a", "GPU-b"),
            memory_per_gpu_mib=24_000,
            expires_at=1_721_238_167.25,
            client_process_group=7600,
            client_session_id=7500,
        ),
    )

    store.save(leases)

    assert store.load() == leases
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert list(store.path.parent.glob("*.tmp")) == []


def test_shutdown_result_is_atomic_and_explicit(tmp_path: Path) -> None:
    store = ShutdownResultStore(tmp_path / "state" / "shutdown.json")
    expected = ShutdownResult(
        success=False,
        error="driver release verification failed",
        timestamp=123.5,
    )

    store.save(expected)

    assert store.load() == expected
    assert store.path.stat().st_mode & 0o777 == 0o600
    store.remove()
    assert store.load() is None
