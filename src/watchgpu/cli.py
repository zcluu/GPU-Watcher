from __future__ import annotations

import asyncio
import contextlib
import getpass
import json
import os
import platform
import re
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import typer

from watchgpu.config import (
    GPUConfig,
    MaintenanceRestartConfig,
    WatchGPUConfig,
    load_config,
    save_config,
)
from watchgpu.console import WatchGPUConsole
from watchgpu.daemon import build_daemon, configure_daemon_cpu
from watchgpu.environment import EnvironmentDiscoveryError, discover_python, run_command
from watchgpu.ipc import AsyncWatchGPUClient, IPCError
from watchgpu.lifecycle import (
    BackgroundMode,
    DetachedPidStore,
    DetachedProcessState,
    RestartStateStore,
    ShutdownResultStore,
    choose_background_mode,
    probe_user_systemd,
    render_systemd_user_unit,
)
from watchgpu.observer import NVMLGPUObserver, resolve_gpu_selectors
from watchgpu.paths import WatchGPUPaths
from watchgpu.profile import ProfileStore
from watchgpu.sdk import GroupMemoryRequest, acquire
from watchgpu.units import parse_user_capacity_mib

app = typer.Typer(no_args_is_help=True, help="User-space GPU reservation coordinator.")
config_app = typer.Typer(no_args_is_help=True, help="Inspect the host-local configuration.")
profile_app = typer.Typer(no_args_is_help=True, help="Inspect measured memory profiles.")
restart_app = typer.Typer(
    invoke_without_command=True,
    help="Restart now or configure transparent scheduled maintenance.",
)
restart_schedule_app = typer.Typer(
    no_args_is_help=True, help="Show, set, or disable the daily maintenance schedule."
)
restart_app.add_typer(restart_schedule_app, name="schedule")
app.add_typer(config_app, name="config")
app.add_typer(profile_app, name="profile")
app.add_typer(restart_app, name="restart")


@app.command()
def doctor(json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        selection = discover_python()
    except EnvironmentDiscoveryError as exc:
        validation = exc.checked[-1]
        source: str | None = None
        checked = [
            {"python": str(item.executable), "valid": item.valid, "error": item.error}
            for item in exc.checked
        ]
        discovery_error: str | None = str(exc)
    else:
        validation = selection.validation
        source = selection.source.value
        checked = [
            {"python": str(item.executable), "valid": item.valid, "error": item.error}
            for item in selection.checked
        ]
        discovery_error = None
    report: dict[str, Any] = {
        "python": str(validation.executable),
        "hostname": platform.node(),
        "python_source": source,
        "checked_pythons": checked,
        "python_version": validation.python_version,
        "torch_version": validation.torch_version,
        "cuda_version": validation.cuda_version,
        "cuda_available": validation.cuda_available,
        "nvml_available": validation.nvml_available,
        "error": discovery_error or validation.error,
        "gpus": [],
    }
    if validation.nvml_available:
        with NVMLGPUObserver() as observer:
            report["driver_version"] = observer.driver_version
            report["gpus"] = [
                {
                    "index": gpu.index,
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "total_mib": gpu.total_mib,
                    "free_mib": gpu.free_mib,
                    "utilization_percent": gpu.utilization_percent,
                    "temperature_c": gpu.temperature_c,
                    "mig_mode": gpu.mig_mode,
                }
                for gpu in observer.snapshots()
            ]
    _print_data(report, json_output=json_output)
    if not validation.valid:
        raise typer.Exit(1)


@app.command()
def start(
    gpus: str | None = typer.Option(None, "--gpus", help="Comma-separated indices/UUIDs or all."),
    leave_free: str | None = typer.Option(None, "--leave-free"),
    background_mode: str | None = typer.Option(None, "--background-mode"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    paths = WatchGPUPaths.discover()
    paths.ensure_directories()
    try:
        parsed_mode = None if background_mode is None else BackgroundMode(background_mode)
    except ValueError as exc:
        raise typer.BadParameter(f"unsupported background mode: {background_mode}") from exc
    config = _resolved_start_config(
        paths, gpus=gpus, leave_free=leave_free, mode=parsed_mode
    )
    _print_start_preview(config)
    if dry_run:
        typer.echo("Dry run only; no configuration was saved and no CUDA worker was started.")
        return
    save_config(config, paths.config_path)
    _start_saved_config(config, paths)


@app.command()
def daemon(foreground: bool = typer.Option(True, "--foreground")) -> None:
    del foreground
    paths = WatchGPUPaths.discover()
    paths.ensure_directories()
    config = load_config(paths.config_path)
    asyncio.run(_daemon_main(config, paths))


@app.command()
def status(json_output: bool = typer.Option(False, "--json")) -> None:
    paths = WatchGPUPaths.discover()
    try:
        snapshot = _client_call(paths.socket_path, "status.get")
    except (IPCError, OSError, TimeoutError):
        pid_status = DetachedPidStore(paths.pid_path).inspect()
        result: dict[str, Any] = {
            "daemon": (
                "DEGRADED"
                if pid_status.state is DetachedProcessState.RUNNING
                else (
                    "STALE_PID"
                    if pid_status.state is DetachedProcessState.STALE
                    else "STOPPED"
                )
            ),
            "socket": str(paths.socket_path),
            "pid_identity": pid_status.state.value,
        }
        _print_data(result, json_output=json_output)
        return
    pid_status = DetachedPidStore(paths.pid_path).inspect()
    daemon_state = snapshot.get("daemon_state", "RUNNING")
    if pid_status.state is not DetachedProcessState.RUNNING:
        daemon_state = "DEGRADED"
    policy = snapshot.get("policy")
    configured_mode: object = None
    if isinstance(policy, dict):
        raw_config = policy.get("config")
        if isinstance(raw_config, dict):
            configured_mode = raw_config.get("background_mode")
    result = {
        "daemon": daemon_state,
        "pid_identity": pid_status.state.value,
        "pid": None if pid_status.record is None else pid_status.record.pid,
        "background_mode": configured_mode,
        **snapshot,
    }
    _print_data(result, json_output=json_output)


@app.command("console")
def console_command() -> None:
    paths = WatchGPUPaths.discover()
    WatchGPUConsole(socket_path=paths.socket_path).run()


@app.command()
def pause(gpu: str | None = typer.Argument(None)) -> None:
    _control("worker.pause", gpu_uuid=gpu)


@app.command()
def resume(gpu: str | None = typer.Argument(None)) -> None:
    _control("worker.resume", gpu_uuid=gpu)


@app.command()
def release(
    gpu: str | None = typer.Argument(None),
    memory: str | None = typer.Argument(None),
) -> None:
    params: dict[str, Any] = {}
    if gpu is not None:
        params["gpu_uuid"] = gpu
    if memory is not None:
        params["memory_mib"] = parse_user_capacity_mib(memory)
    _print_data(
        _client_call(WatchGPUPaths.discover().socket_path, "worker.release", params),
        json_output=False,
    )


@app.command()
def request(
    task: str = typer.Option(..., "--task"),
    memory_per_gpu: str = typer.Option(..., "--memory-per-gpu"),
    count: int = typer.Option(1, "--count"),
    devices: str | None = typer.Option(None, "--devices"),
    ttl_seconds: float = typer.Option(600.0, "--ttl-seconds"),
) -> None:
    selectors = None if devices is None else _split_selectors(devices)
    with acquire(
        GroupMemoryRequest(
            task_name=task,
            count=count,
            mib_per_gpu=parse_user_capacity_mib(memory_per_gpu),
            devices=selectors,
            ttl_seconds=ttl_seconds,
        ),
        socket_path=WatchGPUPaths.discover().socket_path,
    ) as lease:
        typer.echo(
            f"Lease {lease.lease_id} ACTIVE on {','.join(lease.gpu_uuids)}; "
            "this test command now releases it."
        )


@app.command()
def stop(release: bool = typer.Option(False, "--release")) -> None:
    if not release:
        raise typer.BadParameter("use --release to confirm releasing WatchGPU reservations")
    paths = WatchGPUPaths.discover()
    _stop_daemon(paths)


@restart_app.callback()
def restart(ctx: typer.Context, now: bool = typer.Option(False, "--now")) -> None:
    if ctx.invoked_subcommand is not None:
        if now:
            raise typer.BadParameter("--now cannot be combined with a restart subcommand")
        return
    if not now:
        raise typer.BadParameter("use --now for an immediate controlled restart")
    paths = WatchGPUPaths.discover()
    try:
        result = _client_call(paths.socket_path, "daemon.restart")
    except (IPCError, OSError, TimeoutError):
        pass
    else:
        _print_data(result, json_output=False)
        return
    _stop_daemon(paths)
    config = load_config(paths.config_path)
    _start_saved_config(config, paths)


@restart_schedule_app.command("show")
def restart_schedule_show(
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    paths = WatchGPUPaths.discover()
    try:
        snapshot = _client_call(paths.socket_path, "status.get")
    except (IPCError, OSError, TimeoutError):
        config = load_config(paths.config_path)
        last_executed = RestartStateStore(paths.restart_state_path).load()
        result: dict[str, Any] = {
            "daemon": "STOPPED",
            "config": config.maintenance_restart.model_dump(mode="json"),
            "runtime": {
                "state": "DISABLED" if not config.maintenance_restart.enabled else "STOPPED",
                "scheduled_for": None,
                "last_executed_local_date": (
                    None
                    if last_executed.last_executed_local_date is None
                    else last_executed.last_executed_local_date.isoformat()
                ),
            },
        }
    else:
        config, _revision = _runtime_policy(snapshot)
        runtime = snapshot.get("maintenance_restart")
        result = {
            "daemon": "RUNNING",
            "config": config.maintenance_restart.model_dump(mode="json"),
            "runtime": dict(runtime) if isinstance(runtime, dict) else None,
        }
    _print_data(result, json_output=json_output)


@restart_schedule_app.command("set")
def restart_schedule_set(
    at: str | None = typer.Option(None, "--at", help="Local HH:MM fixed time."),
    jitter: str | None = typer.Option(
        None, "--jitter", help="Random window such as 20m, 90s, or 1h."
    ),
    defer_while_leased: bool | None = typer.Option(
        None,
        "--defer-while-leased/--do-not-defer-while-leased",
    ),
) -> None:
    paths = WatchGPUPaths.discover()
    config, revision, running = _schedule_edit_base(paths)
    current = config.maintenance_restart
    try:
        updated = MaintenanceRestartConfig(
            enabled=True,
            at=current.at if at is None else at,
            jitter_seconds=(
                current.jitter_seconds if jitter is None else _parse_duration_seconds(jitter)
            ),
            defer_while_leased=(
                current.defer_while_leased
                if defer_while_leased is None
                else defer_while_leased
            ),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _commit_schedule_edit(paths, config, updated, revision=revision, running=running)


@restart_schedule_app.command("disable")
def restart_schedule_disable() -> None:
    paths = WatchGPUPaths.discover()
    config, revision, running = _schedule_edit_base(paths)
    updated = config.maintenance_restart.model_copy(update={"enabled": False})
    _commit_schedule_edit(paths, config, updated, revision=revision, running=running)


@config_app.command("show")
def config_show(json_output: bool = typer.Option(False, "--json")) -> None:
    config = load_config(WatchGPUPaths.discover().config_path)
    _print_data(config.model_dump(mode="json"), json_output=json_output)


@config_app.command("set")
def config_set(
    leave_free: str | None = typer.Option(None, "--leave-free"),
    gpu: str | None = typer.Option(None, "--gpu"),
    gpus: str | None = typer.Option(None, "--gpus"),
    reserve_limit: str | None = typer.Option(None, "--reserve-limit"),
    poll_interval: float | None = typer.Option(None, "--poll-interval"),
    growth_stability: float | None = typer.Option(None, "--growth-stability"),
    duty_cycle: int | None = typer.Option(None, "--duty-cycle"),
    cpu_budget: int | None = typer.Option(None, "--cpu-budget"),
    maintenance_compute: bool | None = typer.Option(None, "--maintenance-compute"),
    save: bool = typer.Option(True, "--save/--runtime-only"),
) -> None:
    paths = WatchGPUPaths.discover()
    try:
        status_value = _client_call(paths.socket_path, "status.get")
    except (IPCError, OSError, TimeoutError):
        candidate = load_config(paths.config_path)
        revision: int | None = None
        running = False
    else:
        candidate, revision = _runtime_policy(status_value)
        running = True
    updates: dict[str, Any] = {}
    if poll_interval is not None:
        updates["poll_interval_seconds"] = poll_interval
    if growth_stability is not None:
        updates["growth_stability_seconds"] = growth_stability
    if duty_cycle is not None:
        updates["maintenance_duty_cycle_percent"] = duty_cycle
    if cpu_budget is not None:
        updates["cpu_budget_percent"] = cpu_budget
    if maintenance_compute is not None:
        updates["maintenance_compute_enabled"] = maintenance_compute
    gpu_configs = [item.model_copy(deep=True) for item in candidate.gpus]
    if gpus is not None:
        selectors = _split_selectors(gpus)
        if not running:
            with NVMLGPUObserver() as observer:
                selectors = tuple(
                    item.uuid
                    for item in resolve_gpu_selectors(observer.snapshots(), selectors)
                )
        gpu_configs = [GPUConfig(selector=selector) for selector in selectors]
    if gpu is None:
        if leave_free is not None:
            updates["leave_free_mib"] = parse_user_capacity_mib(leave_free)
        if reserve_limit is not None:
            raise typer.BadParameter("--reserve-limit requires --gpu")
    else:
        if not running:
            with NVMLGPUObserver() as observer:
                gpu = resolve_gpu_selectors(observer.snapshots(), (gpu,))[0].uuid
        by_selector = {item.selector: item for item in gpu_configs}
        selected = by_selector.get(gpu, GPUConfig(selector=gpu))
        selected_updates: dict[str, Any] = {}
        if leave_free is not None:
            selected_updates["leave_free_mib"] = parse_user_capacity_mib(leave_free)
        if reserve_limit is not None:
            selected_updates["reserve_limit_mib"] = parse_user_capacity_mib(reserve_limit)
        by_selector[gpu] = selected.model_copy(update=selected_updates)
        gpu_configs = list(by_selector.values())
    updates["gpus"] = gpu_configs
    validated = WatchGPUConfig.model_validate(
        candidate.model_copy(update=updates).model_dump()
    )
    if not running:
        if not save:
            raise typer.BadParameter("--runtime-only requires a running daemon")
        paths.ensure_directories()
        save_config(validated, paths.config_path)
        result = {
            "status": "APPLIED",
            "revision": None,
            "config": validated.model_dump(mode="json"),
            "reason": "saved locally; applies at next start",
        }
    else:
        assert revision is not None
        result = _client_call(
            paths.socket_path,
            "policy.apply",
            {
                "config": validated.model_dump(mode="json"),
                "expected_revision": revision,
                "save": save,
            },
        )
    _print_data(result, json_output=False)


@profile_app.command("list")
def profile_list(json_output: bool = typer.Option(False, "--json")) -> None:
    paths = WatchGPUPaths.discover()
    records = [
        record.to_dict()
        for record in ProfileStore(paths.state_dir / "profiles.jsonl").records()
    ]
    _print_data({"profiles": records}, json_output=json_output)


@app.command()
def logs(
    lines: int = typer.Option(50, "--lines", "-n", min=1, max=10_000),
    follow: bool = typer.Option(False, "--follow", "-f"),
) -> None:
    path = WatchGPUPaths.discover().log_path
    if not path.exists():
        typer.echo(f"No WatchGPU log exists yet: {path}")
        return
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in deque(stream, maxlen=lines):
            typer.echo(line, nl=False)
        if not follow:
            return
        try:
            while True:
                line = stream.readline()
                if line:
                    typer.echo(line, nl=False)
                else:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            return


async def _daemon_main(config: WatchGPUConfig, paths: WatchGPUPaths) -> None:
    configure_daemon_cpu(config)
    runtime = build_daemon(config, paths)
    pid_store = DetachedPidStore(paths.pid_path)
    pid_store.write_for_process(os.getpid())
    stop_event = asyncio.Event()
    runtime.set_shutdown_callback(stop_event.set)
    runtime.set_restart_callback(stop_event.set)
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(signal_number, stop_event.set)
    try:
        await runtime.start()
        await stop_event.wait()
    finally:
        await runtime.stop()
        pid_store.remove()
    if runtime.restart_requested:
        os.execv(
            sys.executable,
            (sys.executable, "-m", "watchgpu.cli", "daemon", "--foreground"),
        )


def _resolved_start_config(
    paths: WatchGPUPaths,
    *,
    gpus: str | None,
    leave_free: str | None,
    mode: BackgroundMode | None,
) -> WatchGPUConfig:
    current = load_config(paths.config_path)
    selectors = (
        tuple(gpu.selector for gpu in current.gpus)
        if gpus is None
        else _split_selectors(gpus)
    )
    if not selectors:
        raise typer.BadParameter("specify --gpus on first start")
    with NVMLGPUObserver() as observer:
        selected = resolve_gpu_selectors(observer.snapshots(), selectors)
    updates: dict[str, Any] = {
        "gpus": [GPUConfig(selector=gpu.uuid) for gpu in selected]
    }
    if leave_free is not None:
        updates["leave_free_mib"] = parse_user_capacity_mib(leave_free)
    if mode is not None:
        updates["background_mode"] = mode.value
    return current.model_copy(update=updates, deep=True)


def _start_saved_config(config: WatchGPUConfig, paths: WatchGPUPaths) -> None:
    try:
        _client_call(paths.socket_path, "status.get")
    except (IPCError, OSError, TimeoutError) as exc:
        if paths.socket_path.exists():
            raise typer.BadParameter(
                "an existing WatchGPU socket is unresponsive; refusing to start a "
                "second daemon. Inspect `watchgpu logs` and the recorded PID first"
            ) from exc
    else:
        raise typer.BadParameter("WatchGPU is already running")
    selection = discover_python()
    probe = probe_user_systemd(username=getpass.getuser())
    mode = choose_background_mode(config.background_mode, probe)
    if mode is BackgroundMode.FOREGROUND:
        asyncio.run(_daemon_main(config, paths))
        return
    if mode is BackgroundMode.SYSTEMD_USER:
        unit = render_systemd_user_unit(
            selection.executable,
            cpu_quota_percent=config.cpu_budget_percent,
            environment={
                "XDG_CONFIG_HOME": str(paths.config_dir.parent),
                "XDG_STATE_HOME": str(paths.state_dir.parent),
                "WATCHGPU_RUNTIME_DIR": str(paths.runtime_dir),
            },
        )
        paths.systemd_unit_path.write_text(unit, encoding="utf-8")
        for command in (
            ("systemctl", "--user", "daemon-reload"),
            ("systemctl", "--user", "enable", "--now", "watchgpu.service"),
        ):
            result = run_command(command)
            if result.returncode != 0:
                raise typer.BadParameter(result.stderr or result.stdout or "systemd start failed")
        _wait_for_daemon_ready(paths)
    else:
        paths.log_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with paths.log_path.open("ab") as log_file:
            process = subprocess.Popen(
                (
                    str(selection.executable),
                    "-m",
                    "watchgpu.cli",
                    "daemon",
                    "--foreground",
                ),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        try:
            _wait_for_daemon_ready(paths, process=process)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                with contextlib.suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=5)
            raise
        typer.echo(f"WatchGPU started in detached mode (PID {process.pid}).")
    if mode is BackgroundMode.DETACHED:
        typer.echo(
            "SESSION_BOUND: detached mode survival after logout depends on this "
            "server's logind/session policy; use systemd-user with Linger=yes for "
            "a logout-safe guarantee."
        )
    elif probe.session_bound:
        typer.echo(
            "SESSION_BOUND: user lingering is unavailable; survival after the last logout "
            "depends on this server's logind policy."
        )


def _wait_for_daemon_ready(
    paths: WatchGPUPaths,
    *,
    process: subprocess.Popen[bytes] | None = None,
    timeout_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise typer.BadParameter(
                f"detached daemon exited during startup with code {process.returncode}; "
                f"inspect {paths.log_path}"
            )
        try:
            asyncio.run(
                AsyncWatchGPUClient(paths.socket_path, timeout_seconds=0.5).call(
                    "status.get"
                )
            )
            return
        except (IPCError, OSError, TimeoutError) as exc:
            last_error = exc
            time.sleep(0.1)
    raise typer.BadParameter(
        f"daemon did not become ready within {timeout_seconds:g}s; "
        f"inspect {paths.log_path}: {last_error}"
    )


def _stop_daemon(paths: WatchGPUPaths) -> None:
    shutdown_store = ShutdownResultStore(paths.shutdown_result_path)
    shutdown_store.remove()
    try:
        rpc_result = _client_call(
            paths.socket_path, "daemon.stop", {"release": True}
        )
    except (IPCError, OSError, TimeoutError):
        pass
    else:
        _print_data(rpc_result, json_output=False)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and paths.socket_path.exists():
            time.sleep(0.1)
        if not paths.socket_path.exists():
            _require_verified_shutdown(shutdown_store)
            typer.echo("WatchGPU stopped; training and external processes were untouched.")
            return
        raise typer.BadParameter("daemon accepted stop but its socket remained for 60 seconds")
    pid_store = DetachedPidStore(paths.pid_path)
    status_value = pid_store.inspect()
    if status_value.state is DetachedProcessState.RUNNING and status_value.record is not None:
        os.kill(status_value.record.pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if pid_store.inspect().state is not DetachedProcessState.RUNNING:
                _require_verified_shutdown(shutdown_store)
                typer.echo("WatchGPU stopped; only its worker reservations were released.")
                return
            time.sleep(0.1)
        raise typer.BadParameter("daemon did not stop within 60 seconds; inspect watchgpu logs")
    probe = probe_user_systemd(username=getpass.getuser())
    if probe.available and paths.systemd_unit_path.exists():
        systemd_result = run_command(
            ("systemctl", "--user", "stop", "watchgpu.service")
        )
        if systemd_result.returncode == 0:
            _require_verified_shutdown(shutdown_store)
            typer.echo("WatchGPU user service stopped; worker reservations were released.")
            return
    typer.echo("WatchGPU is not running.")


def _require_verified_shutdown(store: ShutdownResultStore) -> None:
    try:
        result = store.load()
    except Exception as exc:
        raise typer.BadParameter(f"shutdown result is unreadable: {exc}") from exc
    if result is None:
        raise typer.BadParameter(
            "daemon exited without a verified shutdown result; inspect watchgpu logs"
        )
    if not result.success:
        raise typer.BadParameter(
            f"daemon stopped but driver/worker release verification failed: {result.error}"
        )


def _schedule_edit_base(
    paths: WatchGPUPaths,
) -> tuple[WatchGPUConfig, int | None, bool]:
    try:
        snapshot = _client_call(paths.socket_path, "status.get")
    except (IPCError, OSError, TimeoutError):
        return load_config(paths.config_path), None, False
    config, revision = _runtime_policy(snapshot)
    return config, revision, True


def _runtime_policy(snapshot: dict[str, Any]) -> tuple[WatchGPUConfig, int]:
    raw_policy = snapshot.get("policy")
    if not isinstance(raw_policy, dict):
        raise typer.BadParameter("daemon did not report a runtime policy")
    revision = raw_policy.get("revision")
    raw_config = raw_policy.get("config")
    if not isinstance(revision, int) or isinstance(revision, bool) or not isinstance(
        raw_config, dict
    ):
        raise typer.BadParameter("daemon returned an invalid runtime policy")
    return WatchGPUConfig.model_validate(raw_config), revision


def _commit_schedule_edit(
    paths: WatchGPUPaths,
    config: WatchGPUConfig,
    schedule: MaintenanceRestartConfig,
    *,
    revision: int | None,
    running: bool,
) -> None:
    candidate = WatchGPUConfig.model_validate(
        config.model_copy(update={"maintenance_restart": schedule}).model_dump()
    )
    if running:
        assert revision is not None
        result = _client_call(
            paths.socket_path,
            "policy.apply",
            {
                "config": candidate.model_dump(mode="json"),
                "expected_revision": revision,
                "save": True,
            },
        )
    else:
        save_config(candidate, paths.config_path)
        result = {"status": "APPLIED", "daemon": "STOPPED", "saved": True}
    result = {**result, "maintenance_restart": schedule.model_dump(mode="json")}
    _print_data(result, json_output=False)


_DURATION_PATTERN = re.compile(r"^(?P<value>\d+)(?P<unit>[smh]?)$", re.IGNORECASE)


def _parse_duration_seconds(value: str) -> int:
    match = _DURATION_PATTERN.fullmatch(value.strip())
    if match is None:
        raise ValueError("jitter must be an integer followed by s, m, or h (for example 20m)")
    amount = int(match.group("value"))
    multiplier = {"": 1, "s": 1, "m": 60, "h": 3600}[match.group("unit").lower()]
    return amount * multiplier


def _control(method: str, *, gpu_uuid: str | None) -> None:
    params = {} if gpu_uuid is None else {"gpu_uuid": gpu_uuid}
    _print_data(
        _client_call(WatchGPUPaths.discover().socket_path, method, params),
        json_output=False,
    )


def _client_call(
    socket_path: Path, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    return asyncio.run(AsyncWatchGPUClient(socket_path).call(method, params))


def _split_selectors(value: str) -> tuple[str, ...]:
    if value.strip() == "all":
        return ("all",)
    selectors = tuple(item.strip() for item in value.split(",") if item.strip())
    if not selectors:
        raise typer.BadParameter("GPU selector list cannot be empty")
    return selectors


def _print_start_preview(config: WatchGPUConfig) -> None:
    typer.echo(
        f"GPUs: {','.join(gpu.selector for gpu in config.gpus)} | "
        f"leave-free: {config.leave_free_mib} MiB | chunk: {config.chunk_mib} MiB | "
        f"mode: {config.background_mode}"
    )


def _print_data(value: Any, *, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(value, ensure_ascii=False, sort_keys=True))
    else:
        typer.echo(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
