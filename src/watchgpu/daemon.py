from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path

from watchgpu.config import GPUConfig, WatchGPUConfig
from watchgpu.control import ApplyStatus, RuntimeConfigController
from watchgpu.ipc import UnixSocketServer
from watchgpu.journal import EventJournal
from watchgpu.lifecycle import (
    LeaseStateStore,
    PersistedLease,
    PersistedLeaseState,
    RestartSchedule,
    RestartScheduler,
    RestartScheduleState,
    RestartStateStore,
    ShutdownResult,
    ShutdownResultStore,
)
from watchgpu.models import GPUSnapshot
from watchgpu.observer import GPUObserver, NVMLGPUObserver, resolve_gpu_selectors
from watchgpu.paths import WatchGPUPaths
from watchgpu.policy import ReservationLimits
from watchgpu.profile import ProfileStore
from watchgpu.protocol import SupervisorProtocol
from watchgpu.supervisor import (
    GroupLeaseRequest,
    LeaseState,
    PollingReleaseVerifier,
    ProcessLeaseActivityVerifier,
    Supervisor,
)
from watchgpu.worker import ManagedWorker
from watchgpu.worker_process import (
    ProcessTreeCPUMonitor,
    ProcessTreeCPUUsage,
    WorkerProcessProxy,
    WorkerProcessSpec,
)


class DaemonError(RuntimeError):
    pass


class WatchGPUDaemon:
    """Own the local control socket and drive one Supervisor until shutdown."""

    def __init__(
        self,
        *,
        supervisor: Supervisor,
        config: WatchGPUConfig,
        socket_path: Path,
        config_path: Path,
        poll_interval_seconds: float | None = None,
        lease_store: LeaseStateStore | None = None,
        clock: Callable[[], float] = time.monotonic,
        cleanup: Callable[[], None] | None = None,
        profile_store: ProfileStore | None = None,
        runtime_apply: Callable[[WatchGPUConfig], ApplyStatus] | None = None,
        config_normalizer: Callable[[WatchGPUConfig], WatchGPUConfig] | None = None,
        health_reconcile: Callable[[], None] | None = None,
        wall_clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        event_journal: EventJournal | None = None,
        restart_state_store: RestartStateStore | None = None,
        shutdown_result_store: ShutdownResultStore | None = None,
        cpu_monitor: ProcessTreeCPUMonitor | None = None,
        cpu_affinity_cores: tuple[int, ...] = (),
    ) -> None:
        interval = (
            config.poll_interval_seconds
            if poll_interval_seconds is None
            else poll_interval_seconds
        )
        if interval <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.supervisor = supervisor
        self._clock = clock
        self._poll_interval_seconds = interval
        self._lease_store = lease_store
        self._cleanup = cleanup
        self._runtime_apply = runtime_apply
        self._health_reconcile = health_reconcile
        self._wall_clock = wall_clock
        self._restart_state_store = restart_state_store
        self._shutdown_result_store = shutdown_result_store
        self._cpu_monitor = cpu_monitor
        self._cpu_usage = ProcessTreeCPUUsage(
            percent=0.0, process_count=1, sampled_at=0.0
        )
        self._cpu_budget_percent = config.cpu_budget_percent
        self._maintenance_cpu_target_percent = config.maintenance_cpu_target_percent
        self._cpu_affinity_cores = cpu_affinity_cores
        self._worker_cpu_threads = config.worker_cpu_threads
        restart_state = (
            None
            if restart_state_store is None
            else restart_state_store.load().last_executed_local_date
        )
        self._restart_scheduler = _restart_scheduler(
            config, last_executed_local_date=restart_state
        )
        self._shutdown_callback: Callable[[], None] | None = None
        self._restart_callback: Callable[[], None] | None = None
        self.restart_requested = False
        self._event_journal = event_journal
        self._last_journal_sequence = 0
        self._config_controller = RuntimeConfigController(
            config,
            apply=self._apply_config,
            config_path=config_path,
            normalize=config_normalizer,
        )
        self._protocol = SupervisorProtocol(
            supervisor,
            clock=clock,
            config_controller=self._config_controller,
            profile_store=profile_store,
            shutdown_callback=self._request_shutdown,
            restart_callback=self._request_restart,
            state_change_callback=self._persist_leases,
            restart_status_provider=self._restart_status_payload,
            cpu_status_provider=self._cpu_status_payload,
            runtime_status_provider=self._runtime_status_payload,
        )
        self._server = UnixSocketServer(socket_path, self._protocol)
        self._poll_task: asyncio.Task[None] | None = None
        self._stopped = asyncio.Event()
        self._started = False
        self._closing = False
        self.last_error: str | None = None

    @property
    def config_controller(self) -> RuntimeConfigController:
        return self._config_controller

    async def start(self) -> None:
        if self._started:
            raise DaemonError("daemon is already started")
        self._started = True
        await self._server.start()
        self._poll_task = asyncio.create_task(
            self._poll_loop(), name="watchgpu-supervisor-poll"
        )

    async def stop(self) -> None:
        if self._closing:
            await self._stopped.wait()
            return
        self._closing = True
        errors: list[BaseException] = []
        try:
            task = self._poll_task
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task
                self._poll_task = None
        except BaseException as exc:
            errors.append(exc)
        try:
            self._persist_leases()
        except BaseException as exc:
            errors.append(exc)
        try:
            await asyncio.to_thread(self.supervisor.stop_workers)
        except BaseException as exc:
            errors.append(exc)
        if self._shutdown_result_store is not None:
            error_text = (
                None
                if not errors
                else "; ".join(
                    f"{type(error).__name__}: {error}" for error in errors
                )
            )
            try:
                self._shutdown_result_store.save(
                    ShutdownResult(
                        success=not errors,
                        error=error_text,
                        timestamp=time.time(),
                    )
                )
            except BaseException as exc:
                errors.append(exc)
        # Keep the control socket present until all WatchGPU workers have been
        # released/terminated. CLI stop waits for socket disappearance, so this
        # ordering makes that observation a truthful completion signal.
        try:
            await self._server.close()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._journal_events()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self._event_journal is not None:
                self._event_journal.close()
        except BaseException as exc:
            errors.append(exc)
        try:
            if self._cleanup is not None:
                self._cleanup()
        except BaseException as exc:
            errors.append(exc)
        finally:
            if errors:
                self.last_error = "; ".join(
                    f"{type(error).__name__}: {error}" for error in errors
                )
            self._stopped.set()

    async def wait_stopped(self) -> None:
        await self._stopped.wait()

    def set_restart_callback(self, callback: Callable[[], None]) -> None:
        self._restart_callback = callback

    def set_shutdown_callback(self, callback: Callable[[], None]) -> None:
        self._shutdown_callback = callback

    def _request_shutdown(self) -> None:
        if self._shutdown_callback is not None:
            self._shutdown_callback()

    def _request_restart(self) -> None:
        self.restart_requested = True
        if self._restart_callback is not None:
            self._restart_callback()

    def _apply_config(self, config: WatchGPUConfig) -> ApplyStatus:
        if self._runtime_apply is not None:
            status = self._runtime_apply(config)
            if status is ApplyStatus.APPLIED:
                self._poll_interval_seconds = config.poll_interval_seconds
                self._update_cpu_config(config)
                self._restart_scheduler = _restart_scheduler(
                    config,
                    last_executed_local_date=(
                        self._restart_scheduler.last_executed_local_date
                    ),
                )
            return status
        self.supervisor.update_policy(config, now=self._clock())
        self._poll_interval_seconds = config.poll_interval_seconds
        self._update_cpu_config(config)
        self._restart_scheduler = _restart_scheduler(
            config,
            last_executed_local_date=self._restart_scheduler.last_executed_local_date,
        )
        return ApplyStatus.APPLIED

    async def _poll_loop(self) -> None:
        while True:
            started = self._clock()
            try:
                self.supervisor.tick(now=started)
                if self._cpu_monitor is not None:
                    self._cpu_usage = self._cpu_monitor.sample()
                    self.supervisor.set_maintenance_cpu_pressure(
                        self._cpu_usage.percent > self._cpu_budget_percent
                    )
                self._persist_leases()
                if self._health_reconcile is not None:
                    self._health_reconcile()
                if (
                    self._config_controller.runtime_status is ApplyStatus.PENDING
                    and self._runtime_apply is not None
                ):
                    status = self._runtime_apply(self._config_controller.config)
                    if status is ApplyStatus.APPLIED:
                        self._poll_interval_seconds = (
                            self._config_controller.config.poll_interval_seconds
                        )
                        self._config_controller.mark_runtime_status(status)
                        self._update_cpu_config(self._config_controller.config)
                        self._restart_scheduler = _restart_scheduler(
                            self._config_controller.config,
                            last_executed_local_date=(
                                self._restart_scheduler.last_executed_local_date
                            ),
                        )
                wall_now = self._wall_clock()
                restart = self._restart_scheduler.evaluate(
                    wall_now,
                    leases_active=any(
                        lease.state in {LeaseState.ACTIVE, LeaseState.ORPHANED}
                        for lease in self.supervisor.leases
                    ),
                )
                if restart.state is RestartScheduleState.DUE:
                    restart_state = self._restart_scheduler.mark_executed(wall_now)
                    if self._restart_state_store is not None:
                        self._restart_state_store.save(restart_state)
                    self._request_restart()
                    return
                self._journal_events()
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                if self._event_journal is not None:
                    with suppress(Exception):
                        self._event_journal.append(
                            {
                                "timestamp": time.time(),
                                "type": "DAEMON_POLL_ERROR",
                                "severity": "ERROR",
                                "message": self.last_error,
                            }
                        )
            elapsed = max(0.0, self._clock() - started)
            await asyncio.sleep(max(0.0, self._poll_interval_seconds - elapsed))

    def _restart_status_payload(self) -> dict[str, object]:
        status = self._restart_scheduler.evaluate(
            self._wall_clock(),
            leases_active=any(
                lease.state in {LeaseState.ACTIVE, LeaseState.ORPHANED}
                for lease in self.supervisor.leases
            ),
        )
        last_executed = self._restart_scheduler.last_executed_local_date
        return {
            "state": status.state.value,
            "scheduled_for": (
                None if status.scheduled_for is None else status.scheduled_for.isoformat()
            ),
            "last_executed_local_date": (
                None if last_executed is None else last_executed.isoformat()
            ),
        }

    def _cpu_status_payload(self) -> dict[str, object]:
        over_budget = self._cpu_usage.percent > self._cpu_budget_percent
        if self._maintenance_cpu_target_percent <= 0:
            maintenance_state = "DISABLED"
        elif over_budget:
            maintenance_state = "THROTTLED"
        else:
            maintenance_state = "RUNNING"
        return {
            "process_tree_percent": round(self._cpu_usage.percent, 2),
            "process_count": self._cpu_usage.process_count,
            "sampled_at": self._cpu_usage.sampled_at,
            "budget_percent": self._cpu_budget_percent,
            "maintenance_target_percent": self._maintenance_cpu_target_percent,
            "over_budget": over_budget,
            "affinity_cores": list(self._cpu_affinity_cores),
            "worker_cpu_threads": self._worker_cpu_threads,
            "maintenance_state": maintenance_state,
        }

    def _runtime_status_payload(self) -> dict[str, object]:
        return {
            "started": self._started,
            "closing": self._closing,
            "last_error": self.last_error,
        }

    def _update_cpu_config(self, config: WatchGPUConfig) -> None:
        self._cpu_budget_percent = config.cpu_budget_percent
        self._maintenance_cpu_target_percent = config.maintenance_cpu_target_percent
        self._worker_cpu_threads = config.worker_cpu_threads

    def _journal_events(self) -> None:
        journal = self._event_journal
        if journal is None:
            return
        for event in self.supervisor.events:
            if event.sequence <= self._last_journal_sequence:
                continue
            journal.append(asdict(event))
            self._last_journal_sequence = event.sequence

    def _persist_leases(self) -> None:
        if self._lease_store is None:
            return
        persisted: list[PersistedLease] = []
        for lease in self.supervisor.leases:
            if lease.state not in {
                LeaseState.QUEUED,
                LeaseState.ACTIVE,
                LeaseState.ORPHANED,
            }:
                continue
            persisted.append(
                PersistedLease(
                    lease_id=lease.lease_id,
                    state=PersistedLeaseState(lease.state.value),
                    task_name=lease.request.task_name,
                    client_pid=lease.request.client_pid,
                    client_start_time=lease.request.client_start_time,
                    gpu_uuids=lease.gpu_uuids,
                    memory_per_gpu_mib=lease.request.memory_per_gpu_mib,
                    expires_at=lease.expires_at,
                    gpu_count=lease.request.gpu_count,
                    ttl_seconds=lease.request.ttl_seconds,
                    candidate_uuids=lease.request.candidate_uuids,
                    created_at=lease.created_at,
                    client_process_group=lease.request.client_process_group,
                    client_session_id=lease.request.client_session_id,
                )
            )
        self._lease_store.save(tuple(persisted))


WorkerFactory = Callable[[GPUSnapshot, GPUConfig, WatchGPUConfig], ManagedWorker]


def build_daemon(
    config: WatchGPUConfig,
    paths: WatchGPUPaths,
    *,
    observer: GPUObserver | None = None,
    worker_factory: WorkerFactory | None = None,
) -> WatchGPUDaemon:
    owned_observer = observer is None
    active_observer = NVMLGPUObserver() if observer is None else observer
    snapshots = active_observer.snapshots()
    workers: dict[str, ManagedWorker] = {}
    if not config.gpus:
        if owned_observer:
            _close_observer(active_observer)
        raise DaemonError("no GPUs are configured; run watchgpu start --gpus ...")
    try:
        normalized, selected_with_config = _normalize_runtime_config(config, snapshots)
        cpu_affinity = select_process_cpu_affinity(normalized)
        factory = worker_factory or _default_worker_factory
        workers = {
            snapshot.uuid: factory(snapshot, gpu, normalized)
            for snapshot, gpu in selected_with_config
        }
        supervisor = Supervisor(
            observer=active_observer,
            workers=workers,
            release_verifier=PollingReleaseVerifier(active_observer),
            lease_activity_verifier=ProcessLeaseActivityVerifier(
                active_observer,
                worker_pids=lambda: set(supervisor.worker_pids),
            ),
        )
        lease_store = LeaseStateStore(paths.leases_path)
        for persisted in lease_store.load():
            supervisor.restore_lease(
                GroupLeaseRequest(
                    request_id=persisted.lease_id,
                    task_name=persisted.task_name,
                    gpu_count=persisted.gpu_count,
                    memory_per_gpu_mib=persisted.memory_per_gpu_mib,
                    ttl_seconds=persisted.ttl_seconds,
                    client_pid=persisted.client_pid,
                    candidate_uuids=persisted.candidate_uuids,
                    client_start_time=persisted.client_start_time,
                    client_process_group=persisted.client_process_group,
                    client_session_id=persisted.client_session_id,
                ),
                state=LeaseState(persisted.state.value),
                gpu_uuids=persisted.gpu_uuids,
                created_at=persisted.created_at,
                expires_at=persisted.expires_at,
            )
        current_runtime_config = normalized

        def reconcile_runtime_config(candidate: WatchGPUConfig) -> ApplyStatus:
            nonlocal current_runtime_config
            candidate_snapshots = active_observer.snapshots()
            normalized_candidate, desired_pairs = _normalize_runtime_config(
                candidate, candidate_snapshots
            )
            desired = {snapshot.uuid: (snapshot, gpu) for snapshot, gpu in desired_pairs}
            current_uuids = set(supervisor.managed_gpu_uuids)
            desired_uuids = set(desired)
            restart_retained = _worker_restart_required(
                current_runtime_config, normalized_candidate
            )
            if restart_retained:
                raise ValueError(
                    "chunk size, worker CPU threads, and CPU affinity require "
                    "`watchgpu restart --now`"
                )
            removing = current_uuids - desired_uuids
            if any(supervisor.has_active_lease(gpu_uuid) for gpu_uuid in removing):
                return ApplyStatus.PENDING

            staged: dict[str, ManagedWorker] = {}
            to_create = desired_uuids - current_uuids
            try:
                for gpu_uuid in to_create:
                    snapshot, gpu = desired[gpu_uuid]
                    staged[gpu_uuid] = factory(snapshot, gpu, normalized_candidate)
                for gpu_uuid, new_worker in staged.items():
                    supervisor.add_worker(gpu_uuid, new_worker)
                supervisor.update_policy(normalized_candidate, now=time.monotonic())
                for gpu_uuid in removing:
                    old_worker = supervisor.worker(gpu_uuid)
                    try:
                        old_worker.stop()
                    except BaseException:
                        terminate = getattr(old_worker, "terminate", None)
                        if not callable(terminate):
                            raise
                        terminate()
                    supervisor.remove_worker(gpu_uuid)
            except BaseException:
                for gpu_uuid, staged_worker in staged.items():
                    if gpu_uuid in supervisor.managed_gpu_uuids:
                        with suppress(Exception):
                            supervisor.remove_worker(gpu_uuid)
                    with suppress(Exception):
                        staged_worker.stop()
                raise
            current_runtime_config = normalized_candidate
            return ApplyStatus.APPLIED

        def reconcile_worker_health() -> None:
            snapshots_by_uuid = {
                snapshot.uuid: snapshot for snapshot in active_observer.snapshots()
            }
            config_by_uuid = {
                gpu.selector: gpu for gpu in current_runtime_config.gpus
            }
            for gpu_uuid in supervisor.unhealthy_worker_uuids():
                snapshot = snapshots_by_uuid.get(gpu_uuid)
                gpu = config_by_uuid.get(gpu_uuid)
                if snapshot is None or gpu is None:
                    continue
                replacement = factory(snapshot, gpu, current_runtime_config)
                old_worker = supervisor.replace_failed_worker(gpu_uuid, replacement)
                terminate = getattr(old_worker, "terminate", None)
                if callable(terminate):
                    terminate()

        cleanup: Callable[[], None] | None = None
        if owned_observer:
            def cleanup_owned_observer() -> None:
                _close_observer(active_observer)

            cleanup = cleanup_owned_observer
        return WatchGPUDaemon(
            supervisor=supervisor,
            config=normalized,
            socket_path=paths.socket_path,
            config_path=paths.config_path,
            lease_store=lease_store,
            cleanup=cleanup,
            profile_store=ProfileStore(paths.state_dir / "profiles.jsonl"),
            runtime_apply=reconcile_runtime_config,
            config_normalizer=lambda candidate: _normalize_runtime_config(
                candidate, active_observer.snapshots()
            )[0],
            health_reconcile=reconcile_worker_health,
            event_journal=EventJournal(paths.log_path),
            restart_state_store=RestartStateStore(paths.restart_state_path),
            shutdown_result_store=ShutdownResultStore(paths.shutdown_result_path),
            cpu_monitor=ProcessTreeCPUMonitor(root_pid=os.getpid()),
            cpu_affinity_cores=cpu_affinity,
        )
    except BaseException:
        for worker in workers.values():
            with suppress(Exception):
                worker.stop()
        if owned_observer:
            _close_observer(active_observer)
        raise


def _default_worker_factory(
    snapshot: GPUSnapshot, gpu: GPUConfig, config: WatchGPUConfig
) -> ManagedWorker:
    affinity_cores = select_process_cpu_affinity(config)
    return WorkerProcessProxy(
        WorkerProcessSpec(
            gpu_uuid=snapshot.uuid,
            chunk_mib=config.chunk_mib,
            limits=ReservationLimits(
                leave_free_mib=(
                    config.leave_free_mib
                    if gpu.leave_free_mib is None
                    else gpu.leave_free_mib
                ),
                reserve_limit_mib=gpu.reserve_limit_mib,
                reserve_ratio=config.reserve_ratio,
            ),
            growth_stability_seconds=config.growth_stability_seconds,
            allocation_tolerance_mib=config.allocation_tolerance_mib,
            cpu_affinity_cores=affinity_cores or None,
            worker_cpu_threads=config.worker_cpu_threads,
            maintenance_compute_enabled=config.maintenance_compute_enabled,
            maintenance_duty_cycle_percent=config.maintenance_duty_cycle_percent,
            compute_pause_above_utilization=config.compute_pause_above_utilization,
            cpu_budget_percent=config.cpu_budget_percent,
            maintenance_cpu_target_percent=(
                config.maintenance_cpu_target_percent / max(1, len(config.gpus))
            ),
        )
    )


def select_process_cpu_affinity(config: WatchGPUConfig) -> tuple[int, ...]:
    if not hasattr(os, "sched_getaffinity"):
        return ()
    available_cores = sorted(os.sched_getaffinity(0))
    if not available_cores:
        return ()
    budget_cores = max(1, (config.cpu_budget_percent + 99) // 100)
    allowed_count = min(config.cpu_affinity_cores, budget_cores, len(available_cores))
    return tuple(available_cores[:allowed_count])


def configure_daemon_cpu(config: WatchGPUConfig) -> tuple[int, ...]:
    """Hard-limit the daemon and all subsequently spawned children to one budget set."""

    affinity_cores = select_process_cpu_affinity(config)
    if affinity_cores and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(affinity_cores))
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    return affinity_cores


def _normalize_runtime_config(
    config: WatchGPUConfig, snapshots: tuple[GPUSnapshot, ...]
) -> tuple[WatchGPUConfig, tuple[tuple[GPUSnapshot, GPUConfig], ...]]:
    if not config.gpus:
        return config.model_copy(deep=True), ()
    selected = resolve_gpu_selectors(
        snapshots, tuple(gpu.selector for gpu in config.gpus)
    )
    pairs = tuple(
        (
            snapshot,
            GPUConfig(
                selector=snapshot.uuid,
                reserve_limit_mib=gpu.reserve_limit_mib,
                leave_free_mib=gpu.leave_free_mib,
            ),
        )
        for gpu, snapshot in zip(config.gpus, selected, strict=True)
    )
    normalized = config.model_copy(
        update={"gpus": [gpu for _snapshot, gpu in pairs]}, deep=True
    )
    return normalized, pairs


def _worker_restart_required(
    current: WatchGPUConfig, candidate: WatchGPUConfig
) -> bool:
    return any(
        (
            current.chunk_mib != candidate.chunk_mib,
            current.worker_cpu_threads != candidate.worker_cpu_threads,
            current.cpu_affinity_cores != candidate.cpu_affinity_cores,
        )
    )


def _close_observer(observer: GPUObserver) -> None:
    close = getattr(observer, "close", None)
    if callable(close):
        close()


def _restart_scheduler(
    config: WatchGPUConfig, *, last_executed_local_date: date | None = None
) -> RestartScheduler:
    restart = config.maintenance_restart
    return RestartScheduler(
        RestartSchedule(
            enabled=restart.enabled,
            at=restart.at,
            jitter_seconds=restart.jitter_seconds,
            defer_while_leased=restart.defer_while_leased,
        ),
        last_executed_local_date=last_executed_local_date,
    )
