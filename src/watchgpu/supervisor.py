from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import psutil  # type: ignore[import-untyped]

from watchgpu.config import WatchGPUConfig
from watchgpu.models import GPUSnapshot
from watchgpu.observer import GPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.worker import ManagedWorker, WorkerAction, WorkerState, WorkerStatus


class LeaseRequestError(ValueError):
    pass


class WorkerShutdownError(RuntimeError):
    def __init__(self, failures: Mapping[str, BaseException]) -> None:
        self.failures = dict(failures)
        details = "; ".join(
            f"{gpu_uuid}: {type(error).__name__}: {error}"
            for gpu_uuid, error in self.failures.items()
        )
        super().__init__(f"failed to stop {len(self.failures)} worker(s): {details}")


class LeaseState(StrEnum):
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    ORPHANED = "ORPHANED"
    RELEASED = "RELEASED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class GroupLeaseRequest:
    request_id: str
    task_name: str
    gpu_count: int
    memory_per_gpu_mib: int
    ttl_seconds: float
    client_pid: int
    candidate_uuids: tuple[str, ...] | None = None
    client_start_time: float | None = None
    client_process_group: int | None = None
    client_session_id: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise LeaseRequestError("request_id is required")
        if not self.task_name:
            raise LeaseRequestError("task_name is required")
        if self.gpu_count <= 0:
            raise LeaseRequestError("gpu_count must be positive")
        if self.memory_per_gpu_mib <= 0:
            raise LeaseRequestError("memory_per_gpu_mib must be positive")
        if self.ttl_seconds <= 0:
            raise LeaseRequestError("ttl_seconds must be positive")
        if self.client_pid <= 0:
            raise LeaseRequestError("client_pid must be positive")
        if self.client_process_group is not None and self.client_process_group <= 0:
            raise LeaseRequestError("client_process_group must be positive")
        if self.client_session_id is not None and self.client_session_id <= 0:
            raise LeaseRequestError("client_session_id must be positive")


@dataclass(slots=True)
class Lease:
    lease_id: str
    request: GroupLeaseRequest
    state: LeaseState
    created_at: float
    expires_at: float | None = None
    gpu_uuids: tuple[str, ...] = ()
    released_by_gpu_mib: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True, slots=True)
class ManagedGPUStatus:
    index: int
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    utilization_percent: int
    reserved_mib: int
    leased_mib: int
    worker_state: str
    temperature_c: int | None = None
    mig_mode: str | None = None


@dataclass(frozen=True, slots=True)
class GPUProcessStatus:
    gpu_uuid: str
    pid: int
    used_memory_mib: int | None
    name: str | None
    classification: str
    lease_id: str | None = None
    task_name: str | None = None


@dataclass(frozen=True, slots=True)
class SupervisorEvent:
    sequence: int
    timestamp: float
    type: str
    message: str
    severity: str = "INFO"


@dataclass(frozen=True, slots=True)
class SupervisorStatus:
    gpus: tuple[ManagedGPUStatus, ...]
    leases: tuple[Lease, ...]
    processes: tuple[GPUProcessStatus, ...]
    events: tuple[SupervisorEvent, ...]


class ReleaseVerifier(Protocol):
    def verify(self, gpu_uuid: str, *, expected_free_mib: int, timeout: float) -> bool: ...


class TrustingReleaseVerifier:
    """Test adapter for allocators whose release behavior is deterministic."""

    def verify(self, gpu_uuid: str, *, expected_free_mib: int, timeout: float) -> bool:
        return True


class PollingReleaseVerifier:
    def __init__(
        self,
        observer: GPUObserver,
        *,
        poll_interval_seconds: float = 0.05,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._observer = observer
        self._poll_interval_seconds = poll_interval_seconds
        self._clock = clock
        self._sleep = sleep

    def verify(self, gpu_uuid: str, *, expected_free_mib: int, timeout: float) -> bool:
        deadline = self._clock() + timeout
        while True:
            snapshot = next(
                (
                    item
                    for item in self._observer.snapshots()
                    if item.uuid == gpu_uuid
                ),
                None,
            )
            if snapshot is not None and snapshot.free_mib >= expected_free_mib:
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self._poll_interval_seconds)


class LeaseActivityVerifier(Protocol):
    def is_active(self, lease: Lease) -> bool: ...


class ConservativeLeaseActivityVerifier:
    """Keep orphaned leases reserved when process identity cannot be checked."""

    def is_active(self, lease: Lease) -> bool:
        return True


class ProcessLeaseActivityVerifier:
    """Verify the launcher and conservatively retain GPU-active orphaned ranks."""

    def __init__(
        self,
        observer: GPUObserver | None = None,
        *,
        worker_pids: Callable[[], set[int]] | None = None,
    ) -> None:
        self._observer = observer
        self._worker_pids = worker_pids or (lambda: set())

    def is_active(self, lease: Lease) -> bool:
        try:
            process = psutil.Process(lease.request.client_pid)
            if (
                lease.request.client_start_time is not None
                and abs(process.create_time() - lease.request.client_start_time) > 0.01
            ):
                return False
            return bool(
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.AccessDenied:
            return True
        except psutil.NoSuchProcess:
            pass

        if self._observer is None:
            return False
        excluded = self._worker_pids()
        try:
            candidates = tuple(
                process.pid
                for gpu_uuid in lease.gpu_uuids
                for process in self._observer.processes(gpu_uuid)
                if process.pid not in excluded
            )
        except Exception:
            # Failure to prove that leased GPU processes are gone must not let the
            # reservation worker reclaim training headroom.
            return True
        if (
            lease.request.client_process_group is None
            and lease.request.client_session_id is None
        ):
            return bool(candidates)
        for pid in candidates:
            try:
                if lease.request.client_process_group is not None:
                    if os.getpgid(pid) == lease.request.client_process_group:
                        return True
                elif (
                    lease.request.client_session_id is not None
                    and os.getsid(pid) == lease.request.client_session_id
                ):
                    return True
            except ProcessLookupError:
                continue
            except (PermissionError, OSError):
                return True
        return False


class Supervisor:
    def __init__(
        self,
        *,
        observer: GPUObserver,
        workers: Mapping[str, ManagedWorker],
        release_verifier: ReleaseVerifier,
        lease_activity_verifier: LeaseActivityVerifier | None = None,
        release_timeout_seconds: float = 5.0,
    ) -> None:
        self._observer = observer
        self._workers = dict(workers)
        self._release_verifier = release_verifier
        self._lease_activity_verifier = (
            lease_activity_verifier or ConservativeLeaseActivityVerifier()
        )
        self._release_timeout_seconds = release_timeout_seconds
        self._leases_by_request: dict[str, Lease] = {}
        self._queue: list[Lease] = []
        self._events: list[SupervisorEvent] = []
        self._event_sequence = 0

    @property
    def leases(self) -> tuple[Lease, ...]:
        return tuple(self._leases_by_request.values())

    @property
    def managed_gpu_uuids(self) -> tuple[str, ...]:
        return tuple(self._workers)

    @property
    def worker_pids(self) -> tuple[int, ...]:
        return tuple(
            int(pid)
            for worker in self._workers.values()
            if isinstance((pid := getattr(worker, "pid", None)), int) and pid > 0
        )

    def worker(self, gpu_uuid: str) -> ManagedWorker:
        try:
            return self._workers[gpu_uuid]
        except KeyError as exc:
            raise LeaseRequestError(f"managed GPU not found: {gpu_uuid}") from exc

    @property
    def events(self) -> tuple[SupervisorEvent, ...]:
        return tuple(self._events)

    def has_active_lease(self, gpu_uuid: str) -> bool:
        return any(
            lease.state in {LeaseState.ACTIVE, LeaseState.ORPHANED}
            and gpu_uuid in lease.gpu_uuids
            for lease in self._leases_by_request.values()
        )

    def unhealthy_worker_uuids(self) -> tuple[str, ...]:
        unhealthy: list[str] = []
        for gpu_uuid, worker in self._workers.items():
            is_alive = getattr(worker, "is_alive", None)
            if callable(is_alive) and not is_alive():
                unhealthy.append(gpu_uuid)
        return tuple(unhealthy)

    def replace_failed_worker(
        self, gpu_uuid: str, replacement: ManagedWorker
    ) -> ManagedWorker:
        old_worker = self._workers.get(gpu_uuid)
        if old_worker is None:
            raise LeaseRequestError(f"managed GPU not found: {gpu_uuid}")
        is_alive = getattr(old_worker, "is_alive", None)
        if not callable(is_alive) or is_alive():
            raise LeaseRequestError(f"worker is not failed: {gpu_uuid}")
        if replacement.status.gpu_uuid != gpu_uuid:
            raise LeaseRequestError("replacement worker GPU UUID does not match")
        self._workers[gpu_uuid] = replacement
        self._record_event("WORKER_RESTARTED", f"restarted {gpu_uuid}")
        return old_worker

    def add_worker(self, gpu_uuid: str, worker: ManagedWorker) -> None:
        if gpu_uuid in self._workers:
            raise LeaseRequestError(f"GPU is already managed: {gpu_uuid}")
        if worker.status.gpu_uuid != gpu_uuid:
            raise LeaseRequestError("worker GPU UUID does not match registration UUID")
        self._workers[gpu_uuid] = worker
        self._record_event("WORKER_ADDED", f"added {gpu_uuid}")

    def remove_worker(self, gpu_uuid: str) -> ManagedWorker:
        worker = self._workers.get(gpu_uuid)
        if worker is None:
            raise LeaseRequestError(f"managed GPU not found: {gpu_uuid}")
        if self.has_active_lease(gpu_uuid):
            raise LeaseRequestError(f"GPU has an active lease: {gpu_uuid}")
        for lease in tuple(self._queue):
            candidates = lease.request.candidate_uuids
            if candidates is not None and gpu_uuid in candidates:
                self._queue.remove(lease)
                lease.state = LeaseState.CANCELLED
                lease.error = f"candidate GPU was removed: {gpu_uuid}"
        del self._workers[gpu_uuid]
        self._record_event("WORKER_REMOVED", f"removed {gpu_uuid}")
        return worker

    def status_snapshot(self) -> SupervisorStatus:
        headroom_by_gpu = self._lease_headroom_by_gpu()
        gpus = tuple(
            ManagedGPUStatus(
                index=snapshot.index,
                uuid=snapshot.uuid,
                name=snapshot.name,
                total_mib=snapshot.total_mib,
                free_mib=snapshot.free_mib,
                utilization_percent=snapshot.utilization_percent,
                reserved_mib=self._workers[snapshot.uuid].status.held_mib,
                leased_mib=headroom_by_gpu.get(snapshot.uuid, 0),
                worker_state=self._workers[snapshot.uuid].status.state.value,
                temperature_c=snapshot.temperature_c,
                mig_mode=snapshot.mig_mode,
            )
            for snapshot in self._observer.snapshots()
            if snapshot.uuid in self._workers
        )
        owners = self._managed_process_owners()
        worker_pids = {
            int(pid): gpu_uuid
            for gpu_uuid, worker in self._workers.items()
            if isinstance((pid := getattr(worker, "pid", None)), int) and pid > 0
        }
        processes: list[GPUProcessStatus] = []
        for gpu in gpus:
            for process in self._observer.processes(gpu.uuid):
                owner = owners.get(process.pid)
                if process.pid in worker_pids:
                    classification = "WATCHGPU_WORKER"
                    lease_id = None
                    task_name = None
                elif owner is not None:
                    classification = "MANAGED_TRAINING"
                    lease_id = owner.lease_id
                    task_name = owner.request.task_name
                else:
                    classification = "EXTERNAL"
                    lease_id = None
                    task_name = None
                processes.append(
                    GPUProcessStatus(
                        gpu_uuid=gpu.uuid,
                        pid=process.pid,
                        used_memory_mib=process.used_memory_mib,
                        name=process.name,
                        classification=classification,
                        lease_id=lease_id,
                        task_name=task_name,
                    )
                )
        return SupervisorStatus(
            gpus=gpus,
            leases=self.leases,
            processes=tuple(processes),
            events=tuple(self._events),
        )

    def _managed_process_owners(self) -> dict[int, Lease]:
        owners: dict[int, Lease] = {}
        for lease in self._leases_by_request.values():
            if lease.state not in {LeaseState.ACTIVE, LeaseState.ORPHANED}:
                continue
            try:
                root = psutil.Process(lease.request.client_pid)
                if (
                    lease.request.client_start_time is not None
                    and abs(root.create_time() - lease.request.client_start_time) > 0.01
                ):
                    continue
                process_ids = {root.pid}
                process_ids.update(child.pid for child in root.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                process_ids = set()
                for gpu_uuid in lease.gpu_uuids:
                    try:
                        gpu_processes = self._observer.processes(gpu_uuid)
                    except Exception:
                        continue
                    for gpu_process in gpu_processes:
                        try:
                            if lease.request.client_process_group is not None:
                                matches = (
                                    os.getpgid(gpu_process.pid)
                                    == lease.request.client_process_group
                                )
                            elif lease.request.client_session_id is not None:
                                matches = (
                                    os.getsid(gpu_process.pid)
                                    == lease.request.client_session_id
                                )
                            else:
                                matches = False
                        except (ProcessLookupError, PermissionError, OSError):
                            matches = False
                        if matches:
                            process_ids.add(gpu_process.pid)
            for process_id in process_ids:
                owners.setdefault(process_id, lease)
        return owners

    def update_policy(self, config: WatchGPUConfig, *, now: float) -> None:
        overrides = {gpu.selector: gpu for gpu in config.gpus}
        unknown = set(overrides) - self._workers.keys()
        if unknown:
            raise ValueError(
                f"config references unmanaged GPUs: {', '.join(sorted(unknown))}"
            )
        for gpu_uuid, worker in self._workers.items():
            override = overrides.get(gpu_uuid)
            worker.update_policy(
                limits=ReservationLimits(
                    leave_free_mib=(
                        override.leave_free_mib
                        if override is not None and override.leave_free_mib is not None
                        else config.leave_free_mib
                    ),
                    reserve_limit_mib=(
                        override.reserve_limit_mib if override is not None else None
                    ),
                    reserve_ratio=config.reserve_ratio,
                ),
                growth_stability_seconds=config.growth_stability_seconds,
                allocation_tolerance_mib=config.allocation_tolerance_mib,
            )
            worker.update_maintenance_policy(
                enabled=config.maintenance_compute_enabled,
                duty_cycle_percent=config.maintenance_duty_cycle_percent,
                pause_above_utilization=config.compute_pause_above_utilization,
                cpu_budget_percent=config.cpu_budget_percent,
            )
        self.tick(now=now)
        self._record_event("POLICY_UPDATED", "runtime policy updated")

    def pause_workers(self, gpu_uuid: str | None = None) -> tuple[WorkerStatus, ...]:
        snapshots = {snapshot.uuid: snapshot for snapshot in self._observer.snapshots()}
        statuses = tuple(worker.pause() for worker in self._selected_workers(gpu_uuid))
        self._verify_status_releases(statuses, snapshots)
        self._record_event("WORKER_PAUSED", f"paused {gpu_uuid or 'all'}")
        return statuses

    def resume_workers(self, gpu_uuid: str | None = None) -> tuple[WorkerStatus, ...]:
        statuses = tuple(worker.resume() for worker in self._selected_workers(gpu_uuid))
        self._record_event("WORKER_RESUMED", f"resumed {gpu_uuid or 'all'}")
        return statuses

    def set_maintenance_cpu_pressure(self, over_budget: bool) -> None:
        """Apply one aggregate process-tree CPU gate to every capable worker."""

        for worker in self._workers.values():
            update = getattr(worker, "set_maintenance_cpu_pressure", None)
            if callable(update):
                try:
                    update(over_budget)
                except Exception:
                    # Health reconciliation owns failed-worker replacement; one dead
                    # worker must not prevent the aggregate gate reaching the others.
                    continue

    def release_reservations(
        self, *, gpu_uuid: str | None = None, memory_mib: int | None = None
    ) -> tuple[WorkerStatus, ...]:
        snapshots = {snapshot.uuid: snapshot for snapshot in self._observer.snapshots()}
        statuses: list[WorkerStatus] = []
        for worker in self._selected_workers(gpu_uuid):
            release_mib = worker.status.held_mib if memory_mib is None else memory_mib
            if release_mib == 0:
                statuses.append(worker.status)
            else:
                statuses.append(worker.release_reservation(release_mib))
        self._verify_status_releases(tuple(statuses), snapshots)
        self._record_event("RESERVATION_RELEASED", f"released {gpu_uuid or 'all'}")
        return tuple(statuses)

    def stop_workers(self) -> tuple[WorkerStatus, ...]:
        statuses: list[WorkerStatus] = []
        failures: dict[str, BaseException] = {}
        snapshots = {snapshot.uuid: snapshot for snapshot in self._observer.snapshots()}
        for gpu_uuid, worker in self._workers.items():
            try:
                statuses.append(worker.stop())
            except BaseException as exc:
                failures[gpu_uuid] = exc
                terminate = getattr(worker, "terminate", None)
                if callable(terminate):
                    try:
                        terminate()
                    except BaseException as terminate_exc:
                        failures[gpu_uuid] = RuntimeError(
                            f"stop failed ({type(exc).__name__}: {exc}); "
                            "forced termination also failed "
                            f"({type(terminate_exc).__name__}: {terminate_exc})"
                        )
                self._record_event(
                    "WORKER_STOP_ERROR",
                    f"{gpu_uuid}: {type(failures[gpu_uuid]).__name__}: "
                    f"{failures[gpu_uuid]}",
                    severity="ERROR",
                )
        try:
            self._verify_status_releases(tuple(statuses), snapshots)
        except BaseException as exc:
            failures["driver-verification"] = exc
        if failures:
            raise WorkerShutdownError(failures)
        self._record_event("WORKERS_STOPPED", "all WatchGPU workers stopped")
        return tuple(statuses)

    def _verify_status_releases(
        self,
        statuses: tuple[WorkerStatus, ...],
        snapshots: Mapping[str, GPUSnapshot],
    ) -> None:
        for status in statuses:
            if status.net_released_mib <= 0:
                continue
            snapshot = snapshots.get(status.gpu_uuid)
            if snapshot is None:
                raise LeaseRequestError(
                    f"GPU disappeared while verifying release: {status.gpu_uuid}"
                )
            expected_free_mib = snapshot.free_mib + status.net_released_mib
            if not self._release_verifier.verify(
                status.gpu_uuid,
                expected_free_mib=expected_free_mib,
                timeout=self._release_timeout_seconds,
            ):
                message = f"driver did not confirm release for {status.gpu_uuid}"
                self._record_event("RELEASE_VERIFY_FAILED", message, severity="ERROR")
                raise LeaseRequestError(message)

    def _selected_workers(self, gpu_uuid: str | None) -> tuple[ManagedWorker, ...]:
        if gpu_uuid is None or gpu_uuid == "all":
            return tuple(self._workers.values())
        worker = self._workers.get(gpu_uuid)
        if worker is None:
            raise LeaseRequestError(f"managed GPU not found: {gpu_uuid}")
        return (worker,)

    def request_lease(self, request: GroupLeaseRequest, *, now: float) -> Lease:
        existing = self._leases_by_request.get(request.request_id)
        if existing is not None:
            if existing.request != request:
                raise LeaseRequestError("request_id was already used with different parameters")
            return existing

        lease = Lease(
            lease_id=request.request_id,
            request=request,
            state=LeaseState.QUEUED,
            created_at=now,
        )
        self._leases_by_request[request.request_id] = lease
        selected = self._select_workers(request)
        if selected is None:
            self._queue.append(lease)
            self._record_event("LEASE_QUEUED", f"{request.task_name} queued")
            return lease

        try:
            self._activate(lease, selected, now=now)
        except Exception as exc:
            lease.state = LeaseState.REJECTED
            lease.error = f"lease activation failed: {type(exc).__name__}: {exc}"
            self._record_event("LEASE_REJECTED", lease.error, severity="ERROR")
        return lease

    def restore_lease(
        self,
        request: GroupLeaseRequest,
        *,
        state: LeaseState,
        gpu_uuids: tuple[str, ...],
        created_at: float,
        expires_at: float | None,
    ) -> Lease:
        if request.request_id in self._leases_by_request:
            raise LeaseRequestError(f"lease is already restored: {request.request_id}")
        if state not in {LeaseState.QUEUED, LeaseState.ACTIVE, LeaseState.ORPHANED}:
            raise LeaseRequestError(f"cannot restore terminal lease state {state}")
        if state is LeaseState.QUEUED:
            if gpu_uuids:
                raise LeaseRequestError("queued lease cannot have assigned GPUs")
            restored_state = LeaseState.QUEUED
        else:
            if len(gpu_uuids) != request.gpu_count:
                raise LeaseRequestError("restored lease GPU count does not match request")
            missing = set(gpu_uuids) - self._workers.keys()
            if missing:
                raise LeaseRequestError(
                    f"restored lease references unmanaged GPUs: {', '.join(sorted(missing))}"
                )
            restored_state = LeaseState.ORPHANED
        lease = Lease(
            lease_id=request.request_id,
            request=request,
            state=restored_state,
            created_at=created_at,
            expires_at=expires_at,
            gpu_uuids=gpu_uuids,
        )
        self._leases_by_request[request.request_id] = lease
        if restored_state is LeaseState.QUEUED:
            self._queue.append(lease)
        return lease

    def renew_lease(self, lease_id: str, *, now: float) -> Lease:
        lease = self._leases_by_request.get(lease_id)
        if lease is None:
            raise LeaseRequestError(f"lease not found: {lease_id}")
        if lease.state is not LeaseState.ACTIVE:
            raise LeaseRequestError(f"lease cannot be renewed in state {lease.state}")
        if lease.expires_at is not None and now >= lease.expires_at:
            lease.state = LeaseState.ORPHANED
            raise LeaseRequestError("lease has already expired")
        lease.expires_at = now + lease.request.ttl_seconds
        return lease

    def release_lease(self, lease_id: str, *, now: float) -> Lease:
        lease = self._leases_by_request.get(lease_id)
        if lease is None:
            raise LeaseRequestError(f"lease not found: {lease_id}")
        if lease.state in {LeaseState.RELEASED, LeaseState.CANCELLED}:
            return lease
        if lease.state is LeaseState.QUEUED:
            self._queue.remove(lease)
            lease.state = LeaseState.CANCELLED
            self._record_event("LEASE_CANCELLED", f"{lease.lease_id} cancelled")
        elif lease.state in {LeaseState.ACTIVE, LeaseState.ORPHANED}:
            lease.state = LeaseState.RELEASED
            self._record_event("LEASE_RELEASED", f"{lease.lease_id} released")
        else:
            raise LeaseRequestError(f"lease cannot be released in state {lease.state}")
        lease.expires_at = now
        return lease

    def tick(self, *, now: float) -> tuple[WorkerStatus, ...]:
        for lease in tuple(self._queue):
            if not self._lease_activity_verifier.is_active(lease):
                self.release_lease(lease.lease_id, now=now)
        existing_orphans = tuple(
            lease
            for lease in self._leases_by_request.values()
            if lease.state is LeaseState.ORPHANED
        )
        for lease in self._leases_by_request.values():
            if (
                lease.state is LeaseState.ACTIVE
                and lease.expires_at is not None
                and now >= lease.expires_at
            ):
                lease.state = LeaseState.ORPHANED
        for lease in existing_orphans:
            if not self._lease_activity_verifier.is_active(lease):
                self.release_lease(lease.lease_id, now=now)

        snapshots = {snapshot.uuid: snapshot for snapshot in self._observer.snapshots()}
        headroom_by_gpu = self._lease_headroom_by_gpu()
        statuses: list[WorkerStatus] = []
        for gpu_uuid, worker in self._workers.items():
            snapshot = snapshots.get(gpu_uuid)
            if snapshot is None:
                continue
            try:
                status = worker.tick(
                    snapshot,
                    now=now,
                    lease_headroom_mib=headroom_by_gpu.get(gpu_uuid, 0),
                )
                self._verify_status_releases((status,), snapshots)
            except Exception as exc:
                try:
                    held_mib = worker.status.held_mib
                except Exception:
                    held_mib = 0
                status = WorkerStatus(
                    gpu_uuid=gpu_uuid,
                    state=WorkerState.DEGRADED,
                    action=WorkerAction.ERROR,
                    target_mib=held_mib,
                    held_mib=held_mib,
                    error=f"{type(exc).__name__}: {exc}",
                )
                self._record_event(
                    "WORKER_ERROR", status.error or "worker failed", severity="ERROR"
                )
            statuses.append(status)

        while self._queue:
            lease = self._queue[0]
            selected = self._select_workers(lease.request)
            if selected is None:
                break
            self._queue.pop(0)
            try:
                self._activate(lease, selected, now=now)
            except Exception as exc:
                lease.state = LeaseState.REJECTED
                lease.error = f"lease activation failed: {type(exc).__name__}: {exc}"
                self._record_event("LEASE_REJECTED", lease.error, severity="ERROR")
        return tuple(statuses)

    def _lease_headroom_by_gpu(self) -> dict[str, int]:
        headroom: dict[str, int] = {}
        for lease in self._leases_by_request.values():
            if lease.state not in {LeaseState.ACTIVE, LeaseState.ORPHANED}:
                continue
            for gpu_uuid in lease.gpu_uuids:
                headroom[gpu_uuid] = (
                    headroom.get(gpu_uuid, 0) + lease.request.memory_per_gpu_mib
                )
        return headroom

    def _select_workers(self, request: GroupLeaseRequest) -> tuple[str, ...] | None:
        if request.candidate_uuids is None:
            candidates = tuple(self._workers)
        else:
            missing = set(request.candidate_uuids) - self._workers.keys()
            if missing:
                raise LeaseRequestError(
                    f"candidate GPUs are not managed: {', '.join(sorted(missing))}"
                )
            candidates = request.candidate_uuids

        snapshots = {snapshot.uuid: snapshot for snapshot in self._observer.snapshots()}
        leased_gpus = {
            gpu_uuid
            for lease in self._leases_by_request.values()
            if lease.state in {LeaseState.ACTIVE, LeaseState.ORPHANED}
            for gpu_uuid in lease.gpu_uuids
        }
        capable = tuple(
            uuid
            for uuid in candidates
            if uuid not in leased_gpus
            and uuid in snapshots
            and snapshots[uuid].free_mib + self._workers[uuid].status.held_mib
            >= request.memory_per_gpu_mib
        )
        if len(capable) < request.gpu_count:
            return None
        return capable[: request.gpu_count]

    def _activate(self, lease: Lease, gpu_uuids: tuple[str, ...], *, now: float) -> None:
        snapshots = {snapshot.uuid: snapshot for snapshot in self._observer.snapshots()}
        released: dict[str, int] = {}
        for gpu_uuid in gpu_uuids:
            snapshot = snapshots.get(gpu_uuid)
            if snapshot is None:
                lease.state = LeaseState.REJECTED
                lease.error = f"GPU disappeared before lease activation: {gpu_uuid}"
                self._record_event("LEASE_REJECTED", lease.error, severity="ERROR")
                return
            status = self._workers[gpu_uuid].release_for_lease(
                lease.request.memory_per_gpu_mib
            )
            expected_free_mib = max(
                lease.request.memory_per_gpu_mib,
                snapshot.free_mib + status.net_released_mib,
            )
            if not self._release_verifier.verify(
                gpu_uuid,
                expected_free_mib=expected_free_mib,
                timeout=self._release_timeout_seconds,
            ):
                lease.state = LeaseState.REJECTED
                lease.error = f"driver did not confirm release for {gpu_uuid}"
                self._record_event("LEASE_REJECTED", lease.error, severity="ERROR")
                return
            released[gpu_uuid] = status.net_released_mib

        lease.gpu_uuids = gpu_uuids
        lease.released_by_gpu_mib = released
        lease.expires_at = now + lease.request.ttl_seconds
        lease.state = LeaseState.ACTIVE
        self._record_event(
            "LEASE_APPROVED",
            f"{lease.request.task_name} activated on {','.join(gpu_uuids)}",
        )

    def _record_event(
        self, event_type: str, message: str, *, severity: str = "INFO"
    ) -> None:
        self._event_sequence += 1
        self._events.append(
            SupervisorEvent(
                sequence=self._event_sequence,
                timestamp=time.time(),
                type=event_type,
                message=message,
                severity=severity,
            )
        )
        if len(self._events) > 500:
            del self._events[:-500]
