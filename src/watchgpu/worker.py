from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from watchgpu.allocator import AllocationError, MemoryAllocator, ReconcileResult
from watchgpu.models import GPUSnapshot
from watchgpu.policy import ReservationLimits, calculate_target_hold_mib


class WorkerAction(StrEnum):
    NOOP = "NOOP"
    WAIT_FOR_STABILITY = "WAIT_FOR_STABILITY"
    GROW = "GROW"
    SHRINK = "SHRINK"
    RELEASE_FOR_LEASE = "RELEASE_FOR_LEASE"
    MANUAL_RELEASE = "MANUAL_RELEASE"
    PAUSE = "PAUSE"
    RESUME = "RESUME"
    STOP = "STOP"
    ERROR = "ERROR"


class WorkerState(StrEnum):
    OBSERVING = "OBSERVING"
    HOLDING = "HOLDING"
    PAUSED = "PAUSED"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"


@dataclass(frozen=True, slots=True)
class WorkerStatus:
    gpu_uuid: str
    state: WorkerState
    action: WorkerAction
    target_mib: int
    held_mib: int
    allocated_mib: int = 0
    released_mib: int = 0
    net_released_mib: int = 0
    error: str | None = None
    cpu_affinity_cores: tuple[int, ...] = ()
    worker_cpu_threads: int = 1
    maintenance_cpu_throttled: bool = False
    maintenance_cpu_target_percent: float = 0.0
    maintenance_cpu_active: bool = False


class ManagedWorker(Protocol):
    @property
    def status(self) -> WorkerStatus: ...

    def tick(
        self,
        snapshot: GPUSnapshot,
        *,
        now: float,
        lease_headroom_mib: int = 0,
    ) -> WorkerStatus: ...

    def release_for_lease(self, requested_mib: int) -> WorkerStatus: ...

    def release_reservation(self, requested_mib: int) -> WorkerStatus: ...

    def pause(self) -> WorkerStatus: ...

    def resume(self) -> WorkerStatus: ...

    def stop(self) -> WorkerStatus: ...

    def update_policy(
        self,
        *,
        limits: ReservationLimits,
        growth_stability_seconds: float,
        allocation_tolerance_mib: int,
    ) -> None: ...

    def maintenance_step(self) -> bool: ...

    def update_maintenance_policy(
        self,
        *,
        enabled: bool,
        duty_cycle_percent: int,
        pause_above_utilization: int,
        cpu_budget_percent: int,
        cpu_target_percent: float = 0.0,
    ) -> None: ...


class WorkerController:
    def __init__(
        self,
        *,
        gpu_uuid: str,
        allocator: MemoryAllocator,
        limits: ReservationLimits,
        growth_stability_seconds: float,
        allocation_tolerance_mib: int,
    ) -> None:
        self.gpu_uuid = gpu_uuid
        self._allocator = allocator
        self._limits = limits
        self._growth_stability_seconds = growth_stability_seconds
        self._allocation_tolerance_mib = allocation_tolerance_mib
        self._state = WorkerState.OBSERVING
        self._growth_candidate: tuple[int, float] | None = None
        self._policy_reconcile_pending = False

    @property
    def status(self) -> WorkerStatus:
        return self._status(WorkerAction.NOOP, self._allocator.held_mib)

    def update_policy(
        self,
        *,
        limits: ReservationLimits,
        growth_stability_seconds: float,
        allocation_tolerance_mib: int,
    ) -> None:
        if growth_stability_seconds < 0:
            raise ValueError("growth_stability_seconds cannot be negative")
        if allocation_tolerance_mib < 0:
            raise ValueError("allocation_tolerance_mib cannot be negative")
        limits_changed = limits != self._limits
        self._limits = limits
        self._growth_stability_seconds = growth_stability_seconds
        self._allocation_tolerance_mib = allocation_tolerance_mib
        self._growth_candidate = None
        self._policy_reconcile_pending = limits_changed

    def maintenance_step(self) -> bool:
        if self._state in {WorkerState.PAUSED, WorkerState.STOPPED}:
            return False
        return self._allocator.maintenance_step()

    def update_maintenance_policy(
        self,
        *,
        enabled: bool,
        duty_cycle_percent: int,
        pause_above_utilization: int,
        cpu_budget_percent: int,
        cpu_target_percent: float = 0.0,
    ) -> None:
        del (
            enabled,
            duty_cycle_percent,
            pause_above_utilization,
            cpu_budget_percent,
            cpu_target_percent,
        )

    def tick(
        self,
        snapshot: GPUSnapshot,
        *,
        now: float,
        lease_headroom_mib: int = 0,
    ) -> WorkerStatus:
        if snapshot.uuid != self.gpu_uuid:
            raise ValueError(
                f"worker {self.gpu_uuid} received snapshot for {snapshot.uuid}"
            )
        if self._state in {WorkerState.PAUSED, WorkerState.STOPPED}:
            return self.status

        target_mib = max(
            0,
            calculate_target_hold_mib(
                snapshot,
                current_hold_mib=self._allocator.held_mib,
                limits=self._limits,
                preserve_current_hold=not self._policy_reconcile_pending,
            )
            - max(0, lease_headroom_mib),
        )
        self._policy_reconcile_pending = False
        held_mib = self._allocator.held_mib

        if target_mib < held_mib - self._allocation_tolerance_mib:
            self._growth_candidate = None
            return self._reconcile(target_mib, WorkerAction.SHRINK)

        if target_mib > held_mib + self._allocation_tolerance_mib:
            if self._growth_candidate is None or self._growth_candidate[0] != target_mib:
                self._growth_candidate = (target_mib, now)
                return self._status(WorkerAction.WAIT_FOR_STABILITY, target_mib)
            candidate_target, candidate_since = self._growth_candidate
            if now - candidate_since < self._growth_stability_seconds:
                return self._status(WorkerAction.WAIT_FOR_STABILITY, candidate_target)
            self._growth_candidate = None
            return self._reconcile(target_mib, WorkerAction.GROW)

        self._growth_candidate = None
        if held_mib > 0:
            self._state = WorkerState.HOLDING
        return self._status(WorkerAction.NOOP, target_mib)

    def release_for_lease(self, requested_mib: int) -> WorkerStatus:
        if requested_mib <= 0:
            raise ValueError("requested_mib must be positive")
        if self._state is WorkerState.STOPPED:
            raise RuntimeError("worker is stopped")
        self._growth_candidate = None
        target_mib = max(0, self._allocator.held_mib - requested_mib)
        return self._reconcile(target_mib, WorkerAction.RELEASE_FOR_LEASE)

    def release_reservation(self, requested_mib: int) -> WorkerStatus:
        if requested_mib <= 0:
            raise ValueError("requested_mib must be positive")
        if self._state is WorkerState.STOPPED:
            raise RuntimeError("worker is stopped")
        self._growth_candidate = None
        target_mib = max(0, self._allocator.held_mib - requested_mib)
        return self._reconcile(target_mib, WorkerAction.MANUAL_RELEASE)

    def pause(self) -> WorkerStatus:
        if self._state is WorkerState.STOPPED:
            raise RuntimeError("worker is stopped")
        self._growth_candidate = None
        result = self._allocator.release_all()
        self._state = WorkerState.PAUSED
        return self._status_from_result(WorkerAction.PAUSE, result)

    def resume(self) -> WorkerStatus:
        if self._state is WorkerState.STOPPED:
            raise RuntimeError("worker is stopped")
        self._growth_candidate = None
        self._state = WorkerState.OBSERVING
        return self._status(WorkerAction.RESUME, 0)

    def stop(self) -> WorkerStatus:
        self._growth_candidate = None
        result = self._allocator.release_all()
        self._state = WorkerState.STOPPED
        return self._status_from_result(WorkerAction.STOP, result)

    def _reconcile(self, target_mib: int, action: WorkerAction) -> WorkerStatus:
        try:
            result = self._allocator.reconcile(target_mib)
        except AllocationError as exc:
            self._state = WorkerState.DEGRADED
            return self._status(WorkerAction.ERROR, target_mib, error=str(exc))
        self._state = WorkerState.HOLDING if result.held_mib else WorkerState.OBSERVING
        return self._status_from_result(action, result)

    def _status_from_result(
        self, action: WorkerAction, result: ReconcileResult
    ) -> WorkerStatus:
        return WorkerStatus(
            gpu_uuid=self.gpu_uuid,
            state=self._state,
            action=action,
            target_mib=result.target_mib,
            held_mib=result.held_mib,
            allocated_mib=result.allocated_mib,
            released_mib=result.released_mib,
            net_released_mib=result.net_released_mib,
        )

    def _status(
        self,
        action: WorkerAction,
        target_mib: int,
        *,
        error: str | None = None,
    ) -> WorkerStatus:
        return WorkerStatus(
            gpu_uuid=self.gpu_uuid,
            state=self._state,
            action=action,
            target_mib=target_mib,
            held_mib=self._allocator.held_mib,
            error=error,
        )
