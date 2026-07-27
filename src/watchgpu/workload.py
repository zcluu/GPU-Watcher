from __future__ import annotations

from enum import StrEnum


class MaintenanceState(StrEnum):
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    THROTTLED = "THROTTLED"
    PAUSED = "PAUSED"


class MaintenanceDutyController:
    """Pure duty-cycle and safety gating for a worker's blocking event loop."""

    def __init__(
        self, duty_cycle_percent: int, *, cpu_pause_after_samples: int = 3
    ) -> None:
        if not 1 <= duty_cycle_percent <= 20:
            raise ValueError("duty_cycle_percent must be between 1 and 20")
        if cpu_pause_after_samples <= 0:
            raise ValueError("cpu_pause_after_samples must be positive")
        self._duty_cycle_percent = duty_cycle_percent
        self._cpu_pause_after_samples = cpu_pause_after_samples
        self._cpu_high_samples = 0
        self._state = MaintenanceState.READY
        self._next_delay_seconds = 0.0

    @property
    def state(self) -> MaintenanceState:
        return self._state

    @property
    def next_delay_seconds(self) -> float:
        return self._next_delay_seconds

    def update_duty_cycle(self, duty_cycle_percent: int) -> None:
        if not 1 <= duty_cycle_percent <= 20:
            raise ValueError("duty_cycle_percent must be between 1 and 20")
        self._duty_cycle_percent = duty_cycle_percent

    def allow_compute(
        self,
        *,
        enabled: bool,
        lease_blocked: bool,
        gpu_utilization_percent: int,
        pause_above_utilization: int,
        cpu_percent: float,
        cpu_budget_percent: int,
    ) -> bool:
        if (
            not enabled
            or lease_blocked
            or gpu_utilization_percent >= pause_above_utilization
        ):
            self._cpu_high_samples = 0
            self._state = MaintenanceState.PAUSED
            return False
        if cpu_percent > cpu_budget_percent:
            self._cpu_high_samples += 1
            self._state = (
                MaintenanceState.PAUSED
                if self._cpu_high_samples >= self._cpu_pause_after_samples
                else MaintenanceState.THROTTLED
            )
            return False
        self._cpu_high_samples = 0
        self._state = MaintenanceState.RUNNING
        return True

    def record_compute_slice(self, elapsed_seconds: float) -> float:
        if elapsed_seconds <= 0:
            raise ValueError("elapsed_seconds must be positive")
        work_fraction = self._duty_cycle_percent / 100
        self._next_delay_seconds = elapsed_seconds * (1 - work_fraction) / work_fraction
        self._state = MaintenanceState.WAITING
        return self._next_delay_seconds
