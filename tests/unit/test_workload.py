from __future__ import annotations

import pytest

from watchgpu.worker_process import ProcessTreeCPUMonitor
from watchgpu.workload import (
    CPUMaintenanceDutyController,
    MaintenanceDutyController,
    MaintenanceState,
)


@pytest.mark.parametrize("target", [0, 50, 100])
def test_cpu_maintenance_target_is_bounded_to_one_core(target: int) -> None:
    controller = CPUMaintenanceDutyController(target)
    assert controller.target_percent == target
    assert 0 <= controller.work_seconds <= 0.05
    assert controller.sleep_seconds >= 0


def test_cpu_maintenance_target_can_be_changed_at_runtime() -> None:
    controller = CPUMaintenanceDutyController(50)
    assert controller.work_seconds == pytest.approx(controller.sleep_seconds)
    controller.update_target(100)
    assert controller.sleep_seconds == pytest.approx(0)
    controller.update_target(0)
    assert controller.work_seconds == pytest.approx(0)


def test_duty_controller_converts_compute_time_to_blocking_wait() -> None:
    controller = MaintenanceDutyController(duty_cycle_percent=5)

    delay = controller.record_compute_slice(0.020)

    assert delay == pytest.approx(0.380)
    assert controller.state is MaintenanceState.WAITING


def test_duty_controller_pauses_for_lease_external_load_and_cpu_pressure() -> None:
    controller = MaintenanceDutyController(duty_cycle_percent=5)

    assert not controller.allow_compute(
        enabled=True,
        lease_blocked=True,
        gpu_utilization_percent=0,
        pause_above_utilization=20,
        cpu_percent=0,
        cpu_budget_percent=100,
    )
    assert not controller.allow_compute(
        enabled=True,
        lease_blocked=False,
        gpu_utilization_percent=30,
        pause_above_utilization=20,
        cpu_percent=0,
        cpu_budget_percent=100,
    )
    for _ in range(3):
        assert not controller.allow_compute(
            enabled=True,
            lease_blocked=False,
            gpu_utilization_percent=0,
            pause_above_utilization=20,
            cpu_percent=120,
            cpu_budget_percent=100,
        )
    assert controller.state is MaintenanceState.PAUSED


def test_process_tree_cpu_monitor_aggregates_multiple_workers() -> None:
    samples = iter(
        (
            {(10, 1.0): 0.0, (20, 2.0): 0.0, (30, 3.0): 0.0},
            {(10, 1.0): 0.10, (20, 2.0): 0.45, (30, 3.0): 0.45},
        )
    )
    times = iter((10.0, 11.0))
    monitor = ProcessTreeCPUMonitor(
        root_pid=10,
        clock=lambda: next(times),
        snapshot_reader=lambda _root_pid: next(samples),
    )

    baseline = monitor.sample()
    aggregate = monitor.sample()

    assert baseline.percent == 0
    assert baseline.process_count == 3
    assert aggregate.percent == pytest.approx(100.0)
    assert aggregate.process_count == 3
