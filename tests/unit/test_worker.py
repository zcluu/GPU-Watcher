from __future__ import annotations

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.models import GPUSnapshot
from watchgpu.policy import ReservationLimits
from watchgpu.worker import WorkerAction, WorkerController, WorkerState


def _snapshot(*, free_mib: int) -> GPUSnapshot:
    return GPUSnapshot(
        index=0,
        uuid="GPU-0",
        name="Test GPU",
        total_mib=10_000,
        free_mib=free_mib,
        utilization_percent=0,
    )


def test_worker_delays_growth_and_keeps_its_hold_under_external_pressure() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None),
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    waiting = worker.tick(_snapshot(free_mib=9000), now=0)
    assert waiting.action is WorkerAction.WAIT_FOR_STABILITY
    assert waiting.held_mib == 0

    still_waiting = worker.tick(_snapshot(free_mib=9000), now=9)
    assert still_waiting.action is WorkerAction.WAIT_FOR_STABILITY
    assert still_waiting.held_mib == 0

    grown = worker.tick(_snapshot(free_mib=9000), now=10)
    assert grown.action is WorkerAction.GROW
    assert grown.held_mib == 8000

    held = worker.tick(_snapshot(free_mib=0), now=11)
    assert held.action is WorkerAction.NOOP
    assert held.held_mib == 8000


def test_worker_releases_for_a_lease_and_pause_without_stopping_training() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(2300)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None),
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    released = worker.release_for_lease(700)
    assert released.action is WorkerAction.RELEASE_FOR_LEASE
    assert released.held_mib == 1600
    assert released.net_released_mib == 700

    paused = worker.pause()
    assert paused.action is WorkerAction.PAUSE
    assert paused.state is WorkerState.PAUSED
    assert paused.held_mib == 0

    resumed = worker.resume()
    assert resumed.action is WorkerAction.RESUME
    assert resumed.state is WorkerState.OBSERVING
    assert resumed.held_mib == 0


def test_worker_policy_can_be_updated_without_restart() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(3000)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(
            leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    worker.update_policy(
        limits=ReservationLimits(
            leave_free_mib=3000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=20,
        allocation_tolerance_mib=0,
    )
    status = worker.tick(_snapshot(free_mib=1000), now=0)

    assert status.action is WorkerAction.SHRINK
    assert status.held_mib == 1000


def test_unrelated_policy_update_does_not_unlock_external_pressure_shrink() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(8000)
    limits = ReservationLimits(
        leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
    )
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=limits,
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    worker.update_policy(
        limits=limits,
        growth_stability_seconds=20,
        allocation_tolerance_mib=0,
    )
    status = worker.tick(_snapshot(free_mib=0), now=0)

    assert status.action is WorkerAction.NOOP
    assert status.held_mib == 8000


def test_active_lease_freezes_the_hold_released_during_activation() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(8000)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(
            leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    released = worker.release_for_lease(3000)
    status = worker.tick(_snapshot(free_mib=0), now=0, lease_headroom_mib=3000)

    assert released.action is WorkerAction.RELEASE_FOR_LEASE
    assert released.held_mib == 5000
    assert status.action is WorkerAction.NOOP
    assert status.held_mib == 5000


def test_worker_can_manually_release_its_own_reservation() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(2300)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(
            leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    released = worker.release_reservation(700)

    assert released.action is WorkerAction.MANUAL_RELEASE
    assert released.held_mib == 1600
    assert released.net_released_mib == 700


def test_maintenance_step_reuses_an_existing_reservation_block() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(500)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(
            leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=10,
        allocation_tolerance_mib=0,
    )

    assert worker.maintenance_step()
    assert worker.status.held_mib == 500
    worker.pause()
    assert not worker.maintenance_step()
