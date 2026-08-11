from __future__ import annotations

from watchgpu.models import GPUSnapshot
from watchgpu.policy import ReservationLimits, calculate_target_hold_mib


def test_target_hold_is_stable_before_and_after_watchgpu_allocates() -> None:
    limits = ReservationLimits(leave_free_mib=1024, reserve_limit_mib=None, reserve_ratio=None)

    before = GPUSnapshot(
        index=0,
        uuid="GPU-0",
        name="Test GPU",
        total_mib=44 * 1024,
        free_mib=24 * 1024,
        utilization_percent=0,
    )
    after = GPUSnapshot(
        index=0,
        uuid="GPU-0",
        name="Test GPU",
        total_mib=44 * 1024,
        free_mib=1024,
        utilization_percent=0,
    )

    assert calculate_target_hold_mib(before, current_hold_mib=0, limits=limits) == 23 * 1024
    assert (
        calculate_target_hold_mib(after, current_hold_mib=23 * 1024, limits=limits)
        == 23 * 1024
    )


def test_target_hold_respects_the_strictest_configured_limit() -> None:
    snapshot = GPUSnapshot(
        index=0,
        uuid="GPU-0",
        name="Test GPU",
        total_mib=40 * 1024,
        free_mib=40 * 1024,
        utilization_percent=0,
    )
    limits = ReservationLimits(
        leave_free_mib=1024,
        reserve_limit_mib=30 * 1024,
        reserve_ratio=0.5,
    )

    assert calculate_target_hold_mib(snapshot, current_hold_mib=0, limits=limits) == 20 * 1024


def test_external_usage_cannot_reduce_an_existing_hold() -> None:
    limits = ReservationLimits(
        leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
    )
    pressured = GPUSnapshot(
        index=0,
        uuid="GPU-0",
        name="Test GPU",
        total_mib=10_000,
        free_mib=0,
        utilization_percent=50,
    )

    assert (
        calculate_target_hold_mib(
            pressured,
            current_hold_mib=8000,
            limits=limits,
        )
        == 8000
    )


def test_explicit_policy_reconcile_can_reduce_an_existing_hold() -> None:
    limits = ReservationLimits(
        leave_free_mib=3000, reserve_limit_mib=None, reserve_ratio=None
    )
    snapshot = GPUSnapshot(
        index=0,
        uuid="GPU-0",
        name="Test GPU",
        total_mib=10_000,
        free_mib=1000,
        utilization_percent=0,
    )

    assert (
        calculate_target_hold_mib(
            snapshot,
            current_hold_mib=3000,
            limits=limits,
            preserve_current_hold=False,
        )
        == 1000
    )
