from __future__ import annotations

from dataclasses import dataclass
from math import floor

from watchgpu.models import GPUSnapshot


@dataclass(frozen=True, slots=True)
class ReservationLimits:
    leave_free_mib: int
    reserve_limit_mib: int | None
    reserve_ratio: float | None


def calculate_target_hold_mib(
    snapshot: GPUSnapshot,
    *,
    current_hold_mib: int,
    limits: ReservationLimits,
) -> int:
    """Return how much memory WatchGPU should hold for the observed state."""

    external_used_mib = max(
        0,
        snapshot.total_mib - snapshot.free_mib - max(0, current_hold_mib),
    )
    target_mib = max(0, snapshot.total_mib - external_used_mib - limits.leave_free_mib)

    if limits.reserve_limit_mib is not None:
        target_mib = min(target_mib, limits.reserve_limit_mib)
    if limits.reserve_ratio is not None:
        target_mib = min(target_mib, floor(snapshot.total_mib * limits.reserve_ratio))

    return min(snapshot.total_mib, target_mib)
