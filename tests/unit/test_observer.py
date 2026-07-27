from __future__ import annotations

import pytest

from watchgpu.models import GPUSnapshot
from watchgpu.observer import GPUSelectionError, InMemoryGPUObserver, resolve_gpu_selectors


def _snapshots() -> tuple[GPUSnapshot, ...]:
    return (
        GPUSnapshot(0, "GPU-alpha", "A", 40_000, 30_000, 10),
        GPUSnapshot(1, "GPU-beta", "B", 48_000, 20_000, 70),
    )


def test_gpu_selectors_are_resolved_to_stable_uuids() -> None:
    selected = resolve_gpu_selectors(_snapshots(), ["1", "GPU-alpha"])

    assert [gpu.uuid for gpu in selected] == ["GPU-beta", "GPU-alpha"]


def test_all_selects_every_visible_gpu() -> None:
    assert resolve_gpu_selectors(_snapshots(), ["all"]) == _snapshots()


def test_mig_enabled_gpu_selection_fails_closed() -> None:
    mig_gpu = GPUSnapshot(
        0,
        "GPU-mig-parent",
        "MIG host",
        40_000,
        30_000,
        10,
        mig_mode="ENABLED",
    )

    with pytest.raises(GPUSelectionError, match="MIG-enabled GPU.*not supported"):
        resolve_gpu_selectors((mig_gpu,), ["all"])


@pytest.mark.parametrize(
    "selectors",
    [["9"], ["GPU-missing"], ["all", "0"], ["0", "GPU-alpha"]],
)
def test_invalid_or_duplicate_gpu_selection_is_rejected(selectors: list[str]) -> None:
    with pytest.raises(GPUSelectionError):
        resolve_gpu_selectors(_snapshots(), selectors)


def test_in_memory_observer_exposes_replaceable_snapshots() -> None:
    observer = InMemoryGPUObserver(_snapshots())
    assert observer.snapshots() == _snapshots()

    replacement = (_snapshots()[0],)
    observer.replace(replacement)
    assert observer.snapshots() == replacement
