from __future__ import annotations

import pytest

from watchgpu.observer import GPUObservationError, NVMLGPUObserver


def test_nvml_observer_reports_real_gpu_state_without_allocating() -> None:
    try:
        observer = NVMLGPUObserver()
    except GPUObservationError as exc:
        pytest.skip(f"NVML is unavailable: {exc}")

    try:
        snapshots = observer.snapshots()
        assert snapshots
        assert len({snapshot.uuid for snapshot in snapshots}) == len(snapshots)
        for snapshot in snapshots:
            assert snapshot.total_mib > 0
            assert 0 <= snapshot.free_mib <= snapshot.total_mib
            assert 0 <= snapshot.utilization_percent <= 100
            assert isinstance(observer.processes(snapshot.uuid), tuple)
    finally:
        observer.close()
