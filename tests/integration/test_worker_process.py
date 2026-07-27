from __future__ import annotations

import os
import signal
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import psutil
import pytest

from watchgpu.models import GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.supervisor import Supervisor, TrustingReleaseVerifier
from watchgpu.worker import WorkerAction, WorkerState
from watchgpu.worker_process import (
    WorkerProcessProxy,
    WorkerProcessSpec,
    WorkerProcessTimeoutError,
    _configure_process_environment,
)


def test_spawned_worker_owns_allocator_and_releases_before_exit() -> None:
    worker = WorkerProcessProxy(
        WorkerProcessSpec(
            gpu_uuid="GPU-test",
            chunk_mib=500,
            limits=ReservationLimits(
                leave_free_mib=1000,
                reserve_limit_mib=None,
                reserve_ratio=None,
            ),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
            allocator_kind="memory",
        )
    )
    snapshot = GPUSnapshot(0, "GPU-test", "Test GPU", 10_000, 9000, 0)
    try:
        waiting = worker.tick(snapshot, now=0)
        grown = worker.tick(snapshot, now=1)
        worker.update_maintenance_policy(
            enabled=True,
            duty_cycle_percent=3,
            pause_above_utilization=10,
            cpu_budget_percent=50,
        )
        released = worker.release_for_lease(2000)

        assert worker.pid != 0
        assert waiting.action is WorkerAction.WAIT_FOR_STABILITY
        assert grown.held_mib == 8000
        assert released.held_mib == 6000
    finally:
        stopped = worker.stop()

    assert stopped.state is WorkerState.STOPPED
    assert stopped.held_mib == 0
    assert not worker.is_alive()


def test_one_crashed_worker_does_not_block_other_gpu_ticks() -> None:
    def spec(uuid: str) -> WorkerProcessSpec:
        return WorkerProcessSpec(
            gpu_uuid=uuid,
            chunk_mib=500,
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
            allocator_kind="memory",
        )

    failed = WorkerProcessProxy(spec("GPU-0"))
    healthy = WorkerProcessProxy(spec("GPU-1"))
    snapshots = (
        GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 9000, 0),
        GPUSnapshot(1, "GPU-1", "Test GPU", 10_000, 9000, 0),
    )
    supervisor = Supervisor(
        observer=InMemoryGPUObserver(snapshots),
        workers={"GPU-0": failed, "GPU-1": healthy},
        release_verifier=TrustingReleaseVerifier(),
    )
    failed.terminate()
    try:
        statuses = supervisor.tick(now=0)
    finally:
        healthy.stop()

    assert statuses[0].state is WorkerState.DEGRADED
    assert statuses[0].action is WorkerAction.ERROR
    assert statuses[1].gpu_uuid == "GPU-1"
    assert statuses[1].action is WorkerAction.WAIT_FOR_STABILITY


def test_unresponsive_worker_rpc_times_out_and_only_terminates_that_worker() -> None:
    spec = WorkerProcessSpec(
        gpu_uuid="GPU-hung",
        chunk_mib=500,
        limits=ReservationLimits(1000, None, None),
        growth_stability_seconds=0,
        allocation_tolerance_mib=0,
        allocator_kind="memory",
    )
    hung = WorkerProcessProxy(
        spec,
        command_timeout=0.05,
        termination_timeout=0.1,
    )
    healthy = WorkerProcessProxy(replace(spec, gpu_uuid="GPU-healthy"))
    try:
        os.kill(hung.pid, signal.SIGSTOP)
        started = time.monotonic()

        with pytest.raises(WorkerProcessTimeoutError, match="handling status"):
            _ = hung.status

        assert time.monotonic() - started < 1.0
        assert not hung.is_alive()
        assert healthy.is_alive()
        assert healthy.status.gpu_uuid == "GPU-healthy"
    finally:
        hung.terminate()
        healthy.stop()


def test_multiple_workers_share_one_affinity_and_global_cpu_throttle() -> None:
    if not hasattr(os, "sched_getaffinity"):
        pytest.skip("CPU affinity is unavailable on this platform")
    common_core = min(os.sched_getaffinity(0))

    def spec(uuid: str) -> WorkerProcessSpec:
        return WorkerProcessSpec(
            gpu_uuid=uuid,
            chunk_mib=500,
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
            allocator_kind="memory",
            cpu_affinity_cores=(common_core,),
            worker_cpu_threads=2,
        )

    first = WorkerProcessProxy(spec("GPU-0"))
    second = WorkerProcessProxy(spec("GPU-1"))
    supervisor = Supervisor(
        observer=InMemoryGPUObserver(
            (
                GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 9000, 0),
                GPUSnapshot(1, "GPU-1", "Test GPU", 10_000, 9000, 0),
            )
        ),
        workers={"GPU-0": first, "GPU-1": second},
        release_verifier=TrustingReleaseVerifier(),
    )
    try:
        assert psutil.Process(first.pid).cpu_affinity() == [common_core]
        assert psutil.Process(second.pid).cpu_affinity() == [common_core]
        assert first.status.worker_cpu_threads == 2
        assert second.status.worker_cpu_threads == 2

        supervisor.set_maintenance_cpu_pressure(True)

        assert first.status.maintenance_cpu_throttled is True
        assert second.status.maintenance_cpu_throttled is True
    finally:
        supervisor.stop_workers()


def test_worker_thread_limit_reaches_environment_and_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    fake_torch = SimpleNamespace(
        set_num_threads=lambda count: calls.append(("intra", count)),
        set_num_interop_threads=lambda count: calls.append(("interop", count)),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cores: None)
    spec = WorkerProcessSpec(
        gpu_uuid="GPU-thread-test",
        chunk_mib=500,
        limits=ReservationLimits(1000, None, None),
        growth_stability_seconds=0,
        allocation_tolerance_mib=0,
        allocator_kind="torch",
        cpu_affinity_cores=(7,),
        worker_cpu_threads=3,
    )

    _configure_process_environment(spec)

    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert os.environ["MKL_NUM_THREADS"] == "3"
    assert calls == [("intra", 3), ("interop", 3)]
