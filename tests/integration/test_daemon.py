from __future__ import annotations

import asyncio
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.config import GPUConfig, MaintenanceRestartConfig, WatchGPUConfig, load_config
from watchgpu.control import ApplyStatus
from watchgpu.daemon import WatchGPUDaemon, build_daemon, configure_daemon_cpu
from watchgpu.ipc import AsyncWatchGPUClient
from watchgpu.journal import EventJournal
from watchgpu.lifecycle import (
    LeaseStateStore,
    PersistedLease,
    PersistedLeaseState,
    RestartStateStore,
    ShutdownResultStore,
)
from watchgpu.models import GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.paths import WatchGPUPaths
from watchgpu.policy import ReservationLimits
from watchgpu.supervisor import GroupLeaseRequest, Supervisor, TrustingReleaseVerifier
from watchgpu.worker import WorkerController, WorkerState, WorkerStatus
from watchgpu.worker_process import WorkerProcessProxy, WorkerProcessSpec


class _StopFailingWorker(WorkerController):
    terminated = False

    def stop(self) -> WorkerStatus:
        raise RuntimeError("simulated worker stop failure")

    def terminate(self) -> None:
        self.terminated = True
        super().stop()


class _TrackingJournal(EventJournal):
    closed = False

    def close(self) -> None:
        super().close()
        self.closed = True


def test_daemon_serves_status_and_releases_workers_on_shutdown(tmp_path: Path) -> None:
    async def scenario() -> None:
        observer = InMemoryGPUObserver(
            (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 1000, 0),)
        )
        allocator = InMemoryMemoryAllocator(500)
        allocator.reconcile(3000)
        worker = WorkerController(
            gpu_uuid="GPU-0",
            allocator=allocator,
            limits=ReservationLimits(
                leave_free_mib=1000,
                reserve_limit_mib=None,
                reserve_ratio=None,
            ),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )
        supervisor = Supervisor(
            observer=observer,
            workers={"GPU-0": worker},
            release_verifier=TrustingReleaseVerifier(),
        )
        socket_path = tmp_path / "watchgpu.sock"
        daemon = WatchGPUDaemon(
            supervisor=supervisor,
            config=WatchGPUConfig(gpus=[GPUConfig(selector="GPU-0")]),
            socket_path=socket_path,
            config_path=tmp_path / "config.toml",
            poll_interval_seconds=0.01,
            lease_store=LeaseStateStore(tmp_path / "leases.json"),
        )

        await daemon.start()
        try:
            await asyncio.sleep(0.05)
            status = await AsyncWatchGPUClient(socket_path).call("status.get")
            assert status["gpus"][0]["reserved_mib"] == 3000
            assert status["policy"]["revision"] == 0
            lease = await AsyncWatchGPUClient(socket_path).call(
                "lease.request",
                {
                    "lease_request_id": "persist-before-shutdown",
                    "task_name": "training",
                    "gpu_count": 1,
                    "memory_per_gpu_mib": 500,
                    "ttl_seconds": 60,
                    "client_pid": os.getpid(),
                },
            )
            assert lease["state"] == "ACTIVE"
            persisted = LeaseStateStore(tmp_path / "leases.json").load()
            assert persisted[0].lease_id == "persist-before-shutdown"
            assert persisted[0].state is PersistedLeaseState.ACTIVE
        finally:
            await daemon.stop()

        assert worker.status.state is WorkerState.STOPPED
        assert worker.status.held_mib == 0
        assert not socket_path.exists()

    asyncio.run(scenario())


def test_daemon_cpu_affinity_is_capped_by_one_core_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    applied: list[set[int]] = []
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {9, 4, 7})
    monkeypatch.setattr(
        os, "sched_setaffinity", lambda _pid, cores: applied.append(set(cores))
    )

    selected = configure_daemon_cpu(
        WatchGPUConfig(cpu_affinity_cores=8, cpu_budget_percent=100)
    )

    assert selected == (4,)
    assert applied == [{4}]


def test_daemon_stop_cleans_every_worker_after_first_worker_fails(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer = InMemoryGPUObserver(
            (
                GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 1000, 0),
                GPUSnapshot(1, "GPU-1", "Test GPU", 10_000, 1000, 0),
            )
        )
        failing_allocator = InMemoryMemoryAllocator(500)
        failing_allocator.reconcile(1000)
        healthy_allocator = InMemoryMemoryAllocator(500)
        healthy_allocator.reconcile(1000)
        failing = _StopFailingWorker(
            gpu_uuid="GPU-0",
            allocator=failing_allocator,
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )
        healthy = WorkerController(
            gpu_uuid="GPU-1",
            allocator=healthy_allocator,
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )
        supervisor = Supervisor(
            observer=observer,
            workers={"GPU-0": failing, "GPU-1": healthy},
            release_verifier=TrustingReleaseVerifier(),
        )
        cleanup_calls: list[str] = []
        journal = _TrackingJournal(tmp_path / "events.jsonl")
        shutdown_store = ShutdownResultStore(tmp_path / "shutdown.json")
        daemon = WatchGPUDaemon(
            supervisor=supervisor,
            config=WatchGPUConfig(
                gpus=[GPUConfig(selector="GPU-0"), GPUConfig(selector="GPU-1")]
            ),
            socket_path=tmp_path / "watchgpu.sock",
            config_path=tmp_path / "config.toml",
            cleanup=lambda: cleanup_calls.append("observer"),
            event_journal=journal,
            shutdown_result_store=shutdown_store,
        )

        await daemon.start()
        await daemon.stop()
        await asyncio.wait_for(daemon.wait_stopped(), timeout=0.1)

        assert failing.terminated is True
        assert failing.status.held_mib == 0
        assert healthy.status.state is WorkerState.STOPPED
        assert healthy.status.held_mib == 0
        assert cleanup_calls == ["observer"]
        assert journal.closed is True
        assert daemon.last_error is not None
        assert "simulated worker stop failure" in daemon.last_error
        shutdown_result = shutdown_store.load()
        assert shutdown_result is not None
        assert shutdown_result.success is False
        assert not (tmp_path / "watchgpu.sock").exists()

    asyncio.run(scenario())


def test_build_daemon_resolves_host_gpu_index_to_uuid(tmp_path: Path) -> None:
    observer = InMemoryGPUObserver(
        (GPUSnapshot(0, "GPU-host-specific", "Test GPU", 10_000, 1000, 0),)
    )

    def worker_factory(
        snapshot: GPUSnapshot, _gpu: GPUConfig, _config: WatchGPUConfig
    ) -> WorkerController:
        return WorkerController(
            gpu_uuid=snapshot.uuid,
            allocator=InMemoryMemoryAllocator(500),
            limits=ReservationLimits(
                leave_free_mib=1000,
                reserve_limit_mib=None,
                reserve_ratio=None,
            ),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )

    daemon = build_daemon(
        WatchGPUConfig(gpus=[GPUConfig(selector="0")]),
        WatchGPUPaths(
            runtime_dir=tmp_path / "run",
            config_dir=tmp_path / "config",
            state_dir=tmp_path / "state",
            systemd_user_dir=tmp_path / "systemd",
        ),
        observer=observer,
        worker_factory=worker_factory,
    )

    assert daemon.config_controller.config.gpus[0].selector == "GPU-host-specific"


def test_build_daemon_restores_active_training_as_orphaned(tmp_path: Path) -> None:
    paths = WatchGPUPaths(
        runtime_dir=tmp_path / "run",
        config_dir=tmp_path / "config",
        state_dir=tmp_path / "state",
        systemd_user_dir=tmp_path / "systemd",
    )
    LeaseStateStore(paths.leases_path).save(
        (
            PersistedLease(
                lease_id="lease-before-restart",
                state=PersistedLeaseState.ACTIVE,
                task_name="training",
                client_pid=1234,
                client_start_time=100.0,
                gpu_uuids=("GPU-0",),
                memory_per_gpu_mib=2000,
                expires_at=200.0,
                gpu_count=1,
                ttl_seconds=60,
                created_at=50.0,
            ),
        )
    )
    observer = InMemoryGPUObserver(
        (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 3000, 0),)
    )

    def worker_factory(
        snapshot: GPUSnapshot, _gpu: GPUConfig, _config: WatchGPUConfig
    ) -> WorkerController:
        return WorkerController(
            gpu_uuid=snapshot.uuid,
            allocator=InMemoryMemoryAllocator(500),
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )

    daemon = build_daemon(
        WatchGPUConfig(gpus=[GPUConfig(selector="0")]),
        paths,
        observer=observer,
        worker_factory=worker_factory,
    )

    assert daemon.supervisor.leases[0].state.value == "ORPHANED"
    assert daemon.supervisor.leases[0].gpu_uuids == ("GPU-0",)


def test_gpu_set_change_stays_pending_until_active_lease_releases(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = WatchGPUPaths(
            runtime_dir=tmp_path / "run",
            config_dir=tmp_path / "config",
            state_dir=tmp_path / "state",
            systemd_user_dir=tmp_path / "systemd",
        )
        observer = InMemoryGPUObserver(
            (
                GPUSnapshot(0, "GPU-0", "A40", 10_000, 3000, 0),
                GPUSnapshot(1, "GPU-1", "A40", 10_000, 3000, 0),
            )
        )

        def worker_factory(
            snapshot: GPUSnapshot, _gpu: GPUConfig, _config: WatchGPUConfig
        ) -> WorkerController:
            return WorkerController(
                gpu_uuid=snapshot.uuid,
                allocator=InMemoryMemoryAllocator(500),
                limits=ReservationLimits(1000, None, None),
                growth_stability_seconds=0,
                allocation_tolerance_mib=0,
            )

        daemon = build_daemon(
            WatchGPUConfig(
                poll_interval_seconds=0.01,
                gpus=[GPUConfig(selector="GPU-0")],
            ),
            paths,
            observer=observer,
            worker_factory=worker_factory,
        )
        lease = daemon.supervisor.request_lease(
            GroupLeaseRequest(
                request_id="lease-delays-reconfigure",
                task_name="training",
                gpu_count=1,
                memory_per_gpu_mib=1000,
                ttl_seconds=60,
                client_pid=1234,
            ),
            now=100,
        )

        result = daemon.config_controller.apply(
            WatchGPUConfig(
                poll_interval_seconds=0.01,
                gpus=[GPUConfig(selector="1")],
            ),
            expected_revision=0,
            save=True,
        )
        assert result.status is ApplyStatus.PENDING
        assert daemon.supervisor.managed_gpu_uuids == ("GPU-0",)

        await daemon.start()
        try:
            daemon.supervisor.release_lease(lease.lease_id, now=101)
            await asyncio.sleep(0.05)
            assert daemon.config_controller.runtime_status is ApplyStatus.APPLIED
            assert daemon.supervisor.managed_gpu_uuids == ("GPU-1",)
            assert daemon.config_controller.config.gpus[0].selector == "GPU-1"
            assert load_config(paths.config_path).gpus[0].selector == "GPU-1"
        finally:
            await daemon.stop()

    asyncio.run(scenario())


def test_daemon_restarts_only_the_failed_worker(tmp_path: Path) -> None:
    async def scenario() -> None:
        paths = WatchGPUPaths(
            runtime_dir=tmp_path / "run",
            config_dir=tmp_path / "config",
            state_dir=tmp_path / "state",
            systemd_user_dir=tmp_path / "systemd",
        )
        observer = InMemoryGPUObserver(
            (GPUSnapshot(0, "GPU-0", "A40", 10_000, 1000, 0),)
        )
        created: list[WorkerProcessProxy] = []

        def worker_factory(
            snapshot: GPUSnapshot, _gpu: GPUConfig, _config: WatchGPUConfig
        ) -> WorkerProcessProxy:
            worker = WorkerProcessProxy(
                WorkerProcessSpec(
                    gpu_uuid=snapshot.uuid,
                    chunk_mib=500,
                    limits=ReservationLimits(1000, None, None),
                    growth_stability_seconds=0,
                    allocation_tolerance_mib=0,
                    allocator_kind="memory",
                )
            )
            created.append(worker)
            return worker

        daemon = build_daemon(
            WatchGPUConfig(
                poll_interval_seconds=0.01,
                gpus=[GPUConfig(selector="GPU-0")],
            ),
            paths,
            observer=observer,
            worker_factory=worker_factory,
        )
        await daemon.start()
        try:
            created[0].terminate()
            await asyncio.sleep(0.1)
            assert len(created) >= 2
            assert created[-1].is_alive()
        finally:
            await daemon.stop()

    asyncio.run(scenario())


def test_scheduled_restart_requests_controlled_daemon_shutdown(tmp_path: Path) -> None:
    async def scenario() -> None:
        current_time = datetime(2026, 7, 18, 11, 59, tzinfo=UTC)
        observer = InMemoryGPUObserver(
            (GPUSnapshot(0, "GPU-0", "A40", 10_000, 1000, 0),)
        )
        worker = WorkerController(
            gpu_uuid="GPU-0",
            allocator=InMemoryMemoryAllocator(500),
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )
        supervisor = Supervisor(
            observer=observer,
            workers={"GPU-0": worker},
            release_verifier=TrustingReleaseVerifier(),
        )
        restart_store = RestartStateStore(tmp_path / "restart.json")
        daemon = WatchGPUDaemon(
            supervisor=supervisor,
            config=WatchGPUConfig(
                poll_interval_seconds=0.01,
                gpus=[GPUConfig(selector="GPU-0")],
                maintenance_restart=MaintenanceRestartConfig(
                    enabled=True,
                    at="12:00",
                    jitter_seconds=0,
                    defer_while_leased=True,
                ),
            ),
            socket_path=tmp_path / "watchgpu.sock",
            config_path=tmp_path / "config.toml",
            wall_clock=lambda: current_time,
            restart_state_store=restart_store,
        )
        restart = asyncio.Event()
        daemon.set_restart_callback(restart.set)
        await daemon.start()
        try:
            await asyncio.sleep(0.02)
            status = await AsyncWatchGPUClient(tmp_path / "watchgpu.sock").call(
                "status.get"
            )
            assert status["maintenance_restart"] == {
                "state": "SCHEDULED",
                "scheduled_for": "2026-07-18T12:00:00+00:00",
                "last_executed_local_date": None,
            }
            current_time = datetime(2026, 7, 18, 12, 1, tzinfo=UTC)
            await asyncio.wait_for(restart.wait(), timeout=0.2)
            assert daemon.restart_requested
            assert restart_store.load().last_executed_local_date == date(2026, 7, 18)
        finally:
            await daemon.stop()

    asyncio.run(scenario())
