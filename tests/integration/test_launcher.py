from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.ipc import UnixSocketServer
from watchgpu.launcher import LaunchConfig, launch
from watchgpu.models import GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.profile import ProfileStore
from watchgpu.protocol import SupervisorProtocol
from watchgpu.supervisor import Supervisor, TrustingReleaseVerifier
from watchgpu.worker import WorkerController


class FakePeakMonitor:
    def __init__(self, peaks: dict[str, int] | None = None) -> None:
        self.started = False
        self.peaks = peaks or {"GPU-0": 10_000}

    def start(self) -> None:
        self.started = True

    def stop(self) -> dict[str, int]:
        assert self.started
        return dict(self.peaks)


def test_launcher_acquires_before_runner_and_records_success_profile(tmp_path: Path) -> None:
    async def scenario() -> None:
        allocator = InMemoryMemoryAllocator(500)
        allocator.reconcile(3000)
        worker = WorkerController(
            gpu_uuid="GPU-0",
            allocator=allocator,
            limits=ReservationLimits(1000, None, None),
            growth_stability_seconds=0,
            allocation_tolerance_mib=0,
        )
        supervisor = Supervisor(
            observer=InMemoryGPUObserver(
                (GPUSnapshot(0, "GPU-0", "A40", 10_000, 1000, 0),)
            ),
            workers={"GPU-0": worker},
            release_verifier=TrustingReleaseVerifier(),
        )
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(
            socket_path, SupervisorProtocol(supervisor, clock=time.monotonic)
        )
        await server.start()
        store = ProfileStore(tmp_path / "profiles.jsonl")
        monitor = FakePeakMonitor()

        def runner(arguments: tuple[str, ...]) -> int:
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-0"
            assert arguments == ("--nnodes=1", "--nproc-per-node=1", "train.py")
            return 0

        try:
            exit_code = await asyncio.to_thread(
                launch,
                LaunchConfig(
                    task_name="profiled-training",
                    nproc_per_node=1,
                    memory_per_gpu=2000,
                    devices=("0",),
                    training_script="train.py",
                    training_args=(),
                ),
                socket_path=socket_path,
                profile_store=store,
                runner=runner,
                monitor_factory=lambda _gpus: monitor,
            )
        finally:
            await server.close()

        assert exit_code == 0
        assert store.records()[0].recommended_memory_per_gpu_mib == 11_500
        assert supervisor.leases[0].state.value == "RELEASED"

    asyncio.run(scenario())


def test_launcher_grants_two_gpus_atomically_and_maps_torchrun_environment(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        observer = InMemoryGPUObserver(
            (
                GPUSnapshot(0, "GPU-A", "A40", 10_000, 1000, 0),
                GPUSnapshot(1, "GPU-B", "A40", 10_000, 1000, 0),
            )
        )
        workers: dict[str, WorkerController] = {}
        for gpu_uuid in ("GPU-A", "GPU-B"):
            allocator = InMemoryMemoryAllocator(500)
            allocator.reconcile(3000)
            workers[gpu_uuid] = WorkerController(
                gpu_uuid=gpu_uuid,
                allocator=allocator,
                limits=ReservationLimits(1000, None, None),
                growth_stability_seconds=0,
                allocation_tolerance_mib=0,
            )
        supervisor = Supervisor(
            observer=observer,
            workers=workers,
            release_verifier=TrustingReleaseVerifier(),
        )
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(
            socket_path, SupervisorProtocol(supervisor, clock=time.monotonic)
        )
        await server.start()
        store = ProfileStore(tmp_path / "profiles.jsonl")

        def runner(arguments: tuple[str, ...]) -> int:
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-A,GPU-B"
            assert arguments[:2] == ("--nnodes=1", "--nproc-per-node=2")
            return 0

        try:
            result = await asyncio.to_thread(
                launch,
                LaunchConfig(
                    task_name="two-rank-training",
                    nproc_per_node=2,
                    memory_per_gpu=2000,
                    devices=("0", "1"),
                    training_script="train.py",
                    training_args=("--batch-size", "8"),
                ),
                socket_path=socket_path,
                profile_store=store,
                runner=runner,
                monitor_factory=lambda _gpus: FakePeakMonitor(
                    {"GPU-A": 4000, "GPU-B": 4500}
                ),
            )
        finally:
            await server.close()

        assert result == 0
        lease = supervisor.leases[0]
        assert lease.gpu_uuids == ("GPU-A", "GPU-B")
        assert lease.state.value == "RELEASED"
        assert store.records()[0].world_size == 2
        assert set(store.records()[0].observed_peak_mib_by_gpu) == {"GPU-A", "GPU-B"}

    asyncio.run(scenario())
