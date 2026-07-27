from __future__ import annotations

import asyncio
import json
import stat
from pathlib import Path

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.ipc import MAX_MESSAGE_BYTES, AsyncWatchGPUClient, IPCError, UnixSocketServer
from watchgpu.models import GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.protocol import SupervisorProtocol
from watchgpu.supervisor import Supervisor, TrustingReleaseVerifier
from watchgpu.worker import WorkerController


def _protocol() -> SupervisorProtocol:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(3000)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(
            leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=0,
        allocation_tolerance_mib=0,
    )
    supervisor = Supervisor(
        observer=InMemoryGPUObserver(
            (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 1000, 0),)
        ),
        workers={"GPU-0": worker},
        release_verifier=TrustingReleaseVerifier(),
    )
    return SupervisorProtocol(supervisor, clock=lambda: 100.0)


def test_same_user_can_request_lease_over_unix_socket(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(socket_path, _protocol())
        await server.start()
        try:
            client = AsyncWatchGPUClient(socket_path)
            response = await client.call(
                "lease.request",
                {
                    "lease_request_id": "lease-ipc",
                    "task_name": "training",
                    "gpu_count": 1,
                    "memory_per_gpu_mib": 2000,
                    "ttl_seconds": 60,
                    "client_pid": 1234,
                },
                request_id="rpc-ipc",
            )
            assert response["state"] == "ACTIVE"
            assert response["gpu_uuids"] == ["GPU-0"]
            assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
        finally:
            await server.close()

    asyncio.run(scenario())


def test_second_server_cannot_replace_an_active_socket(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        first = UnixSocketServer(socket_path, _protocol())
        second = UnixSocketServer(socket_path, _protocol())
        await first.start()
        try:
            try:
                await second.start()
            except IPCError as exc:
                assert "instance lock" in str(exc)
            else:
                raise AssertionError("second server unexpectedly acquired the socket")
            await second.close()
            status = await AsyncWatchGPUClient(socket_path).call("status.get")
            assert status["gpus"][0]["uuid"] == "GPU-0"
        finally:
            await first.close()

    asyncio.run(scenario())


def test_request_id_replay_is_idempotent_and_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(socket_path, _protocol())
        await server.start()
        client = AsyncWatchGPUClient(socket_path)
        try:
            first = await client.call(
                "worker.release",
                {"gpu_uuid": "GPU-0", "memory_mib": 500},
                request_id="release-once",
            )
            replay = await client.call(
                "worker.release",
                {"gpu_uuid": "GPU-0", "memory_mib": 500},
                request_id="release-once",
            )
            assert replay == first
            status = await client.call("status.get")
            assert status["gpus"][0]["reserved_mib"] == 2500
            try:
                await client.call(
                    "worker.release",
                    {"gpu_uuid": "GPU-0", "memory_mib": 1000},
                    request_id="release-once",
                )
            except IPCError as exc:
                assert "different content" in str(exc)
            else:
                raise AssertionError("conflicting request_id unexpectedly succeeded")
        finally:
            await server.close()

    asyncio.run(scenario())


def test_oversized_message_gets_a_bounded_protocol_error(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(socket_path, _protocol())
        await server.start()
        try:
            reader, writer = await asyncio.open_unix_connection(socket_path)
            writer.write(b"x" * (MAX_MESSAGE_BYTES + 1) + b"\n")
            await writer.drain()
            response = json.loads(await reader.readline())
            assert response["ok"] is False
            assert response["error"]["code"] == "MESSAGE_TOO_LARGE"
            writer.close()
            await writer.wait_closed()
        finally:
            await server.close()

    asyncio.run(scenario())
