from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.ipc import AsyncWatchGPUClient, UnixSocketServer
from watchgpu.models import GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.protocol import SupervisorProtocol
from watchgpu.sdk import (
    CUDAAlreadyInitializedError,
    LeaseConnectionError,
    LeaseContext,
    LeaseTimeoutError,
    MemoryRequest,
    acquire,
)
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
    return SupervisorProtocol(supervisor, clock=time.monotonic)


def test_sdk_context_requests_before_body_and_releases_on_exit(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(socket_path, _protocol())
        await server.start()
        body_entered = False

        def training() -> None:
            nonlocal body_entered
            with acquire(
                MemoryRequest(
                    task_name="sdk-training",
                    gpu="0",
                    mib=2000,
                    ttl_seconds=1,
                ),
                socket_path=socket_path,
                timeout_seconds=1,
            ) as lease:
                body_entered = True
                assert lease.gpu_uuids == ("GPU-0",)
                assert lease.device == "cuda:0"
                assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-0"

        try:
            await asyncio.to_thread(training)
            response = await AsyncWatchGPUClient(socket_path).call("status.get")
            assert body_entered
            assert response["leases"][0]["state"] == "RELEASED"
        finally:
            await server.close()

    asyncio.run(scenario())


def test_sdk_context_renews_active_lease_in_background(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(socket_path, _protocol())
        await server.start()

        def training() -> tuple[float, float]:
            with acquire(
                MemoryRequest(
                    task_name="heartbeat-training",
                    gpu="GPU-0",
                    mib=2000,
                    ttl_seconds=0.3,
                ),
                socket_path=socket_path,
                timeout_seconds=1,
                heartbeat_interval_seconds=0.05,
            ) as lease:
                assert lease.expires_at is not None
                initial_expiry = lease.expires_at
                time.sleep(0.18)
                assert lease.expires_at is not None
                return initial_expiry, lease.expires_at

        try:
            initial_expiry, renewed_expiry = await asyncio.to_thread(training)
            assert renewed_expiry > initial_expiry
        finally:
            await server.close()

    asyncio.run(scenario())


def test_sdk_releases_grant_when_local_setup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        server = UnixSocketServer(socket_path, _protocol())
        await server.start()

        def fail_heartbeat(_self: LeaseContext, _grant: object) -> None:
            raise RuntimeError("heartbeat setup failed")

        monkeypatch.setattr(LeaseContext, "_start_heartbeat", fail_heartbeat)

        def training() -> None:
            with pytest.raises(RuntimeError, match="heartbeat setup failed"), acquire(
                MemoryRequest(task_name="setup-fails", gpu="GPU-0", mib=2000),
                socket_path=socket_path,
                timeout_seconds=1,
            ):
                pass

        try:
            await asyncio.to_thread(training)
            response = await AsyncWatchGPUClient(socket_path).call("status.get")
            assert response["leases"][0]["state"] == "RELEASED"
        finally:
            await server.close()

    asyncio.run(scenario())


def test_sdk_refuses_to_request_after_cuda_initialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_initialized=lambda: True)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    context = acquire(
        MemoryRequest(task_name="too-late", mib=1000),
        socket_path=tmp_path / "missing.sock",
    )

    with pytest.raises(CUDAAlreadyInitializedError):
        context.__enter__()


def test_sdk_reports_missing_daemon_as_connection_error(tmp_path: Path) -> None:
    with pytest.raises(LeaseConnectionError), acquire(
        MemoryRequest(task_name="no-daemon", mib=1000),
        socket_path=tmp_path / "missing.sock",
    ):
        pass


def test_sdk_timeout_cancels_queued_request(tmp_path: Path) -> None:
    async def scenario() -> None:
        socket_path = tmp_path / "watchgpu.sock"
        protocol = _protocol()
        server = UnixSocketServer(socket_path, protocol)
        await server.start()

        def request_too_much() -> None:
            with pytest.raises(LeaseTimeoutError), acquire(
                MemoryRequest(
                    task_name="queued-timeout",
                    gpu="GPU-0",
                    mib=20_000,
                    request_id="timeout-request",
                ),
                socket_path=socket_path,
                timeout_seconds=0.05,
                poll_interval_seconds=0.01,
            ):
                pass

        try:
            await asyncio.to_thread(request_too_much)
            response = await AsyncWatchGPUClient(socket_path).call("status.get")
            assert response["leases"][0]["state"] == "CANCELLED"
        finally:
            await server.close()

    asyncio.run(scenario())
