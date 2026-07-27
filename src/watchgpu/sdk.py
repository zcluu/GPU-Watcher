from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
import uuid
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Literal

import psutil  # type: ignore[import-untyped]

from watchgpu.ipc import AsyncWatchGPUClient, IPCError
from watchgpu.paths import WatchGPUPaths


class WatchGPUSDKError(RuntimeError):
    pass


class CUDAAlreadyInitializedError(WatchGPUSDKError):
    pass


class LeaseRejectedError(WatchGPUSDKError):
    pass


class LeaseTimeoutError(WatchGPUSDKError):
    pass


class LeaseReleaseError(WatchGPUSDKError):
    pass


class LeaseConnectionError(WatchGPUSDKError):
    pass


@dataclass(frozen=True, slots=True)
class MemoryRequest:
    task_name: str
    mib: int
    gpu: str | None = None
    ttl_seconds: float = 600.0
    request_id: str | None = None

    def as_group(self) -> GroupMemoryRequest:
        devices = None if self.gpu is None else (self.gpu,)
        return GroupMemoryRequest(
            task_name=self.task_name,
            count=1,
            mib_per_gpu=self.mib,
            devices=devices,
            ttl_seconds=self.ttl_seconds,
            request_id=self.request_id,
        )


@dataclass(frozen=True, slots=True)
class GroupMemoryRequest:
    task_name: str
    count: int
    mib_per_gpu: int
    devices: tuple[str, ...] | None = None
    ttl_seconds: float = 600.0
    request_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_name:
            raise ValueError("task_name cannot be empty")
        if self.count <= 0:
            raise ValueError("count must be positive")
        if self.mib_per_gpu <= 0:
            raise ValueError("mib_per_gpu must be positive")
        if self.ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        if self.devices is not None:
            if len(self.devices) < self.count:
                raise ValueError("devices must contain at least count selectors")
            if any(not selector for selector in self.devices):
                raise ValueError("device selectors cannot be empty")
            if len(set(self.devices)) != len(self.devices):
                raise ValueError("device selectors cannot be repeated")


@dataclass(frozen=True, slots=True)
class ManagedGPU:
    index: int
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    reserved_mib: int
    leased_mib: int


@dataclass(slots=True)
class LeaseGrant:
    lease_id: str
    gpu_uuids: tuple[str, ...]
    memory_per_gpu_mib: int
    expires_at: float | None
    heartbeat_error: str | None = None

    @property
    def device(self) -> str:
        return "cuda:0"

    @property
    def devices(self) -> tuple[str, ...]:
        return tuple(f"cuda:{index}" for index in range(len(self.gpu_uuids)))


def default_socket_path() -> Path:
    configured = os.environ.get("WATCHGPU_SOCKET")
    if configured:
        return Path(configured).expanduser()
    return WatchGPUPaths.discover().socket_path


def managed_gpus(*, socket_path: Path | None = None) -> tuple[ManagedGPU, ...]:
    portal = _ClientPortal(socket_path or default_socket_path())
    portal.start()
    try:
        try:
            return _parse_managed_gpus(portal.call("status.get"))
        except (IPCError, OSError, TimeoutError) as exc:
            raise LeaseConnectionError(f"cannot connect to WatchGPU: {exc}") from exc
    finally:
        portal.close()


def acquire(
    request: MemoryRequest | GroupMemoryRequest,
    *,
    socket_path: Path | None = None,
    timeout_seconds: float = 30.0,
    poll_interval_seconds: float = 0.1,
    heartbeat_interval_seconds: float | None = None,
) -> LeaseContext:
    group_request = request.as_group() if isinstance(request, MemoryRequest) else request
    return LeaseContext(
        group_request,
        socket_path=socket_path or default_socket_path(),
        timeout_seconds=timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
    )


class LeaseContext:
    def __init__(
        self,
        request: GroupMemoryRequest,
        *,
        socket_path: Path,
        timeout_seconds: float,
        poll_interval_seconds: float,
        heartbeat_interval_seconds: float | None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if heartbeat_interval_seconds is not None and heartbeat_interval_seconds <= 0:
            raise ValueError("heartbeat_interval_seconds must be positive")
        self.request = request
        self.socket_path = socket_path
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._heartbeat_interval_seconds = heartbeat_interval_seconds
        self._portal = _ClientPortal(socket_path)
        self._grant: LeaseGrant | None = None
        self._lease_request_id = request.request_id or str(uuid.uuid4())
        self._old_environment: dict[str, str | None] = {}
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None

    def __enter__(self) -> LeaseGrant:
        if self._grant is not None:
            raise WatchGPUSDKError("lease context cannot be entered more than once")
        _ensure_cuda_uninitialized()
        self._portal.start()
        try:
            candidate_uuids = self._candidate_uuids()
            params: dict[str, Any] = {
                "lease_request_id": self._lease_request_id,
                "task_name": self.request.task_name,
                "gpu_count": self.request.count,
                "memory_per_gpu_mib": self.request.mib_per_gpu,
                "ttl_seconds": self.request.ttl_seconds,
                "client_pid": os.getpid(),
                "client_start_time": psutil.Process(os.getpid()).create_time(),
                "client_process_group": os.getpgrp(),
                "client_session_id": os.getsid(0),
            }
            if candidate_uuids is not None:
                params["candidate_uuids"] = list(candidate_uuids)
            result = self._wait_for_grant(params)
            grant = _parse_grant(result, expected_gpu_count=self.request.count)
            self._grant = grant
            self._set_lease_environment(grant)
            self._start_heartbeat(grant)
            return grant
        except (IPCError, OSError, TimeoutError) as exc:
            self._cleanup_failed_enter()
            self._portal.close()
            raise LeaseConnectionError(f"cannot connect to WatchGPU: {exc}") from exc
        except BaseException:
            self._cleanup_failed_enter()
            self._portal.close()
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        del exc_value, traceback
        release_error: BaseException | None = None
        try:
            self._stop_heartbeat()
            if self._grant is not None:
                self._portal.call(
                    "lease.release", {"lease_id": self._grant.lease_id}
                )
        except (IPCError, OSError, TimeoutError) as error:
            release_error = error
        finally:
            self._restore_environment()
            self._portal.close()
        if release_error is not None and exc_type is None:
            raise LeaseReleaseError(f"failed to release lease: {release_error}") from release_error
        return False

    def _cleanup_failed_enter(self) -> None:
        self._stop_heartbeat()
        grant = self._grant
        if grant is not None:
            with suppress(IPCError, OSError, TimeoutError):
                self._portal.call("lease.release", {"lease_id": grant.lease_id})
        self._restore_environment()
        self._grant = None

    def _candidate_uuids(self) -> tuple[str, ...] | None:
        if self.request.devices is None:
            return None
        gpus = _parse_managed_gpus(self._portal.call("status.get"))
        by_index = {str(gpu.index): gpu.uuid for gpu in gpus}
        by_uuid = {gpu.uuid: gpu.uuid for gpu in gpus}
        selected: list[str] = []
        for selector in self.request.devices:
            gpu_uuid = by_index.get(selector) or by_uuid.get(selector)
            if gpu_uuid is None:
                raise LeaseRejectedError(f"GPU selector is not managed: {selector}")
            selected.append(gpu_uuid)
        return tuple(selected)

    def _wait_for_grant(self, params: Mapping[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + self._timeout_seconds
        while True:
            result = self._portal.call("lease.request", params)
            state = result.get("state")
            if state == "ACTIVE":
                return result
            if state in {"REJECTED", "CANCELLED", "RELEASED"}:
                reason = result.get("error") or f"lease entered state {state}"
                raise LeaseRejectedError(str(reason))
            if state != "QUEUED":
                raise LeaseRejectedError(f"unexpected lease state: {state!r}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with suppress(IPCError, OSError, TimeoutError):
                    self._portal.call(
                        "lease.release", {"lease_id": self._lease_request_id}
                    )
                raise LeaseTimeoutError(
                    f"timed out waiting for {self.request.count} GPU lease"
                )
            time.sleep(min(self._poll_interval_seconds, remaining))

    def _set_lease_environment(self, grant: LeaseGrant) -> None:
        updates = {
            "CUDA_VISIBLE_DEVICES": ",".join(grant.gpu_uuids),
            "WATCHGPU_LEASE_ID": grant.lease_id,
            "WATCHGPU_SOCKET": str(self.socket_path),
        }
        for name, value in updates.items():
            self._old_environment[name] = os.environ.get(name)
            os.environ[name] = value

    def _restore_environment(self) -> None:
        for name, old_value in self._old_environment.items():
            if old_value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old_value
        self._old_environment.clear()

    def _start_heartbeat(self, grant: LeaseGrant) -> None:
        interval = self._heartbeat_interval_seconds
        if interval is None:
            interval = max(0.01, min(self.request.ttl_seconds / 3, 30.0))
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(grant, interval),
            name=f"watchgpu-heartbeat[{grant.lease_id}]",
            daemon=True,
        )
        self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._heartbeat_thread = None

    def _heartbeat_loop(self, grant: LeaseGrant, interval: float) -> None:
        while not self._heartbeat_stop.wait(interval):
            try:
                result = self._portal.call(
                    "lease.renew", {"lease_id": grant.lease_id}
                )
                expires_at = result.get("expires_at")
                if expires_at is not None and isinstance(expires_at, (int, float)):
                    grant.expires_at = float(expires_at)
                grant.heartbeat_error = None
            except (IPCError, OSError, TimeoutError) as exc:
                grant.heartbeat_error = str(exc)


class _ClientPortal:
    def __init__(self, path: Path) -> None:
        self._client = AsyncWatchGPUClient(path)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="watchgpu-sdk-ipc",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5):
            raise WatchGPUSDKError("timed out starting SDK IPC loop")

    def call(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        loop = self._loop
        if loop is None:
            raise WatchGPUSDKError("SDK IPC loop is not running")
        future = asyncio.run_coroutine_threadsafe(
            self._client.call(method, params), loop
        )
        try:
            return future.result(timeout=10)
        except TimeoutError:
            future.cancel()
            raise

    def close(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None:
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5)
        self._loop = None
        self._thread = None
        self._ready.clear()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


def _parse_grant(result: Mapping[str, Any], *, expected_gpu_count: int) -> LeaseGrant:
    lease_id = result.get("lease_id")
    raw_uuids = result.get("gpu_uuids")
    memory_mib = result.get("memory_per_gpu_mib")
    expires_at = result.get("expires_at")
    if not isinstance(lease_id, str) or not lease_id:
        raise LeaseRejectedError("lease response is missing lease_id")
    if not isinstance(raw_uuids, list) or not all(
        isinstance(item, str) and item for item in raw_uuids
    ):
        raise LeaseRejectedError("lease response has invalid GPU UUIDs")
    if len(raw_uuids) != expected_gpu_count:
        raise LeaseRejectedError("lease response did not approve the full GPU group")
    if not isinstance(memory_mib, int) or isinstance(memory_mib, bool):
        raise LeaseRejectedError("lease response has invalid memory size")
    if expires_at is not None and not isinstance(expires_at, (int, float)):
        raise LeaseRejectedError("lease response has invalid expiry")
    return LeaseGrant(
        lease_id=lease_id,
        gpu_uuids=tuple(raw_uuids),
        memory_per_gpu_mib=memory_mib,
        expires_at=None if expires_at is None else float(expires_at),
    )


def _parse_managed_gpus(result: Mapping[str, Any]) -> tuple[ManagedGPU, ...]:
    raw_gpus = result.get("gpus")
    if not isinstance(raw_gpus, list):
        raise WatchGPUSDKError("status response has invalid GPUs")
    parsed: list[ManagedGPU] = []
    for raw_gpu in raw_gpus:
        if not isinstance(raw_gpu, Mapping):
            raise WatchGPUSDKError("status response has invalid GPU entry")
        try:
            parsed.append(
                ManagedGPU(
                    index=_response_int(raw_gpu, "index"),
                    uuid=_response_string(raw_gpu, "uuid"),
                    name=_response_string(raw_gpu, "name"),
                    total_mib=_response_int(raw_gpu, "total_mib"),
                    free_mib=_response_int(raw_gpu, "free_mib"),
                    reserved_mib=_response_int(raw_gpu, "reserved_mib"),
                    leased_mib=_response_int(raw_gpu, "leased_mib"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise WatchGPUSDKError("status response has invalid GPU entry") from exc
    return tuple(parsed)


def _response_int(value: Mapping[object, object], key: str) -> int:
    item = value[key]
    if not isinstance(item, int) or isinstance(item, bool):
        raise TypeError(f"{key} is not an integer")
    return item


def _response_string(value: Mapping[object, object], key: str) -> str:
    item = value[key]
    if not isinstance(item, str) or not item:
        raise TypeError(f"{key} is not a string")
    return item


def _ensure_cuda_uninitialized() -> None:
    torch_module = sys.modules.get("torch")
    cuda = getattr(torch_module, "cuda", None)
    is_initialized = getattr(cuda, "is_initialized", None)
    if callable(is_initialized) and bool(is_initialized()):
        raise CUDAAlreadyInitializedError(
            "WatchGPU lease must be acquired before CUDA is initialized"
        )
