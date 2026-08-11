from __future__ import annotations

import ctypes
import hashlib
import multiprocessing
import os
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from typing import Literal, cast

import psutil  # type: ignore[import-untyped]

from watchgpu.allocator import (
    InMemoryMemoryAllocator,
    MemoryAllocator,
    TorchMemoryAllocator,
)
from watchgpu.models import GPUSnapshot
from watchgpu.policy import ReservationLimits
from watchgpu.worker import WorkerController, WorkerStatus
from watchgpu.workload import CPUMaintenanceDutyController, MaintenanceDutyController


class WorkerProcessError(RuntimeError):
    pass


class WorkerProcessTimeoutError(WorkerProcessError):
    pass


@dataclass(frozen=True, slots=True)
class ProcessTreeCPUUsage:
    percent: float
    process_count: int
    sampled_at: float


CPUSnapshotReader = Callable[[int], Mapping[tuple[int, float], float]]


class ProcessTreeCPUMonitor:
    """Sample aggregate CPU time for a daemon and all of its descendants."""

    def __init__(
        self,
        *,
        root_pid: int,
        clock: Callable[[], float] = time.monotonic,
        snapshot_reader: CPUSnapshotReader | None = None,
    ) -> None:
        if root_pid <= 0:
            raise ValueError("root_pid must be positive")
        self._root_pid = root_pid
        self._clock = clock
        self._snapshot_reader = snapshot_reader or _read_process_tree_cpu_times
        self._last_sampled_at: float | None = None
        self._last_cpu_times: dict[tuple[int, float], float] = {}

    def sample(self) -> ProcessTreeCPUUsage:
        sampled_at = self._clock()
        current = dict(self._snapshot_reader(self._root_pid))
        if self._last_sampled_at is None:
            percent = 0.0
        else:
            elapsed = max(sampled_at - self._last_sampled_at, 1e-9)
            consumed = sum(
                max(0.0, cpu_time - self._last_cpu_times.get(identity, cpu_time))
                for identity, cpu_time in current.items()
            )
            percent = consumed / elapsed * 100.0
        self._last_sampled_at = sampled_at
        self._last_cpu_times = current
        return ProcessTreeCPUUsage(
            percent=percent,
            process_count=len(current),
            sampled_at=sampled_at,
        )


@dataclass(frozen=True, slots=True)
class WorkerProcessSpec:
    gpu_uuid: str
    chunk_mib: int
    limits: ReservationLimits
    growth_stability_seconds: float
    allocation_tolerance_mib: int
    allocator_kind: Literal["torch", "memory"] = "torch"
    cpu_affinity_cores: tuple[int, ...] | None = None
    worker_cpu_threads: int = 1
    maintenance_compute_enabled: bool = True
    maintenance_duty_cycle_percent: int = 5
    compute_pause_above_utilization: int = 20
    cpu_budget_percent: int = 100
    maintenance_cpu_target_percent: float = 0.0

    def __post_init__(self) -> None:
        if self.worker_cpu_threads <= 0:
            raise ValueError("worker_cpu_threads must be positive")
        if not 0 <= self.maintenance_cpu_target_percent <= 100:
            raise ValueError("maintenance_cpu_target_percent must be between 0 and 100")
        if self.cpu_affinity_cores is not None and (
            not self.cpu_affinity_cores
            or len(self.cpu_affinity_cores) != len(set(self.cpu_affinity_cores))
            or any(core < 0 for core in self.cpu_affinity_cores)
        ):
            raise ValueError("cpu_affinity_cores must contain unique non-negative cores")


class WorkerProcessProxy:
    """Synchronous Supervisor adapter for a spawned per-GPU worker process."""

    def __init__(
        self,
        spec: WorkerProcessSpec,
        *,
        startup_timeout: float = 30.0,
        command_timeout: float = 10.0,
        termination_timeout: float = 5.0,
    ) -> None:
        if not spec.gpu_uuid:
            raise ValueError("gpu_uuid is required")
        if startup_timeout <= 0:
            raise ValueError("startup_timeout must be positive")
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        if termination_timeout <= 0:
            raise ValueError("termination_timeout must be positive")
        context = multiprocessing.get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        self._connection: Connection | None = parent_connection
        self._lock = threading.Lock()
        self._last_status: WorkerStatus | None = None
        self._command_timeout = command_timeout
        self._termination_timeout = termination_timeout
        self._process = context.Process(
            target=_worker_main,
            args=(child_connection, spec),
            name=f"watchgpu-worker[{spec.gpu_uuid}]",
            daemon=False,
        )
        self._process.start()
        child_connection.close()
        if not parent_connection.poll(startup_timeout):
            self._process.terminate()
            self._process.join(timeout=5)
            parent_connection.close()
            self._connection = None
            raise WorkerProcessError(f"worker startup timed out for {spec.gpu_uuid}")
        try:
            self._last_status = self._receive()
        except BaseException:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
            parent_connection.close()
            self._connection = None
            raise

    @property
    def pid(self) -> int:
        return int(self._process.pid or 0)

    @property
    def status(self) -> WorkerStatus:
        if not self.is_alive():
            if self._last_status is not None:
                return self._last_status
            raise WorkerProcessError("worker process is not running")
        return self._exchange("status", None)

    def is_alive(self) -> bool:
        return self._process.is_alive()

    def tick(
        self,
        snapshot: GPUSnapshot,
        *,
        now: float,
        lease_headroom_mib: int = 0,
    ) -> WorkerStatus:
        return self._exchange("tick", (snapshot, now, lease_headroom_mib))

    def release_for_lease(self, requested_mib: int) -> WorkerStatus:
        return self._exchange("release_for_lease", requested_mib)

    def release_reservation(self, requested_mib: int) -> WorkerStatus:
        return self._exchange("release_reservation", requested_mib)

    def pause(self) -> WorkerStatus:
        return self._exchange("pause", None)

    def resume(self) -> WorkerStatus:
        return self._exchange("resume", None)

    def update_policy(
        self,
        *,
        limits: ReservationLimits,
        growth_stability_seconds: float,
        allocation_tolerance_mib: int,
    ) -> None:
        self._exchange(
            "update_policy",
            (limits, growth_stability_seconds, allocation_tolerance_mib),
        )

    def maintenance_step(self) -> bool:
        status = self._exchange("maintenance_step", None)
        return status.held_mib > 0

    def update_maintenance_policy(
        self,
        *,
        enabled: bool,
        duty_cycle_percent: int,
        pause_above_utilization: int,
        cpu_budget_percent: int,
        cpu_target_percent: float = 0.0,
    ) -> None:
        self._exchange(
            "update_maintenance_policy",
            (
                enabled,
                duty_cycle_percent,
                pause_above_utilization,
                cpu_budget_percent,
                cpu_target_percent,
            ),
        )

    def set_maintenance_cpu_pressure(self, over_budget: bool) -> None:
        self._exchange("set_cpu_pressure", over_budget)

    def stop(self) -> WorkerStatus:
        if not self.is_alive():
            if self._last_status is None:
                raise WorkerProcessError("worker exited without a final status")
            return self._last_status
        status = self._exchange("stop", None)
        self._process.join(timeout=self._termination_timeout)
        if self._process.is_alive():
            self.terminate()
            raise WorkerProcessError("worker did not stop after releasing its reservation")
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None
        return status

    def terminate(self) -> None:
        """Terminate a failed worker; never targets training or external processes."""

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=self._termination_timeout)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(timeout=self._termination_timeout)
        connection = self._connection
        if connection is not None:
            connection.close()
            self._connection = None

    def _exchange(self, command: str, payload: object) -> WorkerStatus:
        connection = self._connection
        if connection is None or not self._process.is_alive():
            raise WorkerProcessError(f"worker is unavailable while handling {command}")
        with self._lock:
            try:
                connection.send((command, payload))
                if not connection.poll(self._command_timeout):
                    self.terminate()
                    raise WorkerProcessTimeoutError(
                        f"worker timed out after {self._command_timeout:g}s "
                        f"while handling {command}"
                    )
                status = self._receive()
            except (BrokenPipeError, EOFError, OSError) as exc:
                raise WorkerProcessError(f"worker failed while handling {command}: {exc}") from exc
        self._last_status = status
        return status

    def _receive(self) -> WorkerStatus:
        connection = self._connection
        if connection is None:
            raise WorkerProcessError("worker connection is closed")
        response = connection.recv()
        if not isinstance(response, tuple) or len(response) != 2:
            raise WorkerProcessError("worker returned an invalid response")
        kind, payload = response
        if kind == "error":
            raise WorkerProcessError(str(payload))
        if kind != "ok" or not isinstance(payload, WorkerStatus):
            raise WorkerProcessError("worker returned an invalid status")
        return payload


def _worker_main(connection: Connection, spec: WorkerProcessSpec) -> None:
    _configure_process_environment(spec)
    try:
        allocator: MemoryAllocator
        if spec.allocator_kind == "memory":
            allocator = InMemoryMemoryAllocator(spec.chunk_mib)
        else:
            allocator = TorchMemoryAllocator(spec.chunk_mib, device="cuda:0")
        controller = WorkerController(
            gpu_uuid=spec.gpu_uuid,
            allocator=allocator,
            limits=spec.limits,
            growth_stability_seconds=spec.growth_stability_seconds,
            allocation_tolerance_mib=spec.allocation_tolerance_mib,
        )
        duty = MaintenanceDutyController(spec.maintenance_duty_cycle_percent)
        cpu_duty = CPUMaintenanceDutyController(
            spec.maintenance_cpu_target_percent
        )
        maintenance_enabled = spec.maintenance_compute_enabled
        pause_above_utilization = spec.compute_pause_above_utilization
        cpu_budget_percent = spec.cpu_budget_percent
        global_cpu_pressure = False
        gpu_utilization_percent = 0
        lease_blocked = False
        cpu_maintenance_active = False
        cpu_next_at = time.monotonic()
        last_wall = time.monotonic()
        last_cpu = time.process_time()
        connection.send(
            (
                "ok",
                _worker_status(
                    controller.status,
                    spec,
                    global_cpu_pressure,
                    cpu_maintenance_active,
                    cpu_duty.target_percent,
                ),
            )
        )
        while True:
            now = time.monotonic()
            cpu_delay = max(0.0, cpu_next_at - now)
            poll_timeout = min(
                duty.next_delay_seconds or 0.1,
                cpu_delay if cpu_duty.target_percent > 0 else 0.1,
            )
            if not connection.poll(min(max(poll_timeout, 0.001), 1.0)):
                current_wall = time.monotonic()
                current_cpu = time.process_time()
                wall_delta = max(current_wall - last_wall, 1e-9)
                cpu_percent = (current_cpu - last_cpu) / wall_delta * 100
                last_wall = current_wall
                last_cpu = current_cpu
                allowed = duty.allow_compute(
                    enabled=(
                        maintenance_enabled
                        and controller.status.held_mib > 0
                        and controller.status.state.name not in {"PAUSED", "STOPPED"}
                        and not global_cpu_pressure
                    ),
                    lease_blocked=lease_blocked,
                    gpu_utilization_percent=gpu_utilization_percent,
                    pause_above_utilization=pause_above_utilization,
                    cpu_percent=cpu_percent,
                    cpu_budget_percent=cpu_budget_percent,
                )
                if allowed:
                    started = time.monotonic()
                    if controller.maintenance_step():
                        duty.record_compute_slice(max(time.monotonic() - started, 1e-6))
                cpu_maintenance_active = False
                if (
                    allowed
                    and cpu_duty.target_percent > 0
                    and time.monotonic() >= cpu_next_at
                ):
                    slice_started = time.monotonic()
                    _run_cpu_health_slice(cpu_duty.work_seconds)
                    cpu_maintenance_active = True
                    cpu_next_at = slice_started + (
                        cpu_duty.work_seconds + cpu_duty.sleep_seconds
                    )
                continue

            command, payload = connection.recv()
            if command == "status":
                status = controller.status
            elif command == "tick":
                snapshot, now, headroom = cast(
                    tuple[GPUSnapshot, float, int], payload
                )
                status = controller.tick(
                    snapshot, now=now, lease_headroom_mib=headroom
                )
                gpu_utilization_percent = snapshot.utilization_percent
                lease_blocked = headroom > 0
            elif command == "release_for_lease":
                lease_blocked = True
                status = controller.release_for_lease(cast(int, payload))
            elif command == "release_reservation":
                status = controller.release_reservation(cast(int, payload))
            elif command == "pause":
                status = controller.pause()
            elif command == "resume":
                status = controller.resume()
            elif command == "update_policy":
                limits, stability, tolerance = cast(
                    tuple[ReservationLimits, float, int], payload
                )
                controller.update_policy(
                    limits=limits,
                    growth_stability_seconds=stability,
                    allocation_tolerance_mib=tolerance,
                )
                status = controller.status
            elif command == "maintenance_step":
                controller.maintenance_step()
                status = controller.status
            elif command == "update_maintenance_policy":
                (
                    maintenance_enabled,
                    duty_cycle_percent,
                    pause_above_utilization,
                    cpu_budget_percent,
                    cpu_target_percent,
                ) = cast(tuple[bool, int, int, int, float], payload)
                duty.update_duty_cycle(duty_cycle_percent)
                cpu_duty.update_target(cpu_target_percent)
                cpu_next_at = time.monotonic()
                status = controller.status
            elif command == "set_cpu_pressure":
                global_cpu_pressure = cast(bool, payload)
                status = controller.status
            elif command == "stop":
                status = _worker_status(
                    controller.stop(), spec, global_cpu_pressure, False,
                    cpu_duty.target_percent,
                )
                connection.send(("ok", status))
                return
            else:
                raise WorkerProcessError(f"unknown worker command: {command}")
            connection.send(
                (
                    "ok",
                    _worker_status(
                        status,
                        spec,
                        global_cpu_pressure,
                        cpu_maintenance_active,
                        cpu_duty.target_percent,
                    ),
                )
            )
    except EOFError:
        return
    except BaseException as exc:
        with suppress(BrokenPipeError, OSError):
            connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _configure_process_environment(spec: WorkerProcessSpec) -> None:
    _set_process_name("watchgpu-worker")
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(spec.worker_cpu_threads)
    os.environ["CUDA_VISIBLE_DEVICES"] = spec.gpu_uuid
    if spec.cpu_affinity_cores is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, set(spec.cpu_affinity_cores))
    if spec.allocator_kind == "torch":
        import torch

        torch.set_num_threads(spec.worker_cpu_threads)
        with suppress(RuntimeError):
            torch.set_num_interop_threads(spec.worker_cpu_threads)


def _worker_status(
    status: WorkerStatus,
    spec: WorkerProcessSpec,
    global_cpu_pressure: bool,
    cpu_maintenance_active: bool,
    cpu_target_percent: float,
) -> WorkerStatus:
    return replace(
        status,
        cpu_affinity_cores=spec.cpu_affinity_cores or (),
        worker_cpu_threads=spec.worker_cpu_threads,
        maintenance_cpu_throttled=global_cpu_pressure,
        maintenance_cpu_target_percent=cpu_target_percent,
        maintenance_cpu_active=cpu_maintenance_active,
    )


def _run_cpu_health_slice(duration_seconds: float) -> bytes:
    """Exercise one CPU with repeatable checksum work for a bounded duration."""

    if duration_seconds <= 0:
        return b""
    payload = b"watchgpu-cpu-health-check\0" * 4096
    digest = b""
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        digest = hashlib.sha256(payload + digest).digest()
    return digest


def _read_process_tree_cpu_times(root_pid: int) -> dict[tuple[int, float], float]:
    try:
        root = psutil.Process(root_pid)
        processes = (root, *root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {}
    result: dict[tuple[int, float], float] = {}
    for process in processes:
        try:
            times = process.cpu_times()
            result[(process.pid, process.create_time())] = float(times.user + times.system)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return result


def _set_process_name(name: str) -> None:
    if os.name != "posix":
        return
    with suppress(Exception):
        libc = ctypes.CDLL(None)
        libc.prctl(15, name.encode()[:15], 0, 0, 0)
