from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Protocol, SupportsInt, cast

import pynvml  # type: ignore[import-untyped]

from watchgpu.models import GPUProcess, GPUSnapshot


class GPUSelectionError(ValueError):
    pass


class GPUObservationError(RuntimeError):
    pass


class GPUObserver(Protocol):
    def snapshots(self) -> tuple[GPUSnapshot, ...]: ...

    def processes(self, gpu_uuid: str) -> tuple[GPUProcess, ...]: ...


class InMemoryGPUObserver:
    def __init__(self, snapshots: Iterable[GPUSnapshot]) -> None:
        self._snapshots = tuple(snapshots)
        self._processes: dict[str, tuple[GPUProcess, ...]] = {}

    def snapshots(self) -> tuple[GPUSnapshot, ...]:
        return self._snapshots

    def processes(self, gpu_uuid: str) -> tuple[GPUProcess, ...]:
        return self._processes.get(gpu_uuid, ())

    def replace(self, snapshots: Iterable[GPUSnapshot]) -> None:
        self._snapshots = tuple(snapshots)

    def replace_processes(self, gpu_uuid: str, processes: Iterable[GPUProcess]) -> None:
        self._processes[gpu_uuid] = tuple(processes)


class NVMLGPUObserver:
    _MIB = 1024 * 1024

    def __init__(self) -> None:
        self._closed = False
        self._handles: dict[str, Any] = {}
        try:
            pynvml.nvmlInit()
            self._refresh_handles()
        except pynvml.NVMLError as exc:
            self._closed = True
            raise GPUObservationError(f"NVML initialization failed: {exc}") from exc

    def __enter__(self) -> NVMLGPUObserver:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError as exc:
            raise GPUObservationError(f"NVML shutdown failed: {exc}") from exc
        finally:
            self._closed = True
            self._handles.clear()

    @property
    def driver_version(self) -> str:
        self._ensure_open()
        try:
            return _as_text(pynvml.nvmlSystemGetDriverVersion())
        except pynvml.NVMLError as exc:
            raise GPUObservationError(f"NVML driver query failed: {exc}") from exc

    def snapshots(self) -> tuple[GPUSnapshot, ...]:
        self._ensure_open()
        try:
            self._refresh_handles()
            snapshots: list[GPUSnapshot] = []
            count = pynvml.nvmlDeviceGetCount()
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                try:
                    temperature_c = int(
                        pynvml.nvmlDeviceGetTemperature(
                            handle, pynvml.NVML_TEMPERATURE_GPU
                        )
                    )
                except pynvml.NVMLError:
                    temperature_c = None
                try:
                    current_mig, _pending_mig = pynvml.nvmlDeviceGetMigMode(handle)
                    mig_mode = "ENABLED" if int(current_mig) else "DISABLED"
                except pynvml.NVMLError:
                    mig_mode = None
                snapshots.append(
                    GPUSnapshot(
                        index=index,
                        uuid=_as_text(pynvml.nvmlDeviceGetUUID(handle)),
                        name=_as_text(pynvml.nvmlDeviceGetName(handle)),
                        total_mib=int(memory.total) // self._MIB,
                        free_mib=int(memory.free) // self._MIB,
                        utilization_percent=int(utilization.gpu),
                        temperature_c=temperature_c,
                        mig_mode=mig_mode,
                    )
                )
            return tuple(snapshots)
        except pynvml.NVMLError as exc:
            raise GPUObservationError(f"NVML snapshot failed: {exc}") from exc

    def processes(self, gpu_uuid: str) -> tuple[GPUProcess, ...]:
        self._ensure_open()
        handle = self._handles.get(gpu_uuid)
        if handle is None:
            self._refresh_handles()
            handle = self._handles.get(gpu_uuid)
        if handle is None:
            raise GPUObservationError(f"GPU UUID not found: {gpu_uuid}")

        try:
            running = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        except pynvml.NVMLError as exc:
            raise GPUObservationError(f"NVML process query failed for {gpu_uuid}: {exc}") from exc

        processes: list[GPUProcess] = []
        for process in running:
            pid = int(process.pid)
            raw_memory = getattr(process, "usedGpuMemory", None)
            used_memory_mib = _process_memory_mib(raw_memory, self._MIB)
            try:
                name = _as_text(pynvml.nvmlSystemGetProcessName(pid))
            except pynvml.NVMLError:
                name = None
            processes.append(GPUProcess(pid=pid, used_memory_mib=used_memory_mib, name=name))
        return tuple(processes)

    def _ensure_open(self) -> None:
        if self._closed:
            raise GPUObservationError("NVML observer is closed")

    def _refresh_handles(self) -> None:
        self._handles = {}
        count = pynvml.nvmlDeviceGetCount()
        for index in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            uuid = _as_text(pynvml.nvmlDeviceGetUUID(handle))
            self._handles[uuid] = handle


def resolve_gpu_selectors(
    snapshots: Sequence[GPUSnapshot], selectors: Sequence[str]
) -> tuple[GPUSnapshot, ...]:
    if not selectors:
        raise GPUSelectionError("at least one GPU selector is required")
    if "all" in selectors:
        if len(selectors) != 1:
            raise GPUSelectionError("'all' cannot be combined with other GPU selectors")
        selected_all = tuple(snapshots)
        _reject_unsupported_mig(selected_all)
        return selected_all

    by_index = {str(snapshot.index): snapshot for snapshot in snapshots}
    by_uuid = {snapshot.uuid: snapshot for snapshot in snapshots}
    selected: list[GPUSnapshot] = []
    seen_uuids: set[str] = set()
    for selector in selectors:
        snapshot = by_index.get(selector) or by_uuid.get(selector)
        if snapshot is None:
            raise GPUSelectionError(f"GPU selector not found: {selector}")
        if snapshot.uuid in seen_uuids:
            raise GPUSelectionError(f"GPU selected more than once: {snapshot.uuid}")
        selected.append(snapshot)
        seen_uuids.add(snapshot.uuid)
    resolved = tuple(selected)
    _reject_unsupported_mig(resolved)
    return resolved


def _reject_unsupported_mig(snapshots: Sequence[GPUSnapshot]) -> None:
    enabled = tuple(snapshot for snapshot in snapshots if snapshot.mig_mode == "ENABLED")
    if not enabled:
        return
    details = ", ".join(f"index {gpu.index} ({gpu.uuid})" for gpu in enabled)
    raise GPUSelectionError(
        f"MIG-enabled GPU allocation is not supported: {details}"
    )


def _as_text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value


def _process_memory_mib(value: object, mib: int) -> int | None:
    if value is None:
        return None
    numeric_value: int = int(cast(SupportsInt, value))
    sentinel = getattr(pynvml, "NVML_VALUE_NOT_AVAILABLE_ulonglong", None)
    sentinel_value = getattr(sentinel, "value", sentinel)
    if sentinel_value is not None and numeric_value == int(
        cast(SupportsInt, sentinel_value)
    ):
        return None
    return numeric_value // mib
