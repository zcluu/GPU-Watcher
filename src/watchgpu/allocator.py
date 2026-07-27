from __future__ import annotations

import gc
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    target_mib: int
    held_mib: int
    allocated_mib: int
    released_mib: int

    @property
    def net_released_mib(self) -> int:
        return max(0, self.released_mib - self.allocated_mib)


class MemoryAllocator(Protocol):
    @property
    def held_mib(self) -> int: ...

    @property
    def chunk_sizes(self) -> tuple[int, ...]: ...

    def reconcile(self, target_mib: int) -> ReconcileResult: ...

    def release_all(self) -> ReconcileResult: ...

    def maintenance_step(self) -> bool: ...


class AllocationError(RuntimeError):
    def __init__(self, message: str, *, requested_mib: int, held_mib: int) -> None:
        super().__init__(message)
        self.requested_mib = requested_mib
        self.held_mib = held_mib


@dataclass(slots=True)
class _MemoryBlock:
    size_mib: int
    resource: object


class ChunkMemoryAllocator(ABC):
    def __init__(self, chunk_mib: int) -> None:
        if chunk_mib <= 0:
            raise ValueError("chunk_mib must be positive")
        self._chunk_mib = chunk_mib
        self._blocks: list[_MemoryBlock] = []

    @property
    def held_mib(self) -> int:
        return sum(block.size_mib for block in self._blocks)

    @property
    def chunk_sizes(self) -> tuple[int, ...]:
        return tuple(block.size_mib for block in self._blocks)

    def reconcile(self, target_mib: int) -> ReconcileResult:
        desired_sizes = plan_chunk_sizes(target_mib, self._chunk_mib)
        available = list(self._blocks)
        kept: list[_MemoryBlock] = []
        missing_sizes: list[int] = []

        for desired_size in desired_sizes:
            match_index = next(
                (
                    index
                    for index, block in enumerate(available)
                    if block.size_mib == desired_size
                ),
                None,
            )
            if match_index is None:
                missing_sizes.append(desired_size)
            else:
                kept.append(available.pop(match_index))

        released_mib = sum(block.size_mib for block in available)
        resources_to_release = [block.resource for block in available]
        self._blocks = kept
        if resources_to_release:
            # Drop the temporary _MemoryBlock references before the backend clears
            # its resource list and flushes an allocator cache.  In the torch
            # backend, keeping ``available`` alive until after ``empty_cache()``
            # meant tensors were only destructed after the cache flush, so NVML
            # never observed the reported release while this process stayed alive.
            available.clear()
            self._release_resources(resources_to_release)

        allocated_mib = 0
        for size_mib in missing_sizes:
            resource = self._allocate_resource(size_mib)
            self._blocks.append(_MemoryBlock(size_mib=size_mib, resource=resource))
            allocated_mib += size_mib

        return ReconcileResult(
            target_mib=target_mib,
            held_mib=self.held_mib,
            allocated_mib=allocated_mib,
            released_mib=released_mib,
        )

    def release_all(self) -> ReconcileResult:
        released_mib = self.held_mib
        resources = [block.resource for block in self._blocks]
        self._blocks = []
        if resources:
            self._release_resources(resources)
        return ReconcileResult(
            target_mib=0,
            held_mib=0,
            allocated_mib=0,
            released_mib=released_mib,
        )

    def maintenance_step(self) -> bool:
        if not self._blocks:
            return False
        self._run_maintenance(self._blocks[0].resource)
        return True

    @abstractmethod
    def _allocate_resource(self, size_mib: int) -> object:
        raise NotImplementedError

    @abstractmethod
    def _release_resources(self, resources: list[object]) -> None:
        raise NotImplementedError

    @abstractmethod
    def _run_maintenance(self, resource: object) -> None:
        raise NotImplementedError


class InMemoryMemoryAllocator(ChunkMemoryAllocator):
    def _allocate_resource(self, size_mib: int) -> object:
        return object()

    def _release_resources(self, resources: list[object]) -> None:
        resources.clear()

    def _run_maintenance(self, resource: object) -> None:
        del resource


class TorchMemoryAllocator(ChunkMemoryAllocator):
    _MIB = 1024 * 1024

    def __init__(self, chunk_mib: int, *, device: str = "cuda:0") -> None:
        super().__init__(chunk_mib)
        try:
            import torch
        except ImportError as exc:
            raise AllocationError(
                "PyTorch is not installed",
                requested_mib=0,
                held_mib=0,
            ) from exc
        if not torch.cuda.is_available():
            raise AllocationError(
                "CUDA is not available to PyTorch",
                requested_mib=0,
                held_mib=0,
            )
        self._torch: Any = torch
        self._device = device
        self._device_selected = False

    def _allocate_resource(self, size_mib: int) -> object:
        try:
            if not self._device_selected:
                self._torch.cuda.set_device(self._device)
                self._device_selected = True
            return self._torch.empty(
                size_mib * self._MIB,
                dtype=self._torch.uint8,
                device=self._device,
            )
        except RuntimeError as exc:
            raise AllocationError(
                f"CUDA allocation failed for {size_mib} MiB: {exc}",
                requested_mib=size_mib,
                held_mib=self.held_mib,
            ) from exc

    def _release_resources(self, resources: list[object]) -> None:
        self._torch.cuda.synchronize(self._device)
        resources.clear()
        gc.collect()
        self._torch.cuda.empty_cache()

    def _run_maintenance(self, resource: object) -> None:
        tensor: Any = resource
        elements = min(tensor.numel(), 16 * self._MIB)
        tensor[:elements].bitwise_xor_(1)
        self._torch.cuda.synchronize(self._device)


def plan_chunk_sizes(target_mib: int, chunk_mib: int) -> tuple[int, ...]:
    """Split a target hold into full chunks and at most one tail chunk."""

    if target_mib < 0:
        raise ValueError("target_mib cannot be negative")
    if chunk_mib <= 0:
        raise ValueError("chunk_mib must be positive")

    full_chunks, tail_mib = divmod(target_mib, chunk_mib)
    chunks = [chunk_mib] * full_chunks
    if tail_mib:
        chunks.append(tail_mib)
    return tuple(chunks)
