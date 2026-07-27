from __future__ import annotations

import gc
import weakref

from watchgpu.allocator import ChunkMemoryAllocator, InMemoryMemoryAllocator


class _Resource:
    pass


class _LifecycleTrackingAllocator(ChunkMemoryAllocator):
    def __init__(self, chunk_mib: int) -> None:
        super().__init__(chunk_mib)
        self.resources_alive_after_release: list[bool] = []

    def _allocate_resource(self, size_mib: int) -> object:
        del size_mib
        return _Resource()

    def _release_resources(self, resources: list[object]) -> None:
        references = [weakref.ref(resource) for resource in resources]
        resources.clear()
        gc.collect()
        self.resources_alive_after_release.extend(
            reference() is not None for reference in references
        )

    def _run_maintenance(self, resource: object) -> None:
        del resource


def test_allocator_reconciles_chunks_and_reports_actual_work() -> None:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)

    grown = allocator.reconcile(2300)
    assert grown.held_mib == 2300
    assert grown.allocated_mib == 2300
    assert grown.released_mib == 0
    assert allocator.chunk_sizes == (500, 500, 500, 500, 300)

    shrunk = allocator.reconcile(1600)
    assert shrunk.held_mib == 1600
    assert shrunk.released_mib == 800
    assert shrunk.allocated_mib == 100
    assert allocator.chunk_sizes == (500, 500, 500, 100)

    released = allocator.release_all()
    assert released.released_mib == 1600
    assert released.held_mib == 0
    assert allocator.chunk_sizes == ()


def test_reconcile_drops_hidden_block_references_before_release_hook() -> None:
    allocator = _LifecycleTrackingAllocator(chunk_mib=500)
    allocator.reconcile(1000)

    result = allocator.reconcile(500)

    assert result.released_mib == 500
    assert allocator.resources_alive_after_release == [False]
