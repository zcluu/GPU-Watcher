from __future__ import annotations

import pytest

from watchgpu.allocator import plan_chunk_sizes


@pytest.mark.parametrize(
    ("target_mib", "chunk_mib", "expected"),
    [
        (0, 500, ()),
        (300, 500, (300,)),
        (500, 500, (500,)),
        (2300, 500, (500, 500, 500, 500, 300)),
        (1600, 500, (500, 500, 500, 100)),
    ],
)
def test_target_memory_is_split_into_bounded_chunks(
    target_mib: int, chunk_mib: int, expected: tuple[int, ...]
) -> None:
    chunks = plan_chunk_sizes(target_mib, chunk_mib)

    assert chunks == expected
    assert sum(chunks) == target_mib
    assert all(0 < chunk <= chunk_mib for chunk in chunks)


@pytest.mark.parametrize(("target_mib", "chunk_mib"), [(-1, 500), (1, 0)])
def test_invalid_chunk_plans_are_rejected(target_mib: int, chunk_mib: int) -> None:
    with pytest.raises(ValueError):
        plan_chunk_sizes(target_mib, chunk_mib)
