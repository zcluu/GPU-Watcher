from __future__ import annotations

from collections.abc import Iterable

from watchgpu.models import GPUProcess, GPUSnapshot
from watchgpu.supervisor import PollingReleaseVerifier


class SequenceObserver:
    def __init__(self, free_values: Iterable[int]) -> None:
        self._free_values = iter(free_values)
        self._last = 0

    def snapshots(self) -> tuple[GPUSnapshot, ...]:
        self._last = next(self._free_values, self._last)
        return (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, self._last, 0),)

    def processes(self, gpu_uuid: str) -> tuple[GPUProcess, ...]:
        return ()


def test_release_verifier_waits_for_nvml_free_memory() -> None:
    now = 0.0

    def clock() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        now += seconds

    verifier = PollingReleaseVerifier(
        SequenceObserver((1000, 2000, 2500)),
        poll_interval_seconds=0.1,
        clock=clock,
        sleep=sleep,
    )

    assert verifier.verify("GPU-0", expected_free_mib=2500, timeout=1.0)
    assert now == 0.2
