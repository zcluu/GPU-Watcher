from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GPUSnapshot:
    index: int
    uuid: str
    name: str
    total_mib: int
    free_mib: int
    utilization_percent: int
    temperature_c: int | None = None
    mig_mode: str | None = None


@dataclass(frozen=True, slots=True)
class GPUProcess:
    pid: int
    used_memory_mib: int | None
    name: str | None = None
