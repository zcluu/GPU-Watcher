from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Annotated, Literal

import tomli
import tomli_w
from pydantic import (
    AliasChoices,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    model_validator,
)

from watchgpu.units import parse_capacity_mib

CapacityMiB = Annotated[int, BeforeValidator(parse_capacity_mib)]


class GPUConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    selector: str = Field(min_length=1)
    reserve_limit_mib: CapacityMiB | None = Field(
        default=None,
        validation_alias=AliasChoices("reserve_limit_mib", "reserve_limit"),
        serialization_alias="reserve_limit",
    )
    leave_free_mib: CapacityMiB | None = Field(
        default=None,
        validation_alias=AliasChoices("leave_free_mib", "leave_free"),
        serialization_alias="leave_free",
    )


class MaintenanceRestartConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    at: str = Field(default="04:00", pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    jitter_seconds: int = Field(default=1200, ge=0, le=86_400)
    defer_while_leased: bool = True


class WatchGPUConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    poll_interval_seconds: float = Field(default=2.0, gt=0, le=60)
    growth_stability_seconds: float = Field(default=10.0, ge=0, le=3600)
    background_mode: Literal["auto", "systemd-user", "detached", "foreground"] = "auto"
    leave_free_mib: CapacityMiB = Field(
        default=2048,
        validation_alias=AliasChoices("leave_free_mib", "leave_free"),
        serialization_alias="leave_free",
    )
    chunk_mib: int = Field(default=500, ge=1, le=16_384)
    reserve_ratio: float | None = Field(default=None, gt=0, le=1)
    allocation_tolerance_mib: int = Field(default=32, ge=0, le=1024)
    maintenance_compute_enabled: bool = True
    maintenance_duty_cycle_percent: int = Field(default=5, ge=1, le=20)
    compute_pause_above_utilization: int = Field(default=20, ge=1, le=100)
    cpu_budget_percent: int = Field(default=100, ge=1, le=100)
    worker_cpu_threads: int = Field(default=1, ge=1, le=8)
    cpu_affinity_cores: int = Field(default=1, ge=1, le=8)
    maintenance_restart: MaintenanceRestartConfig = Field(
        default_factory=MaintenanceRestartConfig
    )
    gpus: list[GPUConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def selectors_must_be_unique(self) -> WatchGPUConfig:
        selectors = [gpu.selector for gpu in self.gpus]
        if len(selectors) != len(set(selectors)):
            raise ValueError("GPU selectors must be unique")
        return self


def load_config(path: Path) -> WatchGPUConfig:
    if not path.exists():
        return WatchGPUConfig()
    with path.open("rb") as config_file:
        values = tomli.load(config_file)
    return WatchGPUConfig.model_validate(values)


def save_config(config: WatchGPUConfig, path: Path) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            tomli_w.dump(
                config.model_dump(mode="python", by_alias=True, exclude_none=True),
                temporary_file,
            )
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
