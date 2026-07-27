"""WatchGPU public package."""

from importlib.metadata import PackageNotFoundError, version

from watchgpu.sdk import (
    GroupMemoryRequest,
    LeaseConnectionError,
    LeaseGrant,
    LeaseRejectedError,
    LeaseTimeoutError,
    MemoryRequest,
    acquire,
    managed_gpus,
)
from watchgpu.units import CapacityParseError, parse_capacity_mib

try:
    __version__ = version("gpu-watcher")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = [
    "CapacityParseError",
    "GroupMemoryRequest",
    "LeaseConnectionError",
    "LeaseGrant",
    "LeaseRejectedError",
    "LeaseTimeoutError",
    "MemoryRequest",
    "acquire",
    "managed_gpus",
    "parse_capacity_mib",
    "__version__",
]
