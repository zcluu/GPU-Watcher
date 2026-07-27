#!/usr/bin/env python3
"""Render the README console image with synthetic, non-host data."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from watchgpu.console import WatchGPUConsole


class DemoClient:
    async def call(
        self,
        method: str,
        params: Mapping[str, Any] | None = None,
        *,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        del params, request_id
        if method != "status.get":
            raise ValueError(f"unsupported demo method: {method}")
        return {
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-demo-a",
                    "name": "NVIDIA GPU",
                    "total_mib": 48_000,
                    "free_mib": 3072,
                    "reserved_mib": 24_000,
                    "leased_mib": 12_000,
                    "utilization_percent": 18,
                    "temperature_c": 54,
                    "mig_mode": "DISABLED",
                    "worker_state": "HOLDING",
                },
                {
                    "index": 1,
                    "uuid": "GPU-demo-b",
                    "name": "NVIDIA GPU",
                    "total_mib": 48_000,
                    "free_mib": 4096,
                    "reserved_mib": 31_000,
                    "leased_mib": 0,
                    "utilization_percent": 7,
                    "temperature_c": 49,
                    "mig_mode": "DISABLED",
                    "worker_state": "HOLDING",
                },
            ],
            "leases": [
                {
                    "lease_id": "lease-demo",
                    "task_name": "training-demo",
                    "state": "ACTIVE",
                    "client_pid": 12001,
                    "gpu_uuids": ["GPU-demo-a"],
                    "memory_per_gpu_mib": 12_000,
                    "released_by_gpu_mib": {"GPU-demo-a": 12_000},
                    "created_at": "now",
                    "expires_at": "renewing",
                }
            ],
            "processes": [],
            "profiles": [],
            "events": [],
            "policy": {
                "revision": 3,
                "status": "APPLIED",
                "config": {"leave_free_mib": 2048},
            },
        }

    async def aclose(self) -> None:
        return None


async def render(destination: Path) -> None:
    app = WatchGPUConsole(DemoClient(), poll_interval_seconds=60)
    async with app.run_test(size=(120, 38)) as pilot:
        await pilot.pause(0.2)
        saved = Path(app.save_screenshot(filename=destination.name, path=str(destination.parent)))
    svg = saved.read_text(encoding="utf-8")
    svg = re.sub(r"\s*@font-face \{.*?\}\s*", "\n", svg, flags=re.DOTALL)
    svg = svg.replace("font-family: Fira Code, monospace;", "font-family: monospace;")
    saved.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    asyncio.run(render(Path("docs/assets/console.svg").resolve()))
