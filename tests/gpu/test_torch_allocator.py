from __future__ import annotations

import os
import subprocess
import sys
import time

import pynvml
import pytest


@pytest.mark.skipif(
    os.environ.get("WATCHGPU_RUN_ALLOCATION_TESTS") != "1",
    reason="set WATCHGPU_RUN_ALLOCATION_TESTS=1 for the explicit CUDA allocation smoke test",
)
def test_torch_allocator_releases_memory_when_its_process_exits() -> None:
    code = """
import os
from watchgpu.allocator import TorchMemoryAllocator
allocator = TorchMemoryAllocator(chunk_mib=16, device='cuda:0')
result = allocator.reconcile(16)
assert result.held_mib == 16
print(f'READY {os.getpid()}', flush=True)
input()
result = allocator.release_all()
assert result.held_mib == 0
"""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    free_before = int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
    process = subprocess.Popen(
        [sys.executable, "-c", code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        ready = process.stdout.readline().strip().split()
        assert ready == ["READY", str(process.pid)]
        running_pids = {
            int(item.pid) for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        }
        assert process.pid in running_pids
        assert process.stdin is not None
        process.stdin.write("\n")
        process.stdin.flush()
        _stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            running_pids = {
                int(item.pid)
                for item in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            }
            if process.pid not in running_pids:
                break
            time.sleep(0.05)
        assert process.pid not in running_pids
        free_after = int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
        # External jobs may move concurrently; tolerate 64 MiB of noise while
        # still detecting a leaked allocator/context of hundreds of MiB.
        assert free_after >= free_before - 64 * 1024 * 1024
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        pynvml.nvmlShutdown()
