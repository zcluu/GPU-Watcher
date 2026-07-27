"""Short CUDA workload for validating ``watchgpu-run`` end to end."""

from __future__ import annotations

import argparse
import os
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hold-mib", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    torch.set_num_threads(1)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Keep a predictable amount of memory live while a small matrix multiply
    # gives nvidia-smi/nvitop a real, bounded compute signal.
    held = torch.empty(args.hold_mib * 1024 * 1024, dtype=torch.uint8, device=device)
    left = torch.randn(1024, 1024, device=device)
    right = torch.randn(1024, 1024, device=device)
    for _ in range(args.steps):
        output = left @ right
        held[0].bitwise_xor_(1)
        torch.cuda.synchronize(device)
        time.sleep(0.05)

    print(
        f"watchgpu test completed on {device}: "
        f"held={args.hold_mib} MiB, checksum={output[0, 0].item():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
