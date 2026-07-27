# Responsible Use and Security Boundaries

WatchGPU is intended for cooperative environments where users are authorized to run and reserve GPU resources.

## What WatchGPU Does

- Allocates memory only in its own worker processes.
- Releases only those WatchGPU-owned allocations.
- Verifies release through the NVIDIA driver before approving a lease.
- Identifies managed training through lease and process identity evidence.
- Bounds maintenance compute and CPU use.

## What WatchGPU Does Not Do

- It does not create a hardware-enforced memory quota or exclusive lock.
- It does not prevent unrelated processes from allocating memory.
- It does not preempt, terminate, or signal unrelated GPU processes.
- It does not replace Slurm, Kubernetes, LSF, or administrator policy.
- It does not disguise worker identity or falsify process information.

## Trust Model

The control socket authenticates the Unix peer UID. Processes running under the same UID are therefore inside the same control boundary. Do not deploy WatchGPU where mutually untrusted people share one Unix account.

Runtime paths are created with private permissions and checked for unsafe ownership or symlinks. These protections do not defend against the account owner itself.

## Operational Data

Local state and diagnostics may expose:

- task names and process IDs;
- GPU UUIDs and hardware model;
- absolute Python and XDG paths;
- errors originating from training or CUDA libraries.

Redact this information before attaching `doctor --json`, `status --json`, profiles, or logs to a public issue. Never commit XDG state or config files to the repository.

## Safe Deployment Checklist

1. Confirm that local policy permits user-managed reservation.
2. Prefer the existing scheduler when the host already has one.
3. Start with `--dry-run` and a conservative GPU allowlist.
4. Verify stop/release behavior during a maintenance window.
5. Do not manage MIG-enabled GPUs with the current release.
6. Use distinct Unix accounts for mutually untrusted users.
