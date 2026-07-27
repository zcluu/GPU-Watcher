# Architecture

WatchGPU is a per-host, per-user service. It does not coordinate state across servers and does not assume a shared home directory.

## Components

- **Observer** reads GPU memory, utilization, temperature, process, UUID, and MIG-mode data through NVML.
- **Supervisor** owns reservation policy, lease state, process classification, and runtime events.
- **Workers** are one-process-per-GPU PyTorch allocators. They hold memory in bounded chunks and perform optional low-duty maintenance work.
- **Unix socket API** carries same-UID control and lease requests. The socket and state files use host-local XDG paths.
- **`watchgpu-run`** requests an atomic GPU group lease, launches `torch.distributed.run`, renews the lease, and records observed peak memory.
- **Console and CLI** are clients of the same versioned control protocol.

## Reservation Policy

For each managed GPU, the target reservation is derived from driver-visible free memory, the worker's current hold, `leave_free`, and optional reserve limits. Growth waits for a stability window; shrink and lease release happen immediately.

Allocations use a configurable main chunk size plus an optional tail chunk. The worker destroys released tensors before flushing the PyTorch caching allocator, then the supervisor waits for NVML to confirm the corresponding increase in free memory.

## Lease Lifecycle

```text
QUEUED -> ACTIVE -> RELEASED
             |
             +-> ORPHANED -> RELEASED

QUEUED/activation -> REJECTED or CANCELLED
```

Multi-GPU requests are all-or-nothing. An active lease records the launcher PID, process start time, process group/session, GPU UUIDs, requested memory, and expiry. This identity prevents PID reuse from attaching a lease to an unrelated process.

After daemon recovery, active leases are restored conservatively as `ORPHANED`. WatchGPU does not reclaim their promised headroom until it can establish that the associated training process is gone.

## Automatic Memory Profiling

Without a prior successful profile, `auto` uses the minimum available capacity across selected GPUs after retaining `max(1 GiB, 5%)` and rounding down to 500 MiB. It records the launcher's process-tree NVML peak. A successful recommendation is:

```text
peak + max(15%, 1 GiB), rounded up to 500 MiB
```

The profile fingerprint includes the task, world size, command arguments, referenced file contents, runtime versions, and GPU model/capacity. Failed, interrupted, OOM, and zero-observation runs do not produce a recommendation.

## Persistence

Configuration, logs, profiles, restart state, and leases live under XDG config/state directories. The runtime socket and PID identity live under `XDG_RUNTIME_DIR`, with a private `/tmp` fallback when it is unavailable. Writes that represent control state use atomic replacement.

## Platform Boundary

The implementation depends on Linux `/proc`, `SO_PEERCRED`, Unix sockets, `fcntl`, process groups/sessions, affinity APIs, NVIDIA NVML, and CUDA-enabled PyTorch. Windows and macOS are not supported.

MIG mode is observed for diagnostics, but allocation by MIG instance UUID is not implemented. Selection of a MIG-enabled physical GPU fails closed.
