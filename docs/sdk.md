# Python SDK

The SDK coordinates a Python process with the local WatchGPU daemon. It must acquire a lease before PyTorch initializes CUDA.

## Single GPU

```python
from watchgpu import MemoryRequest, acquire

request = MemoryRequest(
    task_name="experiment-42",
    gpu="0",
    mib=12_000,
    ttl_seconds=600,
)

with acquire(request) as lease:
    import torch

    model = build_model().to(lease.device)
    train(model)
```

The context manager renews the lease in the background and releases it on normal exit or an exception.

## GPU Group

```python
from watchgpu import GroupMemoryRequest, acquire

request = GroupMemoryRequest(
    task_name="distributed-training",
    count=2,
    mib_per_gpu=24_000,
    devices=("0", "1"),
)

with acquire(request) as lease:
    print(lease.gpu_uuids)
```

Group requests are atomic. The daemon queues the request until every GPU can satisfy it; it never returns a partial group.

## Errors

- `CUDAAlreadyInitializedError`: CUDA was initialized before the request.
- `LeaseRejectedError`: the daemon rejected activation.
- `LeaseTimeoutError`: the request did not become active before the client timeout.
- `LeaseConnectionError`: the local daemon/socket is unavailable.
- `LeaseReleaseError`: the client could not confirm lease release.

These errors are available from `watchgpu.sdk`; the most common request and connection types are re-exported by `watchgpu`.
