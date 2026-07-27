# Troubleshooting

Start with these read-only commands:

```bash
watchgpu doctor --json
watchgpu status --json
watchgpu logs --lines 100
watchgpu profile list --json
```

Redact hostnames, paths, PIDs, GPU UUIDs, and task names before sharing output publicly.

## `SESSION_BOUND`

The selected background mode may stop when the last login session ends. Use user systemd with lingering enabled by an administrator, or keep the session alive. Root is not required for normal operation, but users cannot enable lingering where policy forbids it.

## `QUEUED`

The daemon cannot atomically satisfy the requested GPU count, a selected GPU already has an active lease, or there is not enough free plus reclaimable memory. Inspect `watchgpu status` and request fewer GPUs or a smaller explicit amount.

## `REJECTED`

Common causes include an unmanaged GPU selector, invalid request, configuration revision conflict, or a driver-visible release that was not confirmed in time. Check the Events tab and logs before retrying.

## `ORPHANED`

The lease heartbeat expired or the daemon recovered while training identity may still exist. WatchGPU conservatively keeps the promised headroom until it can prove the associated process is gone.

## CUDA Works in the Shell but Not in user systemd

The user systemd manager does not run an interactive shell and may not inherit environment-specific CUDA library variables. Confirm that the selected Python can locate its runtime libraries without shell activation. As a diagnostic, try:

```bash
watchgpu start --gpus 0 --leave-free 2 --background-mode detached
```

## `OVER_LIMIT`

Training exceeded its requested memory. WatchGPU reports this but does not terminate training. Increase the explicit request or run a representative `auto` profiling workload.

## Console Does Not Show the Current Task First

Upgrade to the latest source. Current leases (`ACTIVE`, `ORPHANED`, and `QUEUED`) are sorted before terminal lease history. Restart only the console client after an upgrade; the daemon does not need to restart for a rendering-only change.
