# Configuration

WatchGPU stores host-local configuration at:

```text
$XDG_CONFIG_HOME/watchgpu/config.toml
```

When `XDG_CONFIG_HOME` is unset, `~/.config/watchgpu/config.toml` is used.

## Capacity Inputs

Interactive CLI and console capacity values default to GiB when the unit is omitted:

```text
2        -> 2048 MiB
2.5      -> 2560 MiB
2GiB     -> 2048 MiB
2500MiB  -> 2500 MiB
```

Persisted configuration is normalized to integer MiB values for compatibility.

## Common Settings

| Setting | Default | Meaning |
|---|---:|---|
| `leave_free` | `2048` MiB | Driver-visible memory kept available for other work |
| `chunk_mib` | `500` | Main reservation allocation size |
| `poll_interval_seconds` | `2` | NVML observation interval |
| `growth_stability_seconds` | `10` | Stable target time required before growing |
| `allocation_tolerance_mib` | `32` | Reconciliation deadband |
| `maintenance_compute_enabled` | `true` | Enable bounded maintenance kernel |
| `maintenance_duty_cycle_percent` | `5` | Maximum maintenance duty cycle |
| `compute_pause_above_utilization` | `20` | Pause maintenance above external utilization |
| `cpu_budget_percent` | `100` | Whole WatchGPU process-tree CPU budget |

Use `watchgpu config show` for the complete resolved configuration.

## Runtime Changes

```bash
# Apply until the daemon exits
watchgpu config set --leave-free 3 --runtime-only

# Apply and persist
watchgpu config set --gpu GPU-UUID --leave-free 2 --reserve-limit 30

# Change the managed set
watchgpu config set --gpus GPU-UUID-1,GPU-UUID-2
```

Policy updates use an expected revision so concurrent consoles cannot silently overwrite each other. The result is `APPLIED`, `PENDING`, or `REJECTED`.

## Background Mode

- `auto`: prefer user systemd, otherwise detached.
- `systemd-user`: require a working user systemd manager.
- `detached`: start a new session and track PID identity.
- `foreground`: useful for diagnosis.

Some CUDA environments rely on shell-provided variables that user systemd does not inherit. If `doctor` succeeds interactively but a systemd worker cannot initialize CUDA, use an environment with self-contained runtime libraries or select `detached` mode while investigating.
