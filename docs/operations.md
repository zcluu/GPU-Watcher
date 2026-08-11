# Operations

## Upgrade

The default installer performs a normal, non-editable package installation. Pulling the repository alone does not update the installed commands.

Use a controlled stop so WatchGPU releases only its own reservations, then reinstall and start with the desired policy:

```bash
watchgpu stop --release
git pull --ff-only
./install-watchgpu
watchgpu start -g 0,1 -f 2
```

Stopping WatchGPU does not signal managed training or external GPU processes. During an alpha upgrade, a stop/start is preferred over an in-place daemon restart because protocol and persisted-state compatibility may change.

## Console Upgrade

The console is a client process. Rendering-only updates require closing and reopening `watchgpu console`; the daemon does not need to restart.

## Safe Uninstall

First release WatchGPU-owned allocations:

```bash
watchgpu stop --release
```

Disable a user-systemd unit if one was installed:

```bash
systemctl --user disable --now watchgpu.service 2>/dev/null || true
rm -f ~/.config/systemd/user/watchgpu.service
systemctl --user daemon-reload
```

Remove the Python distribution from the same environment used for installation:

```bash
python -m pip uninstall gpu-watcher
```

Configuration and history are retained intentionally under XDG config/state directories. Review them before optional manual removal:

```text
~/.config/watchgpu/
~/.local/state/watchgpu/
```

## Restart Schedule

```bash
watchgpu restart --now
watchgpu restart schedule set -a 04:00 -j 20m --defer
watchgpu restart schedule show
watchgpu restart schedule disable
```

Immediate and scheduled restarts quiesce new leases, persist state, and release WatchGPU workers. With `defer_while_leased`, scheduled maintenance remains pending while a managed lease is active.
