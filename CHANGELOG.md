# Changelog

All notable changes to WatchGPU will be documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project intends to use semantic versioning after the alpha interface stabilizes.

## [Unreleased]

## [0.1.0] - 2026-07-27

### Added

- Per-GPU elastic PyTorch memory reservation with NVML observation.
- Atomic single- and multi-GPU lease requests.
- `torchrun`-style launcher and automatic peak-memory profiling.
- Textual console, runtime policy editing, process classification, and event journal.
- User-systemd and detached background modes without root privileges.
- Safe stop/restart behavior and conservative orphaned-lease recovery.

### Fixed

- Release tensors before flushing the PyTorch allocator cache so driver-visible memory is restored in-process.
- Keep bootstrap profiling headroom instead of requesting all instantaneously available memory.
- Treat unitless interactive capacities as GiB while preserving normalized MiB configuration.
- Sort current console tasks before terminal lease history.

[Unreleased]: https://github.com/zcluu/GPU-Watcher/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/zcluu/GPU-Watcher/tree/v0.1.0
