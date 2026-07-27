from __future__ import annotations

import argparse
import math
import os
import sys
import threading
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NoReturn, Protocol

import psutil  # type: ignore[import-untyped]

from watchgpu.observer import NVMLGPUObserver
from watchgpu.paths import WatchGPUPaths
from watchgpu.profile import ProfileStore, build_profile_fingerprint
from watchgpu.sdk import GroupMemoryRequest, ManagedGPU, acquire, managed_gpus
from watchgpu.units import parse_user_capacity_mib


class LauncherConfigurationError(ValueError):
    pass


MemorySetting = int | Literal["auto"]
_AUTO_BOOTSTRAP_MIN_HEADROOM_MIB = 1024
_AUTO_BOOTSTRAP_HEADROOM_RATIO = 0.05
_AUTO_BOOTSTRAP_ROUND_MIB = 500


class PeakMonitor(Protocol):
    def start(self) -> None: ...

    def stop(self) -> dict[str, int]: ...


class NVMLProcessTreePeakMonitor:
    """Sample per-GPU NVML usage belonging to the launcher process tree."""

    def __init__(
        self,
        gpu_uuids: Sequence[str],
        *,
        root_pid: int | None = None,
        sample_interval_seconds: float = 0.2,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError("sample_interval_seconds must be positive")
        self._gpu_uuids = tuple(gpu_uuids)
        self._root_pid = os.getpid() if root_pid is None else root_pid
        self._sample_interval_seconds = sample_interval_seconds
        self._observer: NVMLGPUObserver | None = None
        self._peaks = {gpu_uuid: 0 for gpu_uuid in self._gpu_uuids}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("peak monitor is already started")
        try:
            self._observer = NVMLGPUObserver()
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            return
        self._thread = threading.Thread(
            target=self._sample_loop,
            name="watchgpu-profile-monitor",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> dict[str, int]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._sample_interval_seconds * 4))
            self._thread = None
        observer = self._observer
        if observer is not None:
            with suppress(Exception):
                observer.close()
            self._observer = None
        return dict(self._peaks)

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._sample_once()
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
            self._stop_event.wait(self._sample_interval_seconds)

    def _sample_once(self) -> None:
        observer = self._observer
        if observer is None:
            return
        process_ids = _process_tree_pids(self._root_pid)
        for gpu_uuid in self._gpu_uuids:
            used_mib = sum(
                process.used_memory_mib or 0
                for process in observer.processes(gpu_uuid)
                if process.pid in process_ids
            )
            self._peaks[gpu_uuid] = max(self._peaks[gpu_uuid], used_mib)


@dataclass(frozen=True, slots=True)
class LaunchConfig:
    task_name: str
    nproc_per_node: int
    memory_per_gpu: MemorySetting
    devices: tuple[str, ...] | None
    training_script: str
    training_args: tuple[str, ...]
    nnodes: int = 1
    ttl_seconds: float = 600.0
    lease_timeout_seconds: float = 3600.0


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise LauncherConfigurationError(message)


def parse_launch_args(argv: Sequence[str]) -> LaunchConfig:
    parser = _ArgumentParser(prog="watchgpu-run", add_help=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--nproc-per-node", type=int, default=1)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--memory-per-gpu", default="auto")
    parser.add_argument("--devices")
    parser.add_argument("--ttl-seconds", type=float, default=600.0)
    parser.add_argument("--lease-timeout-seconds", type=float, default=3600.0)
    parser.add_argument("training_script")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    values = parser.parse_args(list(argv))

    if values.nnodes != 1:
        raise LauncherConfigurationError("WatchGPU currently supports single-node launches only")
    if values.nproc_per_node <= 0:
        raise LauncherConfigurationError("--nproc-per-node must be positive")
    if values.ttl_seconds <= 0 or values.lease_timeout_seconds <= 0:
        raise LauncherConfigurationError("lease TTL and timeout must be positive")
    devices = _parse_devices(values.devices)
    if devices is not None and len(devices) < values.nproc_per_node:
        raise LauncherConfigurationError(
            "--devices must contain at least --nproc-per-node selectors"
        )
    memory = _parse_memory_setting(values.memory_per_gpu)
    return LaunchConfig(
        task_name=str(values.task),
        nproc_per_node=int(values.nproc_per_node),
        memory_per_gpu=memory,
        devices=devices,
        training_script=str(values.training_script),
        training_args=tuple(str(item) for item in values.training_args),
        nnodes=int(values.nnodes),
        ttl_seconds=float(values.ttl_seconds),
        lease_timeout_seconds=float(values.lease_timeout_seconds),
    )


def launch(
    config: LaunchConfig,
    *,
    socket_path: Path,
    profile_store: ProfileStore,
    runner: Callable[[Sequence[str]], int] | None = None,
    monitor_factory: Callable[[tuple[str, ...]], PeakMonitor] | None = None,
) -> int:
    identity, config_files = _profile_identity(config, socket_path=socket_path)
    fingerprint = build_profile_fingerprint(
        identity,
        config_files=config_files,
    )
    memory_mib = _resolve_memory(
        config,
        fingerprint=fingerprint,
        profile_store=profile_store,
        socket_path=socket_path,
    )
    request = GroupMemoryRequest(
        task_name=config.task_name,
        count=config.nproc_per_node,
        mib_per_gpu=memory_mib,
        devices=config.devices,
        ttl_seconds=config.ttl_seconds,
    )
    torchrun = runner or _run_torch_distributed
    make_monitor = monitor_factory or (
        lambda gpu_uuids: NVMLProcessTreePeakMonitor(gpu_uuids)
    )
    with acquire(
        request,
        socket_path=socket_path,
        timeout_seconds=config.lease_timeout_seconds,
    ) as grant:
        monitor = make_monitor(grant.gpu_uuids)
        monitor.start()
        arguments = (
            "--nnodes=1",
            f"--nproc-per-node={config.nproc_per_node}",
            config.training_script,
            *config.training_args,
        )
        metadata = {
            "requested_memory_per_gpu_mib": memory_mib,
            "memory_setting": str(config.memory_per_gpu),
            "gpu_uuids": list(grant.gpu_uuids),
        }
        try:
            exit_code = torchrun(arguments)
        except BaseException as exc:
            peaks = monitor.stop()
            with suppress(Exception):
                profile_store.record_failure(
                    fingerprint=fingerprint,
                    task_name=config.task_name,
                    world_size=config.nproc_per_node,
                    observed_peak_mib_by_gpu=peaks,
                    exit_code=None,
                    error=f"{type(exc).__name__}: {exc}",
                    metadata=metadata,
                )
            raise
        peaks = monitor.stop()
        observed_max = max(peaks.values(), default=0)
        if observed_max > memory_mib:
            print(
                "OVER_LIMIT: observed training peak "
                f"{observed_max} MiB exceeded the requested {memory_mib} MiB per GPU",
                file=sys.stderr,
            )
        if exit_code == 0 and any(peaks.values()):
            profile_store.record_success(
                fingerprint=fingerprint,
                task_name=config.task_name,
                world_size=config.nproc_per_node,
                observed_peak_mib_by_gpu=peaks,
                exit_code=exit_code,
                metadata=metadata,
            )
        else:
            profile_store.record_failure(
                fingerprint=fingerprint,
                task_name=config.task_name,
                world_size=config.nproc_per_node,
                observed_peak_mib_by_gpu=peaks,
                exit_code=exit_code,
                error=(
                    f"training exited with code {exit_code}"
                    if exit_code != 0
                    else "no GPU memory was observed for the training process tree"
                ),
                metadata=metadata,
            )
        return exit_code


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_launch_args(sys.argv[1:] if argv is None else argv)
    paths = WatchGPUPaths.discover()
    return launch(
        config,
        socket_path=paths.socket_path,
        profile_store=ProfileStore(paths.state_dir / "profiles.jsonl"),
    )


def _resolve_memory(
    config: LaunchConfig,
    *,
    fingerprint: str,
    profile_store: ProfileStore,
    socket_path: Path,
) -> int:
    if isinstance(config.memory_per_gpu, int):
        return config.memory_per_gpu
    recommendation = profile_store.recommend(fingerprint)
    if recommendation is not None:
        return recommendation
    gpus = managed_gpus(socket_path=socket_path)
    selected = _select_bootstrap_gpus(gpus, config.devices, config.nproc_per_node)
    available = min(gpu.free_mib + gpu.reserved_mib for gpu in selected)
    headroom = max(
        _AUTO_BOOTSTRAP_MIN_HEADROOM_MIB,
        math.ceil(available * _AUTO_BOOTSTRAP_HEADROOM_RATIO),
    )
    bootstrap_mib = ((available - headroom) // _AUTO_BOOTSTRAP_ROUND_MIB) * (
        _AUTO_BOOTSTRAP_ROUND_MIB
    )
    if bootstrap_mib <= 0:
        raise LauncherConfigurationError(
            "not enough memory is available for safe bootstrap profiling; "
            "use an explicit --memory-per-gpu value"
        )
    return bootstrap_mib


def _profile_identity(
    config: LaunchConfig, *, socket_path: Path
) -> tuple[dict[str, object], tuple[Path, ...]]:
    runtime: dict[str, object] = {}
    try:
        import torch

        runtime = {
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
        }
    except ImportError:
        runtime = {"torch_version": None, "cuda_version": None}
    try:
        gpus = managed_gpus(socket_path=socket_path)
        selected = _select_bootstrap_gpus(
            gpus, config.devices, config.nproc_per_node
        )
        gpu_types = [(gpu.name, gpu.total_mib) for gpu in selected]
    except Exception:
        gpu_types = []
    files: list[Path] = []
    for argument in (config.training_script, *config.training_args):
        path = Path(argument).expanduser()
        if path.is_file():
            files.append(path)
    identity: dict[str, object] = {
        "task": config.task_name,
        "argv": [config.training_script, *config.training_args],
        "world_size": config.nproc_per_node,
        "runtime": runtime,
        "gpu_types": gpu_types,
    }
    return identity, tuple(files)


def _select_bootstrap_gpus(
    gpus: tuple[ManagedGPU, ...],
    selectors: tuple[str, ...] | None,
    count: int,
) -> tuple[ManagedGPU, ...]:
    by_selector = {str(gpu.index): gpu for gpu in gpus}
    by_selector.update({gpu.uuid: gpu for gpu in gpus})
    if selectors is not None:
        try:
            candidates = tuple(by_selector[selector] for selector in selectors)
        except KeyError as exc:
            raise LauncherConfigurationError(
                f"GPU selector is not managed: {exc.args[0]}"
            ) from exc
    else:
        # Supervisor assigns unspecified candidates in managed order. Mirroring
        # that order keeps auto-profile fingerprints tied to the GPUs that will
        # actually be granted, including on heterogeneous hosts.
        candidates = gpus
    if len(candidates) < count:
        raise LauncherConfigurationError(
            f"requested {count} GPUs but only {len(candidates)} are managed"
        )
    return candidates[:count]


def _parse_memory_setting(value: object) -> MemorySetting:
    if not isinstance(value, str):
        raise LauncherConfigurationError("--memory-per-gpu must be a capacity or auto")
    if value.lower() == "auto":
        return "auto"
    try:
        memory_mib = parse_user_capacity_mib(value)
    except (TypeError, ValueError) as exc:
        raise LauncherConfigurationError(str(exc)) from exc
    if memory_mib <= 0:
        raise LauncherConfigurationError("--memory-per-gpu must be positive")
    return memory_mib


def _parse_devices(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise LauncherConfigurationError("--devices must be comma separated")
    devices = tuple(item.strip() for item in value.split(",") if item.strip())
    if not devices:
        raise LauncherConfigurationError("--devices cannot be empty")
    if len(set(devices)) != len(devices):
        raise LauncherConfigurationError("--devices cannot contain duplicates")
    return devices


def _run_torch_distributed(arguments: Sequence[str]) -> int:
    previous_argv = sys.argv
    try:
        sys.argv = ["torchrun", *arguments]
        from torch.distributed.run import main as torchrun_main

        torchrun_main(list(arguments))
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1
    finally:
        sys.argv = previous_argv
    return 0


def _process_tree_pids(root_pid: int) -> set[int]:
    process_ids = {root_pid}
    try:
        root = psutil.Process(root_pid)
        process_ids.update(child.pid for child in root.children(recursive=True))
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return process_ids


if __name__ == "__main__":
    raise SystemExit(main())
