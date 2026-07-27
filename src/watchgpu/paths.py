from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class WatchGPUPaths:
    """Host-local user paths derived at runtime instead of copied between servers."""

    runtime_dir: Path
    config_dir: Path
    state_dir: Path
    systemd_user_dir: Path

    @classmethod
    def discover(
        cls,
        *,
        environ: Mapping[str, str] = os.environ,
        home: Path | None = None,
        uid: int | None = None,
    ) -> WatchGPUPaths:
        resolved_home = (home or Path.home()).expanduser()
        resolved_uid = os.getuid() if uid is None else uid
        config_home = _xdg_home(environ, "XDG_CONFIG_HOME", resolved_home / ".config")
        state_home = _xdg_home(
            environ,
            "XDG_STATE_HOME",
            resolved_home / ".local" / "state",
        )
        explicit_runtime_value = environ.get("WATCHGPU_RUNTIME_DIR", "").strip()
        explicit_runtime = (
            Path(explicit_runtime_value).expanduser() if explicit_runtime_value else None
        )
        runtime_value = environ.get("XDG_RUNTIME_DIR", "").strip()
        runtime_home = Path(runtime_value).expanduser() if runtime_value else None
        if explicit_runtime is not None and explicit_runtime.is_absolute():
            runtime_dir = explicit_runtime
        elif runtime_home is not None and runtime_home.is_absolute():
            runtime_dir = runtime_home / "watchgpu"
        else:
            standard_runtime = Path("/run/user") / str(resolved_uid)
            runtime_dir = (
                standard_runtime / "watchgpu"
                if standard_runtime.is_dir()
                else Path(tempfile.gettempdir()) / f"watchgpu-runtime-{resolved_uid}"
            )
        return cls(
            runtime_dir=runtime_dir,
            config_dir=config_home / "watchgpu",
            state_dir=state_home / "watchgpu",
            systemd_user_dir=config_home / "systemd" / "user",
        )

    @property
    def socket_path(self) -> Path:
        return self.runtime_dir / "watchgpu.sock"

    @property
    def pid_path(self) -> Path:
        return self.runtime_dir / "supervisor.pid"

    @property
    def config_path(self) -> Path:
        return self.config_dir / "config.toml"

    @property
    def log_path(self) -> Path:
        return self.state_dir / "watchgpu.log"

    @property
    def leases_path(self) -> Path:
        return self.state_dir / "leases.json"

    @property
    def restart_state_path(self) -> Path:
        return self.state_dir / "restart.json"

    @property
    def shutdown_result_path(self) -> Path:
        return self.state_dir / "shutdown.json"

    @property
    def systemd_unit_path(self) -> Path:
        return self.systemd_user_dir / "watchgpu.service"

    def ensure_directories(self) -> None:
        for directory in (
            self.runtime_dir,
            self.config_dir,
            self.state_dir,
            self.systemd_user_dir,
        ):
            try:
                metadata = directory.lstat()
            except FileNotFoundError:
                with suppress(FileExistsError):
                    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
                metadata = directory.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(f"unsafe WatchGPU directory path: {directory}")
            if metadata.st_uid != os.getuid():
                raise RuntimeError(
                    f"WatchGPU directory is not owned by the current user: {directory}"
                )
            directory.chmod(0o700)


def _xdg_home(environ: Mapping[str, str], name: str, default: Path) -> Path:
    value = environ.get(name, "").strip()
    if not value:
        return default
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else default
