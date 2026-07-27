from __future__ import annotations

from pathlib import Path

import pytest

from watchgpu.paths import WatchGPUPaths


def test_ensure_directories_rejects_a_symlinked_runtime_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "attacker-controlled"
    target.mkdir()
    runtime = tmp_path / "runtime"
    runtime.symlink_to(target, target_is_directory=True)
    paths = WatchGPUPaths(
        runtime_dir=runtime,
        config_dir=tmp_path / "config",
        state_dir=tmp_path / "state",
        systemd_user_dir=tmp_path / "systemd" / "user",
    )

    with pytest.raises(RuntimeError, match="unsafe WatchGPU directory"):
        paths.ensure_directories()
