from __future__ import annotations

from pathlib import Path

import pytest

import watchgpu.control as control_module
from watchgpu.config import WatchGPUConfig, load_config, save_config
from watchgpu.control import ApplyStatus, RuntimeConfigController


def test_runtime_config_rejects_stale_revision_without_overwriting(tmp_path: Path) -> None:
    applied: list[int] = []

    def apply(config: WatchGPUConfig) -> ApplyStatus:
        applied.append(config.leave_free_mib)
        return ApplyStatus.APPLIED

    config_path = tmp_path / "config.toml"
    controller = RuntimeConfigController(
        WatchGPUConfig(leave_free="2GiB"),
        apply=apply,
        config_path=config_path,
    )

    first = controller.apply(
        WatchGPUConfig(leave_free="3GiB"),
        expected_revision=0,
        save=True,
    )
    stale = controller.apply(
        WatchGPUConfig(leave_free="4GiB"),
        expected_revision=0,
        save=True,
    )

    assert first.status is ApplyStatus.APPLIED
    assert first.revision == 1
    assert stale.status is ApplyStatus.REJECTED
    assert stale.revision == 1
    assert controller.config.leave_free_mib == 3072
    assert applied == [3072]
    assert load_config(config_path).leave_free_mib == 3072


def test_pending_runtime_config_can_be_marked_applied_after_reconciliation() -> None:
    controller = RuntimeConfigController(
        WatchGPUConfig(leave_free="2GiB"),
        apply=lambda _config: ApplyStatus.PENDING,
    )

    result = controller.apply(
        WatchGPUConfig(leave_free="3GiB"),
        expected_revision=0,
        save=False,
    )
    assert result.status is ApplyStatus.PENDING
    assert controller.runtime_status is ApplyStatus.PENDING

    controller.mark_runtime_status(ApplyStatus.APPLIED)

    assert controller.runtime_status is ApplyStatus.APPLIED


def test_pending_saved_config_is_persisted_only_after_runtime_applies(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    original = WatchGPUConfig(leave_free="2GiB")
    save_config(original, path)
    controller = RuntimeConfigController(
        original,
        apply=lambda _config: ApplyStatus.PENDING,
        config_path=path,
    )

    controller.apply(
        WatchGPUConfig(leave_free="3GiB"),
        expected_revision=0,
        save=True,
    )
    assert load_config(path).leave_free_mib == 2048

    controller.mark_runtime_status(ApplyStatus.APPLIED)
    assert load_config(path).leave_free_mib == 3072


def test_persistence_failure_rolls_runtime_back_without_advancing_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    applied: list[int] = []
    original = WatchGPUConfig(leave_free="2GiB")
    controller = RuntimeConfigController(
        original,
        apply=lambda config: (
            applied.append(config.leave_free_mib) or ApplyStatus.APPLIED
        ),
        config_path=tmp_path / "config.toml",
    )
    monkeypatch.setattr(
        control_module,
        "save_config",
        lambda _config, _path: (_ for _ in ()).throw(OSError("disk full")),
    )

    result = controller.apply(
        WatchGPUConfig(leave_free="3GiB"),
        expected_revision=0,
        save=True,
    )

    assert result.status is ApplyStatus.REJECTED
    assert result.revision == 0
    assert "rolled back" in (result.reason or "")
    assert controller.config.leave_free_mib == 2048
    assert applied == [3072, 2048]
