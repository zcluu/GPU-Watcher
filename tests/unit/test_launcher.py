from __future__ import annotations

import pytest

import watchgpu.launcher as launcher_module
from watchgpu.launcher import (
    LaunchConfig,
    LauncherConfigurationError,
    _resolve_memory,
    parse_launch_args,
)
from watchgpu.profile import ProfileStore
from watchgpu.sdk import ManagedGPU


def test_watchgpu_run_parses_torchrun_style_resource_arguments() -> None:
    config = parse_launch_args(
        [
            "--task",
            "llama-ft",
            "--nproc-per-node=2",
            "--memory-per-gpu=24GiB",
            "--devices=0,2",
            "train.py",
            "--config",
            "llama.yaml",
        ]
    )

    assert config.task_name == "llama-ft"
    assert config.nproc_per_node == 2
    assert config.memory_per_gpu == 24 * 1024
    assert config.devices == ("0", "2")
    assert config.training_script == "train.py"
    assert config.training_args == ("--config", "llama.yaml")


def test_watchgpu_run_rejects_multi_node_launches() -> None:
    with pytest.raises(LauncherConfigurationError, match="single-node"):
        parse_launch_args(
            [
                "--task=distributed-job",
                "--nnodes=2",
                "--memory-per-gpu=8GiB",
                "train.py",
            ]
        )


def test_watchgpu_run_treats_bare_memory_as_gib() -> None:
    config = parse_launch_args(
        ["--task=test", "--memory-per-gpu=12.5", "train.py"]
    )

    assert config.memory_per_gpu == 12_800


def test_auto_bootstrap_keeps_driver_headroom(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        launcher_module,
        "managed_gpus",
        lambda **_kwargs: (
            ManagedGPU(0, "GPU-0", "A40", 46_068, 2048, 23_448, 0),
        ),
    )
    config = LaunchConfig(
        task_name="bootstrap",
        nproc_per_node=1,
        memory_per_gpu="auto",
        devices=("0",),
        training_script="train.py",
        training_args=(),
    )

    memory_mib = _resolve_memory(
        config,
        fingerprint="new-profile",
        profile_store=ProfileStore(tmp_path / "profiles.jsonl"),
        socket_path=tmp_path / "watchgpu.sock",
    )

    assert memory_mib == 24_000
    assert memory_mib < 25_496
