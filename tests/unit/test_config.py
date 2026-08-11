from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from watchgpu.config import WatchGPUConfig, load_config, save_config


def test_user_config_is_normalized_and_defaults_are_applied() -> None:
    config = WatchGPUConfig.model_validate(
        {
            "leave_free": "3GiB",
            "gpus": [
                {"selector": "0"},
                {"selector": "GPU-example", "reserve_limit": "20GiB"},
            ],
        }
    )

    assert config.leave_free_mib == 3072
    assert config.chunk_mib == 500
    assert config.maintenance_compute_enabled is True
    assert config.maintenance_duty_cycle_percent == 5
    assert config.cpu_budget_percent == 100
    assert config.maintenance_cpu_target_percent == 0
    assert config.gpus[1].reserve_limit_mib == 20 * 1024


@pytest.mark.parametrize(
    "updates",
    [
        {"maintenance_duty_cycle_percent": 21},
        {"cpu_budget_percent": 0},
        {"maintenance_cpu_target_percent": -1},
        {"maintenance_cpu_target_percent": 101},
        {"poll_interval_seconds": 0},
        {"gpus": [{"selector": "0"}, {"selector": "0"}]},
    ],
)
def test_unsafe_or_ambiguous_config_is_rejected(updates: dict[str, object]) -> None:
    values: dict[str, object] = {"leave_free": "2GiB", "gpus": [{"selector": "0"}]}
    values.update(updates)

    with pytest.raises(ValidationError):
        WatchGPUConfig.model_validate(values)


def test_config_can_be_saved_and_loaded_atomically(tmp_path: Path) -> None:
    path = tmp_path / "watchgpu" / "config.toml"
    expected = WatchGPUConfig.model_validate(
        {
            "leave_free": "2GiB",
            "maintenance_restart": {"enabled": True, "at": "05:30"},
            "maintenance_cpu_target_percent": 50,
            "gpus": [{"selector": "1", "reserve_limit": "30GiB"}],
        }
    )

    save_config(expected, path)

    assert load_config(path) == expected
    assert load_config(path).maintenance_cpu_target_percent == 50
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(path.parent.glob("*.tmp")) == []
