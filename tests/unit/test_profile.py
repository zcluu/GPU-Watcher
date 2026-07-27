from __future__ import annotations

from pathlib import Path

from watchgpu.profile import (
    ProfileOutcome,
    ProfileStore,
    build_profile_fingerprint,
    recommended_memory_mib,
)


def test_recommendation_adds_safety_margin_and_rounds_up_to_500_mib() -> None:
    assert recommended_memory_mib(10_000) == 11_500
    assert recommended_memory_mib(1_000) == 2_500


def test_profile_fingerprint_is_stable_and_tracks_config_content(tmp_path: Path) -> None:
    config = tmp_path / "train.toml"
    config.write_text("batch = 32\n", encoding="utf-8")
    metadata = {
        "task": "resnet",
        "argv": ["--config", str(config)],
        "world_size": 2,
    }

    first = build_profile_fingerprint(metadata, config_files=(config,))
    reordered = build_profile_fingerprint(
        {"world_size": 2, "argv": ["--config", str(config)], "task": "resnet"},
        config_files=(config,),
    )
    config.write_text("batch = 64\n", encoding="utf-8")
    changed = build_profile_fingerprint(metadata, config_files=(config,))

    assert first == reordered
    assert changed != first


def test_profile_store_persists_failures_but_recommends_only_successes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.jsonl"
    store = ProfileStore(path)

    failed = store.record_failure(
        fingerprint="fp-1",
        task_name="llama-ft",
        world_size=2,
        observed_peak_mib_by_gpu={"GPU-0": 9000, "GPU-1": 9500},
        exit_code=1,
        error="CUDA out of memory",
    )
    assert failed.outcome is ProfileOutcome.FAILED
    assert failed.recommended_memory_per_gpu_mib is None
    assert store.recommend("fp-1") is None

    succeeded = store.record_success(
        fingerprint="fp-1",
        task_name="llama-ft",
        world_size=2,
        observed_peak_mib_by_gpu={"GPU-0": 10_000, "GPU-1": 11_000},
        exit_code=0,
    )

    reloaded = ProfileStore(path)
    assert [record.outcome for record in reloaded.records()] == [
        ProfileOutcome.FAILED,
        ProfileOutcome.SUCCESS,
    ]
    assert succeeded.recommended_memory_per_gpu_mib == 13_000
    assert reloaded.recommend("fp-1") == 13_000
