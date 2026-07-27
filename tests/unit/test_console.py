from __future__ import annotations

from watchgpu.console import (
    ConsoleAction,
    pause_action,
    policy_apply_action,
    release_action,
    render_section,
    restart_action,
    resume_action,
    stop_action,
)


def test_gpu_snapshot_renders_as_narrow_vertical_blocks() -> None:
    snapshot = {
        "gpus": [
            {
                "index": 0,
                "uuid": "GPU-demo-0",
                "name": "NVIDIA A40",
                "total_mib": 46_068,
                "free_mib": 2048,
                "reserved_mib": 30_000,
                "leased_mib": 8192,
                "utilization_percent": 12,
                "worker_state": "HOLDING",
            }
        ]
    }

    rendered = render_section("gpus", snapshot, width=32)

    assert "GPU 0 [HOLDING]" in rendered
    assert "NVIDIA A40" in rendered
    assert "Reserved: 29.3 GiB" in rendered
    assert "Leased: 8.0 GiB" in rendered
    assert "Utilization: 12%" in rendered
    assert all(len(line) <= 32 for line in rendered.splitlines())


def test_control_actions_use_the_versioned_protocol_contract() -> None:
    assert pause_action("GPU-0") == ConsoleAction(
        method="worker.pause", params={"gpu_uuid": "GPU-0"}
    )
    assert resume_action() == ConsoleAction(method="worker.resume", params={})
    assert release_action("GPU-1", memory_mib=1536) == ConsoleAction(
        method="worker.release",
        params={"gpu_uuid": "GPU-1", "memory_mib": 1536},
    )
    assert policy_apply_action(
        {"leave_free": "3GiB", "gpus": [{"selector": "0"}]},
        expected_revision=17,
        save=True,
    ) == ConsoleAction(
        method="policy.apply",
        params={
            "config": {"leave_free": "3GiB", "gpus": [{"selector": "0"}]},
            "expected_revision": 17,
            "save": True,
        },
    )
    assert restart_action() == ConsoleAction(method="daemon.restart", params={})
    assert stop_action() == ConsoleAction(
        method="daemon.stop", params={"release": True}
    )


def test_all_status_sections_render_server_reported_facts_without_wide_rows() -> None:
    snapshot = {
        "leases": [
            {
                "lease_id": "lease-1",
                "task_name": "resnet-training",
                "state": "ACTIVE",
                "client_pid": 4312,
                "gpu_uuids": ["GPU-0"],
                "memory_per_gpu_mib": 24_576,
                "released_by_gpu_mib": {"GPU-0": 23_500},
                "expires_at": 190.0,
            }
        ],
        "processes": [
            {
                "pid": 9821,
                "name": "python",
                "classification": "EXTERNAL",
                "gpu_uuid": "GPU-0",
                "used_memory_mib": 8192,
            }
        ],
        "profiles": [
            {
                "task_key": "resnet:imagenet",
                "world_size": 2,
                "peak_per_gpu_mib": 12_000,
                "margin_mib": 1800,
                "recommended_mib": 14_000,
                "status": "SUCCESS",
                "fingerprint_valid": True,
            }
        ],
        "policy": {
            "revision": 17,
            "status": "PENDING",
            "config": {"leave_free_mib": 3072, "poll_interval_seconds": 2.0},
        },
        "events": [
            {
                "timestamp": "2026-07-14T12:00:00Z",
                "type": "LEASE_APPROVED",
                "message": "lease-1 activated",
            }
        ],
    }

    rendered = {
        section: render_section(section, snapshot, width=36)
        for section in ("tasks", "processes", "profiles", "policy", "events")
    }

    assert "resnet-training [ACTIVE]" in rendered["tasks"]
    assert "Released: GPU-0=22.9 GiB" in rendered["tasks"]
    assert "Process 9821 [EXTERNAL]" in rendered["processes"]
    assert "resnet:imagenet [SUCCESS]" in rendered["profiles"]
    assert "Recommended: 13.7 GiB" in rendered["profiles"]
    assert "Policy r17 [PENDING]" in rendered["policy"]
    assert "leave_free_mib: 3072" in rendered["policy"]
    assert "LEASE_APPROVED" in rendered["events"]
    assert all(
        len(line) <= 36
        for section in rendered.values()
        for line in section.splitlines()
    )


def test_current_tasks_render_before_released_history() -> None:
    snapshot = {
        "leases": [
            {
                "lease_id": "old-1",
                "task_name": "old-run-1",
                "state": "RELEASED",
                "created_at": 1.0,
            },
            {
                "lease_id": "old-2",
                "task_name": "old-run-2",
                "state": "RELEASED",
                "created_at": 2.0,
            },
            {
                "lease_id": "current",
                "task_name": "training-now",
                "state": "ACTIVE",
                "created_at": 3.0,
            },
        ]
    }

    rendered = render_section("tasks", snapshot, width=80)

    assert rendered.index("training-now [ACTIVE]") < rendered.index(
        "old-run-2 [RELEASED]"
    )
    assert rendered.index("old-run-2 [RELEASED]") < rendered.index(
        "old-run-1 [RELEASED]"
    )
