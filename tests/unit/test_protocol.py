from __future__ import annotations

import pytest

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.config import WatchGPUConfig
from watchgpu.control import ApplyStatus, RuntimeConfigController
from watchgpu.models import GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.protocol import ProtocolError, SupervisorProtocol
from watchgpu.supervisor import Supervisor, TrustingReleaseVerifier
from watchgpu.worker import WorkerController


def _supervisor() -> Supervisor:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(3000)
    worker = WorkerController(
        gpu_uuid="GPU-0",
        allocator=allocator,
        limits=ReservationLimits(
            leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None
        ),
        growth_stability_seconds=0,
        allocation_tolerance_mib=0,
    )
    return Supervisor(
        observer=InMemoryGPUObserver(
            (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 1000, 0),)
        ),
        workers={"GPU-0": worker},
        release_verifier=TrustingReleaseVerifier(),
    )


def test_protocol_requests_group_lease_and_returns_serializable_result() -> None:
    protocol = SupervisorProtocol(_supervisor(), clock=lambda: 100.0)

    response = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-1",
            "method": "lease.request",
            "params": {
                "lease_request_id": "lease-1",
                "task_name": "training",
                "gpu_count": 1,
                "memory_per_gpu_mib": 2000,
                "ttl_seconds": 60,
                "client_pid": 1234,
            },
        }
    )

    assert response == {
        "version": 1,
        "request_id": "rpc-1",
        "ok": True,
        "result": {
            "lease_id": "lease-1",
            "state": "ACTIVE",
            "gpu_uuids": ["GPU-0"],
            "memory_per_gpu_mib": 2000,
            "expires_at": 160.0,
            "error": None,
        },
    }


def test_protocol_renews_and_releases_a_lease() -> None:
    now = 100.0
    protocol = SupervisorProtocol(_supervisor(), clock=lambda: now)
    protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-request",
            "method": "lease.request",
            "params": {
                "lease_request_id": "lease-2",
                "task_name": "training",
                "gpu_count": 1,
                "memory_per_gpu_mib": 2000,
                "ttl_seconds": 60,
                "client_pid": 1234,
            },
        }
    )

    now = 130.0
    renewed = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-renew",
            "method": "lease.renew",
            "params": {"lease_id": "lease-2"},
        }
    )
    assert renewed["result"]["expires_at"] == 190.0

    now = 140.0
    released = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-release",
            "method": "lease.release",
            "params": {"lease_id": "lease-2"},
        }
    )
    assert released["result"]["state"] == "RELEASED"


def test_protocol_returns_gpu_and_lease_status_snapshot() -> None:
    protocol = SupervisorProtocol(_supervisor(), clock=lambda: 100.0)
    protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-request",
            "method": "lease.request",
            "params": {
                "lease_request_id": "lease-status",
                "task_name": "visible-training",
                "gpu_count": 1,
                "memory_per_gpu_mib": 2000,
                "ttl_seconds": 60,
                "client_pid": 1234,
            },
        }
    )

    response = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-status",
            "method": "status.get",
            "params": {},
        }
    )

    assert response["result"]["gpus"] == [
        {
            "index": 0,
            "uuid": "GPU-0",
            "name": "Test GPU",
            "total_mib": 10_000,
            "free_mib": 1000,
            "utilization_percent": 0,
            "reserved_mib": 1000,
            "leased_mib": 2000,
            "worker_state": "HOLDING",
            "temperature_c": None,
            "mig_mode": None,
        }
    ]
    assert response["result"]["leases"][0]["task_name"] == "visible-training"


def test_protocol_applies_policy_with_optimistic_revision() -> None:
    controller = RuntimeConfigController(
        WatchGPUConfig(leave_free="2GiB"),
        apply=lambda _config: ApplyStatus.APPLIED,
    )
    protocol = SupervisorProtocol(
        _supervisor(), clock=lambda: 100.0, config_controller=controller
    )
    candidate = WatchGPUConfig(leave_free="3GiB").model_dump(mode="json")

    applied = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-policy-1",
            "method": "policy.apply",
            "params": {
                "config": candidate,
                "expected_revision": 0,
                "save": False,
            },
        }
    )
    stale = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-policy-2",
            "method": "policy.apply",
            "params": {
                "config": candidate,
                "expected_revision": 0,
                "save": False,
            },
        }
    )

    assert applied["result"]["status"] == "APPLIED"
    assert applied["result"]["revision"] == 1
    assert stale["result"]["status"] == "REJECTED"
    assert stale["result"]["revision"] == 1


def test_protocol_controls_only_worker_reservations() -> None:
    protocol = SupervisorProtocol(_supervisor(), clock=lambda: 100.0)

    paused = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-pause",
            "method": "worker.pause",
            "params": {"gpu_uuid": "GPU-0"},
        }
    )
    resumed = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-resume",
            "method": "worker.resume",
            "params": {"gpu_uuid": "GPU-0"},
        }
    )
    released = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-release-worker",
            "method": "worker.release",
            "params": {"gpu_uuid": "GPU-0", "memory_mib": 700},
        }
    )

    assert paused["result"]["workers"][0]["state"] == "PAUSED"
    assert resumed["result"]["workers"][0]["state"] == "OBSERVING"
    assert released["result"]["workers"][0]["held_mib"] == 0


def test_protocol_requests_controlled_restart_and_release_shutdown() -> None:
    actions: list[str] = []
    protocol = SupervisorProtocol(
        _supervisor(),
        clock=lambda: 100.0,
        shutdown_callback=lambda: actions.append("stop"),
        restart_callback=lambda: actions.append("restart"),
    )

    stopped = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-stop",
            "method": "daemon.stop",
            "params": {"release": True},
        }
    )
    status = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-quiescing-status",
            "method": "status.get",
            "params": {},
        }
    )
    with pytest.raises(ProtocolError, match="QUIESCING"):
        protocol.handle(
            {
                "version": 1,
                "request_id": "rpc-restart",
                "method": "daemon.restart",
                "params": {},
            }
        )

    restart_protocol = SupervisorProtocol(
        _supervisor(),
        clock=lambda: 100.0,
        restart_callback=lambda: actions.append("restart"),
    )
    restarted = restart_protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-restart-separate",
            "method": "daemon.restart",
            "params": {},
        }
    )

    assert actions == ["stop", "restart"]
    assert stopped["result"]["status"] == "STOPPING"
    assert status["result"]["daemon_state"] == "QUIESCING"
    assert restarted["result"]["status"] == "RESTARTING"


def test_status_exposes_maintenance_restart_runtime_state() -> None:
    protocol = SupervisorProtocol(
        _supervisor(),
        clock=lambda: 100.0,
        restart_status_provider=lambda: {
            "state": "SCHEDULED",
            "scheduled_for": "2026-07-19T04:15:00+00:00",
            "last_executed_local_date": "2026-07-18",
        },
        cpu_status_provider=lambda: {
            "process_tree_percent": 73.5,
            "process_count": 3,
            "budget_percent": 100,
            "over_budget": False,
            "affinity_cores": [4],
            "worker_cpu_threads": 1,
        },
    )

    response = protocol.handle(
        {
            "version": 1,
            "request_id": "rpc-status-restart",
            "method": "status.get",
            "params": {},
        }
    )

    assert response["result"]["maintenance_restart"] == {
        "state": "SCHEDULED",
        "scheduled_for": "2026-07-19T04:15:00+00:00",
        "last_executed_local_date": "2026-07-18",
    }
    assert response["result"]["cpu"]["process_tree_percent"] == 73.5
    assert response["result"]["cpu"]["affinity_cores"] == [4]
