from __future__ import annotations

import os

import psutil
import pytest

from watchgpu.allocator import InMemoryMemoryAllocator
from watchgpu.config import GPUConfig, WatchGPUConfig
from watchgpu.models import GPUProcess, GPUSnapshot
from watchgpu.observer import InMemoryGPUObserver
from watchgpu.policy import ReservationLimits
from watchgpu.supervisor import (
    GroupLeaseRequest,
    Lease,
    LeaseRequestError,
    LeaseState,
    ProcessLeaseActivityVerifier,
    Supervisor,
    TrustingReleaseVerifier,
)
from watchgpu.worker import WorkerController, WorkerStatus


def _worker(uuid: str, held_mib: int) -> WorkerController:
    allocator = InMemoryMemoryAllocator(chunk_mib=500)
    allocator.reconcile(held_mib)
    return WorkerController(
        gpu_uuid=uuid,
        allocator=allocator,
        limits=ReservationLimits(leave_free_mib=1000, reserve_limit_mib=None, reserve_ratio=None),
        growth_stability_seconds=0,
        allocation_tolerance_mib=0,
    )


def test_orphan_verifier_keeps_reparented_gpu_rank_when_launcher_is_gone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = InMemoryGPUObserver(
        (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 1000, 0),)
    )
    observer.replace_processes(
        "GPU-0", (GPUProcess(pid=4321, used_memory_mib=2000, name="training-rank"),)
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.NoSuchProcess(1)),
    )
    lease = Lease(
        lease_id="orphan-with-rank",
        request=GroupLeaseRequest(
            request_id="orphan-with-rank",
            task_name="training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        state=LeaseState.ORPHANED,
        created_at=0,
        gpu_uuids=("GPU-0",),
    )

    verifier = ProcessLeaseActivityVerifier(observer, worker_pids=lambda: {9999})

    assert verifier.is_active(lease) is True


def test_orphan_verifier_ignores_unrelated_gpu_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observer = InMemoryGPUObserver(
        (GPUSnapshot(0, "GPU-0", "Test GPU", 10_000, 1000, 0),)
    )
    observer.replace_processes(
        "GPU-0", (GPUProcess(pid=4321, used_memory_mib=2000, name="other-user"),)
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.NoSuchProcess(1)),
    )
    monkeypatch.setattr(os, "getpgid", lambda _pid: 88)
    monkeypatch.setattr(os, "getsid", lambda _pid: 99)
    lease = Lease(
        lease_id="orphan-other-process",
        request=GroupLeaseRequest(
            request_id="orphan-other-process",
            task_name="training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
            client_process_group=77,
            client_session_id=78,
        ),
        state=LeaseState.ORPHANED,
        created_at=0,
        gpu_uuids=("GPU-0",),
    )

    assert ProcessLeaseActivityVerifier(observer).is_active(lease) is False


def _supervisor(
    holds: dict[str, int],
    *,
    activity_verifier: MutableLeaseActivityVerifier | None = None,
) -> tuple[Supervisor, dict[str, WorkerController], InMemoryGPUObserver]:
    workers = {uuid: _worker(uuid, held) for uuid, held in holds.items()}
    snapshots = tuple(
        GPUSnapshot(index, uuid, uuid, 10_000, 1000, 0)
        for index, uuid in enumerate(holds)
    )
    observer = InMemoryGPUObserver(snapshots)
    return (
        Supervisor(
            observer=observer,
            workers=workers,
            release_verifier=TrustingReleaseVerifier(),
            lease_activity_verifier=activity_verifier,
        ),
        workers,
        observer,
    )


class MutableLeaseActivityVerifier:
    def __init__(self, active: bool) -> None:
        self.active = active

    def is_active(self, lease: Lease) -> bool:
        return self.active


def test_group_lease_is_approved_atomically_and_is_idempotent() -> None:
    supervisor, workers, _observer = _supervisor({"GPU-0": 3000, "GPU-1": 3000})
    request = GroupLeaseRequest(
        request_id="request-1",
        task_name="two-card-training",
        gpu_count=2,
        memory_per_gpu_mib=2000,
        ttl_seconds=600,
        client_pid=1234,
    )

    lease = supervisor.request_lease(request, now=100)

    assert lease.state is LeaseState.ACTIVE
    assert lease.gpu_uuids == ("GPU-0", "GPU-1")
    assert workers["GPU-0"].status.held_mib == 1000
    assert workers["GPU-1"].status.held_mib == 1000
    assert supervisor.request_lease(request, now=101) is lease


def test_group_lease_does_not_partially_release_when_one_gpu_is_short() -> None:
    supervisor, workers, _observer = _supervisor({"GPU-0": 3000, "GPU-1": 500})
    request = GroupLeaseRequest(
        request_id="request-2",
        task_name="two-card-training",
        gpu_count=2,
        memory_per_gpu_mib=2000,
        ttl_seconds=600,
        client_pid=1234,
    )

    lease = supervisor.request_lease(request, now=100)

    assert lease.state is LeaseState.QUEUED
    assert lease.gpu_uuids == ()
    assert workers["GPU-0"].status.held_mib == 3000
    assert workers["GPU-1"].status.held_mib == 500


def test_lease_can_use_driver_free_memory_plus_worker_reservation() -> None:
    supervisor, workers, _observer = _supervisor({"GPU-0": 1500})
    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="request-free-plus-held",
            task_name="bootstrap-training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=600,
            client_pid=1234,
        ),
        now=100,
    )

    assert lease.state is LeaseState.ACTIVE
    assert workers["GPU-0"].status.held_mib == 0


def test_queued_group_lease_activates_after_workers_regain_capacity() -> None:
    supervisor, workers, observer = _supervisor({"GPU-0": 3000, "GPU-1": 500})
    request = GroupLeaseRequest(
        request_id="request-3",
        task_name="queued-training",
        gpu_count=2,
        memory_per_gpu_mib=2000,
        ttl_seconds=600,
        client_pid=1234,
    )
    lease = supervisor.request_lease(request, now=100)
    assert lease.state is LeaseState.QUEUED

    observer.replace(
        (
            GPUSnapshot(0, "GPU-0", "GPU-0", 10_000, 3000, 0),
            GPUSnapshot(1, "GPU-1", "GPU-1", 10_000, 3000, 0),
        )
    )

    supervisor.tick(now=101)
    supervisor.tick(now=102)

    assert lease.state is LeaseState.ACTIVE
    assert lease.gpu_uuids == ("GPU-0", "GPU-1")
    assert workers["GPU-0"].status.held_mib == 1000
    assert workers["GPU-1"].status.held_mib == 0


def test_active_lease_can_be_renewed() -> None:
    supervisor, _workers, _observer = _supervisor({"GPU-0": 3000})
    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="request-4",
            task_name="long-training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        now=100,
    )
    assert lease.expires_at == 160

    renewed = supervisor.renew_lease(lease.lease_id, now=130)

    assert renewed is lease
    assert renewed.state is LeaseState.ACTIVE
    assert renewed.expires_at == 190


def test_releasing_lease_allows_worker_to_rebuild_reservation() -> None:
    supervisor, workers, observer = _supervisor({"GPU-0": 3000})
    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="request-5",
            task_name="completed-training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        now=100,
    )
    assert workers["GPU-0"].status.held_mib == 1000
    observer.replace((GPUSnapshot(0, "GPU-0", "GPU-0", 10_000, 3000, 0),))

    released = supervisor.release_lease(lease.lease_id, now=110)
    supervisor.tick(now=111)
    supervisor.tick(now=112)

    assert released.state is LeaseState.RELEASED
    assert released.expires_at == 110
    assert workers["GPU-0"].status.held_mib == 3000


def test_expired_lease_stays_orphaned_until_client_exit_is_confirmed() -> None:
    activity = MutableLeaseActivityVerifier(active=True)
    supervisor, workers, observer = _supervisor(
        {"GPU-0": 3000}, activity_verifier=activity
    )
    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="request-6",
            task_name="interrupted-training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        now=100,
    )
    observer.replace((GPUSnapshot(0, "GPU-0", "GPU-0", 10_000, 3000, 0),))

    supervisor.tick(now=160)
    assert lease.state is LeaseState.ORPHANED
    assert workers["GPU-0"].status.held_mib == 1000

    supervisor.tick(now=161)
    assert lease.state is LeaseState.ORPHANED

    activity.active = False
    supervisor.tick(now=162)
    assert lease.state is LeaseState.RELEASED


def test_supervisor_applies_runtime_policy_to_managed_workers() -> None:
    supervisor, workers, _observer = _supervisor({"GPU-0": 3000})

    supervisor.update_policy(
        WatchGPUConfig(
            leave_free="3GiB",
            growth_stability_seconds=20,
            gpus=[GPUConfig(selector="GPU-0")],
        ),
        now=100,
    )

    assert workers["GPU-0"].status.held_mib == 928


def test_active_lease_restores_as_orphaned_and_keeps_headroom() -> None:
    activity = MutableLeaseActivityVerifier(active=True)
    supervisor, workers, observer = _supervisor(
        {"GPU-0": 3000}, activity_verifier=activity
    )
    observer.replace((GPUSnapshot(0, "GPU-0", "GPU-0", 10_000, 3000, 0),))

    restored = supervisor.restore_lease(
        GroupLeaseRequest(
            request_id="restored-request",
            task_name="training-during-restart",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        state=LeaseState.ACTIVE,
        gpu_uuids=("GPU-0",),
        created_at=50,
        expires_at=110,
    )
    supervisor.tick(now=100)

    assert restored.state is LeaseState.ORPHANED
    assert workers["GPU-0"].status.held_mib == 3000


def test_status_classifies_only_lease_process_tree_as_managed_training() -> None:
    supervisor, _workers, observer = _supervisor({"GPU-0": 3000})
    client_pid = os.getpid()
    supervisor.request_lease(
        GroupLeaseRequest(
            request_id="process-owner",
            task_name="known-training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=client_pid,
            client_start_time=psutil.Process(client_pid).create_time(),
        ),
        now=100,
    )
    observer.replace_processes(
        "GPU-0",
        (
            GPUProcess(client_pid, 1500, "python"),
            GPUProcess(999_999, 500, "unknown"),
        ),
    )

    processes = supervisor.status_snapshot().processes

    assert processes[0].classification == "MANAGED_TRAINING"
    assert processes[0].task_name == "known-training"
    assert processes[1].classification == "EXTERNAL"
    assert processes[1].task_name is None


def test_status_classifies_reparented_orphan_rank_by_persisted_process_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    supervisor, _workers, observer = _supervisor({"GPU-0": 0})
    supervisor.restore_lease(
        GroupLeaseRequest(
            request_id="orphan-rank-owner",
            task_name="orphan-training",
            gpu_count=1,
            memory_per_gpu_mib=1000,
            ttl_seconds=60,
            client_pid=1234,
            client_process_group=77,
        ),
        state=LeaseState.ORPHANED,
        gpu_uuids=("GPU-0",),
        created_at=0,
        expires_at=10,
    )
    observer.replace_processes(
        "GPU-0", (GPUProcess(4321, 1000, "torch-rank"),)
    )
    monkeypatch.setattr(
        psutil,
        "Process",
        lambda _pid: (_ for _ in ()).throw(psutil.NoSuchProcess(1)),
    )
    monkeypatch.setattr(os, "getpgid", lambda pid: 77 if pid == 4321 else 88)

    process = supervisor.status_snapshot().processes[0]

    assert process.classification == "MANAGED_TRAINING"
    assert process.lease_id == "orphan-rank-owner"


def test_managed_gpu_cannot_be_removed_until_active_lease_releases() -> None:
    supervisor, _workers, _observer = _supervisor({"GPU-0": 3000})
    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="lease-blocks-remove",
            task_name="training",
            gpu_count=1,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        now=100,
    )

    with pytest.raises(LeaseRequestError, match="active lease"):
        supervisor.remove_worker("GPU-0")

    supervisor.release_lease(lease.lease_id, now=110)
    removed = supervisor.remove_worker("GPU-0")
    assert removed.status.gpu_uuid == "GPU-0"


def test_dead_client_is_removed_from_waiting_queue() -> None:
    activity = MutableLeaseActivityVerifier(active=False)
    supervisor, _workers, _observer = _supervisor(
        {"GPU-0": 0}, activity_verifier=activity
    )
    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="dead-queued-client",
            task_name="never-started",
            gpu_count=1,
            memory_per_gpu_mib=5000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        now=100,
    )
    assert lease.state is LeaseState.QUEUED

    supervisor.tick(now=101)

    assert lease.state is LeaseState.CANCELLED


def test_multi_gpu_activation_exception_has_a_deterministic_rejected_state() -> None:
    class FailingReleaseWorker(WorkerController):
        def release_for_lease(self, requested_mib: int) -> WorkerStatus:
            del requested_mib
            raise RuntimeError("simulated second GPU failure")

    supervisor, workers, _observer = _supervisor({"GPU-0": 3000, "GPU-1": 3000})
    supervisor.remove_worker("GPU-1")
    original = workers["GPU-1"]
    failing_allocator = InMemoryMemoryAllocator(chunk_mib=500)
    failing_allocator.reconcile(original.status.held_mib)
    failing = FailingReleaseWorker(
        gpu_uuid="GPU-1",
        allocator=failing_allocator,
        limits=ReservationLimits(1000, None, None),
        growth_stability_seconds=0,
        allocation_tolerance_mib=0,
    )
    supervisor.add_worker("GPU-1", failing)

    lease = supervisor.request_lease(
        GroupLeaseRequest(
            request_id="second-gpu-fails",
            task_name="two-rank",
            gpu_count=2,
            memory_per_gpu_mib=2000,
            ttl_seconds=60,
            client_pid=1234,
        ),
        now=100,
    )

    assert lease.state is LeaseState.REJECTED
    assert lease.gpu_uuids == ()
    assert "simulated second GPU failure" in (lease.error or "")


def test_manual_release_is_not_reported_successful_without_driver_confirmation() -> None:
    class DenyingVerifier:
        def verify(
            self, gpu_uuid: str, *, expected_free_mib: int, timeout: float
        ) -> bool:
            assert gpu_uuid == "GPU-0"
            assert expected_free_mib == 2000
            assert timeout > 0
            return False

    worker = _worker("GPU-0", 3000)
    supervisor = Supervisor(
        observer=InMemoryGPUObserver(
            (GPUSnapshot(0, "GPU-0", "A40", 10_000, 1000, 0),)
        ),
        workers={"GPU-0": worker},
        release_verifier=DenyingVerifier(),
    )

    with pytest.raises(LeaseRequestError, match="driver did not confirm"):
        supervisor.release_reservations(gpu_uuid="GPU-0", memory_mib=1000)

    assert any(event.type == "RELEASE_VERIFY_FAILED" for event in supervisor.events)
