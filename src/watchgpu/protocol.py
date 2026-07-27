from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict
from typing import Any

from watchgpu.config import WatchGPUConfig
from watchgpu.control import ConfigApplyResult, RuntimeConfigController
from watchgpu.profile import ProfileRecord, ProfileStore
from watchgpu.supervisor import GroupLeaseRequest, Lease, Supervisor

PROTOCOL_VERSION = 1


class ProtocolError(ValueError):
    pass


class SupervisorProtocol:
    """Translate versioned wire messages into Supervisor operations."""

    def __init__(
        self,
        supervisor: Supervisor,
        *,
        clock: Callable[[], float],
        config_controller: RuntimeConfigController | None = None,
        profile_store: ProfileStore | None = None,
        shutdown_callback: Callable[[], None] | None = None,
        restart_callback: Callable[[], None] | None = None,
        restart_status_provider: Callable[[], Mapping[str, Any]] | None = None,
        cpu_status_provider: Callable[[], Mapping[str, Any]] | None = None,
        runtime_status_provider: Callable[[], Mapping[str, Any]] | None = None,
        state_change_callback: Callable[[], None] | None = None,
    ) -> None:
        self._supervisor = supervisor
        self._clock = clock
        self._config_controller = config_controller
        self._profile_store = profile_store
        self._shutdown_callback = shutdown_callback
        self._restart_callback = restart_callback
        self._restart_status_provider = restart_status_provider
        self._cpu_status_provider = cpu_status_provider
        self._runtime_status_provider = runtime_status_provider
        self._daemon_state = "RUNNING"
        self._state_change_callback = state_change_callback

    def handle(self, message: Mapping[str, Any]) -> dict[str, Any]:
        version = message.get("version")
        if version != PROTOCOL_VERSION:
            raise ProtocolError(f"unsupported protocol version: {version}")
        request_id = _required_string(message, "request_id")
        method = _required_string(message, "method")
        params = message.get("params", {})
        if not isinstance(params, Mapping):
            raise ProtocolError("params must be an object")
        if self._daemon_state != "RUNNING" and method != "status.get":
            raise ProtocolError(
                f"daemon is {self._daemon_state}; new mutations and leases are closed"
            )

        if method == "lease.request":
            result = self._request_lease(params)
        elif method == "lease.renew":
            result = lease_to_dict(
                self._supervisor.renew_lease(
                    _required_string(params, "lease_id"), now=self._clock()
                )
            )
        elif method == "lease.release":
            result = lease_to_dict(
                self._supervisor.release_lease(
                    _required_string(params, "lease_id"), now=self._clock()
                )
            )
        elif method == "status.get":
            snapshot = self._supervisor.status_snapshot()
            queued_positions = {
                lease.lease_id: position
                for position, lease in enumerate(
                    (
                        item
                        for item in snapshot.leases
                        if item.state.value == "QUEUED"
                    ),
                    start=1,
                )
            }
            result = {
                "daemon_state": self._daemon_state,
                "gpus": [asdict(gpu) for gpu in snapshot.gpus],
                "leases": [
                    lease_status_to_dict(
                        lease,
                        now=self._clock(),
                        queue_position=queued_positions.get(lease.lease_id),
                    )
                    for lease in snapshot.leases
                ],
                "processes": [asdict(process) for process in snapshot.processes],
                "events": [asdict(event) for event in snapshot.events],
            }
            if self._config_controller is not None:
                result["policy"] = {
                    "revision": self._config_controller.revision,
                    "config": self._config_controller.config.model_dump(mode="json"),
                    "status": self._config_controller.runtime_status.value,
                    "reason": self._config_controller.runtime_reason,
                }
            result["profiles"] = (
                []
                if self._profile_store is None
                else [
                    profile_to_status(record)
                    for record in self._profile_store.records()
                ]
            )
            if self._restart_status_provider is not None:
                result["maintenance_restart"] = dict(self._restart_status_provider())
            if self._cpu_status_provider is not None:
                result["cpu"] = dict(self._cpu_status_provider())
            if self._runtime_status_provider is not None:
                result["runtime"] = dict(self._runtime_status_provider())
        elif method == "policy.apply":
            controller = self._require_config_controller()
            raw_config = params.get("config")
            if not isinstance(raw_config, Mapping):
                raise ProtocolError("config must be an object")
            result = config_result_to_dict(
                controller.apply(
                    WatchGPUConfig.model_validate(raw_config),
                    expected_revision=_required_int(params, "expected_revision"),
                    save=_required_bool(params, "save"),
                )
            )
        elif method == "worker.pause":
            result = {
                "workers": [
                    asdict(status)
                    for status in self._supervisor.pause_workers(
                        _optional_string(params, "gpu_uuid")
                    )
                ]
            }
        elif method == "worker.resume":
            result = {
                "workers": [
                    asdict(status)
                    for status in self._supervisor.resume_workers(
                        _optional_string(params, "gpu_uuid")
                    )
                ]
            }
        elif method == "worker.release":
            result = {
                "workers": [
                    asdict(status)
                    for status in self._supervisor.release_reservations(
                        gpu_uuid=_optional_string(params, "gpu_uuid"),
                        memory_mib=_optional_int(params, "memory_mib"),
                    )
                ]
            }
        elif method == "daemon.stop":
            if not _required_bool(params, "release"):
                raise ProtocolError("daemon.stop requires release=true")
            if self._shutdown_callback is None:
                raise ProtocolError("daemon lifecycle control is not configured")
            self._daemon_state = "QUIESCING"
            self._shutdown_callback()
            result = {"status": "STOPPING"}
        elif method == "daemon.restart":
            if self._restart_callback is None:
                raise ProtocolError("daemon lifecycle control is not configured")
            self._daemon_state = "RESTARTING"
            self._restart_callback()
            result = {"status": "RESTARTING"}
        else:
            raise ProtocolError(f"unknown method: {method}")
        if self._state_change_callback is not None and method != "status.get":
            self._state_change_callback()
        return {
            "version": PROTOCOL_VERSION,
            "request_id": request_id,
            "ok": True,
            "result": result,
        }

    def _request_lease(self, params: Mapping[str, Any]) -> dict[str, Any]:
        raw_candidates = params.get("candidate_uuids")
        candidates: tuple[str, ...] | None
        if raw_candidates is None:
            candidates = None
        elif isinstance(raw_candidates, list) and all(
            isinstance(value, str) for value in raw_candidates
        ):
            candidates = tuple(raw_candidates)
        else:
            raise ProtocolError("candidate_uuids must be a list of strings")

        lease = self._supervisor.request_lease(
            GroupLeaseRequest(
                request_id=_required_string(params, "lease_request_id"),
                task_name=_required_string(params, "task_name"),
                gpu_count=_required_int(params, "gpu_count"),
                memory_per_gpu_mib=_required_int(params, "memory_per_gpu_mib"),
                ttl_seconds=_required_number(params, "ttl_seconds"),
                client_pid=_required_int(params, "client_pid"),
                candidate_uuids=candidates,
                client_start_time=_optional_number(params, "client_start_time"),
                client_process_group=_optional_int(params, "client_process_group"),
                client_session_id=_optional_int(params, "client_session_id"),
            ),
            now=self._clock(),
        )
        return lease_to_dict(lease)

    def _require_config_controller(self) -> RuntimeConfigController:
        if self._config_controller is None:
            raise ProtocolError("runtime policy control is not configured")
        return self._config_controller


def lease_to_dict(lease: Lease) -> dict[str, Any]:
    return {
        "lease_id": lease.lease_id,
        "state": lease.state.value,
        "gpu_uuids": list(lease.gpu_uuids),
        "memory_per_gpu_mib": lease.request.memory_per_gpu_mib,
        "expires_at": lease.expires_at,
        "error": lease.error,
    }


def lease_status_to_dict(
    lease: Lease,
    *,
    now: float | None = None,
    queue_position: int | None = None,
) -> dict[str, Any]:
    result = lease_to_dict(lease)
    result.update(
        {
            "task_name": lease.request.task_name,
            "client_pid": lease.request.client_pid,
            "created_at": lease.created_at,
            "released_by_gpu_mib": dict(lease.released_by_gpu_mib),
            "queue_position": queue_position,
            "heartbeat_age_seconds": (
                None
                if now is None or lease.expires_at is None
                else max(
                    0.0,
                    now - (lease.expires_at - lease.request.ttl_seconds),
                )
            ),
        }
    )
    return result


def config_result_to_dict(result: ConfigApplyResult) -> dict[str, Any]:
    return {
        "status": result.status.value,
        "revision": result.revision,
        "config": result.config.model_dump(mode="json"),
        "reason": result.reason,
    }


def profile_to_status(record: ProfileRecord) -> dict[str, Any]:
    peak = max(record.observed_peak_mib_by_gpu.values(), default=None)
    recommendation = record.recommended_memory_per_gpu_mib
    margin = (
        None
        if peak is None or recommendation is None
        else max(0, recommendation - peak)
    )
    return {
        "task_key": record.task_name,
        "status": record.outcome.value,
        "world_size": record.world_size,
        "peak_per_gpu_mib": peak,
        "margin_mib": margin,
        "recommended_mib": recommendation,
        "fingerprint": record.fingerprint,
        # Validity is evaluated by watchgpu-run against the current launch
        # identity; a generic status request has no candidate argv/config.
        "fingerprint_valid": None,
        "exit_code": record.exit_code,
        "error": record.error,
    }


def _required_string(values: Mapping[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{key} must be a non-empty string")
    return value


def _optional_string(values: Mapping[str, Any], key: str) -> str | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"{key} must be a non-empty string")
    return value


def _required_int(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{key} must be an integer")
    return value


def _optional_int(values: Mapping[str, Any], key: str) -> int | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ProtocolError(f"{key} must be an integer")
    if value <= 0:
        raise ProtocolError(f"{key} must be positive")
    return value


def _required_bool(values: Mapping[str, Any], key: str) -> bool:
    value = values.get(key)
    if not isinstance(value, bool):
        raise ProtocolError(f"{key} must be a boolean")
    return value


def _required_number(values: Mapping[str, Any], key: str) -> float:
    value = values.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError(f"{key} must be a number")
    return float(value)


def _optional_number(values: Mapping[str, Any], key: str) -> float | None:
    value = values.get(key)
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ProtocolError(f"{key} must be a number")
    return float(value)
