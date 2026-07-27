from __future__ import annotations

import json
import os
import random
import re
import tempfile
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from watchgpu.environment import CommandRunner, run_command


class LifecycleError(RuntimeError):
    pass


class BackgroundMode(StrEnum):
    AUTO = "auto"
    SYSTEMD_USER = "systemd-user"
    DETACHED = "detached"
    FOREGROUND = "foreground"


class LingerState(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class DetachedProcessState(StrEnum):
    MISSING = "MISSING"
    RUNNING = "RUNNING"
    STALE = "STALE"


class RestartScheduleState(StrEnum):
    DISABLED = "DISABLED"
    SCHEDULED = "SCHEDULED"
    PENDING = "PENDING"
    DUE = "DUE"


class PersistedLeaseState(StrEnum):
    QUEUED = "QUEUED"
    ACTIVE = "ACTIVE"
    ORPHANED = "ORPHANED"


@dataclass(frozen=True, slots=True)
class UserSystemdProbe:
    available: bool
    linger: LingerState
    detail: str | None = None

    @property
    def session_bound(self) -> bool:
        return not self.available or self.linger is not LingerState.YES


@dataclass(frozen=True, slots=True)
class DetachedPidRecord:
    pid: int
    start_time_ticks: int


@dataclass(frozen=True, slots=True)
class DetachedProcessStatus:
    state: DetachedProcessState
    record: DetachedPidRecord | None
    observed_start_time_ticks: int | None = None


@dataclass(frozen=True, slots=True)
class RestartSchedule:
    enabled: bool = False
    at: str = "04:00"
    jitter_seconds: int = 1200
    defer_while_leased: bool = True

    def __post_init__(self) -> None:
        if not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", self.at):
            raise LifecycleError("restart time must use HH:MM in local time")
        if self.jitter_seconds < 0:
            raise LifecycleError("restart jitter cannot be negative")

    @property
    def local_time(self) -> time:
        hour, minute = (int(part) for part in self.at.split(":"))
        return time(hour=hour, minute=minute)


@dataclass(frozen=True, slots=True)
class RestartScheduleStatus:
    state: RestartScheduleState
    scheduled_for: datetime | None


@dataclass(frozen=True, slots=True)
class RestartRuntimeState:
    """Small host-local checkpoint that prevents duplicate daily maintenance."""

    last_executed_local_date: date | None = None


@dataclass(frozen=True, slots=True)
class PersistedLease:
    lease_id: str
    state: PersistedLeaseState
    task_name: str
    client_pid: int
    client_start_time: float | None
    gpu_uuids: tuple[str, ...]
    memory_per_gpu_mib: int
    expires_at: float | None
    gpu_count: int = 1
    ttl_seconds: float = 600.0
    candidate_uuids: tuple[str, ...] | None = None
    created_at: float = 0.0
    client_process_group: int | None = None
    client_session_id: int | None = None

    def __post_init__(self) -> None:
        if not self.lease_id:
            raise LifecycleDataError("persisted lease ID is required")
        if not self.task_name:
            raise LifecycleDataError("persisted lease task name is required")
        if self.client_pid <= 0:
            raise LifecycleDataError("persisted lease PID must be positive")
        if self.memory_per_gpu_mib <= 0:
            raise LifecycleDataError("persisted lease memory must be positive")
        if self.gpu_count <= 0:
            raise LifecycleDataError("persisted lease GPU count must be positive")
        if self.ttl_seconds <= 0:
            raise LifecycleDataError("persisted lease TTL must be positive")
        if self.client_process_group is not None and self.client_process_group <= 0:
            raise LifecycleDataError("persisted process group must be positive")
        if self.client_session_id is not None and self.client_session_id <= 0:
            raise LifecycleDataError("persisted session ID must be positive")
        if len(self.gpu_uuids) != len(set(self.gpu_uuids)):
            raise LifecycleDataError("persisted lease GPU UUIDs must be unique")


class RestartScheduler:
    def __init__(
        self,
        schedule: RestartSchedule,
        *,
        last_executed_local_date: date | None = None,
        jitter_source: Callable[[int], int] | None = None,
    ) -> None:
        self.schedule = schedule
        random_source = random.SystemRandom()
        self._jitter_source = jitter_source or (
            lambda maximum: random_source.randint(0, maximum)
        )
        self._last_executed_local_date = last_executed_local_date
        self._scheduled_for: datetime | None = None

    @property
    def last_executed_local_date(self) -> date | None:
        return self._last_executed_local_date

    def evaluate(self, now: datetime, *, leases_active: bool) -> RestartScheduleStatus:
        if not self.schedule.enabled:
            return RestartScheduleStatus(RestartScheduleState.DISABLED, None)
        if self._scheduled_for is None:
            self._scheduled_for = _calculate_next_allowed_restart(
                now,
                schedule=self.schedule,
                jitter_source=self._jitter_source,
                last_executed_local_date=self._last_executed_local_date,
            )
        if now < self._scheduled_for:
            state = RestartScheduleState.SCHEDULED
        elif self.schedule.defer_while_leased and leases_active:
            state = RestartScheduleState.PENDING
        else:
            state = RestartScheduleState.DUE
        return RestartScheduleStatus(state, self._scheduled_for)

    def mark_executed(self, now: datetime) -> RestartRuntimeState:
        self._last_executed_local_date = now.date()
        if not self.schedule.enabled:
            self._scheduled_for = None
        else:
            self._scheduled_for = _restart_candidate(
                now + timedelta(days=1),
                schedule=self.schedule,
                jitter_source=self._jitter_source,
            )
        return RestartRuntimeState(
            last_executed_local_date=self._last_executed_local_date
        )


def _calculate_next_allowed_restart(
    now: datetime,
    *,
    schedule: RestartSchedule,
    jitter_source: Callable[[int], int],
    last_executed_local_date: date | None,
) -> datetime:
    if last_executed_local_date is None or last_executed_local_date < now.date():
        return calculate_next_restart(
            now,
            schedule=schedule,
            jitter_source=jitter_source,
        )
    next_allowed_date = last_executed_local_date + timedelta(days=1)
    anchor = datetime.combine(next_allowed_date, time.min, tzinfo=now.tzinfo)
    return _restart_candidate(
        anchor,
        schedule=schedule,
        jitter_source=jitter_source,
    )


def calculate_next_restart(
    now: datetime,
    *,
    schedule: RestartSchedule,
    jitter_source: Callable[[int], int],
) -> datetime:
    candidate = _restart_candidate(now, schedule=schedule, jitter_source=jitter_source)
    if candidate < now:
        tomorrow = now + timedelta(days=1)
        candidate = _restart_candidate(
            tomorrow,
            schedule=schedule,
            jitter_source=jitter_source,
        )
    return candidate


def _restart_candidate(
    on_date: datetime,
    *,
    schedule: RestartSchedule,
    jitter_source: Callable[[int], int],
) -> datetime:
    jitter = jitter_source(schedule.jitter_seconds)
    if not 0 <= jitter <= schedule.jitter_seconds:
        raise LifecycleError("jitter source returned a value outside the configured range")
    fixed_time = datetime.combine(
        on_date.date(),
        schedule.local_time,
        tzinfo=on_date.tzinfo,
    )
    return fixed_time + timedelta(seconds=jitter)


class LifecycleDataError(LifecycleError):
    pass


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    success: bool
    error: str | None
    timestamp: float


class ShutdownResultStore:
    """Host-local acknowledgement written only after worker cleanup finishes."""

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, result: ShutdownResult) -> None:
        _atomic_write_json(
            self.path,
            {
                "version": self.SCHEMA_VERSION,
                "success": result.success,
                "error": result.error,
                "timestamp": result.timestamp,
            },
        )

    def load(self) -> ShutdownResult | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleDataError(f"invalid shutdown result: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != self.SCHEMA_VERSION:
            raise LifecycleDataError("unsupported or missing shutdown result version")
        success = payload.get("success")
        error = payload.get("error")
        timestamp = payload.get("timestamp")
        if not isinstance(success, bool):
            raise LifecycleDataError("shutdown success must be boolean")
        if error is not None and not isinstance(error, str):
            raise LifecycleDataError("shutdown error must be text or null")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise LifecycleDataError("shutdown timestamp must be numeric")
        return ShutdownResult(success, error, float(timestamp))

    def remove(self) -> None:
        with suppress(FileNotFoundError):
            self.path.unlink()


class RestartStateStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, state: RestartRuntimeState) -> None:
        _atomic_write_json(
            self.path,
            {
                "version": self.SCHEMA_VERSION,
                "last_executed_local_date": (
                    None
                    if state.last_executed_local_date is None
                    else state.last_executed_local_date.isoformat()
                ),
            },
        )

    def load(self) -> RestartRuntimeState:
        if not self.path.exists():
            return RestartRuntimeState()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleDataError(f"invalid restart state file: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != self.SCHEMA_VERSION:
            raise LifecycleDataError("unsupported or missing restart state version")
        raw_date = payload.get("last_executed_local_date")
        if raw_date is None:
            return RestartRuntimeState()
        if not isinstance(raw_date, str):
            raise LifecycleDataError("invalid restart state date")
        try:
            parsed_date = date.fromisoformat(raw_date)
        except ValueError as exc:
            raise LifecycleDataError("invalid restart state date") from exc
        return RestartRuntimeState(last_executed_local_date=parsed_date)


class LeaseStateStore:
    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = path

    def save(self, leases: tuple[PersistedLease, ...]) -> None:
        payload = {
            "version": self.SCHEMA_VERSION,
            "leases": [self._to_json(lease) for lease in leases],
        }
        _atomic_write_json(self.path, payload)

    def load(self) -> tuple[PersistedLease, ...]:
        if not self.path.exists():
            return ()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleDataError(f"invalid lease state file: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("version") != self.SCHEMA_VERSION:
            raise LifecycleDataError("unsupported or missing lease state version")
        raw_leases = payload.get("leases")
        if not isinstance(raw_leases, list):
            raise LifecycleDataError("invalid lease state file: leases must be a list")
        return tuple(self._from_json(value) for value in raw_leases)

    @staticmethod
    def _to_json(lease: PersistedLease) -> dict[str, object]:
        return {
            "lease_id": lease.lease_id,
            "state": lease.state.value,
            "task_name": lease.task_name,
            "client_pid": lease.client_pid,
            "client_start_time": lease.client_start_time,
            "gpu_uuids": list(lease.gpu_uuids),
            "memory_per_gpu_mib": lease.memory_per_gpu_mib,
            "expires_at": lease.expires_at,
            "gpu_count": lease.gpu_count,
            "ttl_seconds": lease.ttl_seconds,
            "candidate_uuids": (
                None if lease.candidate_uuids is None else list(lease.candidate_uuids)
            ),
            "created_at": lease.created_at,
            "client_process_group": lease.client_process_group,
            "client_session_id": lease.client_session_id,
        }

    @staticmethod
    def _from_json(value: object) -> PersistedLease:
        if not isinstance(value, dict):
            raise LifecycleDataError("invalid persisted lease: expected an object")
        lease_id = _required_string(value, "lease_id")
        task_name = _required_string(value, "task_name")
        client_pid = _required_positive_int(value, "client_pid")
        memory_per_gpu_mib = _required_positive_int(value, "memory_per_gpu_mib")
        state_value = _required_string(value, "state")
        try:
            state = PersistedLeaseState(state_value)
        except ValueError as exc:
            raise LifecycleDataError(f"invalid persisted lease state: {state_value}") from exc
        raw_uuids = value.get("gpu_uuids")
        if not isinstance(raw_uuids, list) or not all(
            isinstance(uuid, str) and uuid for uuid in raw_uuids
        ):
            raise LifecycleDataError("invalid persisted lease GPU UUIDs")
        client_start_time = _optional_number(value, "client_start_time")
        expires_at = _optional_number(value, "expires_at")
        raw_candidates = value.get("candidate_uuids")
        if raw_candidates is not None and (
            not isinstance(raw_candidates, list)
            or not all(isinstance(item, str) and item for item in raw_candidates)
        ):
            raise LifecycleDataError("invalid persisted lease candidate UUIDs")
        return PersistedLease(
            lease_id=lease_id,
            state=state,
            task_name=task_name,
            client_pid=client_pid,
            client_start_time=client_start_time,
            gpu_uuids=tuple(raw_uuids),
            memory_per_gpu_mib=memory_per_gpu_mib,
            expires_at=expires_at,
            gpu_count=_optional_positive_int(value, "gpu_count", default=1),
            ttl_seconds=_optional_positive_number(value, "ttl_seconds", default=600.0),
            candidate_uuids=(
                None if raw_candidates is None else tuple(raw_candidates)
            ),
            created_at=_optional_number(value, "created_at") or 0.0,
            client_process_group=_optional_positive_int_or_none(
                value, "client_process_group"
            ),
            client_session_id=_optional_positive_int_or_none(
                value, "client_session_id"
            ),
        )


class DetachedPidStore:
    def __init__(
        self,
        path: Path,
        *,
        start_time_reader: Callable[[int], int | None] = lambda pid: (
            read_process_start_time_ticks(pid)
        ),
    ) -> None:
        self.path = path
        self._start_time_reader = start_time_reader

    def write_for_process(self, pid: int) -> DetachedPidRecord:
        if pid <= 0:
            raise LifecycleDataError("PID must be positive")
        start_time = self._start_time_reader(pid)
        if start_time is None:
            raise LifecycleDataError(f"cannot read start time for PID {pid}")
        record = DetachedPidRecord(pid=pid, start_time_ticks=start_time)
        self.save(record)
        return record

    def save(self, record: DetachedPidRecord) -> None:
        _atomic_write_json(
            self.path,
            {"pid": record.pid, "start_time_ticks": record.start_time_ticks},
        )

    def load(self) -> DetachedPidRecord | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise LifecycleDataError(f"invalid detached PID file: {exc}") from exc
        if not isinstance(payload, dict):
            raise LifecycleDataError("invalid detached PID file: expected an object")
        pid = payload.get("pid")
        start_time = payload.get("start_time_ticks")
        if (
            not isinstance(pid, int)
            or isinstance(pid, bool)
            or pid <= 0
            or not isinstance(start_time, int)
            or isinstance(start_time, bool)
            or start_time <= 0
        ):
            raise LifecycleDataError("invalid detached PID file fields")
        return DetachedPidRecord(pid=pid, start_time_ticks=start_time)

    def inspect(self) -> DetachedProcessStatus:
        record = self.load()
        if record is None:
            return DetachedProcessStatus(DetachedProcessState.MISSING, None)
        observed = self._start_time_reader(record.pid)
        state = (
            DetachedProcessState.RUNNING
            if observed == record.start_time_ticks
            else DetachedProcessState.STALE
        )
        return DetachedProcessStatus(state, record, observed)

    def remove(self) -> None:
        self.path.unlink(missing_ok=True)


def probe_user_systemd(
    *,
    username: str,
    runner: CommandRunner = run_command,
) -> UserSystemdProbe:
    manager = runner(("systemctl", "--user", "show-environment"))
    if manager.returncode != 0:
        return UserSystemdProbe(
            available=False,
            linger=LingerState.UNKNOWN,
            detail=manager.stderr.strip() or manager.stdout.strip() or None,
        )

    linger_result = runner(
        ("loginctl", "show-user", username, "-p", "Linger", "--value")
    )
    if linger_result.returncode != 0:
        linger = LingerState.UNKNOWN
        detail = linger_result.stderr.strip() or linger_result.stdout.strip() or None
    else:
        linger = _parse_linger(linger_result.stdout)
        detail = None
    return UserSystemdProbe(available=True, linger=linger, detail=detail)


def choose_background_mode(
    requested: BackgroundMode | str,
    systemd: UserSystemdProbe,
) -> BackgroundMode:
    try:
        mode = BackgroundMode(requested)
    except ValueError as exc:
        raise LifecycleError(f"unsupported background mode: {requested}") from exc
    if mode is BackgroundMode.AUTO:
        return BackgroundMode.SYSTEMD_USER if systemd.available else BackgroundMode.DETACHED
    if mode is BackgroundMode.SYSTEMD_USER and not systemd.available:
        raise LifecycleError(
            f"systemd --user is unavailable: {systemd.detail or 'user manager not running'}"
        )
    return mode


def render_systemd_user_unit(
    python: Path,
    *,
    daemon_args: tuple[str, ...] = (),
    cpu_quota_percent: int = 100,
    environment: Mapping[str, str] | None = None,
) -> str:
    if not python.is_absolute():
        raise LifecycleError("the systemd unit requires an absolute Python path")
    if not 1 <= cpu_quota_percent <= 100:
        raise LifecycleError("cpu_quota_percent must be between 1 and 100")
    command = (
        _systemd_quote(str(python), always=True),
        "-m",
        "watchgpu.cli",
        "daemon",
        "--foreground",
        *(_systemd_quote(argument) for argument in daemon_args),
    )
    environment_lines = tuple(
        f"Environment={_systemd_quote(f'{key}={value}', always=True)}"
        for key, value in sorted((environment or {}).items())
    )
    return "\n".join(
        (
            "[Unit]",
            "Description=WatchGPU user daemon",
            "After=default.target",
            "",
            "[Service]",
            "Type=simple",
            f"ExecStart={' '.join(command)}",
            "Restart=on-failure",
            "RestartSec=5s",
            f"CPUQuota={cpu_quota_percent}%",
            "Environment=OMP_NUM_THREADS=1",
            "Environment=MKL_NUM_THREADS=1",
            "Environment=OPENBLAS_NUM_THREADS=1",
            "Environment=NUMEXPR_NUM_THREADS=1",
            *environment_lines,
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        )
    )


def _parse_linger(output: str) -> LingerState:
    value = output.strip().lower()
    if value.startswith("linger="):
        value = value.partition("=")[2]
    if value == "yes":
        return LingerState.YES
    if value == "no":
        return LingerState.NO
    return LingerState.UNKNOWN


_SYSTEMD_SAFE_ARGUMENT = re.compile(r"^[A-Za-z0-9_@+=:,./-]+$")


def _systemd_quote(value: str, *, always: bool = False) -> str:
    if "\x00" in value or "\n" in value or "\r" in value:
        raise LifecycleError("systemd arguments cannot contain NUL or line breaks")
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
    if not always and _SYSTEMD_SAFE_ARGUMENT.fullmatch(escaped):
        return escaped
    return f'"{escaped}"'


def read_process_start_time_ticks(
    pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> int | None:
    if pid <= 0:
        return None
    try:
        stat_line = (proc_root / str(pid) / "stat").read_text(encoding="utf-8")
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None
    _prefix, separator, suffix = stat_line.rpartition(")")
    if not separator:
        return None
    fields = suffix.split()
    try:
        return int(fields[19])
    except (IndexError, ValueError):
        return None


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=True, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(0o600)
        os.replace(temporary_path, path)
        directory_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _required_string(value: dict[object, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise LifecycleDataError(f"{key} must be a non-empty string")
    return item


def _required_positive_int(value: dict[object, object], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise LifecycleDataError(f"{key} must be a positive integer")
    return item


def _optional_number(value: dict[object, object], key: str) -> float | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, (int, float)) or isinstance(item, bool):
        raise LifecycleDataError(f"{key} must be a number")
    return float(item)


def _optional_positive_int(
    value: dict[object, object], key: str, *, default: int
) -> int:
    item = value.get(key, default)
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise LifecycleDataError(f"{key} must be a positive integer")
    return item


def _optional_positive_int_or_none(
    value: dict[object, object], key: str
) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
        raise LifecycleDataError(f"{key} must be a positive integer or null")
    return item


def _optional_positive_number(
    value: dict[object, object], key: str, *, default: float
) -> float:
    item = value.get(key, default)
    if not isinstance(item, (int, float)) or isinstance(item, bool) or item <= 0:
        raise LifecycleDataError(f"{key} must be a positive number")
    return float(item)
