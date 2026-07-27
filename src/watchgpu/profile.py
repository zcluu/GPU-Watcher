from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TypeAlias, TypeGuard

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


class ProfileStoreError(RuntimeError):
    pass


class ProfileOutcome(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class ProfileRecord:
    fingerprint: str
    task_name: str
    outcome: ProfileOutcome
    world_size: int
    observed_peak_mib_by_gpu: dict[str, int]
    recommended_memory_per_gpu_mib: int | None
    exit_code: int | None
    recorded_at: str
    error: str | None = None
    metadata: dict[str, JsonValue] = field(default_factory=dict)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "fingerprint": self.fingerprint,
            "task_name": self.task_name,
            "outcome": self.outcome.value,
            "world_size": self.world_size,
            "observed_peak_mib_by_gpu": dict(self.observed_peak_mib_by_gpu),
            "recommended_memory_per_gpu_mib": self.recommended_memory_per_gpu_mib,
            "exit_code": self.exit_code,
            "recorded_at": self.recorded_at,
            "error": self.error,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: object) -> ProfileRecord:
        if not isinstance(value, Mapping):
            raise ProfileStoreError("profile record must be an object")
        raw_peaks = value.get("observed_peak_mib_by_gpu")
        if not isinstance(raw_peaks, Mapping):
            raise ProfileStoreError("profile peaks must be an object")
        peaks: dict[str, int] = {}
        for gpu_uuid, peak in raw_peaks.items():
            if not isinstance(gpu_uuid, str) or not _is_int(peak) or peak < 0:
                raise ProfileStoreError("profile peaks contain an invalid entry")
            peaks[gpu_uuid] = peak

        raw_metadata = value.get("metadata", {})
        normalized_metadata = _normalize_json(raw_metadata)
        if not isinstance(normalized_metadata, dict):
            raise ProfileStoreError("profile metadata must be an object")
        try:
            outcome = ProfileOutcome(_required_string(value, "outcome"))
        except ValueError as exc:
            raise ProfileStoreError("profile outcome is invalid") from exc
        return cls(
            fingerprint=_required_string(value, "fingerprint"),
            task_name=_required_string(value, "task_name"),
            outcome=outcome,
            world_size=_required_positive_int(value, "world_size"),
            observed_peak_mib_by_gpu=peaks,
            recommended_memory_per_gpu_mib=_optional_nonnegative_int(
                value, "recommended_memory_per_gpu_mib"
            ),
            exit_code=_optional_int(value, "exit_code"),
            recorded_at=_required_string(value, "recorded_at"),
            error=_optional_string(value, "error"),
            metadata=normalized_metadata,
        )


class ProfileStore:
    """Append-only local record of successful and failed training observations."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def records(self) -> tuple[ProfileRecord, ...]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return ()
        records: list[ProfileRecord] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                decoded = json.loads(line)
                records.append(ProfileRecord.from_dict(decoded))
            except (ValueError, TypeError, ProfileStoreError) as exc:
                raise ProfileStoreError(
                    f"invalid profile record at {self.path}:{line_number}: {exc}"
                ) from exc
        return tuple(records)

    def recommend(self, fingerprint: str) -> int | None:
        recommendations = [
            record.recommended_memory_per_gpu_mib
            for record in self.records()
            if record.fingerprint == fingerprint
            and record.outcome is ProfileOutcome.SUCCESS
            and record.recommended_memory_per_gpu_mib is not None
        ]
        return max(recommendations, default=None)

    def record_success(
        self,
        *,
        fingerprint: str,
        task_name: str,
        world_size: int,
        observed_peak_mib_by_gpu: Mapping[str, int],
        exit_code: int = 0,
        metadata: Mapping[str, object] | None = None,
    ) -> ProfileRecord:
        peaks = _validated_peaks(observed_peak_mib_by_gpu, require_values=True)
        recommendation = recommended_memory_mib(max(peaks.values()))
        record = ProfileRecord(
            fingerprint=_nonempty(fingerprint, "fingerprint"),
            task_name=_nonempty(task_name, "task_name"),
            outcome=ProfileOutcome.SUCCESS,
            world_size=_positive(world_size, "world_size"),
            observed_peak_mib_by_gpu=peaks,
            recommended_memory_per_gpu_mib=recommendation,
            exit_code=exit_code,
            recorded_at=datetime.now(UTC).isoformat(),
            metadata=_metadata(metadata),
        )
        self._append(record)
        return record

    def record_failure(
        self,
        *,
        fingerprint: str,
        task_name: str,
        world_size: int,
        observed_peak_mib_by_gpu: Mapping[str, int],
        exit_code: int | None,
        error: str,
        metadata: Mapping[str, object] | None = None,
    ) -> ProfileRecord:
        record = ProfileRecord(
            fingerprint=_nonempty(fingerprint, "fingerprint"),
            task_name=_nonempty(task_name, "task_name"),
            outcome=ProfileOutcome.FAILED,
            world_size=_positive(world_size, "world_size"),
            observed_peak_mib_by_gpu=_validated_peaks(
                observed_peak_mib_by_gpu, require_values=False
            ),
            recommended_memory_per_gpu_mib=None,
            exit_code=exit_code,
            recorded_at=datetime.now(UTC).isoformat(),
            error=_nonempty(error, "error"),
            metadata=_metadata(metadata),
        )
        self._append(record)
        return record

    def _append(self, record: ProfileRecord) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        payload = json.dumps(
            record.to_dict(),
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.path.open("a", encoding="utf-8") as destination:
            fcntl.flock(destination.fileno(), fcntl.LOCK_EX)
            destination.write(payload + "\n")
            destination.flush()
            os.fsync(destination.fileno())
            fcntl.flock(destination.fileno(), fcntl.LOCK_UN)


def recommended_memory_mib(
    observed_peak_mib: int,
    *,
    margin_ratio: float = 0.15,
    minimum_margin_mib: int = 1024,
    chunk_mib: int = 500,
) -> int:
    """Return a conservative per-GPU request for an observed peak."""

    if observed_peak_mib < 0:
        raise ValueError("observed_peak_mib cannot be negative")
    if margin_ratio < 0:
        raise ValueError("margin_ratio cannot be negative")
    if minimum_margin_mib <= 0:
        raise ValueError("minimum_margin_mib must be positive")
    if chunk_mib <= 0:
        raise ValueError("chunk_mib must be positive")

    margin_mib = max(minimum_margin_mib, math.ceil(observed_peak_mib * margin_ratio))
    requested_mib = observed_peak_mib + margin_mib
    return math.ceil(requested_mib / chunk_mib) * chunk_mib


def build_profile_fingerprint(
    metadata: Mapping[str, object], *, config_files: Sequence[Path] = ()
) -> str:
    """Hash the effective training identity and referenced file contents."""

    files: list[JsonValue] = []
    for path in sorted((item.expanduser().resolve() for item in config_files), key=str):
        files.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    payload: JsonValue = {
        "metadata": _normalize_json(dict(metadata)),
        "config_files": files,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _normalize_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        normalized: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("profile metadata keys must be strings")
            normalized[key] = _normalize_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    raise TypeError(f"unsupported profile metadata value: {type(value).__name__}")


def _metadata(value: Mapping[str, object] | None) -> dict[str, JsonValue]:
    normalized = _normalize_json(dict(value or {}))
    if not isinstance(normalized, dict):
        raise TypeError("profile metadata must be an object")
    return normalized


def _validated_peaks(
    value: Mapping[str, int], *, require_values: bool
) -> dict[str, int]:
    if require_values and not value:
        raise ValueError("successful profiles require at least one GPU peak")
    peaks: dict[str, int] = {}
    for gpu_uuid, peak_mib in value.items():
        if not gpu_uuid:
            raise ValueError("GPU UUID cannot be empty")
        if isinstance(peak_mib, bool) or peak_mib < 0:
            raise ValueError("GPU peak cannot be negative")
        peaks[gpu_uuid] = peak_mib
    return peaks


def _required_string(value: Mapping[object, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ProfileStoreError(f"{key} must be a non-empty string")
    return item


def _optional_string(value: Mapping[object, object], key: str) -> str | None:
    item = value.get(key)
    if item is None:
        return None
    if not isinstance(item, str):
        raise ProfileStoreError(f"{key} must be a string")
    return item


def _required_positive_int(value: Mapping[object, object], key: str) -> int:
    item = value.get(key)
    if not _is_int(item) or item <= 0:
        raise ProfileStoreError(f"{key} must be a positive integer")
    return item


def _optional_int(value: Mapping[object, object], key: str) -> int | None:
    item = value.get(key)
    if item is None:
        return None
    if not _is_int(item):
        raise ProfileStoreError(f"{key} must be an integer")
    return item


def _optional_nonnegative_int(value: Mapping[object, object], key: str) -> int | None:
    item = _optional_int(value, key)
    if item is not None and item < 0:
        raise ProfileStoreError(f"{key} cannot be negative")
    return item


def _is_int(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"{name} cannot be empty")
    return value


def _positive(value: int, name: str) -> int:
    if isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
