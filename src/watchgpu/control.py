from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock

from watchgpu.config import WatchGPUConfig, save_config


class ApplyStatus(StrEnum):
    APPLIED = "APPLIED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


@dataclass(frozen=True, slots=True)
class ConfigApplyResult:
    status: ApplyStatus
    revision: int
    config: WatchGPUConfig
    reason: str | None = None


class RuntimeConfigController:
    """Own revision checks, runtime application, and optional persistence."""

    def __init__(
        self,
        config: WatchGPUConfig,
        *,
        apply: Callable[[WatchGPUConfig], ApplyStatus],
        config_path: Path | None = None,
        normalize: Callable[[WatchGPUConfig], WatchGPUConfig] | None = None,
    ) -> None:
        self._config = config.model_copy(deep=True)
        self._apply_callback = apply
        self._config_path = config_path
        self._normalize = normalize
        self._revision = 0
        self._runtime_status = ApplyStatus.APPLIED
        self._runtime_reason: str | None = None
        self._pending_save = False
        self._lock = Lock()

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def config(self) -> WatchGPUConfig:
        with self._lock:
            return self._config.model_copy(deep=True)

    @property
    def runtime_status(self) -> ApplyStatus:
        with self._lock:
            return self._runtime_status

    @property
    def runtime_reason(self) -> str | None:
        with self._lock:
            return self._runtime_reason

    def mark_runtime_status(
        self, status: ApplyStatus, *, reason: str | None = None
    ) -> None:
        if status is ApplyStatus.REJECTED:
            raise ValueError("a committed runtime config cannot become REJECTED")
        with self._lock:
            if status is ApplyStatus.APPLIED and self._pending_save:
                if self._config_path is None:
                    raise RuntimeError("pending config has no persistence path")
                save_config(self._config, self._config_path)
                self._pending_save = False
            self._runtime_status = status
            self._runtime_reason = reason

    def apply(
        self,
        candidate: WatchGPUConfig,
        *,
        expected_revision: int,
        save: bool,
    ) -> ConfigApplyResult:
        with self._lock:
            if expected_revision != self._revision:
                return self._result(
                    ApplyStatus.REJECTED,
                    reason=(
                        f"revision conflict: expected {expected_revision}, "
                        f"current {self._revision}"
                    ),
                )

            validated = WatchGPUConfig.model_validate(candidate.model_dump())
            if self._normalize is not None:
                validated = self._normalize(validated.model_copy(deep=True))
            config_path = self._config_path
            if save and config_path is None:
                return self._result(
                    ApplyStatus.REJECTED,
                    reason="no config path is available for persistence",
                )
            try:
                status = self._apply_callback(validated.model_copy(deep=True))
            except (RuntimeError, ValueError) as exc:
                return self._result(ApplyStatus.REJECTED, reason=str(exc))
            if status is ApplyStatus.REJECTED:
                return self._result(status, reason="runtime rejected the candidate config")

            if save and status is ApplyStatus.APPLIED:
                assert config_path is not None
                try:
                    save_config(validated, config_path)
                except Exception as exc:
                    try:
                        rollback = self._apply_callback(
                            self._config.model_copy(deep=True)
                        )
                    except Exception as rollback_exc:
                        raise RuntimeError(
                            f"config persistence failed ({exc}); runtime rollback "
                            f"also failed ({rollback_exc})"
                        ) from rollback_exc
                    if rollback is not ApplyStatus.APPLIED:
                        raise RuntimeError(
                            f"config persistence failed ({exc}); runtime rollback "
                            f"returned {rollback.value}"
                        ) from exc
                    return self._result(
                        ApplyStatus.REJECTED,
                        reason=f"config persistence failed; runtime was rolled back: {exc}",
                    )
            self._config = validated.model_copy(deep=True)
            self._revision += 1
            self._runtime_status = status
            self._runtime_reason = (
                "waiting for active leases or worker-safe reconciliation"
                if status is ApplyStatus.PENDING
                else None
            )
            self._pending_save = save and status is ApplyStatus.PENDING
            return self._result(status, reason=self._runtime_reason)

    def _result(
        self, status: ApplyStatus, *, reason: str | None = None
    ) -> ConfigApplyResult:
        return ConfigApplyResult(
            status=status,
            revision=self._revision,
            config=self._config.model_copy(deep=True),
            reason=reason,
        )
