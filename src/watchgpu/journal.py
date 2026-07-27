from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class EventJournal:
    """Append structured local events with bounded user-space log rotation."""

    def __init__(
        self,
        path: Path,
        *,
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 3,
    ) -> None:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = path
        self._logger = logging.Logger(f"watchgpu-journal-{id(self)}")
        self._logger.propagate = False
        self._handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._handler)

    def append(self, event: Mapping[str, Any]) -> None:
        self._logger.info(
            json.dumps(
                dict(event),
                ensure_ascii=True,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )

    def close(self) -> None:
        self._handler.flush()
        self._handler.close()
        self._logger.removeHandler(self._handler)
