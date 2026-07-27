from __future__ import annotations

import json
from pathlib import Path

from watchgpu.journal import EventJournal


def test_event_journal_writes_structured_json_line(tmp_path: Path) -> None:
    journal = EventJournal(tmp_path / "watchgpu.log")
    journal.append({"sequence": 1, "type": "LEASE_APPROVED", "message": "ok"})
    journal.close()

    assert json.loads((tmp_path / "watchgpu.log").read_text()) == {
        "sequence": 1,
        "type": "LEASE_APPROVED",
        "message": "ok",
    }
