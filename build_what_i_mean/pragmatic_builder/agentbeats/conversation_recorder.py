from __future__ import annotations

import datetime as dt
from pathlib import Path
from threading import Lock


class ConversationRecorder:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = Lock()
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, line: str) -> None:
        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
        entry = f"[{timestamp}] {line}\n"
        with self._lock:
            self._path.open("a", encoding="utf-8").write(entry)
