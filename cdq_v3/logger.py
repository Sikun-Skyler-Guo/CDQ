from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class StructuredLogger:
    """JSONL logger for debugging and ad-hoc analysis."""

    def __init__(self, output_dir: Path, run_name: str | None = None, filename: str | None = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_stem = run_name or f"run_{timestamp}"
        if filename:
            self.path = self.output_dir / filename
        else:
            self.path = self.output_dir / f"{log_stem}.jsonl"
        self._lock = threading.Lock()

    def log(self, event: str, **fields: Any) -> None:
        record: Dict[str, Any] = {
            "ts": time.time(),
            "event": event,
        }
        record.update({k: v for k, v in fields.items() if v is not None})
        line = json.dumps(record, ensure_ascii=True)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
