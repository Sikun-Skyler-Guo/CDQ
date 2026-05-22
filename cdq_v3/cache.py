from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional


class PromptCache:
    """SQLite-backed cache keyed by serialized prompt payload."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "cache_key TEXT PRIMARY KEY,"
            "payload TEXT NOT NULL,"
            "response TEXT NOT NULL,"
            "ts REAL NOT NULL)"
        )
        self._conn.commit()

    @staticmethod
    def _normalize_payload(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, ensure_ascii=True)

    @staticmethod
    def _hash(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        serialized = self._normalize_payload(payload)
        cache_key = self._hash(serialized)
        with self._lock:
            cur = self._conn.execute(
                "SELECT response FROM cache WHERE cache_key = ?", (cache_key,)
            )
            row = cur.fetchone()
        if not row:
            return None
        return json.loads(row[0])

    def set(self, payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        serialized = self._normalize_payload(payload)
        cache_key = self._hash(serialized)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache(cache_key, payload, response, ts) VALUES(?,?,?,?)",
                (cache_key, serialized, json.dumps(response, ensure_ascii=True), time.time()),
            )
            self._conn.commit()
