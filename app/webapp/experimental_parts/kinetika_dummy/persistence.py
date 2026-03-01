import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


def _utc_ts() -> float:
    return time.time()


@dataclass
class StepEvent:
    id: int
    ts: float
    tab: str
    event_type: str
    payload_json: str


class ConfigDB:
    """
    SQLite úložiště:
      - current_config (1 řádek): poslední config JSON
      - config_versions: audit trail změn configu
      - steps: audit trail user kroků/událostí
      - blobs: binární/text data (např. uložené FAME/EPO tabulky)
    """

    def __init__(self, db_path: str = "kinetika_config.sqlite") -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS current_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    updated_ts REAL NOT NULL,
                    config_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS config_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    config_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS steps (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    tab TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    key TEXT PRIMARY KEY,
                    updated_ts REAL NOT NULL,
                    mime TEXT NOT NULL,
                    data BLOB NOT NULL
                )
                """
            )
            self._conn.commit()

            # ensure current_config row exists
            cur.execute("SELECT id FROM current_config WHERE id = 1")
            row = cur.fetchone()
            if row is None:
                empty = {}
                cur.execute(
                    "INSERT INTO current_config (id, updated_ts, config_json) VALUES (1, ?, ?)",
                    (_utc_ts(), json.dumps(empty)),
                )
                cur.execute(
                    "INSERT INTO config_versions (ts, config_json) VALUES (?, ?)",
                    (_utc_ts(), json.dumps(empty)),
                )
                self._conn.commit()

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT config_json FROM current_config WHERE id = 1")
            row = cur.fetchone()
            if not row:
                return {}
            try:
                return json.loads(row["config_json"])
            except Exception:
                return {}

    def save_config(self, config: Dict[str, Any]) -> None:
        """
        Uloží current_config + vytvoří novou verzi v config_versions.
        """
        with self._lock:
            ts = _utc_ts()
            config_json = json.dumps(config, ensure_ascii=False)
            cur = self._conn.cursor()
            cur.execute(
                "UPDATE current_config SET updated_ts = ?, config_json = ? WHERE id = 1",
                (ts, config_json),
            )
            cur.execute(
                "INSERT INTO config_versions (ts, config_json) VALUES (?, ?)",
                (ts, config_json),
            )
            self._conn.commit()

    def set_path(self, path: Tuple[str, ...], value: Any) -> Dict[str, Any]:
        """
        Nastaví config na dané cestě (např. ("input", "fame", "row_start")).
        Okamžitě perzistuje do DB (včetně verze).
        Vrací nový config dict.
        """
        with self._lock:
            cfg = self.get_config()
            d = cfg
            for k in path[:-1]:
                if k not in d or not isinstance(d[k], dict):
                    d[k] = {}
                d = d[k]
            d[path[-1]] = value
            self.save_config(cfg)
            return cfg

    def get_path(self, path: Tuple[str, ...], default: Any = None) -> Any:
        cfg = self.get_config()
        d: Any = cfg
        for k in path:
            if not isinstance(d, dict) or k not in d:
                return default
            d = d[k]
        return d

    def log_step(self, tab: str, event_type: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO steps (ts, tab, event_type, payload_json) VALUES (?, ?, ?, ?)",
                (_utc_ts(), tab, event_type, json.dumps(payload, ensure_ascii=False)),
            )
            self._conn.commit()

    def list_steps(self, limit: int = 200) -> list[StepEvent]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT id, ts, tab, event_type, payload_json FROM steps ORDER BY id DESC LIMIT ?",
                (limit,),
            )
            rows = cur.fetchall()
            return [
                StepEvent(
                    id=int(r["id"]),
                    ts=float(r["ts"]),
                    tab=str(r["tab"]),
                    event_type=str(r["event_type"]),
                    payload_json=str(r["payload_json"]),
                )
                for r in rows
            ]

    def put_blob(self, key: str, data: bytes, mime: str = "application/octet-stream") -> None:
        with self._lock:
            ts = _utc_ts()
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO blobs (key, updated_ts, mime, data)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET updated_ts = excluded.updated_ts, mime = excluded.mime, data = excluded.data
                """,
                (key, ts, mime, data),
            )
            self._conn.commit()

    def get_blob(self, key: str) -> Optional[Tuple[bytes, str]]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT data, mime FROM blobs WHERE key = ?", (key,))
            row = cur.fetchone()
            if not row:
                return None
            return (bytes(row["data"]), str(row["mime"]))
