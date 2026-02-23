from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    """Low-level SQLite helper.

    One connection per operation for simplicity.
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def ensure_schema(self) -> None:
        with self.connect() as con:
            con.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    folder TEXT
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            con.execute("""
                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            con.commit()

            cols = {r["name"] for r in con.execute("PRAGMA table_info(experiments)").fetchall()}
            if "folder" not in cols:
                con.execute("ALTER TABLE experiments ADD COLUMN folder TEXT")
                con.commit()
