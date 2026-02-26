from __future__ import annotations

import sqlite3
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, ddl: str) -> None:
    """Add a column to an existing table if it is missing.

    SQLite cannot `ALTER TABLE .. ADD COLUMN` with non-constant defaults
    (e.g. `DEFAULT CURRENT_TIMESTAMP`). If such a default is requested, we add the
    column without the default and backfill existing rows.
    """

    if not _table_exists(conn, table) or _column_exists(conn, table, column):
        return

    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")
        return
    except sqlite3.OperationalError as e:
        # SQLite cannot add a column with a non-constant default (e.g. DEFAULT CURRENT_TIMESTAMP)
        # via ALTER TABLE ... ADD COLUMN. In that case, add the column without the default and
        # backfill existing rows.
        msg = str(e).lower()
        if "non-constant default" not in msg:
            raise

    # Strip DEFAULT CURRENT_TIMESTAMP (and similar) from the ddl for ALTER TABLE.
    ddl_no_default = re.sub(r"\s+DEFAULT\s+CURRENT_TIMESTAMP\b", "", ddl, flags=re.IGNORECASE)
    ddl_no_default = re.sub(r"\s+DEFAULT\s+CURRENT_DATE\b", "", ddl_no_default, flags=re.IGNORECASE)
    ddl_no_default = re.sub(r"\s+DEFAULT\s+CURRENT_TIME\b", "", ddl_no_default, flags=re.IGNORECASE)

    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl_no_default}")

    # Backfill values for existing rows.
    if re.search(r"DEFAULT\s+CURRENT_TIMESTAMP\b", ddl, flags=re.IGNORECASE):
        conn.execute(f"UPDATE {table} SET {column} = COALESCE({column}, CURRENT_TIMESTAMP)")
    elif re.search(r"DEFAULT\s+CURRENT_DATE\b", ddl, flags=re.IGNORECASE):
        conn.execute(f"UPDATE {table} SET {column} = COALESCE({column}, CURRENT_DATE)")
    elif re.search(r"DEFAULT\s+CURRENT_TIME\b", ddl, flags=re.IGNORECASE):
        conn.execute(f"UPDATE {table} SET {column} = COALESCE({column}, CURRENT_TIME)")


class Database:
    """Tiny SQLite helper.

    Repositories own the SQL; this only handles connections.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        ensure_schema(self)


def ensure_schema(db: Database) -> None:
    """Create/migrate database schema (idempotent)."""

    with db.connect() as conn:
        # --- global metadata ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # experiments
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                folder TEXT DEFAULT '',
                description TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # keep experiment names indexed; try to enforce uniqueness if possible
        # (if the existing DB already has duplicate names, a UNIQUE index would fail)
        try:
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)"
            )
        except sqlite3.IntegrityError:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_name ON experiments(name)")

        # legacy settings table (kept for backward compatibility)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )

        # legacy excel blob storage (kept for backward compatibility)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS excel_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL UNIQUE,
                filename TEXT,
                content BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # current file storage used by repositories
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                content BLOB NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # legacy processed tables cache (kept)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS processed_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                cache_key TEXT NOT NULL,
                table_name TEXT NOT NULL,
                df_json TEXT NOT NULL,
                text_md TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(experiment_id, cache_key, table_name),
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # current processed tables cache used by repositories
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_processed_tables (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                cache_key TEXT NOT NULL,
                table_name TEXT NOT NULL,
                df_json TEXT NOT NULL,
                text_md TEXT DEFAULT '',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(experiment_id, cache_key, table_name),
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # legacy computations table (kept)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS computations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                used_columns_json TEXT NOT NULL,
                graph_state_json TEXT NOT NULL,
                ode_text TEXT DEFAULT '',
                state_names_json TEXT DEFAULT '[]',
                param_names_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(experiment_id, name),
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # current computations used by the builder
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_computations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                used_heads_json TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                edge_modes_json TEXT NOT NULL,
                ode_text TEXT DEFAULT '',
                state_names_json TEXT DEFAULT '[]',
                param_names_json TEXT DEFAULT '[]',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(experiment_id, name),
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # --- processing tab persistent settings per experiment ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_processing_settings (
                experiment_id INTEGER PRIMARY KEY,
                initialization TEXT DEFAULT 'TIME_SHIFT',
                t_shift REAL DEFAULT 6.0,
                optim_time_shift INTEGER DEFAULT 0,
                models_to_compute_json TEXT DEFAULT '[]',
                t_max REAL DEFAULT 400.0,
                t_max_plot REAL DEFAULT 400.0,
                last_auto_t_shift REAL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        # --- processing results (grouped by run) ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_processing_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                settings_json TEXT NOT NULL,
                used_t_shift REAL,
                auto_t_shift REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_processing_run_models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                constants_json TEXT NOT NULL,
                plot_png BLOB,
                plot_error TEXT,
                FOREIGN KEY(run_id) REFERENCES experiment_processing_runs(id) ON DELETE CASCADE,
                UNIQUE(run_id, model_name)
            )
            """
        )

        # --- graph tab persistent settings per experiment/model ---
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiment_graph_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                config_json TEXT NOT NULL,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(experiment_id) REFERENCES experiments(id) ON DELETE CASCADE,
                UNIQUE(experiment_id, model_name)
            )
            """
        )

        # migrations for older DBs
        _add_column_if_missing(conn, "experiments", "folder", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "experiments", "description", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "experiments", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        _add_column_if_missing(conn, "experiments", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        _add_column_if_missing(conn, "excel_files", "filename", "TEXT")
        _add_column_if_missing(conn, "excel_files", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        _add_column_if_missing(conn, "excel_files", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        _add_column_if_missing(conn, "processed_tables", "text_md", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "processed_tables", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        _add_column_if_missing(conn, "processed_tables", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        _add_column_if_missing(conn, "experiment_processed_tables", "text_md", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "experiment_processed_tables", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        _add_column_if_missing(conn, "experiment_processed_tables", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        _add_column_if_missing(conn, "computations", "ode_text", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "computations", "state_names_json", "TEXT DEFAULT '[]'")
        _add_column_if_missing(conn, "computations", "param_names_json", "TEXT DEFAULT '[]'")
        _add_column_if_missing(conn, "computations", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        _add_column_if_missing(conn, "computations", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        _add_column_if_missing(conn, "experiment_computations", "ode_text", "TEXT DEFAULT ''")
        _add_column_if_missing(conn, "experiment_computations", "state_names_json", "TEXT DEFAULT '[]'")
        _add_column_if_missing(conn, "experiment_computations", "param_names_json", "TEXT DEFAULT '[]'")
        _add_column_if_missing(conn, "experiment_computations", "created_at", "TEXT DEFAULT CURRENT_TIMESTAMP")
        _add_column_if_missing(conn, "experiment_computations", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        # processing settings migrations (older DBs won't have the table)
        if _table_exists(conn, "experiment_processing_settings"):
            _add_column_if_missing(conn, "experiment_processing_settings", "initialization", "TEXT DEFAULT 'TIME_SHIFT'")
            _add_column_if_missing(conn, "experiment_processing_settings", "t_shift", "REAL DEFAULT 6.0")
            _add_column_if_missing(conn, "experiment_processing_settings", "optim_time_shift", "INTEGER DEFAULT 0")
            _add_column_if_missing(conn, "experiment_processing_settings", "models_to_compute_json", "TEXT DEFAULT '[]'")
            _add_column_if_missing(conn, "experiment_processing_settings", "t_max", "REAL DEFAULT 400.0")
            _add_column_if_missing(conn, "experiment_processing_settings", "t_max_plot", "REAL DEFAULT 400.0")
            _add_column_if_missing(conn, "experiment_processing_settings", "last_auto_t_shift", "REAL")
            _add_column_if_missing(conn, "experiment_processing_settings", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        # graph settings migrations (older DBs won't have the table)
        if _table_exists(conn, "experiment_graph_settings"):
            _add_column_if_missing(conn, "experiment_graph_settings", "updated_at", "TEXT DEFAULT CURRENT_TIMESTAMP")

        conn.commit()
