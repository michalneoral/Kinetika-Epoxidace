from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import List, Optional

from experimental_web.core.paths import EXPERIMENTS_DIR
from experimental_web.core.time import utc_now_iso
from experimental_web.data.database import Database
from experimental_web.data.models import Experiment, ExperimentFile


def _safe_slug(text: str) -> str:
    import re
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_\-]+", "", t)
    return (t[:40] or "experiment")


def _normalize_edge_mode(value) -> int:
    """Normalize edge mode values coming from the UI into small ints.

    The graph editor can hand us bools, ints, floats, strings, or None.
    We normalize to an integer mode:

    - 0 = disabled
    - 1 = enabled (active)
    - 2 = reserved for future use
    """

    if value is None:
        return 0

    # bool must be checked before int, because bool is a subclass of int
    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, (int, float)):
        try:
            v = int(value)
        except Exception:
            return 0
        return 0 if v <= 0 else (2 if v >= 2 else 1)

    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "0", "false", "off", "no", "disabled"}:
            return 0
        if s in {"1", "true", "on", "yes", "enabled"}:
            return 1
        if s.isdigit():
            v = int(s)
            return 0 if v <= 0 else (2 if v >= 2 else 1)
        return 0

    return 0


class ExperimentRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def exists_name(self, name: str) -> bool:
        with self.db.connect() as con:
            row = con.execute("SELECT 1 FROM experiments WHERE name=? LIMIT 1", (name,)).fetchone()
        return row is not None

    def create(self, name: str) -> Experiment:
        now = utc_now_iso()
        with self.db.connect() as con:
            cur = con.execute(
                "INSERT INTO experiments(name, created_at, updated_at) VALUES(?, ?, ?)",
                (name, now, now),
            )
            exp_id = int(cur.lastrowid)
            con.commit()

        folder = EXPERIMENTS_DIR / f"{exp_id:06d}_{_safe_slug(name)}"
        folder.mkdir(parents=True, exist_ok=True)

        with self.db.connect() as con:
            con.execute("UPDATE experiments SET folder=? WHERE id=?", (str(folder), exp_id))
            con.commit()

        exp = self.get(exp_id)
        assert exp is not None
        return exp

    def touch(self, exp_id: int) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute("UPDATE experiments SET updated_at=? WHERE id=?", (now, exp_id))
            con.commit()

    def duplicate(self, source_id: int, new_name: str) -> Experiment:
        src = self.get(source_id)
        if not src:
            raise ValueError("Source experiment not found")
        if self.exists_name(new_name):
            raise ValueError("Experiment with this name already exists")

        new_exp = self.create(new_name)

        # copy folder content (best effort)
        try:
            if src.folder and new_exp.folder:
                src_folder = Path(src.folder)
                dst_folder = Path(new_exp.folder)
                if src_folder.exists() and src_folder.is_dir():
                    for item in src_folder.iterdir():
                        target = dst_folder / item.name
                        if item.is_dir():
                            shutil.copytree(item, target, dirs_exist_ok=True)
                        else:
                            shutil.copy2(item, target)
        except Exception:
            pass

        # copy stored excel blobs
        try:
            ExperimentFileRepository(self.db.path).duplicate_files(source_id, new_exp.id)
        except Exception:
            pass

        return new_exp

    def rename(self, exp_id: int, new_name: str) -> Experiment:
        """Rename an experiment (and best-effort rename its folder slug)."""

        new_name = (new_name or '').strip()
        if not new_name:
            raise ValueError('Name must not be empty')
        if self.exists_name(new_name):
            raise ValueError('Experiment with this name already exists')

        exp = self.get(exp_id)
        if not exp:
            raise ValueError('Experiment not found')

        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute('UPDATE experiments SET name=?, updated_at=? WHERE id=?', (new_name, now, exp_id))
            con.commit()

        # Best-effort: rename folder to match new slug.
        try:
            if exp.folder:
                src = Path(exp.folder)
                dst = EXPERIMENTS_DIR / f"{exp_id:06d}_{_safe_slug(new_name)}"
                if src.exists() and src.is_dir() and src.resolve() != dst.resolve():
                    if not dst.exists():
                        src.rename(dst)
                        with self.db.connect() as con:
                            con.execute('UPDATE experiments SET folder=? WHERE id=?', (str(dst), exp_id))
                            con.commit()
        except Exception:
            # Folder rename is best-effort; DB name is the important part.
            pass

        updated = self.get(exp_id)
        assert updated is not None
        return updated

    def delete(self, exp_id: int, delete_folder: bool = True) -> None:
        exp = self.get(exp_id)

        # delete blobs first (FK not used)
        with self.db.connect() as con:
            con.execute("DELETE FROM experiment_files WHERE experiment_id=?", (exp_id,))
            con.commit()

        with self.db.connect() as con:
            con.execute("DELETE FROM experiments WHERE id=?", (exp_id,))
            con.commit()

        if delete_folder and exp and exp.folder:
            p = Path(exp.folder)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    def list(self, limit: Optional[int] = None) -> List[Experiment]:
        sql = "SELECT id, name, created_at, updated_at, folder FROM experiments ORDER BY updated_at DESC"
        params: tuple = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (int(limit),)

        with self.db.connect() as con:
            rows = con.execute(sql, params).fetchall()
        return [Experiment(**dict(r)) for r in rows]

    def get(self, exp_id: int) -> Optional[Experiment]:
        with self.db.connect() as con:
            row = con.execute(
                "SELECT id, name, created_at, updated_at, folder FROM experiments WHERE id=?",
                (exp_id,),
            ).fetchone()
        return Experiment(**dict(row)) if row else None


class MetaRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def get_last_experiment_id(self) -> Optional[int]:
        with self.db.connect() as con:
            row = con.execute("SELECT value FROM app_meta WHERE key='last_experiment_id'").fetchone()
        return int(row["value"]) if row and str(row["value"]).isdigit() else None

    def set_last_experiment_id(self, exp_id: int) -> None:
        """Persist last selected experiment id.

        The DB schema evolved over time; some installations have `updated_at`,
        others only `created_at` or no timestamp column at all. We therefore
        detect available columns and generate a compatible UPSERT.
        """
        now = utc_now_iso()
        with self.db.connect() as con:
            # Detect optional timestamp columns (backward compatible)
            cols = [row["name"] for row in con.execute("PRAGMA table_info(app_meta)").fetchall()]
            ts_col = "updated_at" if "updated_at" in cols else ("created_at" if "created_at" in cols else None)

            if ts_col:
                sql = f"""
                INSERT INTO app_meta(key, value, {ts_col})
                VALUES('last_experiment_id', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, {ts_col}=excluded.{ts_col}
                """
                con.execute(sql, (str(exp_id), now))
            else:
                con.execute(
                    """
                    INSERT INTO app_meta(key, value)
                    VALUES('last_experiment_id', ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (str(exp_id),),
                )
            con.commit()


class SettingsRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def _upsert(self, key: str, value: str) -> None:
        """UPSERT into user_settings with backward-compatible timestamp handling."""
        now = utc_now_iso()
        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(user_settings)").fetchall()}
            ts_col = "updated_at" if "updated_at" in cols else ("created_at" if "created_at" in cols else None)

            if ts_col:
                con.execute(
                    f"""
                    INSERT INTO user_settings(key, value, {ts_col})
                    VALUES(?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value, {ts_col}=excluded.{ts_col}
                    """,
                    (key, value, now),
                )
            else:
                con.execute(
                    """
                    INSERT INTO user_settings(key, value)
                    VALUES(?, ?)
                    ON CONFLICT(key) DO UPDATE SET value=excluded.value
                    """,
                    (key, value),
                )
            con.commit()

    def _get_raw(self, key: str) -> str | None:
        """Get a setting value.

        Reads primarily from `user_settings` and falls back to the legacy
        `settings` table if the key is missing.

        This makes imports from older DBs (or bundles containing legacy
        `settings`) work without losing configuration.
        """
        with self.db.connect() as con:
            row = con.execute("SELECT value FROM user_settings WHERE key=?", (key,)).fetchone()
            if row:
                return str(row["value"])
            # legacy fallback
            try:
                row2 = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            except Exception:
                row2 = None
            if row2:
                return str(row2["value"])
        return None

    def _get_raw_and_migrate(self, key: str) -> str | None:
        """Like `_get_raw`, but if value exists only in legacy table, migrate it."""
        with self.db.connect() as con:
            row = con.execute("SELECT value FROM user_settings WHERE key=?", (key,)).fetchone()
            if row:
                return str(row["value"])
            try:
                row2 = con.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
            except Exception:
                row2 = None
        if row2:
            val = str(row2["value"])
            # best-effort migrate into user_settings so new code sees it
            try:
                self._upsert(key, val)
            except Exception:
                pass
            return val
        return None

    def get_theme_mode(self, default: str = "auto") -> str:
        raw = self._get_raw_and_migrate('theme_mode')
        return str(raw) if raw is not None else default

    def set_theme_mode(self, mode: str) -> None:
        if mode not in ("auto", "light", "dark"):
            raise ValueError("theme_mode must be one of: auto, light, dark")
        self._upsert('theme_mode', mode)

    # --- Global help/tooltip settings ---
    def get_help_tooltip_delay_ms(self, default: int = 2000) -> int:
        """Return the tooltip delay in milliseconds.

        Convention:
        - <0 => disabled
        - 0 => show immediately
        - >0 => delay before showing
        """
        raw = self._get_raw_and_migrate('help_tooltip_delay_ms')
        if raw is None:
            return int(default)
        try:
            v = int(str(raw).strip())
        except Exception:
            return int(default)
        # allow negative value to represent "disabled"
        return v

    def set_help_tooltip_delay_ms(self, delay_ms: int) -> None:
        # allow negative values ("disabled")
        self._upsert('help_tooltip_delay_ms', str(int(delay_ms)))

    # --- Global UI colors (Quasar/NiceGUI theme palette) ---
    def get_ui_color(self, name: str, default: str) -> str:
        """Return a stored UI color or a default.

        Colors are stored as hex strings (e.g. "#1976D2").
        """
        key = f'ui_color_{name}'
        raw = self._get_raw_and_migrate(key)
        return str(raw) if raw is not None else str(default)

    def set_ui_color(self, name: str, value: str) -> None:
        key = f'ui_color_{name}'
        self._upsert(key, str(value))

    def get_ui_colors(self, defaults: dict[str, str]) -> dict[str, str]:
        """Return a dict of palette colors (merged over defaults)."""
        out = dict(defaults)
        for k in list(defaults.keys()):
            raw = self._get_raw_and_migrate(f'ui_color_{k}')
            if raw is not None:
                out[k] = str(raw)
        return out

    def set_ui_colors(self, colors: dict[str, str]) -> None:
        for k, v in colors.items():
            self.set_ui_color(k, v)


class ExperimentFileRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def delete_for_experiment(self, experiment_id: int) -> None:
        """Delete all stored files for the given experiment (we allow only one)."""
        with self.db.connect() as con:
            con.execute("DELETE FROM experiment_files WHERE experiment_id=?", (experiment_id,))
            con.commit()

    def get_single_for_experiment(self, experiment_id: int) -> Optional[ExperimentFile]:
        """Return the newest stored file for an experiment (or None)."""
        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_files)").fetchall()}
            has_uploaded_at = "uploaded_at" in cols
            has_selected_sheet = "selected_sheet" in cols
            uploaded_expr = "uploaded_at" if has_uploaded_at else ("created_at" if "created_at" in cols else "id")
            selected_expr = "selected_sheet" if has_selected_sheet else "NULL as selected_sheet"
            row = con.execute(
                f"""
                SELECT id, experiment_id, filename, content, sha256, size_bytes,
                       {uploaded_expr} as uploaded_at,
                       {selected_expr}
                FROM experiment_files
                WHERE experiment_id=?
                ORDER BY {uploaded_expr} DESC
                LIMIT 1
                """,
                (experiment_id,),
            ).fetchone()
        return ExperimentFile(**dict(row)) if row else None

    @staticmethod
    def _sha256(data: bytes) -> str:
        import hashlib
        return hashlib.sha256(data).hexdigest()

    def add_excel(self, experiment_id: int, filename: str, content: bytes) -> ExperimentFile:
        now = utc_now_iso()
        sha = self._sha256(content)
        size = len(content)

        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_files)").fetchall()}
            has_uploaded_at = "uploaded_at" in cols
            has_selected_sheet = "selected_sheet" in cols
            if has_uploaded_at and has_selected_sheet:
                cur = con.execute(
                    """
                    INSERT INTO experiment_files(experiment_id, filename, content, sha256, size_bytes, uploaded_at, selected_sheet)
                    VALUES(?, ?, ?, ?, ?, ?, NULL)
                    """,
                    (experiment_id, filename, sqlite3.Binary(content), sha, size, now),
                )
            elif has_uploaded_at:
                cur = con.execute(
                    """
                    INSERT INTO experiment_files(experiment_id, filename, content, sha256, size_bytes, uploaded_at)
                    VALUES(?, ?, ?, ?, ?, ?)
                    """,
                    (experiment_id, filename, sqlite3.Binary(content), sha, size, now),
                )
            else:
                cur = con.execute(
                    """
                    INSERT INTO experiment_files(experiment_id, filename, content, sha256, size_bytes)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (experiment_id, filename, sqlite3.Binary(content), sha, size),
                )
            file_id = int(cur.lastrowid)
            con.commit()

        ef = self.get(file_id)
        assert ef is not None
        return ef

    def list_for_experiment(self, experiment_id: int) -> List[ExperimentFile]:
        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_files)").fetchall()}
            has_uploaded_at = "uploaded_at" in cols
            has_selected_sheet = "selected_sheet" in cols
            uploaded_expr = "uploaded_at" if has_uploaded_at else ("created_at" if "created_at" in cols else "id")
            selected_expr = "selected_sheet" if has_selected_sheet else "NULL as selected_sheet"
            rows = con.execute(
                f"""
                SELECT id, experiment_id, filename, content, sha256, size_bytes,
                       {uploaded_expr} as uploaded_at,
                       {selected_expr}
                FROM experiment_files
                WHERE experiment_id=?
                ORDER BY {uploaded_expr} DESC
                """,
                (experiment_id,),
            ).fetchall()
        return [ExperimentFile(**dict(r)) for r in rows]

    def get(self, file_id: int) -> Optional[ExperimentFile]:
        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_files)").fetchall()}
            has_uploaded_at = "uploaded_at" in cols
            has_selected_sheet = "selected_sheet" in cols
            uploaded_expr = "uploaded_at" if has_uploaded_at else ("created_at" if "created_at" in cols else "id")
            selected_expr = "selected_sheet" if has_selected_sheet else "NULL as selected_sheet"
            row = con.execute(
                f"""
                SELECT id, experiment_id, filename, content, sha256, size_bytes,
                       {uploaded_expr} as uploaded_at,
                       {selected_expr}
                FROM experiment_files
                WHERE id=?
                """,
                (file_id,),
            ).fetchone()
        return ExperimentFile(**dict(row)) if row else None

    def set_selected_sheet(self, file_id: int, sheet: Optional[str]) -> None:
        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_files)").fetchall()}
            if "selected_sheet" in cols:
                con.execute("UPDATE experiment_files SET selected_sheet=? WHERE id=?", (sheet, file_id))
                con.commit()

    def delete(self, file_id: int) -> None:
        with self.db.connect() as con:
            con.execute("DELETE FROM experiment_files WHERE id=?", (file_id,))
            con.commit()

    def duplicate_files(self, source_experiment_id: int, target_experiment_id: int) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT filename, content, sha256, size_bytes, selected_sheet
                FROM experiment_files
                WHERE experiment_id=?
                """,
                (source_experiment_id,),
            ).fetchall()

            for r in rows:
                con.execute(
                    """
                    INSERT INTO experiment_files(experiment_id, filename, content, sha256, size_bytes, uploaded_at, selected_sheet)
                    VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        target_experiment_id,
                        r["filename"],
                        r["content"],
                        r["sha256"],
                        r["size_bytes"],
                        now,
                        r["selected_sheet"],
                    ),
                )
            con.commit()


class TablePickRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def get(self, experiment_id: int, kind: str):
        """Return stored pick range for an experiment.

        Always returns row_start/row_end/col_start/col_end.
        When available in schema, also includes created_at/updated_at.
        """
        with self.db.connect() as con:
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_table_picks)").fetchall()}
            extra_cols = []
            if "created_at" in cols:
                extra_cols.append("created_at")
            if "updated_at" in cols:
                extra_cols.append("updated_at")

            extra_sql = (", " + ", ".join(extra_cols)) if extra_cols else ""
            row = con.execute(
                f"""SELECT row_start, row_end, col_start, col_end{extra_sql}
                     FROM experiment_table_picks
                     WHERE experiment_id=? AND kind=?""",
                (experiment_id, kind),
            ).fetchone()
        return dict(row) if row else None

    def set(self, experiment_id: int, kind: str, row_start: int, row_end: int, col_start: int, col_end: int) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            # Avoid bumping updated_at (and triggering false "Změněno") when the
            # range is identical to the stored one.
            con.execute(
                """INSERT INTO experiment_table_picks(experiment_id, kind, row_start, row_end, col_start, col_end, updated_at)
                     VALUES(?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(experiment_id, kind)
                     DO UPDATE SET row_start=excluded.row_start,
                                   row_end=excluded.row_end,
                                   col_start=excluded.col_start,
                                   col_end=excluded.col_end,
                                   updated_at=excluded.updated_at
                     WHERE row_start!=excluded.row_start
                        OR row_end!=excluded.row_end
                        OR col_start!=excluded.col_start
                        OR col_end!=excluded.col_end""",
                (experiment_id, kind, row_start, row_end, col_start, col_end, now),
            )
            con.commit()


class ProcessedTablesRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def delete_for_experiment(self, experiment_id: int) -> None:
        with self.db.connect() as con:
            con.execute("DELETE FROM experiment_processed_tables WHERE experiment_id=?", (experiment_id,))
            con.commit()

    def list_names(self, experiment_id: int, cache_key: str) -> List[str]:
        with self.db.connect() as con:
            rows = con.execute(
                # DB schema uses `table_name`; keep API stable by aliasing to `name`
                "SELECT table_name AS name FROM experiment_processed_tables WHERE experiment_id=? AND cache_key=? ORDER BY table_name",
                (experiment_id, cache_key),
            ).fetchall()
        return [r["name"] for r in rows]

    def load_tables(self, experiment_id: int, cache_key: str) -> dict:
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT table_name AS name, df_json, text_md
                FROM experiment_processed_tables
                WHERE experiment_id=? AND cache_key=?
                """,
                (experiment_id, cache_key),
            ).fetchall()
        out = {}
        for r in rows:
            out[r["name"]] = (r["df_json"], r["text_md"] or "")
        return out




    def load_latest_tables(self, experiment_id: int) -> dict[str, tuple[str, str]]:
        """Load most recently updated table snapshots for an experiment, regardless of cache_key."""
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT t.table_name AS name, t.df_json, t.text_md
                FROM experiment_processed_tables t
                JOIN (
                    SELECT table_name, MAX(updated_at) AS mx
                    FROM experiment_processed_tables
                    WHERE experiment_id = ?
                    GROUP BY table_name
                ) latest
                ON latest.table_name = t.table_name AND latest.mx = t.updated_at
                WHERE t.experiment_id = ?
                """,
                (experiment_id, experiment_id),
            ).fetchall()
        return {str(r["name"]): (str(r["df_json"]), str(r["text_md"] or "")) for r in rows}

    def save_table(self, experiment_id: int, cache_key: str, name: str, df_json: str, text_md: str = "") -> None:
        """Upsert one processed table snapshot into DB."""
        with self.db.connect() as con:
            # DB schema uses `table_name` (older buggy code used `name`).
            cols = {r[1] for r in con.execute("PRAGMA table_info(experiment_processed_tables)").fetchall()}
            col_table = "name" if "name" in cols else "table_name"
            has_updated_at = "updated_at" in cols

            if has_updated_at:
                sql = (
                    f"INSERT OR REPLACE INTO experiment_processed_tables("
                    f"experiment_id, cache_key, {col_table}, df_json, text_md, updated_at) "
                    f"VALUES(?,?,?,?,?, datetime('now'))"
                )
            else:
                # extremely old schemas
                sql = (
                    f"INSERT OR REPLACE INTO experiment_processed_tables("
                    f"experiment_id, cache_key, {col_table}, df_json, text_md) "
                    f"VALUES(?,?,?,?,?)"
                )
            con.execute(sql, (experiment_id, cache_key, name, df_json, text_md))
            con.commit()
    
# -------------------------
# Computations (saved graphs/config) per experiment
# -------------------------

from dataclasses import dataclass
import json as _json
from typing import List as _List, Tuple as _Tuple

from experimental_web.ui.experiment.compute_builder.graph import GraphState, normalize_graph_state
from experimental_web.domain.ode_generation import generate_ode_model


@dataclass
class ExperimentComputation:
    id: int
    experiment_id: int
    name: str
    table_name: str
    used_heads: _List[str]
    graph_state: GraphState
    ode_text: str | None = None
    state_names: _List[str] | None = None
    param_names: _List[str] | None = None
    created_at: str | None = None
    updated_at: str | None = None


class ExperimentComputationRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def _graphstate_to_payload(self, gs: GraphState) -> _Tuple[str, str]:
        nodes_json = _json.dumps(gs.nodes, ensure_ascii=False)
        edges_out: _List[list] = []
        for (a, b), v in gs.edge_enabled.items():
            try:
                iv = int(v)
            except Exception:
                iv = 0
            iv = 0 if iv < 0 else (2 if iv > 2 else iv)
            edges_out.append([a, b, iv])
        edge_modes_json = _json.dumps(edges_out, ensure_ascii=False)
        return nodes_json, edge_modes_json

    def _graphstate_from_payload(self, nodes_json: str, edge_modes_json: str) -> GraphState:
        nodes = _json.loads(nodes_json) if nodes_json else []
        edges_raw = _json.loads(edge_modes_json) if edge_modes_json else []
        gs = GraphState(nodes=list(nodes))
        for item in edges_raw:
            if not isinstance(item, list) or len(item) != 3:
                continue
            a, b, mode = item
            try:
                iv = int(mode)
            except Exception:
                iv = 0
            iv = 0 if iv < 0 else (2 if iv > 2 else iv)
            gs.edge_enabled[(a, b)] = iv
        return gs

    def list_for_experiment(self, experiment_id: int) -> _List[ExperimentComputation]:
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT
                    id,
                    experiment_id,
                    name,
                    table_name,
                    used_heads_json,
                    nodes_json,
                    edge_modes_json,
                    ode_text,
                    state_names_json,
                    param_names_json,
                    created_at,
                    updated_at
                FROM experiment_computations
                WHERE experiment_id = ?
                ORDER BY updated_at DESC, id DESC
                """,
                (experiment_id,),
            ).fetchall()

        out: _List[ExperimentComputation] = []
        for r in rows:
            used_heads = _json.loads(r["used_heads_json"]) if r["used_heads_json"] else []
            gs = self._graphstate_from_payload(r["nodes_json"], r["edge_modes_json"])
            state_names = _json.loads(r["state_names_json"]) if r["state_names_json"] else None
            param_names = _json.loads(r["param_names_json"]) if r["param_names_json"] else None
            out.append(
                ExperimentComputation(
                    id=int(r["id"]),
                    experiment_id=int(r["experiment_id"]),
                    name=str(r["name"]),
                    table_name=str(r["table_name"]),
                    used_heads=used_heads,
                    graph_state=gs,
                    ode_text=r["ode_text"],
                    state_names=state_names,
                    param_names=param_names,
                    # sqlite3.Row behaves like a mapping but doesn't implement .get
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                )
            )
        return out

    def insert(self, experiment_id: int, name: str, table_name: str, used_heads: _List[str], gs: GraphState) -> int:
        # Ensure we don't persist stale/backward edges (can happen when user reorders nodes).
        normalize_graph_state(gs)
        used_heads_json = _json.dumps(used_heads, ensure_ascii=False)
        nodes_json, edge_modes_json = self._graphstate_to_payload(gs)
        # NOTE: zatím bereme jako aktivní jen hrany s módem=1 (mód=2 doplníme později)
        ode = generate_ode_model(gs.nodes, gs.edge_modes, include_modes=(1,))
        ode_text = ode.ode_text
        state_names_json = _json.dumps(ode.state_names, ensure_ascii=False)
        param_names_json = _json.dumps(ode.param_names, ensure_ascii=False)
        with self.db.connect() as con:
            cur = con.execute(
                """
                INSERT INTO experiment_computations(
                    experiment_id, name, table_name, used_heads_json, nodes_json, edge_modes_json,
                    ode_text, state_names_json, param_names_json, updated_at
                )
                VALUES(?,?,?,?,?,?,?,?,?, datetime('now'))
                """,
                (experiment_id, name, table_name, used_heads_json, nodes_json, edge_modes_json,
                 ode_text, state_names_json, param_names_json),
            )
            con.commit()
            return int(cur.lastrowid)

    def update(self, cid: int, name: str, table_name: str, used_heads: _List[str], gs: GraphState) -> None:
        normalize_graph_state(gs)
        used_heads_json = _json.dumps(used_heads, ensure_ascii=False)
        nodes_json, edge_modes_json = self._graphstate_to_payload(gs)
        # NOTE: zatím bereme jako aktivní jen hrany s módem=1 (mód=2 doplníme později)
        ode = generate_ode_model(gs.nodes, gs.edge_modes, include_modes=(1,))
        ode_text = ode.ode_text
        state_names_json = _json.dumps(ode.state_names, ensure_ascii=False)
        param_names_json = _json.dumps(ode.param_names, ensure_ascii=False)
        with self.db.connect() as con:
            con.execute(
                """
                UPDATE experiment_computations
                SET name=?, table_name=?, used_heads_json=?, nodes_json=?, edge_modes_json=?,
                    ode_text=?, state_names_json=?, param_names_json=?,
                    updated_at=datetime('now')
                WHERE id=?
                """,
                (name, table_name, used_heads_json, nodes_json, edge_modes_json,
                 ode_text, state_names_json, param_names_json, cid),
            )
            con.commit()

    def delete(self, cid: int) -> None:
        with self.db.connect() as con:
            con.execute("DELETE FROM experiment_computations WHERE id=?", (cid,))
            con.commit()


# -------------------------
# Processing tab persistence (settings + last results)
# -------------------------


class ExperimentProcessingSettingsRepository:
    """Persist processing-tab settings per experiment.

    This stores *UI state* (init method, model selection, t_shift, ...), so that when the
    app restarts the tab comes up exactly as the user left it.
    """

    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def get(self, experiment_id: int) -> Optional[dict]:
        with self.db.connect() as con:
            row = con.execute(
                """
                SELECT experiment_id, initialization, t_shift, optim_time_shift,
                       models_to_compute_json, t_max, t_max_plot, last_auto_t_shift, updated_at
                FROM experiment_processing_settings
                WHERE experiment_id=?
                """,
                (experiment_id,),
            ).fetchone()
        return dict(row) if row else None

    def upsert(
        self,
        *,
        experiment_id: int,
        initialization: str,
        t_shift: float,
        optim_time_shift: bool,
        models_to_compute: list[str],
        t_max: float,
        t_max_plot: float,
        last_auto_t_shift: float | None = None,
    ) -> None:
        """Insert/update processing settings.

        Important: bump updated_at only when *user-controlled* settings changed.
        Updating `last_auto_t_shift` (a computed result) must not mark the settings as changed.
        """
        models_json = _json.dumps(list(models_to_compute or []), ensure_ascii=False)

        # Check current row to avoid false "changed" markers.
        with self.db.connect() as con:
            current = con.execute(
                """
                SELECT initialization, t_shift, optim_time_shift, models_to_compute_json,
                       t_max, t_max_plot, last_auto_t_shift
                FROM experiment_processing_settings
                WHERE experiment_id=?
                """,
                (int(experiment_id),),
            ).fetchone()

            if current:
                user_same = (
                    str(current["initialization"]) == str(initialization)
                    and float(current["t_shift"]) == float(t_shift)
                    and int(current["optim_time_shift"]) == (1 if bool(optim_time_shift) else 0)
                    and str(current["models_to_compute_json"] or "[]") == models_json
                    and float(current["t_max"]) == float(t_max)
                    and float(current["t_max_plot"]) == float(t_max_plot)
                )

                if user_same:
                    # Only last_auto_t_shift changed -> update it without bumping updated_at.
                    if last_auto_t_shift is not None:
                        try:
                            cur_last = current["last_auto_t_shift"]
                            if cur_last is None or float(cur_last) != float(last_auto_t_shift):
                                con.execute(
                                    """
                                    UPDATE experiment_processing_settings
                                    SET last_auto_t_shift=?
                                    WHERE experiment_id=?
                                    """,
                                    (float(last_auto_t_shift), int(experiment_id)),
                                )
                                con.commit()
                        except Exception:
                            pass
                    return  # no user-facing change -> keep updated_at as-is

            # No row yet, or user settings changed -> upsert and bump updated_at.
            now = utc_now_iso()
            con.execute(
                """
                INSERT INTO experiment_processing_settings(
                    experiment_id, initialization, t_shift, optim_time_shift, models_to_compute_json,
                    t_max, t_max_plot, last_auto_t_shift, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    initialization=excluded.initialization,
                    t_shift=excluded.t_shift,
                    optim_time_shift=excluded.optim_time_shift,
                    models_to_compute_json=excluded.models_to_compute_json,
                    t_max=excluded.t_max,
                    t_max_plot=excluded.t_max_plot,
                    last_auto_t_shift=COALESCE(excluded.last_auto_t_shift, experiment_processing_settings.last_auto_t_shift),
                    updated_at=excluded.updated_at
                """,
                (
                    int(experiment_id),
                    str(initialization),
                    float(t_shift),
                    1 if bool(optim_time_shift) else 0,
                    models_json,
                    float(t_max),
                    float(t_max_plot),
                    (float(last_auto_t_shift) if last_auto_t_shift is not None else None),
                    now,
                ),
            )
            con.commit()


class ExperimentProcessingResultsRepository:
    """Persist last processing results (constants + plot images) grouped by run."""

    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def create_run(
        self,
        *,
        experiment_id: int,
        settings: dict,
        used_t_shift: float | None,
        auto_t_shift: float | None,
    ) -> int:
        now = utc_now_iso()
        settings_json = _json.dumps(settings or {}, ensure_ascii=False)
        with self.db.connect() as con:
            cur = con.execute(
                """
                INSERT INTO experiment_processing_runs(experiment_id, settings_json, used_t_shift, auto_t_shift, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (int(experiment_id), settings_json, used_t_shift, auto_t_shift, now),
            )
            con.commit()
            return int(cur.lastrowid)

    def add_model_result(
        self,
        *,
        run_id: int,
        model_name: str,
        constants: list[dict],
        plot_png: bytes | None,
        plot_error: str | None,
    ) -> None:
        constants_json = _json.dumps(list(constants or []), ensure_ascii=False)
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO experiment_processing_run_models(run_id, model_name, constants_json, plot_png, plot_error)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(run_id, model_name) DO UPDATE SET
                    constants_json=excluded.constants_json,
                    plot_png=excluded.plot_png,
                    plot_error=excluded.plot_error
                """,
                (int(run_id), str(model_name), constants_json, sqlite3.Binary(plot_png) if plot_png else None, plot_error),
            )
            con.commit()

    def get_latest_run(self, experiment_id: int) -> Optional[dict]:
        with self.db.connect() as con:
            run = con.execute(
                """
                SELECT id, experiment_id, settings_json, used_t_shift, auto_t_shift, created_at
                FROM experiment_processing_runs
                WHERE experiment_id=?
                ORDER BY created_at DESC, id DESC
                LIMIT 1
                """,
                (int(experiment_id),),
            ).fetchone()
            if not run:
                return None
            rows = con.execute(
                """
                SELECT model_name, constants_json, plot_png, plot_error
                FROM experiment_processing_run_models
                WHERE run_id=?
                ORDER BY model_name
                """,
                (int(run["id"]),),
            ).fetchall()

        out = dict(run)
        try:
            out["settings"] = _json.loads(out.get("settings_json") or "{}")
        except Exception:
            out["settings"] = {}
        models: dict[str, dict] = {}
        for r in rows:
            try:
                consts = _json.loads(r["constants_json"] or "[]")
            except Exception:
                consts = []
            png_b = r["plot_png"]
            models[str(r["model_name"])] = {
                "constants": consts,
                "plot_png": bytes(png_b) if png_b is not None else None,
                "plot_error": r["plot_error"],
            }
        out["models"] = models
        return out


class ExperimentGraphSettingsRepository:
    """Persist graph-tab settings per experiment and model.

    The intent is that a user can tweak the appearance (legend, x/y limits, colors,
    annotations, ...) and see the same settings after restart.
    """

    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def get(self, experiment_id: int, model_name: str) -> Optional[dict]:
        with self.db.connect() as con:
            row = con.execute(
                """
                SELECT id, experiment_id, model_name, config_json, updated_at
                FROM experiment_graph_settings
                WHERE experiment_id=? AND model_name=?
                """,
                (int(experiment_id), str(model_name)),
            ).fetchone()
        if not row:
            return None
        out = dict(row)
        try:
            out["config"] = _json.loads(out.get("config_json") or "{}")
        except Exception:
            out["config"] = {}
        return out

    def list_for_experiment(self, experiment_id: int) -> list[dict]:
        """Return all graph settings rows for an experiment (as plain dicts)."""
        # NOTE: The schema evolved over time. Some existing DBs don't have
        # `created_at` in `experiment_graph_settings` (they only have `updated_at`).
        # Be backwards-compatible by falling back gracefully.
        with self.db.connect() as con:
            try:
                rows = con.execute(
                    """
                    SELECT id, experiment_id, model_name, config_json, created_at, updated_at
                    FROM experiment_graph_settings
                    WHERE experiment_id=?
                    ORDER BY id
                    """,
                    (int(experiment_id),),
                ).fetchall()
                has_created_at = True
            except Exception:
                rows = con.execute(
                    """
                    SELECT id, experiment_id, model_name, config_json, updated_at
                    FROM experiment_graph_settings
                    WHERE experiment_id=?
                    ORDER BY id
                    """,
                    (int(experiment_id),),
                ).fetchall()
                has_created_at = False

        out: list[dict] = []
        for r in rows:
            d = dict(r)
            if not has_created_at:
                # Approximation: treat `created_at` as the first known timestamp.
                d["created_at"] = d.get("updated_at")
            try:
                d["config"] = _json.loads(d.get("config_json") or "{}")
            except Exception:
                d["config"] = {}
            out.append(d)
        return out

    def upsert(self, experiment_id: int, model_name: str, config: dict) -> None:
        """Insert/update graph settings, without bumping updated_at if unchanged."""
        cfg_json = _json.dumps(config or {}, ensure_ascii=False)

        with self.db.connect() as con:
            existing = con.execute(
                """
                SELECT config_json
                FROM experiment_graph_settings
                WHERE experiment_id=? AND model_name=?
                """,
                (int(experiment_id), str(model_name)),
            ).fetchone()

            if existing and (existing["config_json"] or "") == cfg_json:
                return  # no real change -> keep updated_at as-is

            now = utc_now_iso()
            con.execute(
                """
                INSERT INTO experiment_graph_settings(experiment_id, model_name, config_json, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(experiment_id, model_name) DO UPDATE SET
                    config_json=excluded.config_json,
                    updated_at=excluded.updated_at
                """,
                (int(experiment_id), str(model_name), cfg_json, now),
            )
            con.commit()

    def delete(self, experiment_id: int, model_name: str) -> None:
        with self.db.connect() as con:
            con.execute(
                "DELETE FROM experiment_graph_settings WHERE experiment_id=? AND model_name=?",
                (int(experiment_id), str(model_name)),
            )
            con.commit()
