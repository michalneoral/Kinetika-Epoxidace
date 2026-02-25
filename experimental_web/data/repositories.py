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
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO app_meta(key, value, updated_at)
                VALUES('last_experiment_id', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (str(exp_id), now),
            )
            con.commit()


class SettingsRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)
        self.db.ensure_schema()

    def get_theme_mode(self, default: str = "auto") -> str:
        with self.db.connect() as con:
            row = con.execute("SELECT value FROM user_settings WHERE key='theme_mode'").fetchone()
        return str(row["value"]) if row else default

    def set_theme_mode(self, mode: str) -> None:
        if mode not in ("auto", "light", "dark"):
            raise ValueError("theme_mode must be one of: auto, light, dark")
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO user_settings(key, value, updated_at)
                VALUES('theme_mode', ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (mode, now),
            )
            con.commit()


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
            row = con.execute(
                """
                SELECT id, experiment_id, filename, content, sha256, size_bytes, uploaded_at, selected_sheet
                FROM experiment_files
                WHERE experiment_id=?
                ORDER BY uploaded_at DESC
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
            cur = con.execute(
                """
                INSERT INTO experiment_files(experiment_id, filename, content, sha256, size_bytes, uploaded_at, selected_sheet)
                VALUES(?, ?, ?, ?, ?, ?, NULL)
                """,
                (experiment_id, filename, sqlite3.Binary(content), sha, size, now),
            )
            file_id = int(cur.lastrowid)
            con.commit()

        ef = self.get(file_id)
        assert ef is not None
        return ef

    def list_for_experiment(self, experiment_id: int) -> List[ExperimentFile]:
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT id, experiment_id, filename, content, sha256, size_bytes, uploaded_at, selected_sheet
                FROM experiment_files
                WHERE experiment_id=?
                ORDER BY uploaded_at DESC
                """,
                (experiment_id,),
            ).fetchall()
        return [ExperimentFile(**dict(r)) for r in rows]

    def get(self, file_id: int) -> Optional[ExperimentFile]:
        with self.db.connect() as con:
            row = con.execute(
                """
                SELECT id, experiment_id, filename, content, sha256, size_bytes, uploaded_at, selected_sheet
                FROM experiment_files
                WHERE id=?
                """,
                (file_id,),
            ).fetchone()
        return ExperimentFile(**dict(row)) if row else None

    def set_selected_sheet(self, file_id: int, sheet: Optional[str]) -> None:
        with self.db.connect() as con:
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
        with self.db.connect() as con:
            row = con.execute(
                """SELECT row_start, row_end, col_start, col_end
                     FROM experiment_table_picks
                     WHERE experiment_id=? AND kind=?""",
                (experiment_id, kind),
            ).fetchone()
        return dict(row) if row else None

    def set(self, experiment_id: int, kind: str, row_start: int, row_end: int, col_start: int, col_end: int) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute(
                """INSERT INTO experiment_table_picks(experiment_id, kind, row_start, row_end, col_start, col_end, updated_at)
                     VALUES(?, ?, ?, ?, ?, ?, ?)
                     ON CONFLICT(experiment_id, kind)
                     DO UPDATE SET row_start=excluded.row_start,
                                   row_end=excluded.row_end,
                                   col_start=excluded.col_start,
                                   col_end=excluded.col_end,
                                   updated_at=excluded.updated_at""",
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
                "SELECT name FROM experiment_processed_tables WHERE experiment_id=? AND cache_key=? ORDER BY name",
                (experiment_id, cache_key),
            ).fetchall()
        return [r["name"] for r in rows]

    def load_tables(self, experiment_id: int, cache_key: str) -> dict:
        with self.db.connect() as con:
            rows = con.execute(
                """
                SELECT name, df_json, text_md
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
                SELECT t.name, t.df_json, t.text_md
                FROM experiment_processed_tables t
                JOIN (
                    SELECT name, MAX(updated_at) AS mx
                    FROM experiment_processed_tables
                    WHERE experiment_id = ?
                    GROUP BY name
                ) latest
                ON latest.name = t.name AND latest.mx = t.updated_at
                WHERE t.experiment_id = ?
                """,
                (experiment_id, experiment_id),
            ).fetchall()
        return {str(r["name"]): (str(r["df_json"]), str(r["text_md"] or "")) for r in rows}

    def save_table(self, experiment_id: int, cache_key: str, name: str, df_json: str, text_md: str = "") -> None:
        """Upsert one processed table snapshot into DB."""
        with self.db.connect() as con:
            con.execute(
                """
                INSERT OR REPLACE INTO experiment_processed_tables(experiment_id, cache_key, name, df_json, text_md, updated_at)
                VALUES(?,?,?,?,?, datetime('now'))
                """,
                (experiment_id, cache_key, name, df_json, text_md),
            )
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
