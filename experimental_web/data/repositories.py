from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, List

from experimental_web.core.paths import EXPERIMENTS_DIR
from experimental_web.core.time import utc_now_iso
from experimental_web.data.database import Database
from experimental_web.data.models import Experiment


def _safe_slug(text: str) -> str:
    import re
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", "_", t)
    t = re.sub(r"[^a-z0-9_\-]+", "", t)
    return (t[:40] or "experiment")


class ExperimentRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)

    def exists_name(self, name: str) -> bool:
        with self.db.connect() as con:
            row = con.execute("SELECT 1 FROM experiments WHERE name = ? LIMIT 1", (name,)).fetchone()
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

    def duplicate(self, source_id: int, new_name: str) -> Experiment:
        """Duplicate experiment row and folder contents."""
        src = self.get(source_id)
        if not src:
            raise ValueError("Source experiment not found")
        if self.exists_name(new_name):
            raise ValueError("Experiment with this name already exists")

        new_exp = self.create(new_name)

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

        return new_exp

    def delete(self, exp_id: int, delete_folder: bool = True) -> None:
        exp = self.get(exp_id)
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

    def touch(self, exp_id: int) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute("UPDATE experiments SET updated_at=? WHERE id=?", (now, exp_id))
            con.commit()


class MetaRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)

    def get(self, key: str, default: str = "") -> str:
        with self.db.connect() as con:
            row = con.execute("SELECT value FROM app_meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO app_meta(key, value, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )
            con.commit()

    def get_last_experiment_id(self) -> Optional[int]:
        v = self.get("last_experiment_id", "")
        return int(v) if v.strip().isdigit() else None

    def set_last_experiment_id(self, exp_id: int) -> None:
        self.set("last_experiment_id", str(exp_id))


class SettingsRepository:
    def __init__(self, db_path: Path) -> None:
        self.db = Database(db_path)

    def get(self, key: str, default: str) -> str:
        with self.db.connect() as con:
            row = con.execute("SELECT value FROM user_settings WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default

    def set(self, key: str, value: str) -> None:
        now = utc_now_iso()
        with self.db.connect() as con:
            con.execute(
                """
                INSERT INTO user_settings(key, value, updated_at)
                VALUES(?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
                """,
                (key, value, now),
            )
            con.commit()

    def get_theme_mode(self, default: str = "auto") -> str:
        return self.get("theme_mode", default)

    def set_theme_mode(self, mode: str) -> None:
        if mode not in ("auto", "light", "dark"):
            raise ValueError("theme_mode must be one of: auto, light, dark")
        self.set("theme_mode", mode)
