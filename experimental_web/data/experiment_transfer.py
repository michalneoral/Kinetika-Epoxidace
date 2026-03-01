from __future__ import annotations

import base64
import datetime as _dt
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Any
import zipfile

from experimental_web.data.database import Database
from experimental_web.data.repositories import ExperimentRepository


FORMAT_VERSION = 1

# Columns stored as base64 strings in the export JSON.
BLOB_COLUMNS: dict[str, list[str]] = {
    'experiment_files': ['content'],
    'excel_files': ['content'],
    'experiment_processing_run_models': ['plot_png'],
}


def _b64e(b: bytes) -> str:
    return base64.b64encode(b).decode('ascii')


def _b64d(s: str) -> bytes:
    return base64.b64decode(s.encode('ascii'))


def _row_to_jsonable(table: str, row: dict) -> dict:
    blob_cols = set(BLOB_COLUMNS.get(table, []))
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in blob_cols and v is not None:
            if isinstance(v, (bytes, bytearray)):
                out[k] = _b64e(bytes(v))
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _jsonable_to_row(table: str, row: dict) -> dict:
    blob_cols = set(BLOB_COLUMNS.get(table, []))
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k in blob_cols and v is not None:
            if isinstance(v, str):
                out[k] = _b64d(v)
            else:
                out[k] = v
        else:
            out[k] = v
    return out


def _safe_filename(name: str) -> str:
    s = ''.join(c if (c.isalnum() or c in '-_.() ') else '_' for c in name).strip()
    return s or 'experiment'


def _unique_name(exp_repo: ExperimentRepository, base: str) -> str:
    base = (base or '').strip() or 'Experiment'
    if not exp_repo.exists_name(base):
        return base
    i = 2
    while True:
        candidate = f'{base} ({i})'
        if not exp_repo.exists_name(candidate):
            return candidate
        i += 1


def export_experiment(db_path: Path, experiment_id: int, include_folder: bool = True) -> tuple[bytes, str]:
    """Export one experiment into a zip archive (bytes) and return (bytes, filename)."""

    db = Database(db_path)
    db.ensure_schema()

    with db.connect() as con:
        exp_row = con.execute('SELECT * FROM experiments WHERE id=?', (experiment_id,)).fetchone()
        if not exp_row:
            raise ValueError('Experiment not found')
        exp = dict(exp_row)

        def table_exists(table: str) -> bool:
            return con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is not None

        def fetch_table(table: str, where_sql: str, params: tuple) -> list[dict]:
            if not table_exists(table):
                return []
            rows = con.execute(f'SELECT * FROM {table} {where_sql}', params).fetchall()
            return [_row_to_jsonable(table, dict(r)) for r in rows]

        tables: dict[str, list[dict]] = {}
        tables['experiments'] = [_row_to_jsonable('experiments', exp)]
        tables['experiment_files'] = fetch_table('experiment_files', 'WHERE experiment_id=?', (experiment_id,))
        tables['excel_files'] = fetch_table('excel_files', 'WHERE experiment_id=?', (experiment_id,))
        tables['experiment_table_picks'] = fetch_table('experiment_table_picks', 'WHERE experiment_id=?', (experiment_id,))
        tables['experiment_processed_tables'] = fetch_table('experiment_processed_tables', 'WHERE experiment_id=?', (experiment_id,))
        tables['processed_tables'] = fetch_table('processed_tables', 'WHERE experiment_id=?', (experiment_id,))
        tables['experiment_computations'] = fetch_table('experiment_computations', 'WHERE experiment_id=?', (experiment_id,))
        tables['computations'] = fetch_table('computations', 'WHERE experiment_id=?', (experiment_id,))
        tables['experiment_processing_settings'] = fetch_table('experiment_processing_settings', 'WHERE experiment_id=?', (experiment_id,))
        tables['experiment_graph_settings'] = fetch_table('experiment_graph_settings', 'WHERE experiment_id=?', (experiment_id,))

        runs = fetch_table('experiment_processing_runs', 'WHERE experiment_id=?', (experiment_id,))
        tables['experiment_processing_runs'] = runs
        run_ids = [r.get('id') for r in runs if r.get('id') is not None]
        if table_exists('experiment_processing_run_models') and run_ids:
            qmarks = ','.join('?' for _ in run_ids)
            rows = con.execute(
                f'SELECT * FROM experiment_processing_run_models WHERE run_id IN ({qmarks})',
                tuple(run_ids),
            ).fetchall()
            tables['experiment_processing_run_models'] = [_row_to_jsonable('experiment_processing_run_models', dict(r)) for r in rows]
        else:
            tables['experiment_processing_run_models'] = []

    manifest = {
        'format_version': FORMAT_VERSION,
        'exported_at': _dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'experiment_id': experiment_id,
        'experiment_name': exp.get('name', ''),
        'blob_columns': BLOB_COLUMNS,
        'include_folder': bool(include_folder and exp.get('folder')),
    }

    payload = {'manifest': manifest, 'tables': tables}

    out = BytesIO()
    with zipfile.ZipFile(out, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('export.json', json.dumps(payload, ensure_ascii=False, indent=2))

        folder = exp.get('folder')
        if include_folder and folder:
            p = Path(folder)
            if p.exists() and p.is_dir():
                for root, _, files in os.walk(p):
                    for fn in files:
                        fp = Path(root) / fn
                        rel = fp.relative_to(p)
                        z.write(fp, arcname=str(Path('folder') / rel))

    safe = _safe_filename(str(exp.get('name', 'experiment')))
    ts = _dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    filename = f'{safe}_export_{ts}.zip'
    return out.getvalue(), filename


def export_all_experiments(db_path: Path, include_folder: bool = True) -> tuple[bytes, str]:
    """Export all experiments into a single zip archive.

    The resulting zip contains:
    - manifest.json (list of exported experiments)
    - global_tables.json (best-effort global app settings)
    - experiments/<experiment_export_zip> (one zip per experiment)

    This keeps the existing single-experiment format intact and makes it easy
    to import individual experiments later.
    """

    db = Database(db_path)
    db.ensure_schema()

    def table_exists(con, table: str) -> bool:
        return con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone() is not None

    with db.connect() as con:
        exp_rows = con.execute('SELECT id, name FROM experiments ORDER BY id').fetchall()
        experiments = [dict(r) for r in exp_rows]

        # Global settings backup
        # NOTE: Keep legacy `settings` table too (older DBs stored UI/global config there).
        global_tables: dict[str, list[dict]] = {}
        for tname in ['user_settings', 'app_meta', 'settings']:
            if table_exists(con, tname):
                trs = con.execute(f'SELECT * FROM {tname}').fetchall()
                global_tables[tname] = [dict(r) for r in trs]

    ts = _dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    manifest = {
        'format_version': FORMAT_VERSION,
        'exported_at': _dt.datetime.utcnow().isoformat(timespec='seconds') + 'Z',
        'include_folder': bool(include_folder),
        'experiments': [],
    }

    out = BytesIO()
    with zipfile.ZipFile(out, mode='w', compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('global_tables.json', json.dumps(global_tables, ensure_ascii=False, indent=2))

        for exp in experiments:
            exp_id = int(exp['id'])
            zip_bytes, filename = export_experiment(db_path, exp_id, include_folder=include_folder)
            arcname = str(Path('experiments') / filename)
            z.writestr(arcname, zip_bytes)
            manifest['experiments'].append({'id': exp_id, 'name': exp.get('name', ''), 'file': arcname})

        z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))

    filename = f'kinetika_all_export_{ts}.zip'
    return out.getvalue(), filename


def import_experiment(db_path: Path, export_zip_bytes: bytes) -> int:
    """Import one experiment from a *single-experiment* export zip.

    Returns the new experiment_id.
    """

    db = Database(db_path)
    db.ensure_schema()
    exp_repo = ExperimentRepository(db_path)

    with zipfile.ZipFile(BytesIO(export_zip_bytes), 'r') as z:
        payload = json.loads(z.read('export.json').decode('utf-8'))
        tables = payload.get('tables') or {}

        exp_rows = tables.get('experiments') or []
        if not exp_rows:
            raise ValueError('Export does not contain experiments row')
        src_exp = exp_rows[0]
        src_name = str(src_exp.get('name') or 'Experiment')
        new_name = _unique_name(exp_repo, src_name)

        # Create experiment via repository (creates folder, ensures consistency)
        new_exp = exp_repo.create(new_name)
        new_experiment_id = int(new_exp.id)

        # Preserve some experiment metadata (best effort)
        with db.connect() as con:
            try:
                con.execute(
                    'UPDATE experiments SET description=COALESCE(?, description), created_at=COALESCE(?, created_at), updated_at=COALESCE(?, updated_at) WHERE id=?',
                    (
                        src_exp.get('description', ''),
                        src_exp.get('created_at'),
                        src_exp.get('updated_at'),
                        new_experiment_id,
                    ),
                )
                con.commit()
            except Exception:
                pass

        def table_exists(con, table: str) -> bool:
            return con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
            ).fetchone() is not None

        def insert_rows(con, table: str, rows: list[dict], run_id_map: dict[int, int] | None = None) -> dict[int, int]:
            run_id_map = run_id_map or {}
            if not rows or not table_exists(con, table):
                return run_id_map

            # tables with autoincrement id where we omit id on insert
            omit_id_tables = {
                'experiment_files',
                'excel_files',
                'experiment_processed_tables',
                'processed_tables',
                'experiment_computations',
                'computations',
                'experiment_graph_settings',
                'experiment_processing_runs',
                'experiment_processing_run_models',
            }

            for rj in rows:
                r = _jsonable_to_row(table, dict(rj))
                if 'experiment_id' in r:
                    r['experiment_id'] = new_experiment_id

                old_id = r.get('id')
                if table in omit_id_tables:
                    r.pop('id', None)

                cols = list(r.keys())
                placeholders = ','.join('?' for _ in cols)
                col_sql = ','.join(cols)
                cur = con.execute(
                    f'INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({placeholders})',
                    tuple(r[c] for c in cols),
                )

                if table == 'experiment_processing_runs' and old_id is not None:
                    run_id_map[int(old_id)] = int(cur.lastrowid)

            return run_id_map

        with db.connect() as con:
            # Simple experiment-scoped tables
            for tname in [
                'experiment_files',
                'excel_files',
                'experiment_table_picks',
                'experiment_processed_tables',
                'processed_tables',
                'experiment_computations',
                'computations',
                'experiment_processing_settings',
                'experiment_graph_settings',
            ]:
                insert_rows(con, tname, tables.get(tname) or [])

            # Runs + models (need id remap)
            run_id_map: dict[int, int] = {}
            run_id_map = insert_rows(con, 'experiment_processing_runs', tables.get('experiment_processing_runs') or [], run_id_map=run_id_map)

            rms = tables.get('experiment_processing_run_models') or []
            if rms:
                rewritten: list[dict] = []
                for r in rms:
                    rr = dict(r)
                    old_run_id = rr.get('run_id')
                    if old_run_id is not None and int(old_run_id) in run_id_map:
                        rr['run_id'] = run_id_map[int(old_run_id)]
                    rewritten.append(rr)
                insert_rows(con, 'experiment_processing_run_models', rewritten)

            con.commit()

        # Restore folder content (best effort)
        try:
            dst_folder = Path(new_exp.folder) if new_exp.folder else None
            if dst_folder and any(n.startswith('folder/') for n in z.namelist()):
                for member in z.namelist():
                    if not member.startswith('folder/'):
                        continue
                    rel = member[len('folder/'):]
                    if not rel or rel.endswith('/'):
                        continue
                    target = dst_folder / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with z.open(member, 'r') as src, open(target, 'wb') as dst:
                        dst.write(src.read())
        except Exception:
            pass

    return new_experiment_id


def import_any(db_path: Path, export_zip_bytes: bytes) -> list[int]:
    """Import either a single experiment export ZIP or a bundle ZIP.

    Supported inputs:
    - ZIP produced by ``export_experiment`` (contains ``export.json`` at the root)
    - ZIP produced by ``export_all_experiments`` (contains ``experiments/*.zip``)

    Returns a list of newly created experiment ids.

    For bundle imports, global tables (``user_settings``, ``app_meta``) are imported
    best-effort from ``global_tables.json`` using ``INSERT OR REPLACE``.
    """

    with zipfile.ZipFile(BytesIO(export_zip_bytes), 'r') as z:
        names = set(z.namelist())

        # Single-experiment export
        if 'export.json' in names:
            return [import_experiment(db_path, export_zip_bytes)]

        # Bundle export (Exportovat vše)
        is_bundle = (
            'manifest.json' in names
            or any(n.startswith('experiments/') and n.lower().endswith('.zip') for n in names)
        )
        if not is_bundle:
            raise ValueError('Neznámý formát exportu: chybí export.json i manifest/experiments')

        # Best-effort import global settings
        try:
            if 'global_tables.json' in names:
                global_tables = json.loads(z.read('global_tables.json').decode('utf-8'))
                if isinstance(global_tables, dict):
                    db = Database(db_path)
                    db.ensure_schema()
                    with db.connect() as con:
                        for tname, rows in global_tables.items():
                            if not isinstance(rows, list):
                                continue
                            exists = con.execute(
                                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                (tname,),
                            ).fetchone() is not None
                            if not exists:
                                continue
                            for row in rows:
                                if not isinstance(row, dict):
                                    continue
                                cols = list(row.keys())
                                if not cols:
                                    continue
                                placeholders = ','.join('?' for _ in cols)
                                col_sql = ','.join(cols)
                                con.execute(
                                    f'INSERT OR REPLACE INTO {tname} ({col_sql}) VALUES ({placeholders})',
                                    tuple(row[c] for c in cols),
                                )
                        con.commit()
        except Exception:
            # Never fail the whole import because of global settings
            pass

        # Determine which experiment ZIPs to import
        exp_zip_paths: list[str] = []
        if 'manifest.json' in names:
            try:
                manifest = json.loads(z.read('manifest.json').decode('utf-8'))
                for item in (manifest.get('experiments') or []):
                    if isinstance(item, dict):
                        p = item.get('file')
                        if isinstance(p, str) and p in names:
                            exp_zip_paths.append(p)
            except Exception:
                exp_zip_paths = []

        if not exp_zip_paths:
            exp_zip_paths = sorted([n for n in names if n.startswith('experiments/') and n.lower().endswith('.zip')])

        if not exp_zip_paths:
            raise ValueError('Exportovat vše ZIP neobsahuje žádné experimenty (experiments/*.zip)')

        new_ids: list[int] = []
        for path in exp_zip_paths:
            try:
                exp_bytes = z.read(path)
                new_id = import_experiment(db_path, exp_bytes)
                new_ids.append(int(new_id))
            except Exception:
                # Continue with remaining experiments; user will see partial import
                continue

        if not new_ids:
            raise ValueError('Import selhal: nepodařilo se naimportovat žádný experiment')

        return new_ids
