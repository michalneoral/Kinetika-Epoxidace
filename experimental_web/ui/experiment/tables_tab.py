from __future__ import annotations

from io import BytesIO, StringIO
from typing import Optional

import pandas as pd
from nicegui import ui

from experimental_web.core.paths import DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import (
    ExperimentFileRepository,
    TablePickRepository,
    ProcessedTablesRepository,
)
from experimental_web.domain.fame_epo import DEFAULT_EPO, DEFAULT_FAME, extract_df
from experimental_web.domain.table_processor_tables import TableProcessorTables
from experimental_web.ui.utils.tables import sanitize_df_for_table
from experimental_web.ui.widgets.sticky_table import StickyTable
from experimental_web.ui.widgets.styled_label import StyledLabel
from experimental_web.logging_setup import get_logger


log = get_logger(__name__)


TABLE_ORDER = [
    "IS_korig",
    "IS_korig_EPO",
    "IS_korig_t",
    "IS_korig_EPO_t",
    "souhrn",
    "C18_1",
    "C18_2",
    "C18_3",
    "C20_1",
]


def _df_to_rows_columns(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    df2 = sanitize_df_for_table(df.copy()).fillna("").infer_objects(copy=False)
    for c in df2.columns:
        df2[c] = df2[c].astype(str)
    columns = [{"name": c, "label": c, "field": c} for c in df2.columns]
    rows = df2.to_dict(orient="records")
    return columns, rows


def _load_pick(pick_repo: TablePickRepository, experiment_id: int, kind: str):
    saved = pick_repo.get(experiment_id, kind)
    if saved:
        return saved["row_start"], saved["row_end"], saved["col_start"], saved["col_end"]
    c = DEFAULT_FAME if kind == "fame" else DEFAULT_EPO
    return c.row_start, c.row_end, c.col_start, c.col_end


def _normalize_first_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df2 = df.copy()
    if df2.columns.size > 0:
        df2 = df2.rename(columns={df2.columns[0]: name})
    return df2


def render_tables_tab(experiment_id: int) -> None:
    """Tab: zpracování -> tabulky (bez kinetiky), s auto-refresh a DB cache."""

    st = get_state()
    file_repo = ExperimentFileRepository(DB_PATH)
    pick_repo = TablePickRepository(DB_PATH)
    cache_repo = ProcessedTablesRepository(DB_PATH)

    status = StyledLabel("Připraveno.", "info")
    container = ui.column().classes("w-full gap-3")

    ui.add_head_html('''
    <style>
      .tables-expansion .q-expansion-item__container > .q-item {
        background: var(--q-grey-2);
        color: #111;
      }
      .tables-expansion .q-expansion-item__content {
        background: white;
        color: #111;
      }
      .body--dark .tables-expansion .q-expansion-item__container > .q-item {
        background: var(--q-dark);
        color: #fff;
      }
      .body--dark .tables-expansion .q-expansion-item__content {
        background: #1d1d1d;
        color: #fff;
      }
    </style>
    ''')

    def load_excel() -> tuple[Optional[object], Optional[pd.ExcelFile]]:
        ef = file_repo.get_single_for_experiment(experiment_id)
        if not ef:
            return None, None
        if not st.current_excel_sheet:
            st.current_excel_sheet = ef.selected_sheet or ""
        try:
            return ef, pd.ExcelFile(BytesIO(ef.content))
        except Exception as e:
            ui.notify(f"Nelze otevřít Excel: {e}", type="negative")
            return ef, None

    def compute_cache_key(excel_sha: str, sheet: str, fame_pick: tuple[int, int, int, int], epo_pick: tuple[int, int, int, int]) -> str:
        import hashlib, json
        payload = {
            "excel_sha": excel_sha,
            "sheet": sheet,
            "fame": fame_pick,
            "epo": epo_pick,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def render_tables(tables: dict[str, tuple[str, str]]) -> None:
        container.clear()
        with container:
            # enforce order, then any extras
            ordered_names = TABLE_ORDER + [n for n in tables.keys() if n not in TABLE_ORDER]
            for name in ordered_names:
                df_json, text_md = tables.get(name, (None, None))
                if df_json is None:
                    continue
                try:
                    df = pd.read_json(StringIO(df_json), orient="split")
                except Exception:
                    continue

                with ui.expansion(name, value=(name == "souhrn")) \
                    .classes("tables-expansion w-full q-mb-sm shadow-1 rounded-borders overflow-hidden") \
                    .props('expand-separator dense'):

                    with ui.card().classes("w-full"):
                        if text_md:
                            ui.markdown(text_md, extras=['latex']).classes("text-caption")
                            ui.separator()

                        cols, rows = _df_to_rows_columns(df)
                        StickyTable.from_rows_and_columns(
                            rows=rows,
                            columns=cols,
                            sticky="both",
                            max_height="520px",
                        ).classes("w-full")

    def compute_and_render() -> None:
        log.info('[UI] tables.compute_and_render: experiment_id=%s', experiment_id)
        ef, xls = load_excel()
        if ef is None or xls is None:
            status.set("Nejdřív nahrajte Excel v tabu „načtení dat“.", "warning")
            container.clear()
            return

        sheets = list(xls.sheet_names or [])
        sheet = st.current_excel_sheet or (sheets[0] if sheets else "")
        if sheet not in sheets:
            sheet = sheets[0] if sheets else ""
        if not sheet:
            status.set("Excel neobsahuje žádné listy.", "error")
            container.clear()
            return

        # load raw sheet
        try:
            df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
        except Exception as e:
            status.set(f"Nelze načíst list: {e}", "error")
            container.clear()
            return

        fame_pick = _load_pick(pick_repo, experiment_id, "fame")
        epo_pick = _load_pick(pick_repo, experiment_id, "epo")

        cache_key = compute_cache_key(ef.sha256 or str(ef.id), sheet, fame_pick, epo_pick)

        cached = cache_repo.load_tables(experiment_id, cache_key)
        if cached:
            log.info('[UI] tables.cache_hit: key=%s', cache_key)
            status.set("Načteno z cache.", "ok")
            render_tables(cached)
            return

        status.set("Počítám tabulky…", "info")
        log.info('[UI] tables.cache_miss: key=%s sheet=%s fame_pick=%s epo_pick=%s', cache_key, sheet, fame_pick, epo_pick)

        # extract fame/epo
        r1, r2, c1, c2 = fame_pick
        fame_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), "FAME")
        r1, r2, c1, c2 = epo_pick
        epo_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), "EPOXIDES")
        # numeric conversion (best effort)
        for _df in (fame_df, epo_df):
            for _col in _df.columns[1:]:
                if _df[_col].dtype == object:
                    converted = pd.to_numeric(_df[_col], errors='coerce')
                    # accept conversion only if it does not introduce new NaNs
                    if converted.notna().sum() == _df[_col].notna().sum():
                        _df[_col] = converted

        processor = TableProcessorTables(fame_df=fame_df, epo_df=epo_df)
        try:
            processor.process()
        except Exception as e:
            log.exception('tables.process.error: %s', e)
            status.set(f"Chyba při výpočtu tabulek: {e}", "error")
            container.clear()
            return

        # serialize to cache payload: name -> (df_json, text_md)
        payload: dict[str, tuple[str, str]] = {}
        for name, df in processor.tables.items():
            df_json = df.to_json(orient="split")
            text_md = processor.tables_text.get(name, "")
            payload[name] = (df_json, text_md)

        for _name, (_df_json, _text_md) in payload.items():
            cache_repo.save_table(experiment_id, cache_key, _name, _df_json, _text_md)
        status.set("Hotovo: uloženo do cache.", "ok")
        log.info('[UI] tables.done: key=%s tables=%s', cache_key, len(payload))
        render_tables(payload)

    # --- auto refresh ---
    last_version = st.data_version

    def tick() -> None:
        nonlocal last_version
        if st.data_version != last_version:
            last_version = st.data_version
            compute_and_render()

    ui.timer(0.5, tick)

    # initial render
    compute_and_render()
