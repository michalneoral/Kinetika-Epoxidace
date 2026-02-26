from __future__ import annotations

from io import BytesIO
from typing import Optional

import pandas as pd
from nicegui import ui

from experimental_web.core.paths import DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import ExperimentFileRepository, TablePickRepository
from experimental_web.domain.fame_epo import DEFAULT_EPO, DEFAULT_FAME, REQUIRED_EPO, REQUIRED_FAME, extract_df, extract_df_dict, validate_table
from experimental_web.ui.utils.tables import sanitize_df_for_table
from experimental_web.ui.widgets.sticky_table import StickyTable
from experimental_web.ui.widgets.styled_label import StyledLabel
from experimental_web.logging_setup import get_logger
from experimental_web.ui.instrumentation import wrap_ui_handler


log = get_logger(__name__)


def _df_to_rows_columns(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    df2 = sanitize_df_for_table(df.copy()).fillna("").infer_objects(copy=False)
    for c in df2.columns:
        df2[c] = df2[c].astype(str)
    columns = [{"name": c, "label": c, "field": c} for c in df2.columns]
    rows = df2.to_dict(orient="records")
    return columns, rows


def render_load_data_tab(experiment_id: int) -> None:
    st = get_state()
    file_repo = ExperimentFileRepository(DB_PATH)
    pick_repo = TablePickRepository(DB_PATH)

    ui.label("Krok 1: Nahrát Excel soubor").classes("text-subtitle1")
    upload = ui.upload(label="Vyberte soubor", auto_upload=True, max_files=1, multiple=False)\
        .props("accept=.xlsx,.xls")\
        .tooltip("Soubor se uloží jako BLOB do SQLite. Nový soubor přepíše starý.")
    ui.separator()

    dynamic = ui.column().classes("w-full gap-4")

    def get_file():
        return file_repo.get_single_for_experiment(experiment_id)

    def load_excel() -> Optional[pd.ExcelFile]:
        ef = get_file()
        if not ef:
            return None
        st.current_excel_file_id = ef.id
        st.current_excel_filename = ef.filename
        st.current_excel_sheet = ef.selected_sheet or ""
        try:
            return pd.ExcelFile(BytesIO(ef.content))
        except Exception as e:
            ui.notify(f"Nelze otevřít Excel: {e}", type="negative")
            return None

    def load_sheet_df(xls: pd.ExcelFile, sheet: str) -> Optional[pd.DataFrame]:
        try:
            return pd.read_excel(xls, sheet_name=sheet, header=None)
        except Exception as e:
            ui.notify(f"Nelze načíst list: {e}", type="negative")
            return None

    def render_preview(df_raw: pd.DataFrame, sheet: str, mount: ui.column) -> None:
        mount.clear()
        with mount:
            ui.label(f"Krok 2: Náhled listu „{sheet}“").classes("text-subtitle1")
            df_prev = df_raw.copy()
            if df_prev.shape[0] > 30:
                df_prev = df_prev.iloc[:30, :]
            if df_prev.shape[1] > 20:
                df_prev = df_prev.iloc[:, :20]
            cols, rows = _df_to_rows_columns(df_prev)
            StickyTable.from_rows_and_columns(rows=rows, columns=cols, sticky="header", max_height="360px").classes("w-full")

    def _load_pick(kind: str):
        saved = pick_repo.get(experiment_id, kind)
        if saved:
            return saved["row_start"], saved["row_end"], saved["col_start"], saved["col_end"]
        c = DEFAULT_FAME if kind=="fame" else DEFAULT_EPO
        return c.row_start, c.row_end, c.col_start, c.col_end

    def render_pick(kind: str, df_raw: pd.DataFrame, mount: ui.column) -> None:
        mount.clear()
        title = "Krok 3: Selekce hodnot pro FAME" if kind=="fame" else "Krok 4: Selekce hodnot pro EPO"
        required = REQUIRED_FAME if kind=="fame" else REQUIRED_EPO
        r1,r2,c1,c2 = _load_pick(kind)

        with mount:
            ui.label(title).classes("text-subtitle1")
            status = StyledLabel("Upravte rozsah.", "info")

            with ui.row().classes("q-gutter-md items-end"):
                rs = ui.number("row_start", value=r1, min=1, step=1).props("dense")
                re_ = ui.number("row_end", value=r2, min=1, step=1).props("dense")
                cs = ui.number("col_start", value=c1, min=1, step=1).props("dense")
                ce = ui.number("col_end", value=c2, min=1, step=1).props("dense")

            table_box = ui.column().classes("w-full gap-2")

            def update() -> None:
                try:
                    rr1=int(rs.value); rr2=int(re_.value); cc1=int(cs.value); cc2=int(ce.value)
                except Exception:
                    status.set("Neplatné hodnoty.", "error"); table_box.clear(); return
                if rr1<1 or cc1<1 or rr2<rr1 or cc2<cc1:
                    status.set("Rozsah není platný.", "warning"); table_box.clear(); return

                pick_repo.set(experiment_id, kind, rr1, rr2, cc1, cc2)
                st.data_version += 1

                df = extract_df(df_raw, rr1, rr2, cc1, cc2)
                if df.empty:
                    status.set("Výřez je prázdný.", "warning"); table_box.clear(); return

                ok, missing = validate_table(extract_df_dict(df_raw, rr1, rr2, cc1, cc2), required)
                status.set("OK: tabulka je validní." if ok else ("Chybí: "+", ".join(missing)), "ok" if ok else "warning")

                table_box.clear()
                with table_box:
                    df_show = sanitize_df_for_table(df.copy()).fillna("").infer_objects(copy=False)
                    for c in df_show.columns:
                        df_show[c] = df_show[c].astype(str)
                    cols, rows = _df_to_rows_columns(df_show)
                    StickyTable.from_rows_and_columns(rows=rows, columns=cols, sticky="both", max_height="420px").classes("w-full")

            for ctrl in (rs,re_,cs,ce):
                ctrl.on("update:model-value", lambda e: update())
            update()

    def render_all() -> None:
        dynamic.clear()
        with dynamic:
            ef = get_file()
            if not ef:
                ui.label("Zatím není nahraný žádný Excel soubor.").classes("text-caption")
                return
            ui.label(f"Aktuální soubor: {ef.filename} ({ef.size_bytes} B)").classes("text-subtitle2")

            xls = load_excel()
            if not xls:
                return

            sheet_box = ui.column().classes("w-full gap-2")
            preview_box = ui.column().classes("w-full gap-2")
            fame_box = ui.column().classes("w-full gap-2")
            epo_box = ui.column().classes("w-full gap-2")

            with sheet_box:
                ui.label("Krok 2: Vyberte list").classes("text-subtitle1")
                sheets = list(xls.sheet_names or [])
                if not sheets:
                    ui.label("Sešit neobsahuje žádné listy.").classes("text-negative")
                    return
                default_sheet = st.current_excel_sheet or sheets[0]
                if default_sheet not in sheets:
                    default_sheet = sheets[0]

                def on_sheet_change(e) -> None:
                    sheet = str(e.value)
                    log.info('[UI] load_data.sheet_change: sheet=%s', sheet)
                    st.current_excel_sheet = sheet
                    if st.current_excel_file_id is not None:
                        file_repo.set_selected_sheet(st.current_excel_file_id, sheet)
                    st.data_version += 1
                    df_raw = load_sheet_df(xls, sheet)
                    if df_raw is None:
                        return
                    render_preview(df_raw, sheet, preview_box)
                    render_pick("fame", df_raw, fame_box)
                    render_pick("epo", df_raw, epo_box)

                sel = ui.select(options=sheets, value=default_sheet, label="Listy", on_change=on_sheet_change).classes("w-96")
                on_sheet_change(type("E", (), {"value": sel.value})())

            preview_box
            ui.separator()
            fame_box
            ui.separator()
            epo_box

    async def handle_upload(e) -> None:
        try:
            data = await e.file.read()
        except Exception as ex:
            ui.notify(f"Nepodařilo se načíst soubor: {ex}", type="negative")
            return
        filename = getattr(e, "name", None) or getattr(e.file, "filename", None) or "upload.xlsx"
        log.info('[UI] load_data.upload: filename=%s bytes=%s', filename, len(data) if data is not None else None)
        file_repo.delete_for_experiment(experiment_id)
        st.current_excel_file_id=None; st.current_excel_filename=""; st.current_excel_sheet=""
        try:
            ef = file_repo.add_excel(experiment_id, filename, data)
        except Exception as ex:
            ui.notify(f"Nepodařilo se uložit do DB: {ex}", type="negative")
            return
        ui.notify(f"Uloženo do DB: {ef.filename} ({ef.size_bytes} B)")
        st.data_version += 1
        render_all()

    upload.on_upload(handle_upload)
    render_all()
