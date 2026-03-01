from __future__ import annotations

import io
from typing import Optional, Tuple

import pandas as pd
from nicegui import ui, events

from abstract_tab import AbstractTab
from utils import df_to_json_bytes, safe_int
from ui_helpers import info, success


class InputTab(AbstractTab):
    name = "input"

    def __init__(self, orchestrator: "Kinetika") -> None:
        super().__init__(orchestrator)
        self.raw_xls_bytes: Optional[bytes] = None
        self.sheet_name: Optional[str] = None
        self.raw_df: Optional[pd.DataFrame] = None

        self.fame_range = {"row_start": 0, "row_end": 10, "col_start": 0, "col_end": 5}
        self.epo_range = {"row_start": 0, "row_end": 10, "col_start": 0, "col_end": 5}

    async def load_from_db(self) -> None:
        db = self.orchestrator.db

        self.fame_range["row_start"] = db.get_path(("input", "fame", "row_start"), 0)
        self.fame_range["row_end"] = db.get_path(("input", "fame", "row_end"), 10)
        self.fame_range["col_start"] = db.get_path(("input", "fame", "col_start"), 0)
        self.fame_range["col_end"] = db.get_path(("input", "fame", "col_end"), 5)

        self.epo_range["row_start"] = db.get_path(("input", "epo", "row_start"), 0)
        self.epo_range["row_end"] = db.get_path(("input", "epo", "row_end"), 10)
        self.epo_range["col_start"] = db.get_path(("input", "epo", "col_start"), 0)
        self.epo_range["col_end"] = db.get_path(("input", "epo", "col_end"), 5)

        self.sheet_name = db.get_path(("input", "sheet_name"), None)

        fame_blob = db.get_blob("tables/fame_df")
        epo_blob = db.get_blob("tables/epo_df")
        if fame_blob and epo_blob:
            try:
                from utils import df_from_json_bytes
                self.orchestrator.processor.add_fame(df_from_json_bytes(fame_blob[0]))
                self.orchestrator.processor.add_epo(df_from_json_bytes(epo_blob[0]))
            except Exception:
                pass

    def _try_slice(self, df: pd.DataFrame, r: dict) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        try:
            rs, re = int(r["row_start"]), int(r["row_end"])
            cs, ce = int(r["col_start"]), int(r["col_end"])
            if rs < 0 or cs < 0 or re <= rs or ce <= cs:
                return False, "Neplatný rozsah (start/end).", None
            cut = df.iloc[rs:re, cs:ce]
            if cut.empty:
                return False, "Výřez je prázdný.", None
            return True, "OK", cut
        except Exception as e:
            return False, f"Chyba při výřezu: {e}", None

    def _persist_range(self, kind: str, key: str, value: int) -> None:
        self.orchestrator.db.set_path(("input", kind, key), value)

    def _persist_sheet(self, value: Optional[str]) -> None:
        self.orchestrator.db.set_path(("input", "sheet_name"), value)

    @ui.refreshable
    def show(self) -> None:
        ui.markdown("## Input (Krok 1) – Upload Excelu + výběr FAME/EPO výřezů")

        with ui.card().classes("w-full"):
            ui.markdown("### Krok 1: Upload Excelu")
            ui.upload(
                label="Nahraj .xlsx/.xls",
                auto_upload=True,
                on_upload=self._on_upload,  # async OK
            ).props("accept=.xlsx,.xls")

            if self.raw_xls_bytes is None:
                info("Nahraj Excel, pak se objeví výběr sheetu a rozsahy.")
                return

            ui.separator()
            ui.markdown("### Krok 2: Výběr sheetu")
            ui.select(
                options=self._list_sheets(),
                value=self.sheet_name,
                label="Sheet",
                on_change=lambda e: self._on_sheet_selected(e.value),
            ).classes("w-80")

            if self.raw_df is None:
                info("Vyber sheet, aby se načetla raw tabulka.")
                return

            ui.separator()
            ui.markdown("### Raw tabulka (preview)")
            ui.table.from_pandas(self.raw_df.head(20)).classes("w-full")

            ui.separator()
            with ui.row().classes("w-full gap-6"):
                with ui.card().classes("w-1/2"):
                    ui.markdown("### FAME rozsah")
                    self._range_editor("fame", self.fame_range)
                    ok, msg, fame_cut = self._try_slice(self.raw_df, self.fame_range)
                    ui.label(f"Validace: {msg}").classes("text-sm")
                    if ok and fame_cut is not None:
                        ui.table.from_pandas(fame_cut.head(20)).classes("w-full")
                        self.orchestrator.processor.add_fame(fame_cut)
                        self.orchestrator.db.put_blob("tables/fame_df", df_to_json_bytes(fame_cut), "application/json")
                        self.orchestrator.db.log_step("input", "fame_cut_valid", {"shape": list(fame_cut.shape)})

                with ui.card().classes("w-1/2"):
                    ui.markdown("### EPO rozsah")
                    self._range_editor("epo", self.epo_range)
                    ok, msg, epo_cut = self._try_slice(self.raw_df, self.epo_range)
                    ui.label(f"Validace: {msg}").classes("text-sm")
                    if ok and epo_cut is not None:
                        ui.table.from_pandas(epo_cut.head(20)).classes("w-full")
                        self.orchestrator.processor.add_epo(epo_cut)
                        self.orchestrator.db.put_blob("tables/epo_df", df_to_json_bytes(epo_cut), "application/json")
                        self.orchestrator.db.log_step("input", "epo_cut_valid", {"shape": list(epo_cut.shape)})

            if self.orchestrator.processor.has_data():
                success("FAME + EPO jsou dostupné. Navazující taby se budou aktualizovat.")
                self.propagate_changes(from_tab=1)

    def _range_editor(self, kind: str, r: dict) -> None:
        def bind_int(field: str, minv: int = 0, maxv: int = 10_000):
            ui.number(
                label=field,
                value=int(r[field]),
                min=minv,
                max=maxv,
                step=1,
                format="%.0f",
                on_change=lambda e: self._on_range_change(kind, field, e.value),
            ).classes("w-32")

        with ui.row().classes("gap-3"):
            bind_int("row_start")
            bind_int("row_end")
            bind_int("col_start")
            bind_int("col_end")

    def _on_range_change(self, kind: str, field: str, value) -> None:
        v = safe_int(value, self.fame_range[field] if kind == "fame" else self.epo_range[field])
        if kind == "fame":
            self.fame_range[field] = v
        else:
            self.epo_range[field] = v
        self._persist_range(kind, field, v)
        self.orchestrator.db.log_step("input", f"{kind}_range_changed", {"field": field, "value": v})
        self.refresh()

    async def _on_upload(self, e: events.UploadEventArguments) -> None:
        data = await e.file.read()  # NiceGUI 3.x
        self.raw_xls_bytes = data
        self.raw_df = None
        self.sheet_name = None
        self.orchestrator.db.log_step("input", "excel_uploaded", {"bytes": len(data)})
        self.refresh()

    def _list_sheets(self) -> list[str]:
        if not self.raw_xls_bytes:
            return []
        try:
            xls = pd.ExcelFile(io.BytesIO(self.raw_xls_bytes))
            return list(xls.sheet_names)
        except Exception:
            return []

    def _on_sheet_selected(self, sheet: Optional[str]) -> None:
        self.sheet_name = sheet
        self._persist_sheet(sheet)
        self.orchestrator.db.log_step("input", "sheet_selected", {"sheet": sheet})
        try:
            if self.raw_xls_bytes and sheet:
                self.raw_df = pd.read_excel(io.BytesIO(self.raw_xls_bytes), sheet_name=sheet)
        except Exception as ex:
            self.raw_df = None
            ui.notify(f"Chyba při čtení Excelu: {ex}", type="negative")
        self.refresh()
        self.propagate_changes(from_tab=1)
