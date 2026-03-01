from __future__ import annotations

from nicegui import ui

from abstract_tab import AbstractTab
from ui_helpers import info, warning, success

class TablesTab(AbstractTab):
    name = "tables"

    def is_accessible(self) -> bool:
        return self.orchestrator.processor.has_data()

    async def load_from_db(self) -> None:
        # expanze per tabulka – uloženo jako dict
        # (lazy load až když existují data)
        return

    @ui.refreshable
    def show(self) -> None:
        ui.markdown("## Tables (Krok 2) – Zpracované tabulky")

        if not self.orchestrator.processor.has_data():
            warning("Tabulky ještě nejsou načteny (nejdřív Input tab).")
            return

        self.orchestrator.processor.process()

        processed = self.orchestrator.processor.processed
        if not processed:
            warning("Zatím nic k zobrazení.")
            return

        ui.markdown("Zde jsou dummy ‘zpracované’ tabulky (processor.process()).")

        for name, df in processed.items():
            # expanze state z configu
            key = f"tables/expanded/{name}"
            expanded = bool(self.orchestrator.db.get_path(("tables", "expanded", name), True))

            with ui.expansion(name, value=expanded).classes("w-full") as exp:
                def _on_toggle(e, tab_name=name):
                    self.orchestrator.db.set_path(("tables", "expanded", tab_name), bool(e.value))
                    self.orchestrator.db.log_step("tables", "table_expand_toggled", {"table": tab_name, "expanded": bool(e.value)})

                exp.on("update:model-value", _on_toggle)

                ui.markdown(f"**Popis:** `{name}` (dummy text / LaTeX můžeš doplnit)")
                ui.table.from_pandas(df.head(50)).classes("w-full")
