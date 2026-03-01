from __future__ import annotations

import threading
import time

from nicegui import ui

from graph_tab import GraphTab
from header import frame
from input_tab import InputTab
from persistence import ConfigDB
from processing_tab import ProcessingTab
from table_processor import TableProcessor
from tables_tab import TablesTab
from utils import schedule


class Kinetika:
    """
    Orchestrátor:
      - drží db + config + processor
      - tvoří taby
      - řídí propagate_changes
      - lazy-load tab state až když je přístupný
    """

    def __init__(self) -> None:
        self.db = ConfigDB("kinetika_config.sqlite")
        self.config = self.db.get_config()

        self.processor = TableProcessor()

        self.input_tab = InputTab(self)
        self.tables_tab = TablesTab(self)
        self.processing_tab = ProcessingTab(self)
        self.graph_tab = GraphTab(self)

        self.tabs = [self.input_tab, self.tables_tab, self.processing_tab, self.graph_tab]

    def set_gui_mode(self, mode: str) -> None:
        self.db.set_path(("gui", "mode"), mode)
        self.db.log_step("app", "gui_mode_changed", {"mode": mode})

        if mode == "dark":
            ui.dark_mode().enable()
        elif mode == "light":
            ui.dark_mode().disable()
        else:
            ui.notify("AUTO režim je v dummy verzi placeholder.", type="info")

    def load_gui_mode(self) -> None:
        mode = self.db.get_path(("gui", "mode"), "auto")
        self.set_gui_mode(mode)

    def propagate_changes(self, from_tab: int) -> None:
        # Pipeline:
        # 1 Input -> refresh Tables + Processing
        # 2 Tables -> refresh Processing
        # 3 Processing -> refresh Graph
        # 4 Graph -> nic
        if from_tab == 1:
            self.tables_tab.refresh()
            self.processing_tab.refresh()
        elif from_tab == 2:
            self.processing_tab.refresh()
        elif from_tab == 3:
            self.graph_tab.refresh()

    def start_background_checker(self) -> None:
        def _worker():
            while True:
                time.sleep(10.0)
                # dummy “check update”
                pass

        threading.Thread(target=_worker, daemon=True).start()


def main() -> None:
    app = Kinetika()
    app.load_gui_mode()
    app.start_background_checker()

    with frame("Kinetika (dummy migrace)", "v0.1"):
        with ui.row().classes("items-center"):
            ui.button("Light", on_click=lambda: app.set_gui_mode("light")).props("outline dense")
            ui.button("Dark", on_click=lambda: app.set_gui_mode("dark")).props("outline dense")
            ui.button("Auto", on_click=lambda: app.set_gui_mode("auto")).props("outline dense")
            ui.separator().classes("mx-3")
            ui.button("Steps log (posledních 50)", on_click=lambda: show_steps_dialog(app)).props("outline dense")

    tabs = ui.tabs().classes("w-full")
    with tabs:
        ui.tab("Input")
        ui.tab("Data")
        ui.tab("Speeds")
        ui.tab("Graphs")

    panels = ui.tab_panels(tabs, value="Input").classes("w-full")
    with panels:
        with ui.tab_panel("Input"):
            ui.timer(0, lambda: schedule(app.input_tab.ensure_loaded()), once=True)
            app.input_tab.show()
        with ui.tab_panel("Data"):
            ui.timer(0, lambda: schedule(app.tables_tab.ensure_loaded()), once=True)
            app.tables_tab.show()
        with ui.tab_panel("Speeds"):
            ui.timer(0, lambda: schedule(app.processing_tab.ensure_loaded()), once=True)
            app.processing_tab.show()
        with ui.tab_panel("Graphs"):
            ui.timer(0, lambda: schedule(app.graph_tab.ensure_loaded()), once=True)
            app.graph_tab.show()

    def _on_tab_change(e):
        mapping = {
            "Input": app.input_tab,
            "Data": app.tables_tab,
            "Speeds": app.processing_tab,
            "Graphs": app.graph_tab,
        }
        tab = mapping.get(e.value)
        if tab:
            schedule(tab.ensure_loaded())

    tabs.on("update:model-value", _on_tab_change)

    ui.run(title="Kinetika dummy", reload=False)


def show_steps_dialog(app: Kinetika) -> None:
    steps = app.db.list_steps(limit=50)
    with ui.dialog() as dialog, ui.card().classes("w-[900px]"):
        ui.markdown("### Steps log (posledních 50)")
        for s in steps:
            ui.markdown(f"- `{s.id}` • `{s.tab}` • `{s.event_type}` • `{time.ctime(s.ts)}`")
        ui.button("Zavřít", on_click=dialog.close).props("outline")
    dialog.open()


if __name__ in {"__main__", "__mp_main__"}:
    main()
