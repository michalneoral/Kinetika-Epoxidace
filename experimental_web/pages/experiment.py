from __future__ import annotations

from nicegui import ui

from experimental_web.ui.layout import frame
from experimental_web.core.state import get_state
from experimental_web.ui.experiment.load_data_tab import render_load_data_tab


@ui.page("/experiment")
def page_experiment() -> None:
    st = get_state()

    with frame(""):
        if st.current_experiment_id is None:
            ui.label("Žádný experiment není otevřen.").classes("text-h6")
            ui.button("Zpět na správu experimentů", on_click=lambda: ui.navigate.to("/")).props("unelevated")
            return


        with ui.tabs().classes("w-full") as tabs:
            tab_load = ui.tab("načtení dat")
            tab_process = ui.tab("zpracování")
            tab_speeds = ui.tab("rychlosti")
            tab_graphs = ui.tab("grafy")

        with ui.tab_panels(tabs, value=tab_load).classes("w-full"):
            with ui.tab_panel(tab_load):
                render_load_data_tab(st.current_experiment_id)

            with ui.tab_panel(tab_process):
                from experimental_web.ui.experiment.tables_tab import render_tables_tab
                render_tables_tab(st.current_experiment_id)

            with ui.tab_panel(tab_speeds):
                from experimental_web.ui.experiment.processing_tab import render_processing_tab
                render_processing_tab(st.current_experiment_id)

            with ui.tab_panel(tab_graphs):
                ui.label("Sem dáme grafy.").classes("text-body1")
