from __future__ import annotations

from nicegui import ui

from experimental_web.ui.layout import frame
from experimental_web.core.state import get_state


@ui.page("/experiment")
def page_experiment() -> None:
    st = get_state()
    if st.current_experiment_id is None:
        ui.navigate.to("/")
        return

    with frame("Experiment"):
        ui.label(f"Aktuální experiment: #{st.current_experiment_id} – {st.current_experiment_name}").classes("text-h6")

        tabs = ui.tabs().classes("w-full")
        tab_load = ui.tab("načtení dat")
        tab_process = ui.tab("zpracování")
        tab_speeds = ui.tab("rychlosti")
        tab_graphs = ui.tab("grafy")

        with ui.tab_panels(tabs, value=tab_load).classes("w-full"):
            with ui.tab_panel(tab_load):
                ui.label("Sem později dáme import/načítání vstupních dat.")
                ui.button("Mock: načíst data", on_click=lambda: ui.notify("Načtení dat (mock)"))

            with ui.tab_panel(tab_process):
                ui.label("Sem později dáme pipeline zpracování.")
                ui.button("Mock: zpracovat", on_click=lambda: ui.notify("Zpracování (mock)"))

            with ui.tab_panel(tab_speeds):
                ui.label("Sem později dáme výpočty rychlostí/metrik.")
                ui.button("Mock: spočítat rychlosti", on_click=lambda: ui.notify("Rychlosti (mock)"))

            with ui.tab_panel(tab_graphs):
                ui.label("Sem později dáme grafy.")
                ui.button("Mock: vykreslit grafy", on_click=lambda: ui.notify("Grafy (mock)"))
