from __future__ import annotations

from contextlib import contextmanager

from nicegui import ui

from experimental_web.core.paths import APP_DIR, DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import SettingsRepository


def _dark_controller():
    """Return the dark mode controller if available (NiceGUI 3.x)."""
    try:
        return ui.dark_mode()
    except TypeError:
        return None


def _apply_theme_mode(mode: str) -> None:
    dm = _dark_controller()
    if dm is not None:
        if mode == "dark":
            if hasattr(dm, "enable"):
                dm.enable()
            else:
                ui.dark_mode(True)
        elif mode == "light":
            if hasattr(dm, "disable"):
                dm.disable()
            else:
                ui.dark_mode(False)
        else:
            if hasattr(dm, "auto"):
                dm.auto()
            else:
                # best-effort: don't force anything
                pass
    else:
        # bool-API fallback
        if mode == "dark":
            ui.dark_mode(True)
        elif mode == "light":
            ui.dark_mode(False)
        else:
            pass


def _apply_theme() -> None:
    settings = SettingsRepository(DB_PATH)
    _apply_theme_mode(settings.get_theme_mode(default="auto"))


def _set_theme(mode: str) -> None:
    settings = SettingsRepository(DB_PATH)
    settings.set_theme_mode(mode)
    _apply_theme_mode(mode)  # apply immediately on this page/client
    ui.notify(f"Režim nastaven: {mode}")


def _close_experiment() -> None:
    st = get_state()
    st.current_experiment_id = None
    st.current_experiment_name = ""
    ui.notify("Experiment zavřen")
    ui.navigate.to("/")


@contextmanager
def frame(title: str):
    """Shared page frame used inside @ui.page functions."""
    _apply_theme()

    with ui.header().classes("items-center bg-primary text-white"):
        ui.label("Experiment Manager").classes("text-h6 text-white")
        ui.space()

        ui.button("Domů", on_click=lambda: ui.navigate.to("/")).props("flat text-color=white")

        if get_state().current_experiment_id is not None:
            ui.button("Zavřít experiment", on_click=_close_experiment).props("flat text-color=white")

        ui.button(
            "Aktuální experiment",
            on_click=lambda: (
                ui.navigate.to("/experiment")
                if get_state().current_experiment_id is not None
                else ui.notify("Žádný experiment není otevřen", type="warning")
            ),
        ).props("flat text-color=white")

        with ui.dropdown_button("Režim", auto_close=True).props("flat text-color=white"):
            ui.item("Auto", on_click=lambda: _set_theme("auto"))
            ui.item("Light", on_click=lambda: _set_theme("light"))
            ui.item("Dark", on_click=lambda: _set_theme("dark"))

        ui.button("O aplikaci", on_click=lambda: ui.notify(f"Data: {APP_DIR} | DB: {DB_PATH}")).props(
            "flat text-color=white"
        )

    with ui.column().classes("w-full q-pa-md"):
        ui.label(title).classes("text-h5")
        yield
