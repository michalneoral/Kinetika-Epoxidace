from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Any

from nicegui import ui, app

from experimental_web.core.paths import APP_DIR, DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import SettingsRepository
from experimental_web.logging_setup import get_logger, is_debug_enabled
from experimental_web.ui.debug_panel import create_debug_log_dialog


log = get_logger(__name__)


THEME_KEY = "theme_mode"          # 'auto' | 'light' | 'dark' (stored in app.storage.user + DB)
DARK_VALUE_KEY = "dark_value"     # None | False | True (stored in app.storage.user, bound to ui.dark_mode)


def _mode_to_dark_value(mode: str) -> Optional[bool]:
    if mode == "dark":
        return True
    if mode == "light":
        return False
    return None  # auto


def _dark_value_to_mode(v: Any) -> str:
    # ui.dark_mode().value can be True/False/None
    if v is True:
        return "dark"
    if v is False:
        return "light"
    return "auto"


def _load_theme_mode() -> str:
    """Get theme mode from per-user storage; fall back to DB on first load."""
    if THEME_KEY not in app.storage.user:
        settings = SettingsRepository(DB_PATH)
        app.storage.user[THEME_KEY] = settings.get_theme_mode(default="auto")
    mode = str(app.storage.user.get(THEME_KEY, "auto"))
    return mode if mode in ("auto", "light", "dark") else "auto"


def _persist_theme_mode(mode: str) -> None:
    settings = SettingsRepository(DB_PATH)
    settings.set_theme_mode(mode)
    app.storage.user[THEME_KEY] = mode
    app.storage.user[DARK_VALUE_KEY] = _mode_to_dark_value(mode)


def _ensure_dark_binding(dark) -> None:
    """Bind ui.dark_mode() to app.storage.user so changes apply immediately (no reload).

    This mirrors the pattern from kinetika_webapp.py:
    - create dark_mode controller
    - bind_value to a model attribute
    - change via set_value(True/False/None)
    """
    mode = _load_theme_mode()
    if DARK_VALUE_KEY not in app.storage.user:
        app.storage.user[DARK_VALUE_KEY] = _mode_to_dark_value(mode)

    # bind_value makes Quasar dark plugin react immediately
    dark.bind_value(app.storage.user, DARK_VALUE_KEY)


def _set_dark_value(dark, value: Optional[bool]) -> None:
    """Set dark mode controller and persist mode to DB + storage."""
    if log.isEnabledFor(10):
        # 10 = DEBUG
        log.debug('[UI] theme: set dark_value=%s', value)
    dark.set_value(value)
    mode = _dark_value_to_mode(value)
    _persist_theme_mode(mode)
    ui.notify(f"Režim nastaven: {mode}")


def _close_experiment() -> None:
    st = get_state()
    log.info('[UI] close_experiment: id=%s name=%s', st.current_experiment_id, st.current_experiment_name)
    st.current_experiment_id = None
    st.current_experiment_name = ""
    st.current_excel_file_id = None
    st.current_excel_filename = ""
    st.current_excel_sheet = ""
    ui.notify("Experiment zavřen")
    ui.navigate.to("/")


@contextmanager
def frame(title: str):
    """Shared page frame used inside @ui.page functions."""

    dark = ui.dark_mode()
    _ensure_dark_binding(dark)



    debug_dialog = create_debug_log_dialog()

    with ui.header().classes("items-center bg-primary text-white"):
        st = get_state()
        ui.label().bind_text_from(
            st,
            'current_experiment_name',
            lambda name: f'Aktuální experiment: {name}' if st.current_experiment_id is not None else 'Kinetika-Epoxidace',
        ).classes('text-h6 text-white')
        ui.space()

        ui.button("Domů", on_click=lambda: (log.info('[UI] nav: home'), ui.navigate.to("/"))).props("flat text-color=white")

        if get_state().current_experiment_id is not None:
            ui.button("Zavřít experiment", on_click=_close_experiment).props("flat text-color=white")

        ui.button(
            "Aktuální experiment",
            on_click=lambda: (
                log.info('[UI] nav: experiment'),
                ui.navigate.to("/experiment")
                if get_state().current_experiment_id is not None
                else ui.notify("Žádný experiment není otevřen", type="warning")
            ),
        ).props("flat text-color=white")

        # Debug log button (only in debug mode)
        if debug_dialog is not None and is_debug_enabled():
            ui.button(icon='bug_report', on_click=lambda: (log.debug('[UI] open debug dialog'), debug_dialog.open()))\
                .props('flat fab-mini color=white')\
                .tooltip('Debug log')

        # --- Theme controls (instant, no reload) ---
        with ui.element().tooltip("Přepnout režim: dark → auto → light (okamžitě)"):
            # Pattern copied/adapted from kinetika_webapp.py
            ui.button(icon="dark_mode", on_click=lambda: _set_dark_value(dark, None)) \
                .props("flat fab-mini color=white") \
                .bind_visibility_from(dark, "value", value=True)
            ui.button(icon="light_mode", on_click=lambda: _set_dark_value(dark, True)) \
                .props("flat fab-mini color=white") \
                .bind_visibility_from(dark, "value", value=False)
            ui.button(icon="brightness_auto", on_click=lambda: _set_dark_value(dark, False)) \
                .props("flat fab-mini color=white") \
                .bind_visibility_from(dark, "value", lambda mode: mode is None)

        ui.button("O aplikaci", on_click=lambda: ui.notify(f"Data: {APP_DIR} | DB: {DB_PATH}")).props(
            "flat text-color=white"
        )

    with ui.column().classes("w-full q-pa-md"):
        if title:
            ui.label(title).classes("text-h5")
        yield
