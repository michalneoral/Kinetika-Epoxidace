from __future__ import annotations

from contextlib import contextmanager
from typing import Optional, Any

from nicegui import ui, app

from experimental_web.core.paths import APP_DIR, DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import SettingsRepository
from experimental_web.logging_setup import get_logger, is_debug_enabled
from experimental_web.ui.debug_panel import create_debug_log_dialog
from experimental_web.ui.styled_elements.custom_color_picker import ColorPickerButton
from experimental_web.ui.utils.tooltips import attach_tooltip, push_tooltip_settings_to_client


log = get_logger(__name__)


THEME_KEY = "theme_mode"          # 'auto' | 'light' | 'dark' (stored in app.storage.user + DB)
DARK_VALUE_KEY = "dark_value"     # None | False | True (stored in app.storage.user, bound to ui.dark_mode)

# Global UI help settings (stored in app.storage.user + DB)
# Convention:
# - <0 => disabled
# - 0  => show immediately
# - >0 => delay in ms
HELP_TOOLTIP_DELAY_KEY = 'help_tooltip_delay_ms'

# Global UI colors (stored in app.storage.user + DB)
UI_COLORS_KEY = 'ui_colors'

# --- UI color palettes ---
# NiceGUI/Quasar-ish defaults (baseline).
UI_COLORS_NICEGUI_DEFAULTS: dict[str, str] = {
    'primary': '#1976D2',
    'secondary': '#26A69A',
    'accent': '#9C27B0',
    'positive': '#21BA45',
    'negative': '#C10015',
    'info': '#31CCEC',
    'warning': '#F2C037',
}

# UPCE defaults: keep the baseline palette, but use UPCE primary.
# RGB(152, 25, 78) => #98194E
UI_COLORS_UPCE_DEFAULTS: dict[str, str] = {
    **UI_COLORS_NICEGUI_DEFAULTS,
    'primary': '#98194E',
}

# Project default palette.
UI_COLORS_DEFAULTS: dict[str, str] = UI_COLORS_UPCE_DEFAULTS


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


def _load_help_tooltip_delay_ms() -> int:
    """Load global tooltip delay from per-user storage; fall back to DB on first load."""
    if HELP_TOOLTIP_DELAY_KEY not in app.storage.user:
        settings = SettingsRepository(DB_PATH)
        app.storage.user[HELP_TOOLTIP_DELAY_KEY] = settings.get_help_tooltip_delay_ms(default=2000)
    try:
        return int(app.storage.user.get(HELP_TOOLTIP_DELAY_KEY, 2000))
    except Exception:
        return 2000


def _persist_help_tooltip_delay_ms(delay_ms: int) -> None:
    delay_ms = int(delay_ms)
    settings = SettingsRepository(DB_PATH)
    settings.set_help_tooltip_delay_ms(delay_ms)
    app.storage.user[HELP_TOOLTIP_DELAY_KEY] = delay_ms


def _normalize_hex_color(value: str, default: str) -> str:
    import re

    s = (value or '').strip()
    if re.fullmatch(r'#([0-9a-fA-F]{6}|[0-9a-fA-F]{8})', s):
        return s
    return default


def _load_ui_colors() -> dict[str, str]:
    """Load UI palette colors from per-user storage; fall back to DB on first load."""
    if UI_COLORS_KEY not in app.storage.user:
        settings = SettingsRepository(DB_PATH)
        colors = settings.get_ui_colors(UI_COLORS_DEFAULTS)
        # normalize to valid hex strings
        colors = {k: _normalize_hex_color(v, UI_COLORS_DEFAULTS[k]) for k, v in colors.items()}
        app.storage.user[UI_COLORS_KEY] = colors
    colors = dict(app.storage.user.get(UI_COLORS_KEY, UI_COLORS_DEFAULTS))
    # ensure all keys exist
    for k, d in UI_COLORS_DEFAULTS.items():
        colors[k] = _normalize_hex_color(colors.get(k, d), d)
    app.storage.user[UI_COLORS_KEY] = colors
    return colors


def _persist_ui_color(name: str, color: str) -> str:
    """Persist one UI color and return the normalized stored value."""
    name = str(name)
    if name not in UI_COLORS_DEFAULTS:
        return ''
    normalized = _normalize_hex_color(str(color), UI_COLORS_DEFAULTS[name])
    settings = SettingsRepository(DB_PATH)
    settings.set_ui_color(name, normalized)
    colors = _load_ui_colors()
    colors[name] = normalized
    app.storage.user[UI_COLORS_KEY] = colors
    return normalized


def _persist_ui_colors(colors: dict[str, str]) -> dict[str, str]:
    settings = SettingsRepository(DB_PATH)
    out = {}
    for k, d in UI_COLORS_DEFAULTS.items():
        out[k] = _normalize_hex_color(str(colors.get(k, d)), d)
        settings.set_ui_color(k, out[k])
    app.storage.user[UI_COLORS_KEY] = dict(out)
    return out


def _apply_ui_colors(colors: dict[str, str]) -> None:
    """Apply Quasar palette colors to the current client."""
    # ui.colors() updates Quasar CSS variables which are used by classes like bg-primary.
    ui.colors(**colors)


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

    # Apply global UI palette colors early so header/tabs use them.
    _apply_ui_colors(_load_ui_colors())

    # Publish tooltip settings early so all tooltips on this page follow it.
    push_tooltip_settings_to_client(delay_ms=_load_help_tooltip_delay_ms())



    debug_dialog = create_debug_log_dialog()

    # Settings dialog (global)
    settings_dialog = ui.dialog()

    def _open_settings() -> None:
        settings_dialog.open()

    with ui.header().classes("items-center bg-primary text-white"):
        st = get_state()
        ui.label().bind_text_from(
            st,
            'current_experiment_name',
            lambda name: f'Aktuální experiment: {name}' if st.current_experiment_id is not None else 'Kinetika-Epoxidace',
        ).classes('text-h6 text-white')
        ui.space()

        btn_home = ui.button(
            "Domů",
            on_click=lambda: (log.info('[UI] nav: home'), ui.navigate.to("/")),
        ).props("flat text-color=white")
        attach_tooltip(
            btn_home,
            "Domů",
            "Přejde na úvodní stránku se seznamem experimentů.",
        )

        if get_state().current_experiment_id is not None:
            btn_close = ui.button("Zavřít experiment", on_click=_close_experiment).props("flat text-color=white")
            attach_tooltip(
                btn_close,
                "Zavřít experiment",
                "Uzavře aktuálně otevřený experiment a vrátí tě domů. Data v databázi zůstanou zachovaná.",
            )

        btn_current = ui.button(
            "Aktuální experiment",
            on_click=lambda: (
                log.info('[UI] nav: experiment'),
                ui.navigate.to("/experiment")
                if get_state().current_experiment_id is not None
                else ui.notify("Žádný experiment není otevřen", type="warning")
            ),
        ).props("flat text-color=white")
        attach_tooltip(
            btn_current,
            "Aktuální experiment",
            "Otevře hlavní stránku experimentu (záložky: Načtení dat, Tabulky, Rychlosti, Grafy, …). Pokud není nic otevřeno, zobrazí se varování.",
        )

        # Debug log button (only in debug mode)
        if debug_dialog is not None and is_debug_enabled():
            btn_debug = ui.button(
                icon='bug_report',
                on_click=lambda: (log.debug('[UI] open debug dialog'), debug_dialog.open()),
            ).props('flat fab-mini color=white')

            attach_tooltip(
                btn_debug,
                "Debug log",
                "Otevře panel s interními logy aplikace. Hodí se při ladění a při hlášení chyb. "
                "Tlačítko je viditelné jen v debug režimu.",
            )

        # --- Theme controls (instant, no reload) ---
        with ui.element() as theme_controls:
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

        attach_tooltip(
            theme_controls,
            "Vzhled (dark/auto/light)",
            "Přepíná režim zobrazení: tmavý / automatický / světlý. Změna se projeví okamžitě bez reloadu a uloží se pro tohoto uživatele.",
        )

        btn_settings = ui.button(icon='settings', on_click=_open_settings).props('flat fab-mini color=white')
        attach_tooltip(
            btn_settings,
            'Nastavení',
            'Otevře dialog globálních nastavení aplikace (např. chování nápovědy/tooltipů). Nastavení je nezávislé na experimentech.',
        )


    # --- Settings dialog content ---
    with settings_dialog, ui.card().classes('w-[720px]'):
        title_lbl = ui.label('Nastavení').classes('text-h6')
        attach_tooltip(title_lbl, 'Globální nastavení', 'Nastavení platí pro celou aplikaci a ukládá se do databáze (není svázané s experimentem).')

        ui.separator()

        tabs = ui.tabs().classes('w-full')
        with tabs:
            ui.tab('Vzhled').props('name="appearance"')
            ui.tab('Nápověda').props('name="help"')
            ui.tab('O aplikaci').props('name="about"')

        with ui.tab_panels(tabs, value='appearance').classes('w-full'):

            # --- Appearance tab ---
            with ui.tab_panel('appearance'):
                ui.label('Vzhled').classes('text-subtitle1')
                ui.label('Barvy rozhraní (Quasar/NiceGUI paleta)').classes('text-caption text-grey')

                colors_model = _load_ui_colors()
                value_labels: dict[str, ui.label] = {}
                pickers: dict[str, ColorPickerButton] = {}

                def _apply_and_notify() -> None:
                    _apply_ui_colors(colors_model)
                    ui.notify('Nastavení vzhledu uloženo')

                def _on_color_pick(color_name: str, new_color: str, value_lbl: ui.label) -> None:
                    stored = _persist_ui_color(color_name, new_color)
                    if stored:
                        colors_model[color_name] = stored
                        value_lbl.set_text(stored)
                        _apply_and_notify()

                # Display palette controls.
                palette = [
                    ('primary', 'Primární', 'Hlavní barva UI (např. header a zvýraznění).'),
                    ('secondary', 'Sekundární', 'Doplňková barva UI (sekundární prvky).'),
                    ('accent', 'Akcent', 'Akcentní barva pro zvýraznění vybraných prvků.'),
                    ('positive', 'Pozitivní', 'Barva pro úspěch/OK stavy.'),
                    ('negative', 'Negativní', 'Barva pro chyby/varování.'),
                    ('info', 'Info', 'Barva pro informační hlášky.'),
                    ('warning', 'Varování', 'Barva pro upozornění.'),
                ]

                for key, label, desc in palette:
                    with ui.row().classes('items-center w-full q-py-xs'):
                        lbl = ui.label(label).classes('w-40')
                        value_lbl = ui.label(colors_model.get(key, UI_COLORS_DEFAULTS[key])).classes('text-caption text-grey')
                        value_labels[key] = value_lbl

                        picker = ColorPickerButton(
                            icon='palette',
                            color=colors_model.get(key, UI_COLORS_DEFAULTS[key]),
                            on_pick=lambda e, k=key, vl=value_lbl: _on_color_pick(k, e.color, vl),
                        )
                        pickers[key] = picker

                        # Tooltip on the visible button.
                        attach_tooltip(picker.button, label, desc)

                with ui.row().classes('justify-end w-full q-pt-md'):
                    def _reset_with_palette(palette: dict[str, str]) -> None:
                        # Persist exact palette values (normalize per-key) and apply.
                        settings = SettingsRepository(DB_PATH)
                        stored: dict[str, str] = {}
                        for k in UI_COLORS_NICEGUI_DEFAULTS.keys():
                            fallback = str(palette.get(k, UI_COLORS_DEFAULTS.get(k, '#000000')))
                            stored[k] = _normalize_hex_color(str(palette.get(k, fallback)), fallback)
                            settings.set_ui_color(k, stored[k])
                        app.storage.user[UI_COLORS_KEY] = dict(stored)

                        colors_model.update(stored)
                        for k, v in stored.items():
                            if k in value_labels:
                                value_labels[k].set_text(v)
                            if k in pickers:
                                pickers[k].set_color(v)
                        _apply_and_notify()

                    btn_reset_nicegui = ui.button('Reset na NiceGUI', on_click=lambda: _reset_with_palette(UI_COLORS_NICEGUI_DEFAULTS)).props('flat')
                    attach_tooltip(
                        btn_reset_nicegui,
                        'Reset na NiceGUI',
                        'Nastaví barvy na výchozí paletu NiceGUI/Quasar (původní výchozí barvy).',
                    )

                    btn_reset_upce = ui.button('Reset na UPCE', on_click=lambda: _reset_with_palette(UI_COLORS_UPCE_DEFAULTS)).props('flat')
                    attach_tooltip(
                        btn_reset_upce,
                        'Reset na UPCE',
                        'Nastaví barvy na paletu UPCE. Primární barva je RGB(152, 25, 78) = #98194E. Tohle je i výchozí nastavení aplikace.',
                    )

            # --- Help tab ---
            with ui.tab_panel('help'):
                ui.label('Nápověda (tooltips)').classes('text-subtitle1')

                # Tooltip delay choices (label v UI -> prodleva v ms).
                # Convention:
                # - -1 => disabled
                # - 0  => show immediately
                # - >0 => delay
                delay_choices = [
                    ('Nezobrazovat', -1),
                    ('Zobrazit hned', 0),
                    ('Zobrazit po 1 s', 1000),
                    ('Zobrazit po 2 s', 2000),
                    ('Zobrazit po 5 s', 5000),
                ]
                label_to_ms = {label: ms for (label, ms) in delay_choices}

                def _delay_ms_to_label(ms: int) -> str:
                    try:
                        ms_i = int(ms)
                    except Exception:
                        ms_i = 2000
                    for label, v in delay_choices:
                        if int(v) == ms_i:
                            return label
                    # Fallback: default 2 s
                    return 'Zobrazit po 2 s'

                def _on_help_delay_change(e) -> None:
                    label = str(getattr(e, 'value', '') or '')
                    delay_ms = int(label_to_ms.get(label, 2000))
                    _persist_help_tooltip_delay_ms(delay_ms)
                    push_tooltip_settings_to_client(delay_ms=delay_ms)
                    ui.notify('Nastavení nápovědy uloženo')

                help_toggle = ui.toggle(
                    options=[label for (label, _ms) in delay_choices],
                    value=_delay_ms_to_label(_load_help_tooltip_delay_ms()),
                    on_change=_on_help_delay_change,
                ).props('spread')
                attach_tooltip(
                    help_toggle,
                    'Nastavení nápovědy',
                    'Určuje, jestli se mají zobrazovat tooltippy a po jaké prodlevě. "Nezobrazovat" je vypne úplně. "Zobrazit hned" zobrazí tooltip bez čekání.',
                )

            # --- About tab ---
            with ui.tab_panel('about'):
                import os
                from experimental_web.core.config import APP_NAME, UPDATE_REPO_URL, UPDATE_OWNER, UPDATE_REPO
                from experimental_web.core.version import __version__
                from experimental_web.core.runtime_control import default_port, open_in_browser, read_saved_port, request_shutdown

                about_title = ui.label('O aplikaci').classes('text-subtitle1')
                attach_tooltip(
                    about_title,
                    'O aplikaci',
                    'Základní info o běhu aplikace (umístění dat a cesta k databázi).',
                )
                ui.label('Základní informace o běhu aplikace.').classes('text-caption text-grey')

                with ui.column().classes('q-gutter-xs q-mt-sm'):
                    ui.label(f'Verze: {__version__}').classes('text-body2')
                    repo_link = ui.link(f'GitHub: {UPDATE_OWNER}/{UPDATE_REPO}', UPDATE_REPO_URL, new_tab=True).classes('text-body2')
                    attach_tooltip(
                        repo_link,
                        'Repozitář na GitHubu',
                        'Odkaz na projekt na GitHubu. Z tohoto repozitáře se stahují nové verze při automatické aktualizaci.',
                    )

                    ui.label(f'Data: {APP_DIR}').classes('text-body2')
                    ui.label(f'Databáze: {DB_PATH}').classes('text-body2')
                    ui.label(f'Debug: {"ON" if is_debug_enabled() else "OFF"}').classes('text-body2')

                def _current_port() -> int:
                    # Prefer current process port, then saved port, then default.
                    env_port = os.getenv('EXPERIMENTAL_WEB_ACTIVE_PORT', '').strip()
                    if env_port.isdigit():
                        p = int(env_port)
                        if 1 <= p <= 65535:
                            return p
                    return read_saved_port(APP_DIR) or default_port()

                quit_dialog = ui.dialog()
                with quit_dialog, ui.card().classes('w-[520px]'):
                    lbl = ui.label('Ukončit aplikaci?').classes('text-subtitle1')
                    attach_tooltip(
                        lbl,
                        'Ukončit aplikaci',
                        'Aplikace běží jako lokální server na pozadí. '
                        'Tato volba ukončí běžící proces (užitečné, když je zavřený prohlížeč).',
                    )
                    ui.label('Aplikace se ukončí a webová stránka přestane reagovat.').classes('text-caption text-grey')
                    with ui.row().classes('justify-end q-gutter-sm q-mt-md'):
                        ui.button('Zrušit', on_click=quit_dialog.close).props('flat')
                        ui.button(
                            'Ukončit',
                            icon='power_settings_new',
                            on_click=lambda: (request_shutdown(_current_port()), ui.notify('Ukončuji aplikaci…', type='warning')),
                        ).props('color=negative')

                ui.separator().classes('q-mt-md')
                with ui.row().classes('items-center q-gutter-sm q-mt-sm'):
                    btn_open = ui.button(
                        'Otevřít v prohlížeči',
                        icon='open_in_new',
                        on_click=lambda: open_in_browser(_current_port()),
                    ).props('outline')
                    attach_tooltip(
                        btn_open,
                        'Otevřít v prohlížeči',
                        'Otevře (nový tab) s běžící aplikací. Hodí se, když se tab zavřel, ale server pořád běží.',
                    )

                    btn_quit = ui.button(
                        'Ukončit aplikaci',
                        icon='power_settings_new',
                        on_click=quit_dialog.open,
                    ).props('color=negative outline')
                    attach_tooltip(
                        btn_quit,
                        'Ukončit aplikaci',
                        'Ukončí běžící proces aplikace. Použij, když nechceš, aby aplikace běžela na pozadí.',
                    )

        with ui.row().classes('justify-end w-full q-pt-md'):
            close_btn = ui.button('Zavřít', on_click=settings_dialog.close).props('flat')
            attach_tooltip(close_btn, 'Zavřít', 'Zavře dialog nastavení bez dalších změn.')

    with ui.column().classes("w-full q-pa-md"):
        if title:
            ui.label(title).classes("text-h5")
        yield