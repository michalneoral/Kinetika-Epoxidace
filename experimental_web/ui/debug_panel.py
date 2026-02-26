from __future__ import annotations

from dataclasses import dataclass

from nicegui import ui

from experimental_web.logging_setup import get_log_buffer_handler, get_logger, is_debug_enabled


log = get_logger(__name__)


@dataclass
class DebugPanelState:
    last_id: int = 0
    paused: bool = False
    max_lines: int = 400


def create_debug_log_dialog() -> ui.dialog | None:
    """Create a debug log dialog.

    - Only created when debug is enabled.
    - Shows last N log lines from the in-memory ring buffer.
    """
    if not is_debug_enabled():
        return None

    buf = get_log_buffer_handler()
    if buf is None:
        return None

    state = DebugPanelState()
    dialog = ui.dialog().props('maximized')

    with dialog, ui.card().classes('w-full h-full'):
        with ui.row().classes('w-full items-center justify-between'):
            ui.label('Debug log').classes('text-h6')

            # Placeholders; buttons will be wired after log_view exists.
            buttons_row = ui.row().classes('items-center gap-2')

        ui.label('Zobrazuje poslední logy ze serverového procesu.').classes('text-caption text-grey-7')
        ui.separator().classes('q-mt-sm q-mb-sm')

        # Textarea is a safe, dependency-free way to render logs.
        log_view = ui.textarea(value='')\
            .props('readonly autogrow')\
            .classes('w-full h-full font-mono text-xs')

        with buttons_row:
            pause_btn = ui.button('Pause', icon='pause', on_click=lambda: _toggle_pause(state, pause_btn)).props('outline')
            ui.button('Clear view', icon='delete', on_click=lambda: _clear_view(state, log_view)).props('outline')
            ui.button('Close', icon='close', on_click=dialog.close).props('flat')

    def poll() -> None:
        if state.paused:
            return
        try:
            new = buf.get_since(state.last_id)
        except Exception:
            return
        if not new:
            return

        state.last_id = new[-1][0]
        lines = (log_view.value or '').splitlines()
        lines.extend([s for _, s in new])
        if len(lines) > state.max_lines:
            lines = lines[-state.max_lines :]
        log_view.value = '\n'.join(lines)

    # Poll logs periodically (only updates when there are new lines).
    ui.timer(0.5, poll)
    return dialog


def _toggle_pause(state: DebugPanelState, btn: ui.button) -> None:
    state.paused = not state.paused
    if state.paused:
        btn.text = 'Resume'
        btn.props('icon=play_arrow')
    else:
        btn.text = 'Pause'
        btn.props('icon=pause')


def _clear_view(state: DebugPanelState, view: ui.textarea) -> None:
    state.last_id = 0
    view.value = ''
