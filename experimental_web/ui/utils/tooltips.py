from __future__ import annotations

from typing import Any

from nicegui import ui


# Keep tooltips readable and consistent across the app.
DEFAULT_TOOLTIP_STYLE = 'max-width: 460px; white-space: normal;'

# Quasar's QTooltip supports a `delay` (ms) before showing the tooltip.
# We default to 1s so tooltips don't pop up immediately on casual hovers.
DEFAULT_TOOLTIP_DELAY_MS = 1000


def _ensure_idle_tooltip_js() -> None:
    """Install a tiny client-side helper so tooltip timers reset while the mouse moves.

    Desired UX:
    - Tooltip appears after `delay_ms` only if the pointer stays over the element
      and the mouse is *not moving*.

    We keep using `ui.tooltip()` and Quasar's built-in delay, but on `mousemove`
    we synthetically dispatch `mouseleave`+`mouseenter` on the current target to
    restart Quasar's internal delay timer.
    """

    ui.run_javascript(
        r"""
(() => {
  const FN = '__kinetika_tt_reset_on_move';
  if (window[FN]) return;

  window[FN] = (evt) => {
    const el = evt && evt.currentTarget;
    if (!el) return;

    // Throttle: reset at most ~15×/s per element.
    const now = (window.performance && performance.now) ? performance.now() : Date.now();
    const last = el.__kinetika_tt_last_move || 0;
    if (now - last < 66) return;
    el.__kinetika_tt_last_move = now;

    try {
      // Reset Quasar tooltip timers by faking leave+enter.
      // (mouseenter/mouseleave do not bubble; Quasar listens directly on the element.)
      el.dispatchEvent(new MouseEvent('mouseleave', { bubbles: false, cancelable: true }));
      el.dispatchEvent(new MouseEvent('mouseenter', { bubbles: false, cancelable: true }));
    } catch (_) {
      // Ignore synthetic event failures
    }
  };
})();
"""
    )


def attach_tooltip(
    target: Any,
    title: str,
    detail: str,
    *,
    style: str = DEFAULT_TOOLTIP_STYLE,
    delay_ms: int = DEFAULT_TOOLTIP_DELAY_MS,
) -> None:
    """Attach a tooltip to a NiceGUI element.

    Content convention:
    - a short title on top
    - a more detailed explanation below
    """

    # Install the client-side helper once per document.
    _ensure_idle_tooltip_js()

    # Reset tooltip delay while the mouse moves over the target.
    # We call via `window.` so it's available from Vue template expressions.
    try:
        target.props('@mousemove="window.__kinetika_tt_reset_on_move($event)"')
    except Exception:
        # Some NiceGUI elements may not expose `.props` (very rare); ignore.
        pass

    def _render() -> None:
        # Use ":delay" so Vue treats it as a number prop (not a string).
        with ui.tooltip().props(f':delay="{int(delay_ms)}"').classes('q-pa-sm').style(style):
            ui.markdown(f'**{title}**\n\n{detail}')

    # Prefer attaching inside the element context (most reliable).
    try:
        with target:
            _render()
    except Exception:
        # Fallback: create a tooltip in the current context (NiceGUI usually binds it to the last element).
        _render()
