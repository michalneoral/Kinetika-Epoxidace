from __future__ import annotations

from typing import Any

from nicegui import ui


# Keep tooltips readable and consistent across the app.
DEFAULT_TOOLTIP_STYLE = 'max-width: 460px; white-space: normal;'

# Quasar's QTooltip supports a `delay` (ms) before showing the tooltip.
# Default is 2s (can be overridden by global settings in the UI).
DEFAULT_TOOLTIP_DELAY_MS = 2000


def push_tooltip_settings_to_client(*, delay_ms: int, enabled: bool | None = None) -> None:
    """Publish tooltip settings to the browser.

    We keep tooltip delay/enable flags in `window.*` so all tooltips can bind to
    them dynamically (no need to re-render components).
    """
    _ensure_idle_tooltip_js()

    # Convention:
    # - delay_ms < 0  => tooltips disabled
    # - delay_ms == 0 => show immediately
    # - delay_ms > 0  => show after the delay
    stored_delay = int(delay_ms)
    delay_ms = int(max(0, stored_delay))
    if enabled is None:
        enabled = stored_delay >= 0
    ui.run_javascript(
        """
(() => {
  window.__kinetika_tt_delay_ms = %d;
  window.__kinetika_tt_disable = %s;
  // Hard-disable via CSS class to stay compatible across Quasar/NiceGUI versions.
  // (Some versions don't expose a reliable `disable` prop on QTooltip.)
  try {
    document.body.classList.toggle('kinetika-tooltips-disabled', window.__kinetika_tt_disable);
  } catch (_) {}
})();
"""
        % (delay_ms, "false" if enabled else "true")
    )


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

  // Defaults (can be overridden by push_tooltip_settings_to_client).
  if (window.__kinetika_tt_delay_ms === undefined) window.__kinetika_tt_delay_ms = 2000;
  if (window.__kinetika_tt_disable === undefined) window.__kinetika_tt_disable = false;

  // Ensure a CSS rule exists which can fully hide tooltips.
  // This is used for the "Nezobrazovat" mode and is more robust than relying
  // on a component prop.
  const STYLE_ID = 'kinetika-tooltips-style';
  if (!document.getElementById(STYLE_ID)) {
    const style = document.createElement('style');
    style.id = STYLE_ID;
    style.textContent = `
      body.kinetika-tooltips-disabled .q-tooltip,
      body.kinetika-tooltips-disabled .q-tooltip__content {
        display: none !important;
      }
    `;
    document.head.appendChild(style);
  }

  try {
    document.body.classList.toggle('kinetika-tooltips-disabled', window.__kinetika_tt_disable);
  } catch (_) {}

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
    delay_ms: int | None = None,
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
        # Bind delay dynamically to global window settings.
        # Disabling is handled robustly via a body CSS class.
        if delay_ms is None:
            delay_expr = 'window.__kinetika_tt_delay_ms'
        else:
            delay_expr = str(int(delay_ms))

        with ui.tooltip().props(
            f':delay="{delay_expr}"'
        ).classes('q-pa-sm').style(style):
            ui.markdown(f'**{title}**\n\n{detail}')

    # Prefer attaching inside the element context (most reliable).
    try:
        with target:
            _render()
    except Exception:
        # Fallback: create a tooltip in the current context (NiceGUI usually binds it to the last element).
        _render()
