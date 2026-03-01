from __future__ import annotations

from typing import Any, Callable, Optional
from typing_extensions import Self

from nicegui.element import Element
from nicegui.events import ColorPickEventArguments, GenericEventArguments, Handler, handle_event
from nicegui.elements.menu import Menu
from nicegui import ui

# NOTE:
# We intentionally avoid dynamic Tailwind classes like "bg-[#ff00aa]".
# Tailwind CSS is compiled ahead-of-time; dynamic values selected at runtime would
# not have CSS rules and the preview would stop updating after a few picks.
# Instead we use an inline CSS variable which is always available.

def _ensure_css() -> None:
    """Ensure the ColorPickerButton CSS exists in the *current* browser document.

    NiceGUI routes can trigger full page reloads in the browser while the server
    process keeps modules cached. A module-level "already added" flag would then
    incorrectly skip injecting CSS after reopening an experiment.

    We therefore inject idempotently via JS (per-document).
    """

    css = r"""
  /*
    Quasar's q-btn does not always paint the visible background on the root
    element; depending on the variant, the visible paint can be on
    `.q-btn__content` and/or via pseudo elements.

    Therefore we set the color variable on the root and paint the preview
    on BOTH the root and the content to be robust across Quasar versions.
  */
  .cpb-btn {
    background: none !important;
  }

  .cpb-btn,
  .cpb-btn::before,
  .cpb-btn::after,
  .cpb-btn .q-btn__content,
  .cpb-btn .q-btn__content::before,
  .cpb-btn .q-btn__content::after {
    background-color: var(--cpb-color) !important;
  }

  .cpb-btn,
  .cpb-btn .q-btn__content {
    color: white !important; /* icon contrast on dark colors */
  }

  .cpb-btn .q-btn__content {
    border-radius: inherit;
  }
"""

    ui.run_javascript(
        """
(() => {
  const id = 'cpb-style';
  if (document.getElementById(id)) return;
  const style = document.createElement('style');
  style.id = id;
  style.textContent = %s;
  document.head.appendChild(style);
})();
""" % (repr(css),)
    )


class ColorPickerButton:
    def __init__(
        self,
        *,
        icon: str = 'palette',
        color: str = '#00ff00',
        color_type: str = 'rgb',
        on_pick: Optional[Callable[[ColorPickEventArguments], None]] = None,
    ):
        _ensure_css()

        self._color = color
        self._bound: Optional[tuple[Any, str]] = None
        self.color_type = color_type.lower()

        with ui.button(icon=icon).classes('cpb-btn').style(f'--cpb-color: {self._color};') as self.button:
            self.picker = CustomColorPicker(
                on_pick=self._handle_pick,
                color_type=self.color_type,
            )
            self.picker.set_color(self._color)

        self.user_callback = on_pick

    def _apply_color_to_button(self, color: str) -> None:
        # Update via CSS variable (works for any runtime color).
        self.button.style(f'--cpb-color: {color};')

    def _handle_pick(self, e: ColorPickEventArguments):
        self._color = e.color
        self._apply_color_to_button(e.color)
        self.picker.set_color(e.color)

        if self._bound:
            obj, key = self._bound
            if hasattr(obj, key):
                setattr(obj, key, e.color)
            else:
                obj[key] = e.color

        if self.user_callback:
            self.user_callback(e)

    def bind_color(self, obj: Any, key: str):
        """Bind to a dict or object so color updates are stored externally."""
        self._bound = (obj, key)

        initial_color = getattr(obj, key) if hasattr(obj, key) else obj[key]
        self._color = str(initial_color)
        self.picker.set_color(self._color)
        self._apply_color_to_button(self._color)
        return self

    @property
    def value(self):
        return self._color

    def set_color(self, color: str):
        self._color = str(color)
        self.picker.set_color(self._color)
        self._apply_color_to_button(self._color)


class CustomColorPicker(Menu):

    def __init__(
        self,
        *,
        color: Optional[str] = None,
        on_pick: Optional[Handler[ColorPickEventArguments]] = None,
        value: bool = False,
        color_type: str = 'rgb',
    ) -> None:
        super().__init__(value=value)
        self._pick_handlers = [on_pick] if on_pick else []
        self._color_value = color if color is not None else "#ffffff"  # default fallback color
        self.color_type = color_type.lower()
        with self:
            def handle_change(e: GenericEventArguments):
                self._color_value = e.args
                for handler in self._pick_handlers:
                    handle_event(handler, ColorPickEventArguments(sender=self, client=self.client, color=e.args))

            self.q_color = Element('q-color').on('change', handle_change)
            self.q_color.props(f'v-model="{self.color_type}"')

    @property
    def color(self) -> str:
        """Get the current selected color."""
        return self._color_value

    def set_color(self, color: str) -> None:
        """Set the color of the picker."""
        self._color_value = color
        # Quasar q-color expects model-value for external set.
        self.q_color.props(f'model-value="{color}"')

    def on_pick(self, callback: Handler[ColorPickEventArguments]) -> Self:
        """Add a callback to be invoked when a color is picked."""
        self._pick_handlers.append(callback)
        return self

    def bind_color(self, obj: Any, key: str) -> Self:
        """Bind this color picker to a value in a dict or dataclass."""
        initial_color = getattr(obj, key) if hasattr(obj, key) else obj[key]
        self.set_color(str(initial_color))

        # NOTE: forward and backward binding is not implemented; we only push on change.
        def update_model(e: ColorPickEventArguments):
            if hasattr(obj, key):
                setattr(obj, key, e.color)
            else:
                obj[key] = e.color

        self.on_pick(update_model)
        return self
