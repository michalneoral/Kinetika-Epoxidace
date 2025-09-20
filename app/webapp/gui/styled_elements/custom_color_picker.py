from typing import Any, Callable, Optional
from typing_extensions import Self

from nicegui.element import Element
from nicegui.events import ColorPickEventArguments, GenericEventArguments, Handler, handle_event
from nicegui.elements.menu import Menu
from nicegui import ui
from webapp.utils.colors import convert_color

class ColorPickerButton:
    def __init__(self, *,
                 icon: str = 'palette',
                 color: str = '#00ff00',
                 color_type: str = 'rgb',
                 on_pick: Optional[Callable[[ColorPickEventArguments], None]] = None):
        self._color = color
        self._bound = None
        self.color_type = color_type.lower()

        with ui.button(icon=icon).classes(f'!bg-[{self._color}]') as self.button:
            self.picker = CustomColorPicker(
                on_pick=self._handle_pick,
                color_type=self.color_type,
            )
            self.picker.set_color(self._color)

        self.user_callback = on_pick

    def _handle_pick(self, e: ColorPickEventArguments):
        self._color = e.color
        self.button.classes(f'!bg-[{e.color}]')
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
        self._color = initial_color
        self.picker.set_color(initial_color)
        self.button.classes(f'!bg-[{initial_color}]')
        return self

    @property
    def value(self):
        return self._color

    def set_color(self, color: str):
        self._color = color
        self.picker.set_color(color)
        self.button.classes(f'!bg-[{color}]')


class CustomColorPicker(Menu):

    def __init__(self, *,
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
        self.q_color.props(f'model-value="{color}"')

    def on_pick(self, callback: Handler[ColorPickEventArguments]) -> Self:
        """Add a callback to be invoked when a color is picked."""
        self._pick_handlers.append(callback)
        return self

    def bind_color(self, obj: Any, key: str) -> Self:
        """Bind this color picker to a value in a dict or dataclass."""
        initial_color = getattr(obj, key) if hasattr(obj, key) else obj[key]
        self.set_color(initial_color)

        # TODO: forward and backward is not implemented

        def update_model(e: ColorPickEventArguments):
            if hasattr(obj, key):
                setattr(obj, key, e.color)
            else:
                obj[key] = e.color

        self.on_pick(update_model)
        return self
