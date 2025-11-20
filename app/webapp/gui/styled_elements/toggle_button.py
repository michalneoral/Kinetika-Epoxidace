from nicegui import ui


class ToggleButton(ui.button):

    def __init__(self, config=None, *args, **kwargs) -> None:
        self._state = False
        super().__init__(*args, **kwargs)
        self.on('click', self.toggle)
        self.config = config

    def toggle(self) -> None:
        """Toggle the button state."""
        self._state = not self._state
        self.update()

    def turn_off(self) -> None:
        """Turn off the button state."""
        self._state = False
        self.update()

    def turn_on(self) -> None:
        """Turn on the button state."""
        self._state = True
        self.update()

    def update(self) -> None:
        with self.props.suspend_updates():
            self.props(f'color={"green" if self._state else "gray"}')
        super().update()

    def is_on(self) -> bool:
        return self._state

    def get_config(self):
        if self._state:
            return self.config
        return None
