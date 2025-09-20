from nicegui import ui
from typing import Any, Callable, cast

from typing_extensions import Self

from nicegui.binding import BindableProperty, bind, bind_from, bind_to
from nicegui.element import Element

class StyledLabel(ui.label):
    status = BindableProperty(
        on_change=lambda sender, status: cast(Self, sender)._handle_status_change(status))  # pylint: disable=protected-access

    STATUS_CLASSES = {
        'info': {
            'bg': 'bg-blue-100', 'border': 'border-blue-500', 'text': 'text-blue-600'
        },
        'positive': {
            'bg': 'bg-green-100', 'border': 'border-green-500', 'text': 'text-green-700'
        },
        'ok': {
            'bg': 'bg-green-100', 'border': 'border-green-500', 'text': 'text-green-700'
        },
        'warning': {
            'bg': 'bg-yellow-100', 'border': 'border-yellow-500', 'text': 'text-yellow-700'
        },
        'negative': {
            'bg': 'bg-red-100', 'border': 'border-red-500', 'text': 'text-red-700'
        },
        'error': {
            'bg': 'bg-red-100', 'border': 'border-red-500', 'text': 'text-red-700'
        },
        'default': {
            'bg': 'bg-gray-100', 'border': 'border-gray-400', 'text': 'text-gray-800'
        },
    }

    def __init__(self, text: str = '', status: str = 'default', style_text: bool = True, **kwargs):
        # super().__init__(text, **kwargs)
        super().__init__(text=text)
        self.status = status
        self._style_text = style_text
        self.update_style()

    def update_style(self) -> None:
        if self.status is None:
            self.status = 'default'
        elif isinstance(self.status, bool) and self.status:
            self.status = 'ok'
        elif isinstance(self.status, bool) and not self.status:
            self.status = 'error'

        cls = self.STATUS_CLASSES.get(self.status, self.STATUS_CLASSES['default'])

        # Remove any previously set classes
        self._classes.clear()

        class_str = ' '.join([
            cls['bg'],
            cls['border'],
            'border',
            'rounded-lg',
            'px-3',
            'py-1',
            'inline-block',
        ])
        if self._style_text:
            class_str += " " + cls['text']

        self.classes(class_str)

    def bind_status_from(self,
                       target_object: Any,
                       target_name: str = 'status',
                       backward: Callable[..., Any] = lambda x: x,
                       ) -> Self:
        """Bind the text of this element from the target object's target_name property.

        The binding works one way only, from the target to this element.
        The update happens immediately and whenever a value changes.

        :param target_object: The object to bind from.
        :param target_name: The name of the property to bind from.
        :param backward: A function to apply to the value before applying it to this element.
        """
        bind_from(self, 'status', target_object, target_name, backward)
        return self

    def set_status(self, status: str) -> None:
        self.status = status
        self.update_style()

    def set_text(self, text: str) -> None:
        self.text = text

    def set_style_text(self, value: bool) -> None:
        self._style_text = value
        self.update_style()

    def _handle_status_change(self, status: str) -> None:
        """Called when the status of this element changes.

        :param status: The new status.
        """
        self._status_to_model_status(status)
        self.update_style()
        self.update()

    def _status_to_model_status(self, status: str) -> None:
        self.status = status


