from __future__ import annotations

from nicegui import ui

_STATUS_CLASSES = {
    "default": "text-grey-8 dark:text-grey-3",
    "info": "text-primary",
    "ok": "text-positive",
    "warning": "text-warning",
    "error": "text-negative",
}

_BG_CLASSES = {
    "default": "bg-grey-2 dark:bg-grey-9",
    "info": "bg-blue-1 dark:bg-blue-10",
    "ok": "bg-green-1 dark:bg-green-10",
    "warning": "bg-yellow-1 dark:bg-yellow-10",
    "error": "bg-red-1 dark:bg-red-10",
}


class StyledLabel(ui.label):
    """Lightweight status label with proper class switching."""

    def __init__(self, text: str = "", status: str = "default") -> None:
        super().__init__(text)
        self._status = "default"
        self.classes("q-px-sm q-py-xs rounded-borders text-caption")
        self.set(text, status)

    def set(self, text: str, status: str = "default") -> None:
        # update text
        self.text = text

        if status not in _STATUS_CLASSES:
            status = "default"

        # remove previous status classes, then add new ones
        prev = self._status
        if prev in _STATUS_CLASSES:
            self.classes(remove=_STATUS_CLASSES[prev])
        if prev in _BG_CLASSES:
            self.classes(remove=_BG_CLASSES[prev])

        self._status = status
        self.classes(_STATUS_CLASSES[status])
        self.classes(_BG_CLASSES[status])
