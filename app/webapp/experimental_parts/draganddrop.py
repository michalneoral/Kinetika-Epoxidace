from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, Optional, Any

from nicegui import ui


class Item(Protocol):
    title: str


@dataclass
class DragState:
    dragged: Optional["Card"] = None


class Column(ui.column):
    def __init__(
        self,
        name: str,
        state: DragState,
        on_drop: Callable[[Item, str], None] | None = None,
        mover: Callable[["Card"], Any] | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.state = state
        self.on_drop = on_drop
        self.mover = mover

        # už sem nedáváme ui.label(name) — nadpis bude nad sloupcem v app.py
        self.classes("bg-blue-grey-2 w-72 p-4 rounded shadow-2 gap-2")

        self.on("dragover.prevent", self.highlight)
        self.on("dragleave", self.unhighlight)
        self.on("drop", self.move_card)

    def highlight(self) -> None:
        self.classes(remove="bg-blue-grey-2", add="bg-blue-grey-3")

    def unhighlight(self) -> None:
        self.classes(remove="bg-blue-grey-3", add="bg-blue-grey-2")

    def move_card(self) -> None:
        self.unhighlight()
        dragged = self.state.dragged
        if dragged is None:
            return

        self.accept_card(dragged.item)
        dragged.parent_slot.parent.remove(dragged)
        self.state.dragged = None

    def accept_card(self, item: Item) -> None:
        with self:
            Card(item=item, state=self.state, mover=self.mover)
        if self.on_drop is not None:
            self.on_drop(item, self.name)


class Card(ui.card):
    def __init__(
        self,
        item: Item,
        state: DragState,
        mover: Callable[["Card"], Any] | None,
    ) -> None:
        super().__init__()
        self.item = item
        self.state = state
        self._mover = mover

        with self.props("draggable").classes("w-full cursor-pointer bg-grey-1 p-2"):
            ui.label(item.title)

        self.on("dragstart", self.handle_dragstart)
        self.on("dblclick", self.handle_double_click)

    def handle_dragstart(self) -> None:
        self.state.dragged = self

    def handle_double_click(self) -> None:
        if self._mover is not None:
            self._mover(self)
