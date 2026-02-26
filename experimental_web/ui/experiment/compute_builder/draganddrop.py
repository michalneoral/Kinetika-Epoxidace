from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol, Optional, Any

from nicegui import ui


class Item(Protocol):
    title: str


@dataclass
class DragState:
    dragged_item: Optional[Item] = None
    source_column: Optional[str] = None


# Inject CSS once for dark/light styling
if not getattr(ui, "_dd_css_injected", False):
    ui.add_head_html('''
    <style>
      .dd-column {
        background: var(--q-grey-2);
        border: 1px solid rgba(0,0,0,0.08);
      }
      .dd-column.dd-highlight { background: var(--q-grey-3); }
      .dd-card {
        background: white;
        border: 1px solid rgba(0,0,0,0.10);
      }
      .dd-card.dd-hover { outline: 2px dashed var(--q-primary); outline-offset: 2px; }
      .dd-card.dd-drop-before::before {
        content: '';
        position: absolute;
        left: 6px;
        right: 6px;
        top: -1px;
        height: 2px;
        background: var(--q-primary);
        border-radius: 1px;
      }


      .body--dark .dd-column {
        background: #1d1d1d;
        border: 1px solid rgba(255,255,255,0.10);
      }
      .body--dark .dd-column.dd-highlight { background: #262626; }
      .body--dark .dd-card {
        background: #2a2a2a;
        border: 1px solid rgba(255,255,255,0.12);
      }
    </style>
    ''')
    setattr(ui, "_dd_css_injected", True)


DropHandler = Callable[[Item, str, str, Optional[int]], None]
# args: item, source_column, target_column, target_index (None => append)


class Column(ui.column):
    def __init__(
        self,
        name: str,
        state: DragState,
        on_drop: DropHandler | None = None,
        mover: Callable[["Card"], Any] | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.state = state
        self.on_drop = on_drop
        self.mover = mover

        self.classes("dd-column w-72 p-4 rounded shadow-2 gap-2")

        self.on("dragover.prevent", self.highlight)
        self.on("dragleave", self.unhighlight)
        self.on("drop", self.drop_on_column)

    def highlight(self) -> None:
        self.classes(add="dd-highlight")

    def unhighlight(self) -> None:
        self.classes(remove="dd-highlight")

    def drop_on_column(self) -> None:
        """Drop on empty area: append to end."""
        self.unhighlight()
        if self.on_drop is None:
            return
        item = self.state.dragged_item
        src = self.state.source_column
        if item is None or src is None:
            return
        self.state.dragged_item = None
        self.state.source_column = None
        self.on_drop(item, src, self.name, None)

    def accept_card(self, item: Item, index: int) -> None:
        with self:
            Card(item=item, state=self.state, mover=self.mover, column_name=self.name, index=index, on_drop=self.on_drop)


class Card(ui.card):
    def __init__(
        self,
        item: Item,
        state: DragState,
        mover: Callable[["Card"], Any] | None,
        column_name: str,
        index: int,
        on_drop: DropHandler | None,
    ) -> None:
        super().__init__()
        self.item = item
        self.state = state
        self._mover = mover
        self.column_name = column_name
        self.index = index
        self._on_drop = on_drop

        with self.props("draggable").classes("dd-card relative w-full cursor-pointer p-2"):
            ui.label(item.title)

        self.on("dragstart", self.handle_dragstart)
        self.on("dblclick", self.handle_double_click)

        # allow dropping onto card to insert before it
        self.on("dragover.prevent", self._hover_dropline)
        self.on("dragleave", self._unhover)
        self.on("drop", self._drop_on_card)

    def handle_dragstart(self) -> None:
        self.state.dragged_item = self.item
        self.state.source_column = self.column_name

    def handle_double_click(self) -> None:
        if self._mover is not None:
            self._mover(self)

    def _hover_dropline(self) -> None:
        self.classes(add="dd-drop-before")

    def _unhover(self) -> None:
        self.classes(remove="dd-drop-before")

    def _drop_on_card(self) -> None:
        self._unhover()
        if self._on_drop is None:
            return
        item = self.state.dragged_item
        src = self.state.source_column
        if item is None or src is None:
            return
        self.state.dragged_item = None
        self.state.source_column = None
        self._on_drop(item, src, self.column_name, self.index)
