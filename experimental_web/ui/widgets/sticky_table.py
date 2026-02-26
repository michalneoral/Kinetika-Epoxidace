from __future__ import annotations

from typing import Dict, List, Literal, Optional, Union
from typing_extensions import Self

from nicegui import ui

StickyOption = Literal['header', 'column', 'both', 'none']


_STICKY_CSS = '''
<style>
/* Helper: make the table body scrollable when max-height is set */
.sticky-scroll {
    overflow: auto;
}

/* Sticky header: style all header cells */
.sticky-header .q-table__middle thead th,
.sticky-both   .q-table__middle thead th {
    position: sticky;
    top: 0;
    background: var(--q-primary) !important;
    color: white !important;
    z-index: 3;
}

/* Sticky first column (body + header) */
.sticky-column .q-table__middle tbody td:first-child,
.sticky-column .q-table__middle thead th:first-child,
.sticky-both   .q-table__middle tbody td:first-child,
.sticky-both   .q-table__middle thead th:first-child {
    position: sticky;
    left: 0;
    background: var(--q-primary) !important;
    color: white !important;
    z-index: 2;
}

/* When both are sticky, top-left cell must be above everything */
.sticky-both .q-table__middle thead th:first-child {
    z-index: 4;
}

/* Keep borders visible */
.sticky-header .q-table__middle thead th,
.sticky-column .q-table__middle tbody td:first-child,
.sticky-both .q-table__middle thead th,
.sticky-both .q-table__middle tbody td:first-child {
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.25);
}
</style>
'''


class StickyTable(ui.table):
    """A NiceGUI table with optional sticky header/first column.

    Important note about NiceGUI multi-page navigation:
    - The browser may rebuild the <head> when navigating between pages.
    - Therefore we MUST ensure the CSS is present whenever a StickyTable is created.
    We simply inject the CSS every time (duplicate <style> tags are harmless).
    """

    def __init__(self, *args,
                 sticky: StickyOption = 'none',
                 max_height: Optional[str] = "420px",
                 **kwargs):
        # Always inject; this is required because the page head may be rebuilt across navigation.
        ui.add_head_html(_STICKY_CSS)

        sticky_class = {
            'header': 'sticky-header',
            'column': 'sticky-column',
            'both': 'sticky-both',
            'none': '',
        }[sticky]

        super().__init__(*args, **kwargs)
        if sticky_class:
            self.classes(sticky_class)
        if max_height:
            self.classes('sticky-scroll')
            self.style(f'max-height: {max_height};')

    @classmethod
    def from_rows_and_columns(
        cls,
        *,
        rows: List[Dict],
        columns: List[Dict],
        sticky: StickyOption = 'none',
        max_height: Optional[str] = "420px",
        row_key: Optional[str] = None,
        pagination: Optional[Union[int, dict]] = None,
        selection: Optional[Literal['single', 'multiple']] = None,
    ) -> Self:
        return cls(
            rows=rows,
            columns=columns,
            row_key=row_key,
            pagination=pagination,
            selection=selection,
            sticky=sticky,
            max_height=max_height,
        )
