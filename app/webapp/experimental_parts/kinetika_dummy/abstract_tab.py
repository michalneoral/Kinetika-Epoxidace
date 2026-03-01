from __future__ import annotations

import asyncio
from nicegui import ui

from utils import schedule


class AbstractTab:
    """
    Base tab:
      - show() je refreshable
      - refresh() coalescing (slučuje burst změn)
      - propagate helper do orchestrátoru
      - lazy load tab state (ensure_loaded)
    """

    name: str = "tab"

    def __init__(self, orchestrator: "Kinetika") -> None:
        self.orchestrator = orchestrator
        self._refresh_scheduled = False
        self._loaded_once = False

    def is_accessible(self) -> bool:
        return True

    async def ensure_loaded(self) -> None:
        if self._loaded_once:
            return
        if not self.is_accessible():
            return
        await self.load_from_db()
        self._loaded_once = True

    async def load_from_db(self) -> None:
        return

    def propagate_changes(self, from_tab: int) -> None:
        self.orchestrator.propagate_changes(from_tab)

    def refresh(self) -> None:
        if self._refresh_scheduled:
            return
        self._refresh_scheduled = True

        async def _do() -> None:
            await asyncio.sleep(0.02)
            self._refresh_scheduled = False
            self.show.refresh()

        ui.timer(0.0, lambda: schedule(_do()), once=True)

    @ui.refreshable
    def show(self) -> None:
        raise NotImplementedError
