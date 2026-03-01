from contextlib import contextmanager
from nicegui import ui


@contextmanager
def frame(title: str, version: str) -> None:
    with ui.header().classes("items-center justify-between"):
        ui.label(f"{title}  •  {version}").classes("text-lg font-bold")
        with ui.row().classes("items-center"):
            ui.button("Reload", on_click=lambda: ui.run_javascript("location.reload()")).props("outline")
    yield
