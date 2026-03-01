from nicegui import ui


def info(text: str) -> None:
    with ui.card().classes("w-full"):
        ui.label(text).classes("text-blue-700")


def warning(text: str) -> None:
    with ui.card().classes("w-full"):
        ui.label(text).classes("text-orange-700 font-medium")


def success(text: str) -> None:
    with ui.card().classes("w-full"):
        ui.label(text).classes("text-green-700 font-medium")
