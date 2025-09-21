from nicegui import ui
from app.webapp.config import ColorConfig
from abc import ABC, abstractmethod
from dataclasses import asdict


class ColorTheme:
    def __init__(self, theme):
        self.theme = None
        self.theme_name = theme
        self.__call__(theme)

    def __call__(self, theme, *args, **kwargs):
        self.theme_name = theme
        if theme.lower() == 'upce':
            self.theme = ColorConfig(primary="#98194e",
                                     secondary="#3da345")
        else:
            raise NotImplementedError(f"Theme '{theme}' is not supported")

        ui.colors(**asdict(self.theme))


def load_style():
    theme = ColorTheme('UPCE')
    # ui.button.default_props()
    ui.input.default_props('filled')
    ui.select.default_props('filled')
    ui.number.default_props('filled')
    # ui.table.default_props("my-sticky-header-column-table")
    # ui.table.default_props(':rows="rows" :columns="columns"')

    return theme
