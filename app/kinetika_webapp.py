#!/usr/bin/env python3
from dataclasses import asdict
import argparse
import os

import threading
import time
from app.version import __version__
from app.updater import maybe_update_in_background

from app.webapp.config import GuiConfig
from app.webapp.gui.utils.latex import LatexLabel, typeset_latex
from app.webapp.gui.processing_tab import ProcessingTab
from app.webapp.gui.tables_tab import TableTab
from app.webapp.processing import TableProcessor
from concurrent.futures import ThreadPoolExecutor

from app.webapp.config import Config
import httpx

from nicegui import events, ui
from app.webapp.gui.style.load_style import load_style
from app.webapp.gui.input_tab import InputTab
from app.webapp.gui.graph_tab import GraphTab
import matplotlib
matplotlib.use('Agg')

executor = ThreadPoolExecutor(max_workers=2)

main_page_title = 'KINETIKA'
sub_title = None

class Kinetika():
    def __init__(self, theme, config):
        self.theme = theme
        self.tab_value = 'input'
        self.main_page_title = 'KINETIKA'

        self.config = config
        self.gui_config = GuiConfig()
        self.dark_mode = ui.dark_mode()
        self.dark_mode.bind_value(self.gui_config, 'dark')

        # Central definition of tabs
        self.tabs_meta = {
            'input': {'label': 'Vstupní data', 'icon': '📥'},
            'data': {'label': 'Předzpracování dat', 'icon': '🧪'},
            'speeds': {'label': 'Počítání rychlostí', 'icon': '⚡'},
            'graphs': {'label': 'Grafy', 'icon': '📈'},
        }

        self.page_title_label = ui.label('').classes('text-4xl font-bold')

        processor = TableProcessor()
        self.input_tab = InputTab(processor, theme=theme)
        self.tables_tab = TableTab(processor, theme=theme)
        self.input_tab.add_refreshable_element(self.tables_tab)
        self.processing_tab = ProcessingTab(processor, executor, theme=theme)
        self.input_tab.add_refreshable_element(self.processing_tab)
        self.graphs_tab = GraphTab(processor, executor, theme=theme)
        self.processing_tab.add_refreshable_element(self.graphs_tab)

        # Build the UI
        # with ui.header().classes(replace='row items-center h-16'):
        #     ui.image('https://fcht.upce.cz/sites/default/files/themes/fcht/logo.svg').classes("w-12")

        # with ui.header().props('bordered') as header:
        #     header.classes('bg-page')
        #     header.style('background-color: var(--q-color-page);')
        #     # with ui.row().classes('w-full'):
        #     #     ui.label('Kinetika')
        #     with ui.row().classes('justify-center w-full'):
        #         with ui.page_sticky(position='top-left', x_offset=10, y_offset=10):
        #             ui.image('https://fcht.upce.cz/sites/default/files/themes/fcht/logo.svg').classes("w-12")
        #         self.create_tabs()

        with ui.header().style("background-color: #585858d0;"):
            with ui.row().classes('items-center justify-between w-full no-wrap'):
                # LEFT SECTION
                with ui.row().classes('items-center gap-2'):
                    ui.image('https://fcht.upce.cz/sites/default/files/themes/fcht/logo.svg').classes("w-12")
                    ui.icon('menu')
                    ui.button('Home', icon='home', on_click=lambda: ui.notify('Home'))

                # CENTER TABS
                with ui.row().classes('absolute-center'):
                    self.create_tabs()

                # RIGHT SECTION
                with ui.row().classes('items-center gap-2'):
                    with ui.element().classes('max-[420px]:hidden').tooltip(
                            'Cycle theme mode through dark, light, and system/auto.'):
                        ui.button(icon='dark_mode', on_click=lambda: self.dark_mode.set_value(None)) \
                            .props('flat fab-mini color=white'
                                   ).bind_visibility_from(self.dark_mode, 'value', value=True)
                        ui.button(icon='light_mode', on_click=lambda: self.dark_mode.set_value(True)) \
                            .props('flat fab-mini color=white'
                                   ).bind_visibility_from(self.dark_mode, 'value', value=False)
                        ui.button(icon='brightness_auto', on_click=lambda: self.dark_mode.set_value(False)) \
                            .props('flat fab-mini color=white'
                                   ).bind_visibility_from(self.dark_mode, 'value', lambda mode: mode is None)

        self.create_panels()
        self.update_page_title(self.tab_value)

    def update_dark_mode(self, value=None, button=None, toggle=False):
        if isinstance(toggle, bool) and toggle:
            self.dark_mode.toggle()
        if button is not None:
            button.props(f"icon={'light_mode' if self.dark_mode.value else 'dark_mode'}")

    def create_tabs(self):
        with ui.tabs(on_change=lambda e: self.update_page_title(e.value)) as self.tabs:
            self.tabs.props('align=center')
            for key, meta in self.tabs_meta.items():
                ui.tab(key, label=meta['label'], icon=meta['icon'])

    def set_label(self, t):
        c_tab = self.tabs_meta[t._props['name']]
        with ui.row().classes('justify-center w-full'):
            ui.label(f'{c_tab["icon"]} {c_tab["label"]}').classes('text-4xl font-bold')


    def create_panels(self):
        with ui.tab_panels(self.tabs, value=self.tab_value, on_change=lambda e: typeset_latex()).classes('w-full justify-center'):
            with ui.tab_panel('input') as t:
                self.set_label(t)
                self.input_tab()

            with ui.tab_panel('data') as t:
                self.set_label(t)
                self.tables_tab()

            with ui.tab_panel('speeds') as t:
                self.set_label(t)
                self.processing_tab()

            with ui.tab_panel('graphs') as t:
                self.set_label(t)
                self.graphs_tab()

    def update_page_title(self, value):
        ui.page_title(f"{self.main_page_title} | { self.tabs_meta[value]['label']}")


# @ui.page('/')
# async def page():

def run_nicegui_app():
    # page()
    config = Config()
    theme = load_style()

    Kinetika(theme, config)

    ui.run(title=main_page_title, favicon="📊", **asdict(config.gui),
           reload=False, port=8082, show=True)


if __name__ in {"__main__", "__mp_main__"}:
    print(f"Kinetika - Epoxidace v{__version__} starting…")

    parser = argparse.ArgumentParser(description="Aplikace pro počítání kinetiky")
    parser.add_argument('--debug_cli', metavar='EXCEL_FILE', help='Run in CLI mode using the specified Excel file')
    args = parser.parse_args()

    # fire-and-forget update check (won't block startup)
    threading.Thread(target=maybe_update_in_background, daemon=True).start()

    if args.debug_cli:
        from app.webapp.cli_debug import run_debug_cli

        if os.path.exists(args.debug_cli):
            run_debug_cli(args.debug_cli)
        else:
            print(f"❌ File '{args.debug_cli}' not found.")
    else:
        ##from test_nicegui import run_nicegui_app

        run_nicegui_app()
