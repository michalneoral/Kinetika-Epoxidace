from dataclasses import asdict

from nicegui import events, ui
from typing import List
import os

from app.webapp.config import TablePickConfigEPO, TablePickConfigFAME
from app.webapp.gui.abstract_tab import AbstractTab
import pandas as pd
from app.webapp.core import extract_df, extract_df_dict, validate_table
from app.webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from app.webapp.processing import TableProcessor
from app.webapp.kinetic_model import InitConditions
import logging
from io import BytesIO
from app.webapp.gui.styled_elements.label import StyledLabel
from app.webapp.gui.utils.tables import sanitize_df_for_table
from app.webapp.gui.styled_elements.table import StickyTable

# from nicegui_tabulator import tabulator, use_theme
# use the theme for all clients
# use_theme('bootstrap4')


class InputTab(AbstractTab):
    def __init__(self, processor: TableProcessor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor

        self.fame_table_defaults = TablePickConfigFAME()
        self.epo_table_defaults = TablePickConfigEPO()

        self.table_data = {
            'raw_content': None,
            'raw_df': None,
            'raw_xls': None,
            'status_fame_text': 'Tabulka ještě nenačtena',
            'status_fame': None,
            'status_epo_text': 'Tabulka ještě nenačtena',
            'status_epo': None,
            'fame_df': None,
            'epo_df': None,
        }

    @ui.refreshable
    def __call__(self, *args, **kwargs):
        with ui.column().classes('justify-center w-full'):
            with ui.row().classes('justify-center w-full'):
                ui.label('Krok 1: Nahrajte Excel soubor (.xlsx)').classes('text-lg font-semibold')
            self.display_upload_dialog()
            self.display_sheet_picker()
            self.print_raw_file_table()
            self.display_load_subtables_buttons()
            self.place_fame_epo_tables()


    def get_table_data(self):
        return self.table_data

    def handle_upload(self, e, path: str = None):
        if path is None:
            self.table_data['raw_content'] = BytesIO(e.content.read())  # binary buffer
        else:
            self.table_data['raw_content'] = path
        self.table_data['raw_xls'] = pd.ExcelFile(self.table_data['raw_content'])
        self.display_sheet_picker.refresh()


    @ui.refreshable
    def display_upload_dialog(self):
        with ui.row().classes('justify-center w-full'):
            ui.upload(
                label='Vyberte soubor',
                auto_upload=True,
                max_files=1,
                on_upload=self.handle_upload,
                multiple=False,

            ).props('accept=.xlsx,.xls').tooltip('bobika')
            try:
                default_file_path = "/home/neoramic/repos/tynka_bakalarka_grafy/BP/kinetika/data/MERO_FAME+Epoxides_0-24h_60C.xlsx"
                self.handle_upload(None, default_file_path)
            except:
                try:
                    default_file_path = "C:/Users/micha/PycharmProjects/BP_with_kinetika/kinetika/data/MERO_FAME+Epoxides_0-24h_60C.xlsx"
                    self.handle_upload(None, default_file_path)
                except:
                    pass

    @ui.refreshable
    def display_sheet_picker(self):
        if self.table_data['raw_xls'] is not None:
            with ui.row().classes('justify-center w-full'):
                ui.label('Krok 2: Vyberte list').classes('text-lg font-semibold')

            with ui.row().classes('justify-center w-full'):
                c_value = self.table_data['raw_xls'].sheet_names[0]
                sheet = ui.select(options=self.table_data['raw_xls'].sheet_names,
                                  with_input=False,
                                  label="Listy v souboru:",
                                  value=c_value,
                                  on_change=self.handle_read_excel,
                                  ).classes('w-40')
                self.handle_read_excel(sheet)


    def handle_read_excel(self, e):
        self.table_data['raw_df'] = pd.read_excel(self.table_data['raw_xls'], sheet_name=e.value, header=None)
        self.print_raw_file_table.refresh()

    @ui.refreshable
    def print_raw_file_table(self):
        if self.table_data['raw_df'] is not None:
            with ui.row().classes('justify-center w-full'):
                StickyTable.from_pandas(self.table_data['raw_df'],
                                        sticky='both', theme=self.theme, title='BOBIKA').classes('max-h-80 w-full')
                self.display_load_subtables_buttons.refresh()

    @ui.refreshable
    def display_load_subtables_buttons(self):
        if self.table_data['raw_df'] is not None:
            # Reactive values

            ui.number.default_classes('w-32')
            with ui.row().classes('w-full flex-wrap gap-4'):
                with ui.column().classes('flex-1 min-w-[300px] items-center'):
                    ui.label('Krok 3: Selekce hodnot pro FAME').classes('text-lg font-semibold mx-16 text-center')
                    with ui.row().classes('gap-2'):
                        ui.number(label='řádek začátek', on_change=lambda e: self.display_fame_table.refresh(),
                                  min=1).bind_value(self.fame_table_defaults, "row_start")
                        ui.number(label='řádek konec', on_change=lambda e: self.display_fame_table.refresh(),
                                  min=1).bind_value(self.fame_table_defaults, "row_end")
                        ui.number(label='sloupec začátek', on_change=lambda e: self.display_fame_table.refresh(),
                                  min=1).bind_value(self.fame_table_defaults, "col_start")
                        ui.number(label='sloupec konec', on_change=lambda e: self.display_fame_table.refresh(),
                                  min=1).bind_value(self.fame_table_defaults, "col_end")

                with ui.column().classes('flex-1 min-w-[300px] items-center'):
                    ui.label('Krok 4: Selekce hodnot pro EPO').classes('text-lg font-semibold mx-16 text-center')
                    with ui.row().classes('gap-2'):
                        ui.number(label='řádek začátek', on_change=lambda e: self.display_epo_table.refresh(),
                                  min=1).bind_value(self.epo_table_defaults, "row_start")
                        ui.number(label='řádek konec', on_change=lambda e: self.display_epo_table.refresh(),
                                  min=1).bind_value(self.epo_table_defaults, "row_end")
                        ui.number(label='sloupec začátek', on_change=lambda e: self.display_epo_table.refresh(),
                                  min=1).bind_value(self.epo_table_defaults, "col_start")
                        ui.number(label='sloupec konec', on_change=lambda e: self.display_epo_table.refresh(),
                                  min=1).bind_value(self.epo_table_defaults, "col_end")




    def place_fame_epo_tables(self):
        with ui.row().classes('w-full flex-wrap gap-4'):
            with ui.column().classes('flex-1 min-w-[300px] items-center'):
                StyledLabel(status='info').bind_text_from(self.table_data, 'status_fame_text').bind_status_from(self.table_data, 'status_fame')
                self.display_fame_table()
            with ui.column().classes('flex-1 min-w-[300px] items-center'):
                StyledLabel(status='info').bind_text_from(self.table_data, 'status_epo_text').bind_status_from(self.table_data, 'status_epo')
                self.display_epo_table()

    @ui.refreshable
    def display_fame_table(self):
        if self.table_data['raw_df'] is not None:
            fame_df = extract_df_dict(self.table_data['raw_df'], asdict(self.fame_table_defaults))
            clean_df = sanitize_df_for_table(fame_df)
            StickyTable.from_pandas(clean_df,
                                    sticky='both', theme=self.theme, title='FAME').classes('max-h-80 w-full')
            # tabulator.from_pandas(clean_df).classes('max-h-80 w-full')
            self.table_data['status_fame'], self.table_data['status_fame_text'] = validate_table(fame_df, 'FAME', REQUIRED_FAME)
            if self.table_data['status_fame']:
                self.processor.add_fame(fame_df=fame_df)
            else:
                self.processor.add_fame(fame_df=None)
            self.call_refreshable_elements()

    @ui.refreshable
    def display_epo_table(self):
        if self.table_data['raw_df'] is not None:
            epo_df = extract_df_dict(self.table_data['raw_df'], asdict(self.epo_table_defaults))
            clean_df = sanitize_df_for_table(epo_df)
            StickyTable.from_pandas(clean_df,
                                    sticky='both', theme=self.theme, title='EPO').classes('max-h-80 w-full')
            self.table_data['status_epo'], self.table_data['status_epo_text'] = validate_table(epo_df, 'EPOXIDES', REQUIRED_EPO)
            if self.table_data['status_epo']:
                self.processor.add_epo(epo_df=epo_df)
            else:
                self.processor.add_epo(epo_df=None)
            self.call_refreshable_elements()
