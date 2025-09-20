from nicegui import events, ui
from typing import List
import os
import pandas as pd

from webapp.core import extract_df, extract_df_dict, validate_table
from webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from webapp.processing import TableProcessor
from webapp.kinetic_model import InitConditions
import logging
from io import BytesIO
from webapp.gui.styled_elements.label import StyledLabel
from webapp.gui.utils.tables import sanitize_df_for_table
from webapp.gui.abstract_tab import AbstractTab
from webapp.gui.styled_elements.table import StickyTable

class TableTab(AbstractTab):
    def __init__(self, processor: TableProcessor = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.expanded = {'all': True}

    @ui.refreshable
    def __call__(self, *args, **kwargs):
        with ui.column().classes('justify-center w-full'):
            with ui.row().classes('justify-center w-full'):
                if not self.processor.has_data():
                    StyledLabel('Tabulky ještě nejsou načteny', status='warning')
                    return
                ui.label('Tabulky').classes('text-lg font-semibold')
            self.processor.process()
            self.show_tables()

    def refresh(self):
        self.__call__.refresh()

    def show_tables(self):
        with ui.row().classes('justify-center w-full'):
            with ui.column().classes('min-w-[300px] items-center'):
                for k,v in self.processor.tables.items():
                    if not k in self.expanded:
                        self.expanded[k] = True
                    with ui.expansion(f'Zpracovaná tabulka: {k}', icon='table_chart'
                                      ).classes('w-full rounded-lg shadow-sm p-4 bg-neutral-100 dark:bg-neutral-800').bind_value(self.expanded, k):
                        ui.markdown(self.processor.tables_text[k], extras=['latex']).classes('w-full text-center')
                        clean_df = sanitize_df_for_table(v)
                        # ui.table.from_pandas(clean_df).classes('w-full overflow-x-auto')
                        StickyTable.from_pandas(clean_df,
                                                sticky='both', theme=self.theme).classes('w-full overflow-x-auto')
