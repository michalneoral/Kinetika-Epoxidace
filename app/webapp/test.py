from nicegui import ui
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from PIL import Image
import argparse
import os

# === GLOBAL STATE ===
curve_configs = []
plot_config = {}

from app.webapp.advanced_plotting import AdvancedPlotter
import random
import pandas as pd
from app.webapp.core import extract_df, validate_table
from app.webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from app.webapp.processing import TableProcessor
from rich import print
from concurrent.futures import ThreadPoolExecutor
from app.webapp.config import AdvancedGraphConfig
from app.webapp.gui.graph_tab import GraphTab
from app.webapp.gui.style.load_style import load_style


executor = ThreadPoolExecutor(max_workers=2)

import matplotlib
matplotlib.use('Agg')


def test():
    try:
        path_to_excel = "/home/neoramic/repos/tynka_bakalarka_grafy/BP/kinetika/data/MERO_FAME+Epoxides_0-24h_60C.xlsx"
        xls = pd.ExcelFile(path_to_excel)
    except:
        path_to_excel = "C:/Users/micha/PycharmProjects/BP_with_kinetika/kinetika/data/MERO_FAME+Epoxides_0-24h_60C.xlsx"
        xls = pd.ExcelFile(path_to_excel)


    # print("Sheets:", xls.sheet_names)
    df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)

    col_end = 12

    fame_df = extract_df(df_raw, 3, 16, 2, col_end)
    print("\nFAME:")
    # print(fame_df.head())
    valid, msg = validate_table(fame_df, 'FAME', REQUIRED_FAME)
    print(msg)

    epo_df = extract_df(df_raw, 28, 37, 2, col_end)
    print("\nEPO:")
    # print(epo_df.head())
    valid, msg = validate_table(epo_df, 'EPOXIDES', REQUIRED_EPO)
    print(msg)

    processor = TableProcessor(fame_df, epo_df)
    processor.process()
    # processor.debug_print()
    processor.compute_all_kinetics()

    graph_tab = GraphTab(processor, executor)

    graph_tab()



def run_nicegui_app():
    # page()
    load_style()
    test()

    ui.run(title='Test', favicon="📊", dark=None,
           reload=False, port=8082, show=False)


def main():
    parser = argparse.ArgumentParser(description="Aplikace pro počítání kinetiky")
    parser.add_argument('--debug_cli', metavar='EXCEL_FILE', help='Run in CLI mode using the specified Excel file')
    args = parser.parse_args()

    if args.debug_cli:
        from cli_debug import run_debug_cli

        if os.path.exists(args.debug_cli):
            run_debug_cli(args.debug_cli)
        else:
            print(f"❌ File '{args.debug_cli}' not found.")
    else:
        ##from test_nicegui import run_nicegui_app

        run_nicegui_app()


if __name__ in {"__main__", "__mp_main__"}:
    main()

    # print(AdvancedGraphConfig.get_choices('legend_mode'))
