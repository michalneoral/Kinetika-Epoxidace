# cli_debug.py
import pandas as pd
from app.webapp.core import extract_df, validate_table
from app.webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from app.webapp.processing import TableProcessor
from rich import print
from app.webapp.graphs import plot_kinetic_fit
import matplotlib
print(matplotlib.get_backend())
# matplotlib.use("TkAgg")
# print(matplotlib.get_backend())
from app.webapp.advanced_plotting import AdvancedPlotter

def run_debug_cli(path_to_excel):
    print("=== DEBUG MODE ===")
    xls = pd.ExcelFile(path_to_excel)
    print("Sheets:", xls.sheet_names)
    df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)

    col_end = 10

    fame_df = extract_df(df_raw, 3, 16, 2, col_end)
    print("\nFAME:")
    print(fame_df.head())
    valid, msg = validate_table(fame_df, 'FAME', REQUIRED_FAME)
    print(msg)

    epo_df = extract_df(df_raw, 28, 37, 2, col_end)
    print("\nEPO:")
    print(epo_df.head())
    valid, msg = validate_table(epo_df, 'EPOXIDES', REQUIRED_EPO)
    print(msg)

    processor = TableProcessor(fame_df, epo_df)
    processor.process()
    processor.debug_print()

    processor.compute_all_kinetics()
    # processor.plot_all_kinetics_debug()
    plotters = [AdvancedPlotter(name, model) for name, model in processor.k_models.items()]
    for p in plotters:
        p.plot(ui=False)
