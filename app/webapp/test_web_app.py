import streamlit as st
import os
from app.webapp.advanced_plotting import AdvancedPlotter
import random
import pandas as pd
from app.webapp.core import extract_df, validate_table
from app.webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from app.webapp.processing import TableProcessor
from rich import print

st.set_page_config(layout="wide")

if "processor" not in st.session_state:
    path_to_excel = "kinetika/data/MERO_FAME+Epoxides_0-24h_60C.xlsx"
    xls = pd.ExcelFile(path_to_excel)
    print("Sheets:", xls.sheet_names)
    df_raw = pd.read_excel(xls, sheet_name=xls.sheet_names[0], header=None)

    col_end = 12

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
    st.session_state["processor"] = processor
else:
    processor = st.session_state["processor"]

# Create models and wrap in plotters
plotters = [AdvancedPlotter(name, model) for name, model in processor.k_models.items()]

# Hold all generated figures
figures = []





def st_normal():
    _, col, _ = st.columns([1, 2.5, 1])
    return col

for idx, plotter in enumerate(plotters):
    main_col1, main_col2 = st.columns([1, 1])
    with main_col2:
        with st.expander(f"🧪 Nastavení grafu {plotter.name}", expanded=True):
            # === PLOT PARAMS ===

            col1, col2 = st.columns([1, 4])
            with col1:
                show_title = st.checkbox("Zobrazit titulek", value=True, key=f"show_title_{idx}")
            with col2:
                title = st.text_input("Titulek grafu", value=plotter.name, key=f"title_{idx}", disabled=not show_title)

            # === LEGEND ===
            st.markdown("**Nastavení Legendy**")
            legend_mode = st.selectbox("Typ legendy", ["components_only", "both", "single", "None"], key=f"legend_{idx}")

            # === FIGSIZE ===
            st.markdown("**Nastavení velikosti grafu**")
            col1, col2 = st.columns(2)
            with col1:
                fig_width = st.slider("Šířka obrázku (inch)", 4, 20, 8, key=f"fig_width_{idx}")
            with col2:
                fig_height = st.slider("Výška obrázku (inch)", 2, 10, 4, key=f"fig_height_{idx}")
            figsize = (fig_width, fig_height)

            # === LIMITS ===
            st.markdown("**Nastavení limitů grafu**")
            col1, col2, col3 = st.columns(3)
            xlim_base_min, xlim_base_max = plotter.get_xlim_data_minmax()
            with col1:
                xlim_mode = st.selectbox("osa X", ["default", "auto", "manual", "all_data"])
            with col2:
                xlim0 = st.number_input("Min X", xlim_base_min, xlim_base_max, xlim_base_min, key=f"xlim0_{idx}", disabled=xlim_mode!="manual")
            with col3:
                xlim1 = st.number_input("Max X", xlim_base_min, None, xlim_base_max, key=f"xlim1_{idx}", disabled=xlim_mode!="manual")
            xlim = (min(xlim0, xlim1), max(xlim0, xlim1))

            # === STYLING FOR EACH CURVE ===
            st.markdown("🎨 **Styly jednotlivých křivek**")
            n_curves = plotter.get_n_curves()
            curve_styles = []

            linestyles = ["solid", "dashed", "dashdot", "dotted", "None"]
            marker_styles = ["s", "o", "v", "^", "D", "x", "+", "*", ".", "None"]
            default_colors = ["#FF0000", "#000000", "#5B0687", "#777777", "#0000FF", "#00FF00", "#00FFFF"]

            for i in range(n_curves):
                st.markdown(f"##### Křivka {i + 1}")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                with col1:
                    color = st.color_picker("Barva",
                                            value=default_colors[i],
                                            key=f"color_{idx}_{i}")
                with col2:
                    linestyle = st.selectbox("Styl čáry", linestyles, index=0, key=f"linestyle_{idx}_{i}")
                with col3:
                    linewidth = st.slider("Tloušťka čáry", 0.5, 5.0, 1.5, step=0.1, key=f"linewidth_{idx}_{i}")

                # col4, col5, col6 = st.columns(3)
                with col4:
                    marker = st.selectbox("Marker", marker_styles, index=0, key=f"marker_{idx}_{i}")
                with col5:
                    marker_size = st.slider("Velikost markeru", 2, 20, 6, step=1, key=f"markersize_{idx}_{i}")
                with col6:
                    label = st.text_input("Popisek", value=plotter.data['column_names'][i], key=f"label_{idx}_{i}")

                curve_styles.append({
                    "color": color,
                    "linestyle": linestyle,
                    "linewidth": linewidth,
                    "marker": marker,
                    "markersize": marker_size,
                    "label": label
                })

    with main_col1:
        # === PLOT + DISPLAY ===
        fig = plotter.plot(
            title=title,
            figsize=figsize,
            curve_styles=curve_styles,
            legend_mode=legend_mode,
            show_title=show_title,
            xlim_mode=xlim_mode,
            xlim=xlim,
            ui=True
        )
        # st_normal().pyplot(fig, use_container_width=True)
        st.pyplot(fig)
    figures.append((f"{plotter.name}_plot", fig))

# Export settings
export_format = st.selectbox("Formát exportu", ["png", "pdf", "svg", "jpeg", "webp", "eps", "tiff"])
export_dir = st.text_input("Složka pro export", value="exported_graphs")

# Export
if st.button("📤 Exportovat grafy"):
    os.makedirs(export_dir, exist_ok=True)
    for name, fig in figures:
        filename = f"{name}.{export_format}"
        path = os.path.join(export_dir, filename)
        fig.savefig(path, format=export_format, bbox_inches='tight')
    st.success(f"{len(figures)} grafů exportováno do {export_dir}")
