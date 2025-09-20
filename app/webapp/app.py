import argparse
import os
import streamlit as st
import pandas as pd
from webapp.core import extract_df, validate_table
from webapp.utils import REQUIRED_FAME, REQUIRED_EPO
from webapp.processing import TableProcessor
from webapp.kinetic_model import InitConditions

from webapp.config_manager import (
    load_config, save_config, list_configs,
    CONFIG_DIR, RECENT_CONFIG, get_timestamped_filename
)

# Widget keys
WIDGET_KEYS = [
    "fame_row_start", "fame_row_end", "fame_col_start", "fame_col_end",
    "epo_row_start", "epo_row_end", "epo_col_start", "epo_col_end"
]

DEFAULTS = {
    "fame_row_start": 3, "fame_row_end": 16,
    "fame_col_start": 2, "fame_col_end": 15,
    "epo_row_start": 28, "epo_row_end": 37,
    "epo_col_start": 2, "epo_col_end": 15
}

# Load recent config
config = load_config()

def run_streamlit_app():
    st.set_page_config(
        page_title="Excel Validator",
        page_icon="📊",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.markdown("### ⚙️ Konfigurace")
    top_left, top_right = st.columns([1, 1.5])

    with top_left:
        available = list_configs()
        selected_file = st.selectbox("📂 Načíst konfiguraci", ["(výchozí)"] + available)
        if selected_file != "(výchozí)":
            selected_path = os.path.join(CONFIG_DIR, selected_file)
            loaded_config = load_config(selected_path)
            for k, v in loaded_config.items():
                st.session_state[k] = v
            st.rerun()

    with top_right:
        with st.expander("💾 Uložit aktuální konfiguraci jako..."):
            default_name = get_timestamped_filename()
            new_filename = st.text_input("Název souboru pro uložení:", value=default_name, key="save_name")
            if st.button("✅ Uložit", key="confirm_save"):
                save_path = os.path.join(CONFIG_DIR, new_filename)
                to_save = {k: st.session_state.get(k) for k in WIDGET_KEYS}
                save_config(to_save, save_path)
                st.success(f"Uloženo do `{new_filename}`")

    if st.button("🗑️ Obnovit výchozí hodnoty"):
        for k, v in DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()

    tab1, tab2, tab3, tab4 = st.tabs(["📥 Vstupní data", "🧪 Předzpracování dat", "⚡ Spočítané rychlosti", "📈 Grafy"])

    with tab1:
        st.title("📅 Načítání a validace tabulek z Excelu (.xlsx)")

        default_file_path = "kinetika/data/MERO_FAME+Epoxides_0-24h_60C.xlsx"

        uploaded_file = st.file_uploader("Krok 1: Nahrajte Excel soubor (.xlsx)",
                                         type=["xlsx", "xls"],
                                         help='bobika')
        # Fallback to default file if nothing is uploaded
        if uploaded_file is not None:
            loaded_file = uploaded_file
            st.success("Soubor byl úspěšně nahrán uživatelem.")
        elif os.path.exists(default_file_path):
            loaded_file = default_file_path
            st.info(f"Použit výchozí soubor: `{default_file_path}`")
        else:
            st.warning("Žádný soubor nebyl nahrán a výchozí soubor nebyl nalezen.")
            return

        try:
            xls = pd.ExcelFile(loaded_file)
            sheet = st.selectbox("Krok 2: Vyberte list", xls.sheet_names)
            df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
            st.write("📄 Náhled celého listu:")
            st.dataframe(df_raw)

            st.markdown("### 🔍 Krok 3: Výběr tabulek FAME a EPO")
            fame_col, epo_col = st.columns(2)

            with fame_col:
                st.markdown("#### 🧪 FAME")
                fame_row_start = st.number_input("Řádek začátek:",
                                                 value=config.get("fame_row_start", DEFAULTS["fame_row_start"]),
                                                 min_value=1, key="fame_row_start")
                fame_row_end = st.number_input("Řádek konec:",
                                               value=config.get("fame_row_end", DEFAULTS["fame_row_end"]), min_value=1,
                                               key='fame_row_end')
                fame_col_start = st.number_input("Sloupec začátek:",
                                                 value=config.get("fame_col_start", DEFAULTS["fame_col_start"]),
                                                 min_value=1, key='fame_col_start')
                fame_col_end = st.number_input("Sloupec konec:",
                                               value=config.get("fame_col_end", DEFAULTS["fame_col_end"]), min_value=1,
                                               key='fame_col_end')

                # if st.button("📋 Načíst tabulku FAME"):
                fame_df = extract_df(df_raw, fame_row_start, fame_row_end, fame_col_start, fame_col_end)
                st.session_state["fame_df"] = fame_df
                st.session_state["fame_valid"], st.session_state["fame_msg"] = validate_table(fame_df, 'FAME',
                                                                                              REQUIRED_FAME)

                if "fame_df" in st.session_state:
                    st.dataframe(st.session_state["fame_df"], use_container_width=True)
                    msg = st.session_state.get("fame_msg", "")
                    valid = st.session_state.get("fame_valid", False)
                    st.success(msg) if valid else st.warning(msg)

            with epo_col:
                st.markdown("#### 🧬 EPO")
                epo_row_start = st.number_input("Řádek začátek:",
                                                value=config.get("epo_row_start", DEFAULTS["epo_row_start"]),
                                                min_value=1, key="epo_row_start")
                epo_row_end = st.number_input("Řádek konec:", value=config.get("epo_row_end", DEFAULTS["epo_row_end"]),
                                              min_value=1, key='epo_row_end')
                epo_col_start = st.number_input("Sloupec začátek:",
                                                value=config.get("epo_col_start", DEFAULTS["epo_col_start"]),
                                                min_value=1, key='epo_col_start')
                epo_col_end = st.number_input("Sloupec konec:",
                                              value=config.get("epo_col_end", DEFAULTS["epo_col_end"]), min_value=1,
                                              key='epo_col_end')

                # if st.button("📋 Načíst tabulku EPO"):
                epo_df = extract_df(df_raw, epo_row_start, epo_row_end, epo_col_start, epo_col_end)
                st.session_state["epo_df"] = epo_df
                st.session_state["epo_valid"], st.session_state["epo_msg"] = validate_table(epo_df, 'EPOXIDES',
                                                                                            REQUIRED_EPO)

                if "epo_df" in st.session_state:
                    st.dataframe(st.session_state["epo_df"], use_container_width=True)
                    msg = st.session_state.get("epo_msg", "")
                    valid = st.session_state.get("epo_valid", False)
                    st.success(msg) if valid else st.warning(msg)

            current_state = {k: st.session_state.get(k) for k in WIDGET_KEYS}
            save_config(current_state)

        except Exception as e:
            st.error(f"❌ Chyba při načítání Excel souboru: {e}")

    with tab2:
        st.header("🧪 Předzpracování dat")
        if st.session_state.get("fame_df") is None or st.session_state.get("epo_df") is None:
            st.warning("Tabulky FAME a EPO nejsou načteny.")

        fame_df = st.session_state.get("fame_df")
        epo_df = st.session_state.get("epo_df")
        fame_valid = st.session_state.get("fame_valid", False)
        epo_valid = st.session_state.get("epo_valid", False)

        if fame_df is None or epo_df is None:
            st.warning("Tabulky FAME a EPO nejsou načteny.")
        elif not fame_valid:
            st.warning("Tabulka FAME není validní. Prosím načtěte tabulku správně.")
        elif not epo_valid:
            st.warning("Tabulka EPO není validní. Prosím načtěte tabulku správně.")
        else:
            processor = TableProcessor(fame_df, epo_df)
            processor.process()
            st.session_state["processor"] = processor

            # === Inicializace session state ===
            if "expand_all" not in st.session_state:
                st.session_state["expand_all"] = True

            # === Tlačítko pro přepínání rozbalení ===
            def toggle_expanders():
                st.session_state["expand_all"] = not st.session_state["expand_all"]

            st.button("📂 Rozbalit/Sbalit vše", on_click=toggle_expanders)

            # === Expandery s tabulkami ===
            with st.expander("✅ Zpracovaná tabulka: IS_korig", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                Tabulka `IS_korig` vzniká normalizací všech koncentrací FAME podle interního standardu **C17**.

                $$
                \text{IS}_\text{korig}(i) = \frac{c_i}{c_{C17}}
                $$

                kde \\( c_i \\) je původní hodnota pro danou FAME sloučeninu. Poté jsou sloučeniny *C18:1 I* a *C18:1 II* sloučeny do jedné nové řádky *C18:1* součtem.
                """)
                st.dataframe(processor.IS_korig)

            with st.expander("✅ Zpracovaná tabulka: IS_korig_EPO", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `IS_korig_EPO` je vytvořena stejně jako `IS_korig`, ale pro epoxidované deriváty (EPO). Každý řádek je vydělen koncentrací C17:

                        $$
                        \text{IS}_\text{korig}^{\text{EPO}}(j) = \frac{c_j^{\text{EPO}}}{c_{C17}}
                        $$
                        """)
                st.dataframe(processor.IS_korig_EPO)

            with st.expander("✅ IS_korig_t", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `IS_korig_t` je transpozice `IS_korig`, kde každý řádek reprezentuje jeden časový bod. To umožňuje sledovat změny koncentrací jednotlivých FAME v čase.
                        """)
                st.dataframe(processor.IS_korig_t)

            with st.expander("✅ IS_korig_EPO_t", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `IS_korig_EPO_t` je transpozice `IS_korig_EPO` a obsahuje vývoj koncentrací jednotlivých EPO v čase.
                        """)
                st.dataframe(processor.IS_korig_EPO_t)

            with st.expander("✅ Souhrnná tabulka", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `souhrn` sumarizuje celkové množství:

                        - nenasycených FAME (bez epoxidace),
                        - všech epoxidovaných derivátů,
                        - celkový součet.

                        Z těchto hodnot jsou odvozeny relativní podíly nenasycených FAME a EPO:

                        $$
                        \text{zastoupení}_\text{unFAME} = \frac{\Sigma_\text{unFAME}}{\Sigma_\text{total}}, \quad
                        \text{zastoupení}_\text{EPO} = \frac{\Sigma_\text{EPO}}{\Sigma_\text{total}}
                        $$
                        """)
                st.dataframe(processor.souhrn)

            with st.expander("✅ C18:1", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `C18:1` obsahuje vývoj koncentrací nenasycené FAME sloučeniny a jejího odpovídajícího epoxidu:

                        $$
                        \Sigma = c_{\text{FAME}} + c_{\text{EPO}}
                        $$

                        Také je zde spočítána změna hydroxylových skupin:

                        $$
                        \text{hydroxyly}(t) = \Sigma(0) - \Sigma(t)
                        $$

                        a relativní zastoupení každé složky vzhledem k výchozí koncentraci.
                        """)
                st.dataframe(processor.C18_1)

            with st.expander("✅ C20:1", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `C20:1` obsahuje vývoj koncentrací nenasycené FAME sloučeniny a jejího odpovídajícího epoxidu:

                        $$
                        \Sigma = c_{\text{FAME}} + c_{\text{EPO}}
                        $$

                        Také je zde spočítána změna hydroxylových skupin:

                        $$
                        \text{hydroxyly}(t) = \Sigma(0) - \Sigma(t)
                        $$

                        a relativní zastoupení každé složky vzhledem k výchozí koncentraci.
                        """)
                st.dataframe(processor.C20_1)

            with st.expander("✅ C18:2", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `C18:2` obsahuje součet sloučeniny *C18:2* a jejích epoxidovaných derivátů:

                        - úplný součet (FAME + všechny EPO),
                        - částečný součet (1-EPO I + 1-EPO II),
                        - rozšířený součet (včetně 2-EPO).

                        Z těchto údajů se opět vypočítávají hydroxylové skupiny a relativní zastoupení jednotlivých složek.
                        """)
                st.dataframe(processor.C18_2)

            with st.expander("✅ C18:3", expanded=st.session_state["expand_all"]):
                st.markdown(r"""
                        Tabulka `C18:3` zpracovává *C18:3* a její epoxidované deriváty. Jsou zde tři varianty součtů:

                        - úplný součet všech derivátů (1-EPO I–III, 2-EPO I),
                        - částečný součet (pouze 1-EPO I–III),
                        - rozšířený součet (včetně 2-EPO I).

                        Opět se počítají hydroxylové skupiny a relativní poměry.
                        """)
                st.dataframe(processor.C18_3)

    with tab3:
        st.header("⚡ Spočítané rychlosti")
        st.info("Zobrazené grafy jsou pouze pro validaci výsledků. Grafy pro export jsou na další záložce.")

        fame_df = st.session_state.get("fame_df")
        epo_df = st.session_state.get("epo_df")
        fame_valid = st.session_state.get("fame_valid", False)
        epo_valid = st.session_state.get("epo_valid", False)

        if fame_df is None or epo_df is None:
            st.warning("Tabulky FAME a EPO nejsou načteny.")
        elif not fame_valid:
            st.warning("Tabulka FAME není validní. Prosím načtěte tabulku správně.")
        elif not epo_valid:
            st.warning("Tabulka EPO není validní. Prosím načtěte tabulku správně.")
        else:
            if "processor" not in st.session_state:
                processor = TableProcessor(fame_df, epo_df)
                processor.process()
                st.session_state["processor"] = processor
            else:
                processor = st.session_state["processor"]

            with st.expander("⚙️ Nastavení parametrů pro výpočet", expanded=False):
                # UI: Picker based on InitConditions
                options = list(InitConditions)
                labels = [opt.name for opt in options]
                help_text = "### Nastavení volby\n\n"
                help_text += "\n".join([f"- **{opt.name.upper()}**: {opt.description}" for opt in options])
                help_text += "\nDoporučeno: Použít **TIME_SHIFT**"
                selected_label = st.selectbox("Zvolte způsob inicializace:", labels, help=help_text, index=2)

                # Map label to enum
                selected_enum = InitConditions[selected_label]

                # Description under picker
                st.markdown(f"**Popis:** {selected_enum.description}")

                t_shift = st.slider(
                    label="Posun počátečního času (t₀)",
                    min_value=0.0,
                    max_value=10.0,
                    value=1.0,  # default starting value
                    step=0.1,
                    format="%.2f"
                )

            processor.compute_all_kinetics(initialization=selected_enum, t_shift=t_shift)
            if hasattr(processor, "k_models"):
                for name, model in processor.k_models.items():
                    st.markdown(f"#### Výsledky pro {name}")

                    if model.k_fit is not None:
                        # latex_eq = r",\quad ".join([f"k_{{{i + 1}}} = {k:.5f}" for i, k in enumerate(model.k_fit)])
                        latex_eq = r"\begin{aligned}" + "\n" + "\n".join(
                            [r"k_{" + f"{idx + 1}" + r"}" + f" &= {k['latex']}&= {k['value']:.8f} \\\\" for idx, k in
                             enumerate(model.get_constants_with_names())]
                        ) + r"\end{aligned}"

                        st.latex(latex_eq)

                        # Simulate and plot
                        try:
                            fig = model.plot_debug(ui=True, legend_mode='components_only')
                            st.pyplot(fig)
                        except Exception as e:
                            import traceback
                            st.error("Chyba při vykreslování grafu:")
                            st.text(traceback.format_exc())
                    else:
                        st.warning("Model nebyl fitován.")
            else:
                st.warning("Kinetické modely nebyly spočítány.")


    with tab4:
        st.info("Zde budou vykresleny grafy (TODO)")

def main():
    parser = argparse.ArgumentParser(description="Excel validator app")
    parser.add_argument('--debug_cli', metavar='EXCEL_FILE', help='Run in CLI mode using the specified Excel file')
    args = parser.parse_args()

    if args.debug_cli:
        from cli_debug import run_debug_cli
        if os.path.exists(args.debug_cli):
            run_debug_cli(args.debug_cli)
        else:
            print(f"❌ File '{args.debug_cli}' not found.")
    else:
        from app import run_streamlit_app
        run_streamlit_app()

if __name__ == "__main__":
    main()
