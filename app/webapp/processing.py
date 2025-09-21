import pandas as pd
from app.webapp.kinetic_model import KineticModel, InitConditions, OptimumTShift
import numpy as np
from rich import print
import traceback
from collections import OrderedDict


def get_possible_models():
    return ['C18:1_simplified', 'C18:1', 'C18:2_simplified', 'C18:2_eps_and_others', 'C18:2', 'C18:2_with_k_uh',
            'C20:1_simplified']


# Retry defining the TableProcessor now that pandas is imported
class TableProcessor:
    def __init__(self, fame_df: pd.DataFrame = None, epo_df: pd.DataFrame = None):
        self.fame_df = fame_df.copy() if fame_df is not None else None
        self.epo_df = epo_df.copy() if epo_df is not None else None

        self.kinetic_results = {}
        self.k_models = {}
        self.tables = OrderedDict()
        self.tables_text = OrderedDict()

    def has_data(self):
        if self.fame_df is None or self.epo_df is None:
            return False
        return True

    def add_fame(self, fame_df: pd.DataFrame = None):
        self.fame_df = fame_df.copy() if fame_df is not None else None

    def add_epo(self, epo_df: pd.DataFrame = None):
        self.epo_df = epo_df.copy() if epo_df is not None else None

    def process(self):
        if not self.has_data():
            return
        self.tables = OrderedDict()

        # POMOCNÉ TABULKY
        C17 = self.fame_df[self.fame_df["FAME"] == "C17"].iloc[0].drop("FAME")
        IS_korig = self.fame_df.drop(columns="FAME") / C17
        IS_korig.insert(0, "FAME", self.fame_df["FAME"])
        IS_korig_EPO = self.epo_df.drop(columns="EPOXIDES") / C17
        IS_korig_EPO.insert(0, "EPOXIDES", self.epo_df["EPOXIDES"])
        IS_korig = IS_korig[IS_korig["FAME"] != "C17"].reset_index(drop=True)

        idx1 = IS_korig.index[IS_korig["FAME"] == "C18:1 I"][0]
        idx2 = IS_korig.index[IS_korig["FAME"] == "C18:1 II"][0]
        sum_row = IS_korig.loc[[idx1, idx2]].drop(columns="FAME").sum()
        new_row = pd.Series({"FAME": "C18:1", **sum_row.to_dict()})
        upper = IS_korig.iloc[:idx1]
        lower = IS_korig.iloc[idx1:]
        IS_korig = pd.concat([upper, pd.DataFrame([new_row]), lower], ignore_index=True)

        IS_korig_t = IS_korig.set_index("FAME").T
        IS_korig_t.index.name = "time"
        IS_korig_t = IS_korig_t.reset_index()

        IS_korig_EPO_t = IS_korig_EPO.set_index("EPOXIDES").T
        IS_korig_EPO_t.index.name = "time"
        IS_korig_EPO_t = IS_korig_EPO_t.reset_index()
        self.tables['IS_korig'] = IS_korig
        self.tables['IS_korig_EPO'] = IS_korig_EPO
        self.tables['IS_korig_t'] = IS_korig_t
        self.tables['IS_korig_EPO_t'] = IS_korig_EPO_t

        # TEXTY POMOCNÝCH TABULEK
        self.tables_text['IS_korig'] = r"""
            ### IS korig
        
            Tabulka `IS_korig` vzniká normalizací všech koncentrací FAME podle interního standardu **C17**.

            $$
            \text{IS}_\text{korig}(i) = \frac{c_i}{c_{C17}}
            $$

            kde $c_i$ je původní hodnota pro danou FAME sloučeninu. Poté jsou sloučeniny *C18:1 I* a *C18:1 
            II* sloučeny do jedné nové řádky *C18:1* součtem."""

        self.tables_text['IS_korig_EPO'] = r"""
            Tabulka `IS_korig_EPO` je vytvořena stejně jako `IS_korig`, ale pro 
            epoxidované deriváty (EPO). Každý řádek je vydělen koncentrací C17:

            $$
            \text{IS}_\text{korig}^{\text{EPO}}(j) = \frac{c_j^{\text{EPO}}}{c_{C17}}
            $$
            """

        self.tables_text['IS_korig_t'] = r"""
            Tabulka `IS_korig_t` je transpozice `IS_korig`, kde každý řádek 
            reprezentuje jeden časový bod. To umožňuje sledovat změny koncentrací jednotlivých FAME v čase."""

        self.tables_text['IS_korig_EPO_t'] = r"""
            Tabulka `IS_korig_EPO_t` je transpozice `IS_korig_EPO` a obsahuje vývoj koncentrací jednotlivých EPO v čase.
            """

        # VÝSTUPNÍ TABULKY
        C18_1 = self._compute_unfame("C18:1")
        C20_1 = self._compute_unfame("C20:1", custom_suffix="1-EPO")

        C18_2 = self._compute_c18_2()
        C18_3 = self._compute_c18_3()
        souhrn = self._compute_summary()

        self.tables['souhrn'] = souhrn
        self.tables['C18_1'] = C18_1
        self.tables['C18_2'] = C18_2
        self.tables['C18_3'] = C18_3
        self.tables['C20_1'] = C20_1

        # TEXT VÝSTUPNÍCH TABULEK
        self.tables_text['souhrn'] = r"""
            Tabulka `souhrn` sumarizuje celkové množství:

            - nenasycených FAME (bez epoxidace),
            - všech epoxidovaných derivátů,
            - celkový součet.

            Z těchto hodnot jsou odvozeny relativní podíly nenasycených FAME a EPO:

            $$
            \text{zastoupení}_\text{unFAME} = \frac{\Sigma_\text{unFAME}}{\Sigma_\text{total}}, \quad
            \text{zastoupení}_\text{EPO} = \frac{\Sigma_\text{EPO}}{\Sigma_\text{total}}
            $$
            """
        self.tables_text['C18_1'] = r"""
            Tabulka `C18:1` obsahuje vývoj koncentrací nenasycené FAME sloučeniny a jejího odpovídajícího epoxidu:

            $$
            \Sigma = c_{\text{FAME}} + c_{\text{EPO}}
            $$

            Také je zde spočítána změna hydroxylových skupin:

            $$
            \text{hydroxyly}(t) = \Sigma(0) - \Sigma(t)
            $$

            a relativní zastoupení každé složky vzhledem k výchozí koncentraci.
            """
        self.tables_text['C18_2'] = r"""
            Tabulka `C18:2` obsahuje součet sloučeniny *C18:2* a jejích epoxidovaných derivátů:

            - úplný součet (FAME + všechny EPO),
            - částečný součet (1-EPO I + 1-EPO II),
            - rozšířený součet (včetně 2-EPO).

            Z těchto údajů se opět vypočítávají hydroxylové skupiny a relativní zastoupení jednotlivých složek.
            """
        self.tables_text['C18_3'] = r"""
            Tabulka `C18:3` zpracovává *C18:3* a její epoxidované deriváty. Jsou zde tři varianty součtů:

            - úplný součet všech derivátů (1-EPO I–III, 2-EPO I),
            - částečný součet (pouze 1-EPO I–III),
            - rozšířený součet (včetně 2-EPO I).

            Opět se počítají hydroxylové skupiny a relativní poměry.
            """
        self.tables_text['C20_1'] = r"""
            Tabulka `C20:1` obsahuje vývoj koncentrací nenasycené FAME sloučeniny a jejího odpovídajícího epoxidu:

            $$
            \Sigma = c_{\text{FAME}} + c_{\text{EPO}}
            $$

            Také je zde spočítána změna hydroxylových skupin:

            $$
            \text{hydroxyly}(t) = \Sigma(0) - \Sigma(t)
            $$

            a relativní zastoupení každé složky vzhledem k výchozí koncentraci.
            """

    def debug_print(self):
        for k, v in self.tables.items():
            print(f'tabulka {k}:')
            print(v)
            print()

    def _compute_unfame(self, name, custom_suffix="EPO"):
        IS_korig_t = self.tables['IS_korig_t']
        IS_korig_EPO_t = self.tables['IS_korig_EPO_t']

        new_column = IS_korig_t[name].astype(float).fillna(0.0) + IS_korig_EPO_t[
            f"{name} {custom_suffix}"].astype(
            float).fillna(0.0)
        df = pd.concat([IS_korig_t[["time", name]], IS_korig_EPO_t[[f"{name} {custom_suffix}"]],
                        pd.DataFrame({f"Σ {name} FAME + EPO": new_column})], axis=1)

        # Get initial value at time = 0
        s0 = df.loc[df['time'] == 0, f'Σ {name} FAME + EPO'].values[0]

        # Compute hydroxyl group formation over time
        df['hydroxyly'] = s0 - df[f'Σ {name} FAME + EPO']

        df[f"zastoupení {name}"] = df[name] / s0
        df[f"zastoupení {name} EPO"] = df[f"{name} {custom_suffix}"] / s0
        df["zastoupení hydroxyly"] = df["hydroxyly"] / s0
        df[f"zastoupení {name} EPO + hydroxyly"] = df[f"zastoupení {name} EPO"] + df["zastoupení hydroxyly"]
        return df

    def _compute_c18_2(self):
        IS_korig_t = self.tables['IS_korig_t']
        IS_korig_EPO_t = self.tables['IS_korig_EPO_t']

        cols_all = [
            ("IS_korig_t", "C18:2"),
            ("IS_korig_EPO_t", "C18:2 1-EPO I"),
            ("IS_korig_EPO_t", "C18:2 1-EPO II"),
            ("IS_korig_EPO_t", "C18:2 2-EPO"),
        ]
        new_column = sum(
            (IS_korig_t[col] if src == "IS_korig_t" else IS_korig_EPO_t[col]).astype(float).fillna(0.0)
            for src, col in cols_all
        )

        cols_2 = ["C18:2 1-EPO I", "C18:2 1-EPO II"]
        new_column_2 = sum(IS_korig_EPO_t[col].astype(float).astype(float).fillna(0.0) for col in cols_2)
        new_column_3 = new_column_2 + IS_korig_EPO_t["C18:2 2-EPO"].astype(float).astype(float).fillna(0.0)

        s0 = new_column.loc[IS_korig_t['time'] == 0].values[0]
        C18_2 = pd.concat([
            IS_korig_t[["time", "C18:2"]],
            IS_korig_EPO_t[["C18:2 1-EPO I", "C18:2 1-EPO II", "C18:2 2-EPO"]],
            pd.DataFrame({"Σ C18:2 FAME + EPO": new_column}),
            pd.DataFrame({"Σ C18:2 1-EPO": new_column_2}),
            pd.DataFrame({"Σ C18:2 vše EPO": new_column_3}),
        ], axis=1)

        C18_2["hydroxyly"] = s0 - C18_2["Σ C18:2 FAME + EPO"]
        C18_2["Σ C18:2 EPO + hydroxyly"] = C18_2["hydroxyly"] + C18_2["Σ C18:2 vše EPO"]

        for col in ["C18:2", "C18:2 1-EPO I", "C18:2 1-EPO II", "C18:2 2-EPO", "Σ C18:2 1-EPO", "Σ C18:2 vše EPO",
                    "hydroxyly", "Σ C18:2 EPO + hydroxyly"]:
            C18_2[f"zastoupení {col}"] = C18_2[col] / s0

        return C18_2

    def _compute_c18_3(self):
        IS_korig_t = self.tables['IS_korig_t']
        IS_korig_EPO_t = self.tables['IS_korig_EPO_t']

        cols_all = [
            ("IS_korig_t", "C18:3"),
            ("IS_korig_EPO_t", "C18:3 1-EPO I"),
            ("IS_korig_EPO_t", "C18:3 1-EPO II"),
            ("IS_korig_EPO_t", "C18:3 1-EPO III"),
            ("IS_korig_EPO_t", "C18:3 2-EPO I"),
        ]
        new_column = sum(
            (IS_korig_t[col] if src == "IS_korig_t" else IS_korig_EPO_t[col]).astype(float).fillna(0.0)
            for src, col in cols_all
        )

        cols_2 = ["C18:3 1-EPO I", "C18:3 1-EPO II", "C18:3 1-EPO III"]
        new_column_2 = sum(IS_korig_EPO_t[col].astype(float).astype(float).fillna(0.0) for col in cols_2)
        new_column_3 = new_column_2 + IS_korig_EPO_t["C18:3 2-EPO I"].astype(float).astype(float).fillna(0.0)

        s0 = new_column.loc[IS_korig_t['time'] == 0].values[0]
        C18_3 = pd.concat([
            IS_korig_t[["time", "C18:3"]],
            IS_korig_EPO_t[["C18:3 1-EPO I", "C18:3 1-EPO II", "C18:3 1-EPO III", "C18:3 2-EPO I"]],
            pd.DataFrame({"Σ C18:3 FAME + EPO": new_column}),
            pd.DataFrame({"Σ C18:3 1-EPO": new_column_2}),
            pd.DataFrame({"Σ C18:3 vše EPO": new_column_3}),
        ], axis=1)

        C18_3["hydroxyly"] = s0 - C18_3["Σ C18:3 FAME + EPO"]

        for col in ["C18:3", "C18:3 1-EPO I", "C18:3 1-EPO II", "C18:3 1-EPO III", "C18:3 2-EPO I", "Σ C18:3 vše EPO",
                    "Σ C18:3 1-EPO", "hydroxyly"]:
            C18_3[f"zastoupení {col}"] = C18_3[col] / s0

        return C18_3

    def _compute_summary(self):
        IS_korig_t = self.tables['IS_korig_t']
        IS_korig_EPO_t = self.tables['IS_korig_EPO_t']

        fame_cols = ["C16:1", "C18:1", "C18:2", "C18:3", "C20:1", "C22:1"]
        epo_cols = [
            "C18:1 EPO", "C18:2 1-EPO I", "C18:2 1-EPO II", "C18:2 2-EPO",
            "C18:3 1-EPO I", "C18:3 1-EPO II", "C18:3 1-EPO III",
            "C18:3 2-EPO I", "C20:1 1-EPO"
        ]

        sum_fame = sum(IS_korig_t[col].astype(float).astype(float).fillna(0.0) for col in fame_cols)
        sum_epo = sum(IS_korig_EPO_t[col].astype(float).astype(float).fillna(0.0) for col in epo_cols)
        sum_all = sum_fame + sum_epo
        s0 = sum_all.loc[IS_korig_t["time"] == 0].values[0]

        df = pd.concat([
            IS_korig_t[["time"]],
            pd.DataFrame({"Σ všech unFAME": sum_fame}),
            pd.DataFrame({"Σ všech EPO": sum_epo}),
            pd.DataFrame({"Σ sum": sum_all}),
        ], axis=1)

        df["zastoupení Σ všech unFAME"] = df["Σ všech unFAME"] / df["Σ sum"]
        df["zastoupení Σ všech EPO"] = df["Σ všech EPO"] / df["Σ sum"]

        return df

    def create_model(self,
                     df, columns,
                     **kwargs):

        assert len(columns) >= 2
        c_columns = ['time'] + columns
        conc_data = df[c_columns].astype(np.float64).fillna(0.0).values

        model = KineticModel(
            concentration_data=conc_data,
            column_names=columns,
            **kwargs,
        )
        return model

    def compute_all_kinetics(self,
                             initialization=InitConditions.TIME_SHIFT,
                             t_shift=9.0,
                             optim_time_shift=False,
                             models_to_compute=None,
                             ):
        """
        Compute kinetic constants only for C18_1 and store in self.k_models.
        """
        self.k_models = {}
        if models_to_compute is None:
            models_to_compute = []

        # NOTE: all c_model strings has to be defined in get_possible_models() function

        c_model = 'C18:1_simplified'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C18_1'],
                                                              columns=[
                                                                  'zastoupení C18:1',
                                                                  'zastoupení C18:1 EPO + hydroxyly'
                                                              ],
                                                              init_method=initialization, t_shift=t_shift)

        c_model = 'C18:1'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C18_1'],
                                                   columns=[
                                                       'zastoupení C18:1',
                                                       'zastoupení C18:1 EPO',
                                                       'zastoupení hydroxyly',
                                                   ],
                                                   init_method=initialization, t_shift=t_shift, k_uh=False)

        c_model = 'C18:2_simplified'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C18_2'],
                                                              columns=[
                                                                  'zastoupení C18:2',
                                                                  'zastoupení Σ C18:2 vše EPO',
                                                                  'zastoupení hydroxyly',
                                                              ],
                                                              init_method=initialization, t_shift=t_shift)

        c_model = 'C18:2_eps_and_others'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C18_2'],
                                                                  columns=[
                                                                      'zastoupení C18:2',
                                                                      'zastoupení Σ C18:2 vše EPO',
                                                                      'zastoupení hydroxyly',
                                                                      'zastoupení Σ C18:2 EPO + hydroxyly'
                                                                  ],
                                                                  special_model='C18:2_eps_others',
                                                                  init_method=initialization, t_shift=t_shift)

        c_model = 'C18:2'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C18_2'],
                                                   columns=[
                                                       'zastoupení C18:2',
                                                       'zastoupení Σ C18:2 1-EPO',
                                                       'zastoupení C18:2 2-EPO',
                                                       'zastoupení hydroxyly',
                                                   ],
                                                   init_method=initialization, t_shift=t_shift)

        c_model = 'C18:2_with_k_uh'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C18_2'],
                                                             columns=[
                                                                 'zastoupení C18:2',
                                                                 'zastoupení Σ C18:2 1-EPO',
                                                                 'zastoupení C18:2 2-EPO',
                                                                 'zastoupení hydroxyly',
                                                             ],
                                                             init_method=initialization, t_shift=t_shift, k_uh=True)

        c_model = 'C20:1_simplified'
        if c_model in models_to_compute:
            self.k_models[c_model] = self.create_model(self.tables['C20_1'],
                                                              columns=[
                                                                  'zastoupení C20:1',
                                                                  'zastoupení C20:1 EPO + hydroxyly'
                                                              ],
                                                              init_method=initialization, t_shift=t_shift)

        if optim_time_shift:
            OTS = OptimumTShift(models=self.k_models, t_shift_init=t_shift, verbose=False)
            t_shift, t_shift_errorsum = OTS.fit()
            print(t_shift, t_shift_errorsum)

        for k, v in self.k_models.items():
            v.reinit_kinetic_model(t_shift=t_shift)
            v.verbose = True
            c_k, _ = v.fit()
            # print(v.bounds)
            print(f"{k}: {c_k}")

        # for k, model in self.k_models.items():
        #     model_k_names = model.get_constants_with_names()
        #     print(model_k_names)
        #
        #     latex_eq = r",\quad ".join([f"{k['latex']} = {k['value']:.5f}" for k in model.get_constants_with_names()])
        #     print(latex_eq)

        print('DONE')
        return True

    def plot_all_kinetics_debug(self):
        for k, v in self.k_models.items():
            # v.plot_debug_plotly(legend_mode='components_only')
            v.plot_debug(legend_mode='components_only')
