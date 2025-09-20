import copy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from webapp.constants import K_MIN, K_MAX
from webapp.utils import StrEnum
from matplotlib.lines import Line2D
import plotly.graph_objects as go


class InitConditions(StrEnum):
    NO_INIT = ("no_init", "Bez inicializace, vstupní data v čase t=0 jsou použita jako počáteční podmínka.")
    FROM_MATLAB = ("from_matlab",
                   "(Nedoporučeno) Stejná inicializace jako v původním souboru Matlab. To znamená, že v čase t=0 je "
                   "vložen vektor s hodnotami U=1.0 a 0.0 pro všechny ostatní složky. Původní měření v čase t=0 je "
                   "posunuto na t=1.")
    TIME_SHIFT = ("time_shift",
                  "(Doporučeno) Vloží se vektor s počátečními hodnotami měření v čase t=t₀ (U=1.0 a 0.0 pro ostatní). "
                  "Čas t₀ může být odhadnut minimalizací účelové funkce přes více měření z celého datasetu.")



class OptimumTShift:
    def __init__(self, models, t_shift_init=1.0, t_shift_bounds=(0, 20), verbose=True):
        self.models = models
        self.t_shift_init = t_shift_init
        self.t_shift_bounds = np.atleast_2d(np.array(t_shift_bounds, dtype=np.float64))
        self.minimize_method = 'Nelder-Mead'
        self.verbose = verbose

    def loss(self, t_shift):
        fun_total = 0
        for k, v in self.models.items():
            v.reinit_kinetic_model(t_shift=t_shift)
            _, fun = v.fit()
            fun_total += fun
        return fun_total

    def fit(self):
        result = minimize(self.loss,
                          x0=self.t_shift_init,
                          method=self.minimize_method,
                          bounds=self.t_shift_bounds,
                          tol=1e-8,
                          options={'disp': self.verbose})
        self.result = result
        self.t_shift_fit = result.x
        if self.verbose:
            print("\nOptimal coefficients:", self.t_shift_fit)
        return self.t_shift_fit, result.fun


class KineticModel:

    def __init__(self,
                 concentration_data,
                 k_init=None,
                 t_max=400,
                 init_method="time_shift",
                 t_shift=1,
                 k_min=None,
                 k_max=None,
                 column_names=None,
                 special_model=None,
                 k_uh=False,
                 verbose=False,
                 stars=False):

        self.init_method = init_method
        original_data = np.array(concentration_data, dtype=np.float64)
        self.original_data = original_data
        self.original_data[:, 1:] = np.clip(self.original_data[:, 1:], 0, 1)

        self.data = None
        self.time_exp = None
        self.y_exp = None
        self.y_0 = None

        self.minimize_method = 'Nelder-Mead'
        # self.minimize_method = 'L-BFGS-B'
        # self.ivp_method = 'RK23'
        self.ivp_method = 'RK45'

        self.verbose = verbose
        self.stars = stars
        self.k_fit = None
        self.result = None

        self.data = self._init_data(self.original_data, init_method=self.init_method, t_shift=t_shift)
        self.reinit_kinetic_model(t_shift=t_shift)
        self.t_max = t_max

        # print(self.data)
        # print(self.y_exp)

        n_cols = self.data.shape[1] - 1
        if special_model is not None:
            if special_model == 'C18:2_eps_others':
                assert n_cols == 4, f"For special model {special_model} you have to have {n_cols} columns in your data."
                self.odes = self.odes_k_ue_eh_ueh
                k_init_default = [0.005, 0.001, 0.005]
            else:
                raise NotImplementedError(f"Special model {special_model} not implemented.")
        else:
            if n_cols == 2:
                self.odes = self.odes_k_uh
                k_init_default = [0.005]
            elif n_cols == 3:
                if not k_uh:
                    self.odes = self.odes_k_um_mh
                    k_init_default = [0.005, 0.001]
                else:
                    self.odes = self.odes_k_um_uh_mh
                    k_init_default = [0.005, 0.001, 0.001]
            elif n_cols == 4:
                if not k_uh:
                    self.odes = self.odes_k_um_md_mh_dh
                    k_init_default = [0.005, 0.005, 0.001, 0.001]
                else:
                    self.odes = self.odes_k_um_uh_md_mh_dh
                    k_init_default = [0.005, 0.001, 0.005, 0.001, 0.001]
            elif n_cols == 5:
                self.odes = self.odes_k_um_md_dt_mh_dh_th
                k_init_default = [0.005, 0.005, 0.005, 0.001, 0.001, 0.001]
            else:
                raise NotImplementedError("For this kind of data are not implemented ODES equations (YET).")

        self.column_names = column_names if column_names is not None else [str(i+1) for i in range(n_cols)]
        # print(self.column_names)

        k_len = len(k_init_default)
        k_init = k_init if k_init is not None else k_init_default
        self.k_init = np.array(k_init, dtype=np.float64)
        assert k_len == len(self.k_init), (f"Number of initial values for constants ({len(self.k_init)}) does not fit "
                                           f"the number of kinetic constants ({k_len}).")

        self.k_min = self._set_k_bounds(k_min, k_len, K_MIN)
        self.k_max = self._set_k_bounds(k_max, k_len, K_MAX)
        self.bounds = np.stack([self.k_min, self.k_max], axis=0).T

    # U: Unsaturated FAME (e.g., methyl oleate)
    # M: Monoepoxide
    # D: Diepoxide
    # T: Triepoxide or side product (e.g., hydroxy-epoxide)
    # H: Final product (fully epoxidized or degraded compound)

    def get_constants_with_names(self):
        if self.odes is None:
            print('No odes set yet')
            return None

        c_name = str(self.odes.__name__)
        if c_name.startswith('odes_k_'):
            c_name = c_name.replace('odes_k_', '')
        else:
            print('Unknown format of coefficients.')
            return None

        k_split = c_name.split('_')
        k_names = []
        for c_k in k_split:
            c_k_name = {'name': c_k, 'from': c_k[0].upper(), 'to': c_k[1:2].upper(),
                        'latex': r"k_{\mathrm{" + f"{c_k[0].upper()}" + r"}\rightarrow \mathrm{" + f"{c_k[1:].upper()}" + "}}"}
            k_names.append(c_k_name)

        if self.k_fit is None:
            print('Model is not fitted yet')
        elif len(k_split) == len(self.k_fit):
            for idx, k_value in enumerate(self.k_fit):
                k_names[idx]['value'] = float(k_value)
        else:
            print('Model does not have the right number of kinetic constants')
        return k_names

    def reinit_kinetic_model(self, t_shift):
        self.data = self._init_data(self.original_data, init_method=self.init_method, t_shift=t_shift)
        self.time_exp = self.data[:, 0]
        self.y_exp = np.clip(self.data[:, 1:], 0, 1)
        self.y_0 = self.y_exp[0]
        self.result = None
        self.k_fit = None

    def _init_data(self, data, init_method, t_shift=None):
        data = copy.deepcopy(data)
        if init_method == InitConditions.NO_INIT:
            pass

        elif init_method == InitConditions.FROM_MATLAB:
            assert t_shift is not None, """t_shift cannot be None when initializing Kinetic Model."""

            d = np.zeros_like(data[0:1])
            d[0, 1] = 1.0  # data[0][1] is the first measurement
            data[0, 0] += float(t_shift)  # data[:][0] is time
            data = np.concatenate([d, data], axis=0)

        elif init_method == InitConditions.TIME_SHIFT:
            assert t_shift is not None, """t_shift cannot be None when initializing Kinetic Model."""

            d = np.zeros_like(data[0:1])
            d[0, 1] = 1.0  # data[0][1] is the first measurement
            data[:, 0] += float(t_shift)  # data[:][0] is time
            data = np.concatenate([d, data], axis=0)

        else:
            raise NotImplementedError(f"Initialization method {init_method} not implemented yet.")
        return data

    def odes_k_uh(self, t, conc, k):
        U, H = conc
        k_uh = k[0]
        dU = -k_uh * U
        dH = k_uh * U
        return [dU, dH]

    def odes_k_um_mh(self, t, conc, k):
        U, M, H = conc
        k_um, k_mh = k
        dU = -k_um * U
        dM = k_um * U - k_mh * M
        dH = k_mh * M
        return [dU, dM, dH]

    def odes_k_um_uh_mh(self, t, conc, k):
        U, M, H = conc
        k_um, k_uh, k_mh = k
        dU = -k_um * U - k_uh * U
        dM = k_um * U - k_mh * M
        dH = k_mh * M + k_uh * U
        return [dU, dM, dH]

    def odes_k_ue_eh_ueh(self, t, conc, k):
        U, E, H, EH = conc
        k_ue, k_eh, k_ueh = k
        # dU = - k_ueh * U
        dU = - k_ue * U
        dE = k_ue * U - k_eh * E
        dH = k_eh * E
        dEH = dE + dH
        return [dU, dE, dH, dEH]

    def odes_k_um_md_mh_dh(self, t, conc, k):
        U, M, D, H = conc
        k_um, k_md, k_mh, k_dh = k
        dU = -k_um * U
        dM = k_um * U - k_mh * M - k_md * M
        dD = k_md * M - k_dh * D
        dH = k_mh * M + k_dh * D
        return [dU, dM, dD, dH]

    def odes_k_um_uh_md_mh_dh(self, t, conc, k):
        U, M, D, H = conc
        k_um, k_uh, k_md, k_mh, k_dh = k
        dU = -k_um * U - k_uh * U
        dM = k_um * U - k_mh * M - k_md * M
        dD = k_md * M - k_dh * D
        # dH = k_mh * M + k_dh * D + k_uh * U
        dH = -dU - dM - dD
        return [dU, dM, dD, dH]


    def odes_k_um_md_dt_mh_dh_th(self, t, conc, k):
        U, M, D, T, H = conc
        k_um, k_md, k_dt, k_mh, k_dh, k_th = k
        dU = -k_um * U
        dM = k_um * U - (k_md + k_mh) * M
        dD = k_md * M - (k_dt + k_dh) * D
        dT = k_dt * D - k_th * T
        dH = k_mh * M + k_dh * D + k_th * T
        return [dU, dM, dD, dT, dH]

    def solve_ivp(self, k):
        # print(k)
        sol = solve_ivp(lambda t, y: self.odes(t, y, k),
                        t_span=[self.time_exp[0], self.time_exp[-1]],
                        y0=self.y_0,
                        t_eval=self.time_exp,
                        method=self.ivp_method)
        return sol.t, sol.y.T

    def loss(self, k):
        if np.any(np.array(k) < 0):
            return 1e10  # penalize negative rates
        _, y_sim = self.solve_ivp(k)
        loss = np.sum((y_sim - self.y_exp) ** 2)
        return loss

    def fit(self):
        """
        Perform optimization of kinetic constants using Nelder-Mead.

        Returns:
        - Optimal kinetic constants (as NumPy array)
        """
        result = minimize(self.loss,
                          self.k_init,
                          method=self.minimize_method,
                          bounds=self.bounds,
                          tol=1e-8,
                          options={'disp': self.verbose})
        self.result = result
        self.k_fit = result.x
        if self.verbose:
            print("\nOptimal coefficients:", self.k_fit)
        return self.k_fit, result.fun

    def simulate(self, k_values=None, time_points=100, t_max=None):
        """
        Simulate the model using provided or fitted kinetic constants.

        Parameters:
        - k_values: list of rate constants [k1–k6]; if None, use fitted
        - time_points: number of time points for simulation

        Returns:
        - Tuple (time, concentrations) as NumPy arrays
        """
        t_max = t_max if t_max is not None else self.t_max
        if k_values is None:
            if self.k_fit is None:
                raise ValueError("No fitted coefficients. Call fit() or pass k_values.")
            k_values = self.k_fit
        t_eval = np.linspace(0, t_max, time_points)
        y0 = self.data[0, 1:]
        sol = solve_ivp(self.odes,
                        t_span=[0, t_max],
                        y0=y0,
                        t_eval=t_eval,
                        args=(k_values,),
                        method=self.ivp_method)
        return sol.t, sol.y

    def _set_k_bounds(self, value, k_len, default_value):
        value = value if value is not None else np.ones((k_len,), dtype=np.float64) * default_value
        if isinstance(value, list):
            value = np.array(value, dtype=np.float64)
        elif isinstance(value, np.ndarray):
            value = copy.deepcopy(value).astype(np.float64)
        else:
            raise ValueError(f"value should be a list of size corresponding to model {self.odes.__name__}")

        assert np.all(value >= K_MIN), f"all k constants should be >= {K_MIN}"
        assert np.all(value <= K_MAX), f"all k constants should be <= {K_MAX}"
        return value

    def plot_debug(self, sim_t=None, sim_y=None, ui=False, legend_mode="both", show_title=False):
        """
        Plot simulated concentrations and experimental data with selectable legend styles.

        Parameters:
        - sim_t: array of simulation time points
        - sim_y: array of concentrations over time
        - ui: if True, return figure object instead of showing it
        - legend_mode: str, one of ["both", "single", "components_only"]
            - "both": two legends (components + Simulace/Naměřené hodnoty)
            - "single": single legend with both component names and symbols
            - "components_only": single legend with components, no title
        """
        if sim_t is None and sim_y is None:
            sim_t, sim_y = self.simulate()
        elif sim_t is None or sim_y is None:
            raise ValueError("Both sim_t and sim_y must be set or not set at all.")

        fig = plt.figure()
        ax = plt.gca()
        num_curves = sim_y.shape[0]
        colors = plt.cm.tab10.colors

        custom_lines = []  # for combined symbol handles
        for i in range(num_curves):
            color = colors[i % len(colors)]
            name = self.column_names[i]
            ax.plot(sim_t, sim_y[i], color=color)
            ax.plot(self.data[:, 0], self.data[:, i + 1], 'o', color=color)
            custom_lines.append(Line2D([0], [0], color=color, marker='o', label=name))

        # Add legends depending on the mode
        if legend_mode == "both":
            first_legend = ax.legend(handles=custom_lines, title="Komponenty", loc="upper right")
            ax.add_artist(first_legend)
            ax.legend(handles=[
                Line2D([0], [0], color='gray', linestyle='-', label='Simulace'),
                Line2D([0], [0], color='gray', linestyle='None', marker='o', label='Naměřené hodnoty')
            ], title="Typ dat", loc="lower right")

        elif legend_mode == "single":
            legend_lines = custom_lines + [
                Line2D([0], [0], color='gray', linestyle='-', label='Simulace'),
                Line2D([0], [0], color='gray', linestyle='None', marker='o', label='Naměřené hodnoty')
            ]
            ax.legend(handles=legend_lines, title=None)

        elif legend_mode == "components_only":
            ax.legend(handles=custom_lines, title=None)

        plt.xlabel("Čas")
        plt.ylabel("Koncentrace")
        plt.ylim([0, 1.0])
        if show_title:
            plt.title("Kinetické rovnice (Epoxidace FAME)")
        plt.grid(True)
        plt.tight_layout()

        if not ui:
            plt.show()
        else:
            return fig

    def run_trial(self, k_trial, y0, t_range=(0, 500), time_points=100):
        """
        Run and return a trial simulation with arbitrary initial values.

        Parameters:
        - k_trial: list of kinetic constants
        - y0: initial concentrations [A, B, C, D, E]
        - t_range: tuple of time span (start, end)
        - time_points: resolution of output

        Returns:
        - Tuple (time, concentrations) of the trial simulation
        """
        t_eval = np.linspace(*t_range, time_points)
        sol = solve_ivp(self.odes, t_range, y0, t_eval=t_eval, args=(k_trial,))
        return sol.t, sol.y

    @staticmethod
    def test():
        r"""
        Run example fit + plot with hardcoded FAME-like data.
        This is a simplified test with normalized concentrations.
        """
        k_init = [0.005, 0.005, 0.005, 0.001, 0.001, 0.001]
        tmax = 500.0
        data = [
            [0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [1, 0.955882, 0.014706, 0.014706, 0.014706, 0.0],
            [10, 0.822581, 0.048387, 0.064516, 0.064516, 0.0],
            [20, 0.690909, 0.072727, 0.127273, 0.109091, 0.0],
            [30, 0.58, 0.1, 0.16, 0.16, 0.0],
            [45, 0.431818, 0.136364, 0.204545, 0.204545, 0.022727],
            [60, 0.324324, 0.162162, 0.243243, 0.243243, 0.027027],
            [80, 0.2, 0.2, 0.266667, 0.266667, 0.066667],
            [100, 0.130435, 0.217391, 0.304348, 0.26087, 0.086957],
            [120, 0.1, 0.2, 0.3, 0.25, 0.15],
        ]
        model = KineticModel(k_init, tmax, data, verbose=True, stars=True)
        model.fit()
        t_sim, y_sim = model.simulate()
        model.plot_debug(t_sim, y_sim)

        # Optional trial simulation
        t_trial, y_trial = model.run_trial(
            k_trial=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            y0=[1.0, 0.0, 0.0, 0.0, 0.0]
        )
        model.plot_debug(t_trial, y_trial)



def main():
    """
    Entry point: Run test simulation and fitting with example data.
    """
    KineticModel.test()


if __name__ == "__main__":
    main()
