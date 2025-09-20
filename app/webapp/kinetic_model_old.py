import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt


class KineticModel:
    r"""
    A class for simulating and fitting a first-order reaction kinetics model
    for FAME (Fatty Acid Methyl Esters) epoxidation or similar systems.

    The model assumes a cascade of irreversible reactions:

        A → B → E
        A → C → E
        A → D → E

    Typical interpretation for epoxidation:

        A: Unsaturated FAME (e.g., methyl oleate)
        B: Monoepoxide
        C: Diepoxide
        D: Triepoxide or side product (e.g., hydroxy-epoxide)
        E: Final product (fully epoxidized or degraded compound)

    Rate constants (first-order kinetics):

    - \( k_1 \): A → B
    - \( k_2 \): A → C
    - \( k_3 \): A → D
    - \( k_4 \): B → E
    - \( k_5 \): C → E
    - \( k_6 \): D → E

    """

    def __init__(self, k_init, tmax, concentration_data, bounds=(0.0, 10.0),
                 verbose=False, stars=False):
        """
        Initialize the kinetic model.

        Parameters:
        - k_init: list of initial guesses for rate constants [k1, k2, ..., k6]
        - tmax: maximum time for simulation
        - concentration_data: list or array with shape (n_timepoints, 6)
          where columns = [t, A, B, C, D, E]
        - bounds: tuple (min, max) to clip kinetic constants during fitting
        - verbose: if True, print fitting progress in detail
        - stars: if True, print * after each objective function evaluation
        """
        self.k_init = np.array(k_init, dtype=np.float64)
        self.tmax = tmax
        self.data = np.array(concentration_data)
        self.bounds = bounds
        self.verbose = verbose
        self.stars = stars
        self.k_fit = None

    def odes(self, t, conc, k):
        r"""
        Defines the system of ODEs for the kinetic model.

        Equations:
        \[
        \begin{aligned}
        \frac{dA}{dt} &= -(k_1 + k_2 + k_3) A \\
        \frac{dB}{dt} &= +k_1 A - k_4 B \\
        \frac{dC}{dt} &= +k_2 A - k_5 C \\
        \frac{dD}{dt} &= +k_3 A - k_6 D \\
        \frac{dE}{dt} &= +k_4 B + k_5 C + k_6 D
        \end{aligned}
        \]

        Parameters:
        - t: time
        - conc: list of concentrations [A, B, C, D, E]
        - k: kinetic constants [k1, k2, ..., k6]

        Returns:
        - list of derivatives [dA/dt, dB/dt, ..., dE/dt]
        """
        A, B, C, D, E = conc
        dAdt = -k[0] * A - k[1] * A - k[2] * A
        dBdt = k[0] * A - k[3] * B
        dCdt = k[1] * A - k[4] * C
        dDdt = k[2] * A - k[5] * D
        dEdt = k[3] * B + k[4] * C + k[5] * D
        return [dAdt, dBdt, dCdt, dDdt, dEdt]

    def objective(self, k):
        """
        Objective function for parameter fitting.

        Computes mean squared error (MSE) between model-predicted and
        experimental concentrations over all timepoints.

        Parameters:
        - k: list of kinetic constants

        Returns:
        - Scalar MSE error
        """
        k = np.clip(k, *self.bounds)
        error = 0
        for i in range(1, len(self.data)):
            t_span = [0, self.data[i, 0]]
            y0 = self.data[0, 1:]
            sol = solve_ivp(self.odes, t_span, y0, t_eval=[t_span[1]], args=(k,))
            pred = sol.y[:, -1]
            true = self.data[i, 1:]
            error += np.mean((true - pred) ** 2)
            if self.verbose:
                print(f"Time={t_span[1]:.1f}, True={true}, Pred={pred}")
            if self.stars:
                print("*", end="")
        return error

    def fit(self):
        """
        Perform optimization of kinetic constants using Nelder-Mead.

        Returns:
        - Optimal kinetic constants (as NumPy array)
        """
        c_bounds = [self.bounds] * len(self.k_init)  # 6 bounds for k1–k6
        result = minimize(self.objective,
                          self.k_init,
                          method='Nelder-Mead',
                          bounds=c_bounds)
        self.k_fit = result.x
        print("\nOptimal coefficients:", self.k_fit)
        return self.k_fit

    def simulate(self, k_values=None, time_points=100):
        """
        Simulate the model using provided or fitted kinetic constants.

        Parameters:
        - k_values: list of rate constants [k1–k6]; if None, use fitted
        - time_points: number of time points for simulation

        Returns:
        - Tuple (time, concentrations) as NumPy arrays
        """
        if k_values is None:
            if self.k_fit is None:
                raise ValueError("No fitted coefficients. Call fit() or pass k_values.")
            k_values = self.k_fit
        t_eval = np.linspace(0, self.tmax, time_points)
        y0 = self.data[0, 1:]
        sol = solve_ivp(self.odes,
                        t_span=[0, self.tmax],
                        y0=y0,
                        t_eval=t_eval,
                        args=(k_values,))
        return sol.t, sol.y

    def plot(self, sim_t, sim_y):
        """
        Plot simulated concentrations and experimental data.

        Parameters:
        - sim_t: array of simulation time points
        - sim_y: array of concentrations over time
        """
        plt.plot(sim_t, sim_y.T, label='Simulated')
        for i in range(1, self.data.shape[1]):
            plt.plot(self.data[:, 0], self.data[:, i], 'o', label=f'Exp {i}')
        plt.xlabel("Time")
        plt.ylabel("Concentration")
        plt.ylim([0, 1.0])
        plt.title("Kinetické rovnice (Epoxidace FAME)")
        plt.legend()
        plt.grid(True)
        plt.show()

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
        """
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
        model.plot(t_sim, y_sim)

        # Optional trial simulation
        t_trial, y_trial = model.run_trial(
            k_trial=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
            y0=[1.0, 0.0, 0.0, 0.0, 0.0]
        )
        model.plot(t_trial, y_trial)


def fit_epoxidation_kinetics(time, A_data, B_data, C_data, D_data, H_data):
    """
    Fit kinetic coefficients (k1 to k6) to GC/MS data of epoxidation and hydrolysis of C18:3.

    Reaction model:
        A --(k1)--> B --(k2)--> C --(k3)--> D
                      |          |         |
                     (k4)       (k5)      (k6)
                      v          v         v
                      ------------> H (hydrolysis pool)

    Parameters:
        time     : 1D array of time points
        A_data   : 1D array of A (C18:3) concentrations
        B_data   : 1D array of monoepoxide concentrations
        C_data   : 1D array of diepoxide concentrations
        D_data   : 1D array of triepoxide concentrations
        H_data   : 1D array of hydroxy product concentrations

    Returns:
        result: dict with keys:
            'k': fitted kinetic constants [k1, k2, k3, k4, k5, k6]
            'success': optimizer success flag
            'y_fit': fitted concentration trajectories at input time points (shape: [N, 5])
            'residual': mean squared error of fit
    """

    # Stack experimental data
    y_exp = np.stack([A_data, B_data, C_data, D_data, H_data], axis=1)

    # Initial conditions from first data point
    y0 = y_exp[0]

    def ode_system(t, y, k):
        A, B, C, D, E = y
        dAdt = -k[0] * A - k[1] * A - k[2] * A
        dBdt = k[0] * A - k[3] * B
        dCdt = k[1] * A - k[4] * C
        dDdt = k[2] * A - k[5] * D
        dEdt = k[3] * B + k[4] * C + k[5] * D
        return [dAdt, dBdt, dCdt, dDdt, dEdt]

    def simulate(k):
        sol = solve_ivp(lambda t, y: ode_system(t, y, k), [time[0], time[-1]], y0, t_eval=time, method='LSODA')
        return sol.y.T

    def loss(k):
        if np.any(np.array(k) < 0):
            return 1e10  # penalize negative rates
        y_sim = simulate(k)
        return np.mean((y_sim - y_exp) ** 2)

    # Initial guess for k1 to k6
    k0 = np.array([0.01, 0.01, 0.005, 0.001, 0.001, 0.001])

    # Bounds to ensure physical validity (non-negative rates)
    bounds = [(0, 1)] * 6

    result = minimize(loss, k0, method='L-BFGS-B', bounds=bounds)

    k_fit = result.x
    y_fit = simulate(k_fit)

    return {
        'k': k_fit,
        'success': result.success,
        'y_fit': y_fit,
        'residual': loss(k_fit)
    }


def main():
    """
    Entry point: Run test simulation and fitting with example data.
    """
    KineticModel.test()


if __name__ == "__main__":
    main()
