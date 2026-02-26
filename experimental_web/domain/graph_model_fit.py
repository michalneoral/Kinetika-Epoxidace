from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class FitResult:
    state_names: List[str]
    param_names: List[str]
    optimal_params: List[float]
    t: List[float]
    y_hat: List[List[float]]  # shape: (len(t), n_states)
    y_data: List[List[float]]  # same shape
    plot_png_base64: str


def _detect_time_column(columns: Sequence[str]) -> Optional[str]:
    lowered = [c.lower() for c in columns]
    for key in ("čas", "cas", "time", "t"):
        for col, low in zip(columns, lowered):
            if key in low:
                return col
    return None


def fit_graph_ode_model(
    df,
    used_columns: List[str],
    ode_text: str,
    state_names: List[str],
    param_names: List[str],
    *,
    initialization: str = "TIME_SHIFT",
    t_shift: float = 9.0,
    optim_time_shift: bool = False,
    max_iter: int = 300,
) -> FitResult:
    """Fit a user-defined ODE system (from graph) to measured concentrations.

    IMPORTANT: This uses the same fitting pipeline as the prebuilt models:
      KineticModel + (optional) OptimumTShift + KineticModel.plot_debug()
    """

    # Configure logging inside the subprocess (run.cpu_bound).
    debug_enabled = False
    try:
        import os
        import logging
        from experimental_web.logging_setup import setup_logging, get_logger

        debug_env = os.environ.get('EXPERIMENTAL_WEB_DEBUG')
        debug_level = int(debug_env) if debug_env is not None else None
        setup_logging(debug_level)
        log = get_logger(__name__)
        debug_enabled = log.isEnabledFor(logging.DEBUG)
    except Exception:  # pragma: no cover
        log = None

    # Local imports to keep import footprint small.
    import matplotlib.pyplot as plt
    from io import BytesIO

    from experimental_web.domain.ode_compiler import compile_ode_equations
    from experimental_web.domain.kinetic_model import InitConditions, KineticModel, OptimumTShift

    if not used_columns:
        raise ValueError("Nejsou vybrané žádné sloupce se stavy (použitá data).")

    if log is not None:
        try:
            log.info('Fit graph ODE model: cols=%s; optim_time_shift=%s', len(used_columns), optim_time_shift)
        except Exception:
            pass

    missing_cols = [c for c in used_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Chybí sloupce v tabulce: {', '.join(missing_cols)}")

    time_col = _detect_time_column(list(df.columns))
    if time_col is None:
        t = np.arange(len(df), dtype=float)
        t_name = "time"
    else:
        t = np.asarray(df[time_col], dtype=float)
        t_name = str(time_col)

    y_data = np.asarray(df[used_columns], dtype=float)
    if y_data.ndim != 2:
        raise ValueError("Neočekávaný tvar dat pro stavy.")
    if len(t) != y_data.shape[0]:
        raise ValueError("Nesedí počet řádků času a stavů.")

    odes = compile_ode_equations(
        ode_text,
        state_names=state_names,
        param_names=param_names,
        order=["d" + s for s in state_names],
    )

    conc_data = np.concatenate([t.reshape(-1, 1), y_data], axis=1)

    try:
        init_enum = InitConditions[str(initialization)]
    except Exception:
        init_enum = InitConditions.TIME_SHIFT

    model = KineticModel(
        concentration_data=conc_data,
        column_names=list(used_columns),
        init_method=init_enum,
        t_shift=float(t_shift),
        custom_odes=odes,
        param_names_override=list(param_names),
        k_init=[0.01] * len(param_names),
        verbose=debug_enabled,
    )

    if optim_time_shift:
        ots = OptimumTShift(models={"custom": model}, t_shift_init=float(t_shift), verbose=False)
        t_shift_fit, _ = ots.fit()
        # minimize returns array-like
        try:
            t_shift = float(np.asarray(t_shift_fit).ravel()[0])
        except Exception:
            t_shift = float(t_shift)

    model.reinit_kinetic_model(t_shift=float(t_shift))
    model.fit()

    # Fit result on experimental time points (incl. inserted init row depending on init method)
    t_fit, y_hat = model.solve_ivp(model.k_fit)
    y_exp = model.y_exp

    fig = model.plot_debug(ui=True, legend_mode="components_only")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    k_fit = model.k_fit if model.k_fit is not None else np.zeros((len(param_names),), dtype=float)

    return FitResult(
        state_names=list(state_names),
        param_names=list(param_names),
        optimal_params=[float(x) for x in np.asarray(k_fit).tolist()],
        t=[float(x) for x in np.asarray(t_fit).tolist()],
        y_hat=[[float(v) for v in row] for row in np.asarray(y_hat).tolist()],
        y_data=[[float(v) for v in row] for row in np.asarray(y_exp).tolist()],
        plot_png_base64=b64,
    )
