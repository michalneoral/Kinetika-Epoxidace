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


def fit_graph_ode_model_split(
    df,
    used_columns: List[str],
    graph_state,
    *,
    initialization: str = "TIME_SHIFT",
    t_shift: float = 9.0,
    optim_time_shift: bool = False,
    max_iter: int = 300,
) -> FitResult:
    """Fit a graph-defined computation split into main (mode=1) + control (mode=2).

    The main part is fitted first (optionally with OptimumTShift).
    The control part is then fitted separately using the same ProcessingConfig and
    the same final t_shift.

    A single plot is returned; if a curve exists in both computations, the main
    curve is kept (control is only added for curves not present in main).

    Note: the returned time-series (t/y_hat/y_data) correspond to the main model.
    """

    # Local imports to keep import footprint small.
    import matplotlib.pyplot as plt
    from io import BytesIO

    import numpy as np

    from experimental_web.domain.ode_compiler import compile_ode_equations
    from experimental_web.domain.kinetic_model import InitConditions, KineticModel, OptimumTShift
    from experimental_web.domain.ode_generation import generate_ode_model
    from experimental_web.domain.plot_merge import plot_debug_merged

    if not used_columns:
        raise ValueError("Nejsou vybrané žádné sloupce se stavy (použitá data).")

    missing_cols = [c for c in used_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Chybí sloupce v tabulce: {', '.join(missing_cols)}")

    # Prefer graph_state.nodes ordering if available
    try:
        nodes_all = list(getattr(graph_state, 'nodes', []) or [])
    except Exception:
        nodes_all = []
    if nodes_all:
        used_columns = [c for c in nodes_all if c in set(used_columns)]

    time_col = _detect_time_column(list(df.columns))
    if time_col is None:
        t = np.arange(len(df), dtype=float)
    else:
        t = np.asarray(df[time_col], dtype=float)

    y_data = np.asarray(df[used_columns], dtype=float)
    if y_data.ndim != 2:
        raise ValueError("Neočekávaný tvar dat pro stavy.")
    if len(t) != y_data.shape[0]:
        raise ValueError("Nesedí počet řádků času a stavů.")

    conc_data = np.concatenate([t.reshape(-1, 1), y_data], axis=1)

    # Main ODE (mode=1)
    try:
        edge_modes = dict(getattr(graph_state, 'edge_modes', {}) or {})
    except Exception:
        edge_modes = {}

    ode_main = generate_ode_model(nodes_all or list(used_columns), edge_modes, include_modes=(1,))

    odes_main = compile_ode_equations(
        ode_main.ode_text,
        state_names=list(ode_main.state_names),
        param_names=list(ode_main.param_names),
        order=["d" + s for s in list(ode_main.state_names)],
    )

    try:
        init_enum = InitConditions[str(initialization)]
    except Exception:
        init_enum = InitConditions.TIME_SHIFT

    model_main = KineticModel(
        concentration_data=conc_data,
        column_names=list(used_columns),
        init_method=init_enum,
        t_shift=float(t_shift),
        custom_odes=odes_main,
        param_names_override=list(ode_main.param_names),
        k_init=[0.01] * (len(ode_main.param_names) if len(ode_main.param_names) > 0 else 1),
        verbose=False,
    )

    if optim_time_shift:
        ots = OptimumTShift(models={"main": model_main}, t_shift_init=float(t_shift), verbose=False)
        t_shift_fit, _ = ots.fit()
        try:
            t_shift = float(np.asarray(t_shift_fit).ravel()[0])
        except Exception:
            t_shift = float(t_shift)

    model_main.reinit_kinetic_model(t_shift=float(t_shift))
    model_main.fit()

    # Control induced subgraph (mode=2)
    endpoints = set()
    edge_ctrl = {}
    for (a, b), m in edge_modes.items():
        try:
            mv = int(m)
        except Exception:
            mv = 0
        if mv == 2:
            endpoints.add(a)
            endpoints.add(b)
            edge_ctrl[(a, b)] = 2

    nodes_ctrl = [n for n in (nodes_all or list(used_columns)) if n in endpoints]

    model_ctrl = None
    ode_ctrl = None
    if nodes_ctrl and edge_ctrl:
        cols_ctrl = [c for c in used_columns if c in set(nodes_ctrl)]
        if cols_ctrl:
            y_ctrl = np.asarray(df[cols_ctrl], dtype=float)
            conc_ctrl = np.concatenate([t.reshape(-1, 1), y_ctrl], axis=1)

            ode_ctrl = generate_ode_model(nodes_ctrl, edge_ctrl, include_modes=(2,))
            odes_ctrl = compile_ode_equations(
                ode_ctrl.ode_text,
                state_names=list(ode_ctrl.state_names),
                param_names=list(ode_ctrl.param_names),
                order=["d" + s for s in list(ode_ctrl.state_names)],
            )

            model_ctrl = KineticModel(
                concentration_data=conc_ctrl,
                column_names=list(cols_ctrl),
                init_method=init_enum,
                t_shift=float(t_shift),
                custom_odes=odes_ctrl,
                param_names_override=list(ode_ctrl.param_names),
                k_init=[0.01] * (len(ode_ctrl.param_names) if len(ode_ctrl.param_names) > 0 else 1),
                verbose=False,
            )
            model_ctrl.reinit_kinetic_model(t_shift=float(t_shift))
            model_ctrl.fit()

    # Active columns for merged plotting: control replaces main for nodes which
    # are present in the global node list but inactive in the main graph.
    main_active_cols: set[str] = set()
    try:
        for (a, b), m in edge_modes.items():
            try:
                mv = int(m)
            except Exception:
                mv = 0
            if mv == 1:
                main_active_cols.add(str(a))
                main_active_cols.add(str(b))
        main_active_cols = {c for c in main_active_cols if c in set(used_columns)}
    except Exception:
        main_active_cols = set(str(c) for c in used_columns)

    ctrl_active_cols: set[str] = set()
    try:
        if model_ctrl is not None:
            ctrl_active_cols = set(str(c) for c in getattr(model_ctrl, 'column_names', []) or [])
    except Exception:
        ctrl_active_cols = set()

    fig = plot_debug_merged(
        model_main,
        model_ctrl,
        ui=True,
        legend_mode="components_only",
        prefer_main=True,
        main_active_columns=main_active_cols,
        control_active_columns=ctrl_active_cols,
    )
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    # Merge parameters (prefer main)
    pnames = list(ode_main.param_names)
    k_main = model_main.k_fit if model_main.k_fit is not None else np.zeros((len(pnames),), dtype=float)
    pvals = [float(x) for x in np.asarray(k_main).tolist()]

    if model_ctrl is not None and ode_ctrl is not None:
        k_ctrl = model_ctrl.k_fit if model_ctrl.k_fit is not None else np.zeros((len(ode_ctrl.param_names),), dtype=float)
        seen = set(pnames)
        for n, v in zip(list(ode_ctrl.param_names), np.asarray(k_ctrl).tolist()):
            if n in seen:
                continue
            seen.add(n)
            pnames.append(str(n))
            try:
                pvals.append(float(v))
            except Exception:
                pvals.append(float('nan'))

    # Keep time-series from main
    t_fit, y_hat = model_main.solve_ivp(model_main.k_fit)
    y_exp = model_main.y_exp

    return FitResult(
        state_names=list(ode_main.state_names),
        param_names=pnames,
        optimal_params=pvals,
        t=[float(x) for x in np.asarray(t_fit).tolist()],
        y_hat=[[float(v) for v in row] for row in np.asarray(y_hat).tolist()],
        y_data=[[float(v) for v in row] for row in np.asarray(y_exp).tolist()],
        plot_png_base64=b64,
    )
