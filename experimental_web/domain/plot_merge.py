from __future__ import annotations

"""Helpers for rendering merged plots.

When a computation is split into two independent fits ("main" and "control"),
we want to render a single figure that contains outputs from both.

We intentionally keep :meth:`experimental_web.domain.kinetic_model.KineticModel.plot_debug`
untouched and re-implement the same visual style here.
"""


def plot_debug_merged(
    main_model,
    control_model=None,
    *,
    legend_mode: str = "components_only",
    t_max_plot: float | None = None,
    prefer_main: bool = True,
    main_active_columns: set[str] | None = None,
    control_active_columns: set[str] | None = None,
    ui: bool = True,
):
    """Plot a merged debug figure.

    - Curves from ``main_model`` are plotted for its active columns.
    - Curves from ``control_model`` are plotted for its active columns.

    When a column exists in both models:

    - If it is active in the *main* computation (``main_active_columns``), the
      main curve wins (control is skipped) when ``prefer_main=True``.
    - If it is **not** active in the main computation but is active in control,
      the control curve is plotted instead (this happens for nodes which are
      present in the global node list but have ``dX = 0`` in the main part).

    Returns a matplotlib Figure when ``ui=True``.
    """

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    sim_t_main, sim_y_main = main_model.simulate()
    cols_main = list(getattr(main_model, "column_names", []) or [])
    data_main = getattr(main_model, "data", None)

    cols_ctrl = []
    sim_t_ctrl = None
    sim_y_ctrl = None
    data_ctrl = None
    if control_model is not None:
        cols_ctrl = list(getattr(control_model, "column_names", []) or [])
        sim_t_ctrl, sim_y_ctrl = control_model.simulate()
        data_ctrl = getattr(control_model, "data", None)

    # Determine which columns are "active" in each computation.
    # If the caller doesn't provide active sets, fall back to "all columns".
    main_active = {str(c) for c in (main_active_columns or set(cols_main))}
    ctrl_active = {str(c) for c in (control_active_columns or set(cols_ctrl))}

    # Mapping for consistent colors: if a column exists in main, reuse its color
    # index even when plotting it from the control computation.
    main_index_by_name = {str(c): i for i, c in enumerate(cols_main)}

    # Decide which main curves to plot. If a column is not active in main but is
    # active in control, we skip plotting it from main so control can supply it.
    main_indices: list[int] = []
    for i, c in enumerate(cols_main):
        name = str(c)
        if control_model is not None and name in ctrl_active and name not in main_active:
            continue
        main_indices.append(i)

    # Decide which control curves to plot.
    ctrl_indices: list[int] = []
    if control_model is not None:
        for i, c in enumerate(cols_ctrl):
            name = str(c)
            if name not in ctrl_active:
                continue
            if prefer_main and name in main_active:
                continue
            ctrl_indices.append(i)

    fig = plt.figure()
    ax = plt.gca()
    colors = plt.cm.tab10.colors

    legend_handles: list[Line2D] = []

    def _plot_one(idx: int, name: str, t, y, data, color) -> None:
        ax.plot(t, y, color=color)
        if data is not None:
            try:
                ax.plot(data[:, 0], data[:, idx + 1], "o", color=color)
            except Exception:
                pass
        legend_handles.append(Line2D([0], [0], color=color, marker="o", label=name))

    # main curves (only selected indices)
    for i in main_indices:
        name = str(cols_main[i]) if i < len(cols_main) else str(i + 1)
        _plot_one(i, name, sim_t_main, sim_y_main[i], data_main, colors[i % len(colors)])

    # control curves (selected indices)
    if control_model is not None and sim_y_ctrl is not None and sim_t_ctrl is not None:
        base = len(legend_handles)
        for j, i in enumerate(ctrl_indices):
            name = str(cols_ctrl[i]) if i < len(cols_ctrl) else f"control_{i+1}"
            # If the same name exists in main, reuse its color index.
            if name in main_index_by_name:
                col = colors[main_index_by_name[name] % len(colors)]
            else:
                col = colors[(base + j) % len(colors)]
            _plot_one(i, name, sim_t_ctrl, sim_y_ctrl[i], data_ctrl, col)

    # legend styles match KineticModel.plot_debug
    if legend_mode == "both":
        first_legend = ax.legend(handles=legend_handles, title="Komponenty", loc="upper right")
        ax.add_artist(first_legend)
        ax.legend(
            handles=[
                Line2D([0], [0], color="gray", linestyle="-", label="Simulace"),
                Line2D([0], [0], color="gray", linestyle="None", marker="o", label="Naměřené hodnoty"),
            ],
            title="Typ dat",
            loc="lower right",
        )
    elif legend_mode == "single":
        legend_lines = legend_handles + [
            Line2D([0], [0], color="gray", linestyle="-", label="Simulace"),
            Line2D([0], [0], color="gray", linestyle="None", marker="o", label="Naměřené hodnoty"),
        ]
        ax.legend(handles=legend_lines, title=None)
    else:  # components_only
        ax.legend(handles=legend_handles, title=None)

    plt.xlabel("čas [s]")
    plt.ylabel("koncentrace [-]")

    # Apply x-limit (xlim) like plot_debug
    try:
        if t_max_plot is not None:
            mx = 0.0
            try:
                mx = float(getattr(sim_t_main, "max", lambda: 0)())
            except Exception:
                mx = 0.0
            try:
                mx_data = float(data_main[:, 0].max()) if data_main is not None else 0.0
            except Exception:
                mx_data = 0.0
            mx = max(mx, mx_data)
            req = float(t_max_plot)
            if mx > 0 and req > 0:
                ax.set_xlim(0.0, min(req, mx))
    except Exception:
        pass

    plt.ylim([0, 1.0])
    plt.grid(True)
    plt.tight_layout()

    if ui:
        return fig
    plt.show()
    return None
