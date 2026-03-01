from __future__ import annotations

from dataclasses import asdict
from io import BytesIO
from typing import Any, Optional
from multiprocessing import Manager
import queue as pyqueue

import pandas as pd
from nicegui import ui, run

from experimental_web.core.paths import DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import (
    ExperimentFileRepository,
    TablePickRepository,
    ProcessedTablesRepository,
    ExperimentProcessingSettingsRepository,
    ExperimentProcessingResultsRepository,
)
from experimental_web.domain.fame_epo import DEFAULT_EPO, DEFAULT_FAME, extract_df
from experimental_web.domain.processing import TableProcessor, get_possible_models
from experimental_web.domain.processing_config import ProcessingConfig
from experimental_web.domain.kinetic_model import InitConditions
from experimental_web.ui.widgets.styled_label import StyledLabel
from experimental_web.logging_setup import get_logger, log_scope
from experimental_web.ui.instrumentation import wrap_ui_handler
from experimental_web.ui.utils.staleness import compute_staleness, is_model_stale


log = get_logger(__name__)


def _load_pick(pick_repo: TablePickRepository, experiment_id: int, kind: str):
    saved = pick_repo.get(experiment_id, kind)
    if saved:
        return saved["row_start"], saved["row_end"], saved["col_start"], saved["col_end"]
    c = DEFAULT_FAME if kind == "fame" else DEFAULT_EPO
    return c.row_start, c.row_end, c.col_start, c.col_end


def _normalize_first_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df2 = df.copy()
    if df2.columns.size > 0:
        df2 = df2.rename(columns={df2.columns[0]: name})
    return df2


def _fig_to_png_bytes(fig) -> bytes:
    import matplotlib.pyplot as plt
    from io import BytesIO as _BIO

    buf = _BIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def _compute_kinetics_job(
    *,
    excel_bytes: bytes,
    sheet: str,
    fame_pick: tuple[int, int, int, int],
    epo_pick: tuple[int, int, int, int],
    params: dict[str, Any],
    custom_computations: Optional[list[dict[str, Any]]] = None,
    progress_queue: Any | None = None,
) -> dict[str, Any]:
    """Runs in subprocess via run.cpu_bound. Must return only pickleable/JSONable data."""
    import os
    import base64 as _b64

    # Configure logging inside the subprocess (inherits EXPRIMENTAL_WEB_DEBUG from parent).
    try:
        from experimental_web.logging_setup import setup_logging, get_logger

        debug_env = os.environ.get('EXPERIMENTAL_WEB_DEBUG')
        debug_level = int(debug_env) if debug_env is not None else None
        setup_logging(debug_level)
        log = get_logger(__name__)
    except Exception:  # pragma: no cover
        log = None

    xls = pd.ExcelFile(BytesIO(excel_bytes))
    df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)

    r1, r2, c1, c2 = fame_pick
    fame_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), "FAME")

    r1, r2, c1, c2 = epo_pick
    epo_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), "EPOXIDES")

    for df in (fame_df, epo_df):
        for col in df.columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="ignore")

    from experimental_web.domain.kinetic_model import KineticModel
    from experimental_web.domain.ode_compiler import compile_ode_equations
    from experimental_web.domain.control_graph import control_subgraph, parse_edge_modes_payload
    from experimental_web.domain.ode_generation import generate_ode_model
    from experimental_web.domain.plot_merge import plot_debug_merged

    processor = TableProcessor(fame_df=fame_df, epo_df=epo_df)
    processor.process()

    if log is not None:
        try:
            log.info('Compute job: sheet=%s; custom_computations=%s', sheet, len(list(custom_computations or [])))
        except Exception:
            pass

    init_name = params.get("initialization", "TIME_SHIFT")
    params2 = dict(params)
    params2["initialization"] = InitConditions[init_name]


    # Support for optional control (mode=2) computations.
    control_specs: dict[str, Any] = {}
    effective_t_shift: float = float(params2.get('t_shift', 1.0) or 0.0)

    # Attach configurable (graph-defined) model(s), if any.
    custom_models: dict[str, Any] = {}
    try:
        items = list(custom_computations or [])
        for item in items:
            if not (item and item.get("ode_text") and item.get("used_heads") and item.get("state_names") and item.get("param_names")):
                continue

            table_name = str(item.get("table_name") or "")
            table = processor.tables.get(table_name)
            if table is None:
                continue

            # detect time column
            def _detect_time_column(columns):
                lowered = [str(c).lower() for c in columns]
                for key in ("čas", "cas", "time", "t"):
                    for col, low in zip(columns, lowered):
                        if key in low:
                            return col
                return None

            time_col = _detect_time_column(list(table.columns))
            cols = list(item.get("used_heads") or [])
            # enforce deterministic ordering: prefer graph_state.nodes if present
            try:
                nodes = list(item.get("nodes") or [])
                if nodes:
                    cols = [c for c in nodes if c in set(cols)]
            except Exception:
                pass

            if not cols:
                continue

            # Determine which columns are active in the main/control parts.
            # (Used for merged plotting: control replaces main for nodes which are
            # present in the global node list but inactive in the main graph.)
            main_active_cols: set[str] = set()
            control_active_cols: set[str] = set()
            try:
                edges_parsed = parse_edge_modes_payload(item.get('edge_modes'))
                for a, b, m in edges_parsed:
                    if int(m) == 1:
                        main_active_cols.add(str(a))
                        main_active_cols.add(str(b))
                    elif int(m) == 2:
                        control_active_cols.add(str(a))
                        control_active_cols.add(str(b))
                colset = set(str(c) for c in cols)
                main_active_cols = {c for c in main_active_cols if c in colset}
                control_active_cols = {c for c in control_active_cols if c in colset}
            except Exception:
                main_active_cols = set(str(c) for c in cols)
                control_active_cols = set()

            if time_col is None:
                t = pd.Series(range(len(table)), name="time", dtype=float)
                df_sel = pd.concat([t, table[cols]], axis=1)
                df_sel = df_sel.rename(columns={"time": "time"})
            else:
                df_sel = table[[time_col] + cols].copy()
                df_sel = df_sel.rename(columns={time_col: "time"})

            for c in cols:
                if c in df_sel.columns:
                    df_sel[c] = pd.to_numeric(df_sel[c], errors="coerce")
            df_sel["time"] = pd.to_numeric(df_sel["time"], errors="coerce")
            df_sel = df_sel.fillna(0.0)

            conc_data = df_sel[["time"] + cols].astype(float).values

            odes = compile_ode_equations(
                str(item.get("ode_text")),
                state_names=list(item.get("state_names") or []),
                param_names=list(item.get("param_names") or []),
                order=["d" + s for s in list(item.get("state_names") or [])],
            )

            pnames = list(item.get("param_names") or [])
            # Clamp t_max to the available time range in the selected data.
            try:
                max_time = float(conc_data[:, 0].max())
                if not (max_time > 0):
                    max_time = 0.0
            except Exception:
                max_time = 0.0
            try:
                req_tmax = float(params2.get('t_max', 400.0) or 0)
            except Exception:
                req_tmax = 0.0
            if max_time > 0:
                tmax_eff = min(max(1.0, req_tmax if req_tmax > 0 else max_time), max_time)
            else:
                tmax_eff = max(1.0, req_tmax if req_tmax > 0 else 400.0)

            model = KineticModel(
                concentration_data=conc_data,
                column_names=cols,
                init_method=params2["initialization"],
                t_shift=float(params2.get("t_shift", 1.0)),
                t_max=float(tmax_eff),
                custom_odes=odes,
                param_names_override=pnames,
                k_init=[0.01] * len(pnames),
                verbose=False,
            )

            display_name = str(item.get("display_name") or item.get("name") or "(bez názvu)")
            # Ensure uniqueness (can happen with duplicate names)
            base_key = f"Konfigurovatelný: {display_name}"
            key = base_key
            suffix = 2
            while key in custom_models:
                key = f"{base_key} ({suffix})"
                suffix += 1
            custom_models[key] = model

            # Build a separate control (mode=2) computation for this graph, if any.
            try:
                nodes_all = list(item.get('nodes') or []) or list(cols)
                nodes_ctrl, edge_modes_ctrl = control_subgraph(nodes_all, item.get('edge_modes'), mode=2)
                if nodes_ctrl and edge_modes_ctrl:
                    cols_ctrl = [c for c in nodes_ctrl if c in set(cols)]
                    if cols_ctrl:
                        if time_col is None:
                            t2 = pd.Series(range(len(table)), name='time', dtype=float)
                            df_ctrl = pd.concat([t2, table[cols_ctrl]], axis=1)
                            df_ctrl = df_ctrl.rename(columns={'time': 'time'})
                        else:
                            df_ctrl = table[[time_col] + cols_ctrl].copy()
                            df_ctrl = df_ctrl.rename(columns={time_col: 'time'})

                        for c in cols_ctrl:
                            if c in df_ctrl.columns:
                                df_ctrl[c] = pd.to_numeric(df_ctrl[c], errors='coerce')
                        df_ctrl['time'] = pd.to_numeric(df_ctrl['time'], errors='coerce')
                        df_ctrl = df_ctrl.fillna(0.0)
                        conc_ctrl = df_ctrl[['time'] + cols_ctrl].astype(float).values

                        ode_ctrl = generate_ode_model(nodes_ctrl, edge_modes_ctrl, include_modes=(2,))
                        odes_ctrl = compile_ode_equations(
                            ode_ctrl.ode_text,
                            state_names=list(ode_ctrl.state_names),
                            param_names=list(ode_ctrl.param_names),
                            order=['d' + s for s in list(ode_ctrl.state_names)],
                        )

                        # Clamp t_max for control data as well
                        try:
                            max_time2 = float(conc_ctrl[:, 0].max())
                            if not (max_time2 > 0):
                                max_time2 = 0.0
                        except Exception:
                            max_time2 = 0.0
                        if max_time2 > 0:
                            tmax_ctrl = min(max(1.0, req_tmax if req_tmax > 0 else max_time2), max_time2)
                        else:
                            tmax_ctrl = max(1.0, req_tmax if req_tmax > 0 else 400.0)

                        control_specs[key] = {
                            'cols': cols_ctrl,
                            'conc': conc_ctrl,
                            'odes': odes_ctrl,
                            'param_names': list(ode_ctrl.param_names),
                            't_max': float(tmax_ctrl),
                            'main_active_cols': set(main_active_cols),
                            'control_active_cols': set(cols_ctrl),
                        }
            except Exception:
                pass
    except Exception:
        # custom model is optional; ignore errors here
        pass

    results: dict[str, Any] = {"models": {}}

    def _emit(msg: Any) -> None:
        if progress_queue is None:
            return
        try:
            progress_queue.put_nowait(msg)
        except Exception:
            try:
                progress_queue.put(msg)
            except Exception:
                pass

    def _post_fit_callback(name: str, model: Any, idx: int, total: int) -> None:
        entry: dict[str, Any] = {}

        # Optional control model (computed separately, same ProcessingConfig).
        ctrl = None
        if name in control_specs:
            spec = control_specs.get(name) or {}
            try:
                pnames_ctrl = list(spec.get('param_names') or [])
                # If there are no control params, we still build the model to generate curves.
                ctrl = KineticModel(
                    concentration_data=spec.get('conc'),
                    column_names=list(spec.get('cols') or []),
                    init_method=params2['initialization'],
                    t_shift=float(effective_t_shift),
                    t_max=float(spec.get('t_max') or params2.get('t_max', 400.0)),
                    custom_odes=spec.get('odes'),
                    param_names_override=pnames_ctrl,
                    k_init=[0.01] * len(pnames_ctrl) if pnames_ctrl else [0.01],
                    verbose=False,
                )
                ctrl.reinit_kinetic_model(t_shift=float(effective_t_shift))
                ctrl.fit()
            except Exception as e:
                ctrl = None
                entry['control_error'] = str(e)

        # Merge constants (prefer main when duplicates exist).
        consts_out: list[dict[str, Any]] = []
        try:
            main_consts = list(model.get_constants_with_names() or [])
        except Exception:
            main_consts = []
        try:
            ctrl_consts = list(ctrl.get_constants_with_names() or []) if ctrl is not None else []
        except Exception:
            ctrl_consts = []

        seen = set()
        for k in main_consts:
            latex = str(k.get('latex', ''))
            seen.add(latex)
            try:
                consts_out.append({'latex': latex, 'value': float(k.get('value'))})
            except Exception:
                pass
        for k in ctrl_consts:
            latex = str(k.get('latex', ''))
            if latex in seen:
                continue
            seen.add(latex)
            try:
                consts_out.append({'latex': latex, 'value': float(k.get('value'))})
            except Exception:
                pass

        # Add indices expected by UI
        entry['constants'] = [
            {'idx': i + 1, 'latex': c.get('latex', ''), 'value': float(c.get('value', 0.0))}
            for i, c in enumerate(consts_out)
        ]

        # Plot merged figure: control curves are added only if not present in main.
        try:
            fig = plot_debug_merged(
                model,
                ctrl,
                legend_mode='components_only',
                t_max_plot=params2.get('t_max_plot'),
                prefer_main=True,
                main_active_columns=set((spec.get('main_active_cols') or [])) if name in control_specs else None,
                control_active_columns=set((spec.get('control_active_cols') or [])) if name in control_specs else None,
                ui=True,
            )
            png = _fig_to_png_bytes(fig)
            entry['plot_png_b64'] = _b64.b64encode(png).decode('ascii')
        except Exception as e:
            entry['plot_error'] = str(e)

        results['models'][name] = entry
        _emit({'type': 'model_result', 'name': name, 'entry': entry, 'done': idx, 'total': total})

    def _progress_callback(event: Any) -> None:
        """Forward progress events from the domain pipeline to the UI."""
        nonlocal effective_t_shift
        if isinstance(event, dict):
            # Capture the final t_shift from OptimumTShift to reuse for control models.
            try:
                if event.get('type') == 'phase' and event.get('phase') == 'optim_time_shift' and event.get('state') == 'end':
                    if 't_shift' in event and event.get('t_shift') is not None:
                        effective_t_shift = float(event.get('t_shift'))
            except Exception:
                pass
            _emit(event)

    # Run the original pipeline (incl. OptimumTShift) and stream progress per-model.
    processor.compute_all_kinetics(
        **params2,
        custom_models=custom_models,
        post_fit_callback=_post_fit_callback,
        progress_callback=_progress_callback,
    )

    # Fallback: ensure we return all models even if streaming failed for some reason.
    for name, model in (processor.k_models or {}).items():
        if name in results["models"]:
            continue
        try:
            _post_fit_callback(name, model, idx=0, total=max(1, len(processor.k_models or {})))
        except Exception:
            pass

    return results


def render_processing_tab(experiment_id: int) -> None:
    """Tab: rychlosti / processing (náročné -> spouští se tlačítkem)."""

    st = get_state()
    file_repo = ExperimentFileRepository(DB_PATH)
    pick_repo = TablePickRepository(DB_PATH)
    settings_repo = ExperimentProcessingSettingsRepository(DB_PATH)
    results_repo = ExperimentProcessingResultsRepository(DB_PATH)
    tables_cache_repo = ProcessedTablesRepository(DB_PATH)

    # --- load persisted settings (best effort) ---
    saved: dict | None = None
    params_init = InitConditions.TIME_SHIFT
    models: list[str] = []
    auto_t_shift_value: float | None = None
    try:
        saved = settings_repo.get(experiment_id)
        if saved:
            try:
                init_name = str(saved.get('initialization') or 'TIME_SHIFT')
                params_init = InitConditions[init_name]
            except Exception:
                params_init = InitConditions.TIME_SHIFT

            try:
                import json as _json2
                models = _json2.loads(saved.get('models_to_compute_json') or '[]')
                if not isinstance(models, list):
                    models = []
                models = [str(x) for x in models]
            except Exception:
                models = []

            auto_t_shift_value = saved.get('last_auto_t_shift')
    except Exception:
        saved = None
        params_init = InitConditions.TIME_SHIFT
        models = []

    had_saved_settings = bool(saved)

    params = ProcessingConfig()
    # apply loaded settings
    try:
        if saved:
            params.initialization = params_init
            try:
                params.t_shift = float(saved.get('t_shift') or params.t_shift)
            except Exception:
                pass
            params.optim_time_shift = bool(int(saved.get('optim_time_shift') or 0))
            try:
                params.t_max = float(saved.get('t_max') or params.t_max)
            except Exception:
                pass
            try:
                params.t_max_plot = float(saved.get('t_max_plot') or params.t_max_plot)
            except Exception:
                pass
            params.models_to_compute = list(models or [])
    except Exception:
        pass
    params_gui = {"expanded": True}

    status = StyledLabel("Nastavte parametry a klikněte na Spočítat.", "info")
    spinner = ui.spinner(size="lg", color="primary")
    spinner.visible = False

    # --- persist settings with a light debounce (no spam writes) ---
    # NOTE: We must not write immediately on page open. Otherwise updated_at changes even without user edits,
    # and the UI will incorrectly show "Změněno".
    last_settings_snapshot: dict | None = None

    def _settings_snapshot() -> dict:
        return {
            'initialization': params.initialization.name,
            't_shift': float(params.t_shift),
            'optim_time_shift': bool(params.optim_time_shift),
            'models_to_compute': list(params.models_to_compute or []),
            't_max': float(getattr(params, 't_max', 400.0)),
            't_max_plot': float(getattr(params, 't_max_plot', 400.0)),
            'last_auto_t_shift': float(auto_t_shift_value) if auto_t_shift_value is not None else None,
        }

    # Initialize snapshot right after render to prevent an automatic upsert on first timer tick.
    last_settings_snapshot = _settings_snapshot()

    def _persist_settings_if_changed() -> None:
        nonlocal last_settings_snapshot
        snap = _settings_snapshot()
        if last_settings_snapshot == snap:
            return
        try:
            settings_repo.upsert(
                experiment_id=experiment_id,
                initialization=snap['initialization'],
                t_shift=snap['t_shift'],
                optim_time_shift=snap['optim_time_shift'],
                models_to_compute=snap['models_to_compute'],
                t_max=snap['t_max'],
                t_max_plot=snap['t_max_plot'],
                last_auto_t_shift=snap['last_auto_t_shift'],
            )
            last_settings_snapshot = snap
        except Exception:
            # persistence is best-effort and must not break UI
            pass

    progress_queue: Any | None = None
    progress_total = 0
    progress_done = 0
    progress_note = ''
    model_bodies: dict[str, Any] = {}
    manager: Manager | None = None

    # Progress is shown only transiently in a dialog (not as a permanent part of the UI).
    progress_dialog = ui.dialog().props('persistent')
    with progress_dialog, ui.card().classes('w-[520px] max-w-[92vw]'):
        with ui.row().classes('items-center justify-between'):
            ui.label('Počítám grafy…').classes('text-h6')
            phase_spinner = ui.spinner(size='sm', color='primary')
            phase_spinner.visible = False
        progress_label = ui.label('').classes('text-caption text-grey-7')
        progress_bar = ui.linear_progress(value=0).classes('w-full')
        ui.separator().classes('q-mt-sm q-mb-sm')
        ui.label('Výsledky se doplňují průběžně.').classes('text-caption text-grey-6')

    def _expected_model_order(selected: list[str], custom_names: list[str]) -> list[str]:
        order = [
            'C18:1_simplified',
            'C18:1',
            'C18:2_simplified',
            'C18:2_eps_and_others',
            'C18:2',
            'C18:2_separated',
            'C18:2_with_k_uh',
            'C20:1_simplified',
        ]
        models = [m for m in order if m in set(selected or [])]
        for cn in (custom_names or []):
            models.append(f"Konfigurovatelný: {cn}")
        return models

    def _render_model_body(body: Any, entry: dict[str, Any]) -> None:
        body.clear()
        with body:
            if entry.get('plot_error'):
                StyledLabel(f"Chyba při vykreslování grafu: {entry['plot_error']}", "warning")
            else:
                b64 = entry.get('plot_png_b64', '')
                if not b64 and entry.get('plot_png_bytes'):
                    try:
                        import base64
                        b64 = base64.b64encode(entry.get('plot_png_bytes') or b'').decode('ascii')
                    except Exception:
                        b64 = ''
                if b64:
                    ui.image(f"data:image/png;base64,{b64}").classes('max-w-4xl')

            consts = entry.get('constants') or []
            if consts:
                ui.label('Konstanty:').classes('text-subtitle2')
                for c in consts:
                    ui.markdown(
                        r"$$k_{" + f"{c['idx']}" + r"}" + f" = {c['latex']} = {c['value']:.8f}$$",
                        extras=['latex'],
                    )

    def _update_progress_label() -> None:
        if progress_total <= 0:
            progress_label.set_text('')
            return
        pct = 100.0 * (progress_done / progress_total)
        note = (progress_note or '').strip()
        if note:
            progress_label.set_text(f"{progress_done}/{progress_total} ({pct:.0f}%) — {note}")
        else:
            progress_label.set_text(f"{progress_done}/{progress_total} ({pct:.0f}%)")

    def _poll_progress_queue() -> None:
        nonlocal progress_queue, progress_total, progress_done, progress_note, auto_t_shift_value
        if progress_queue is None:
            return
        while True:
            try:
                msg = progress_queue.get_nowait()
            except pyqueue.Empty:
                break
            except Exception:
                break

            if not isinstance(msg, dict):
                continue

            mtype = msg.get('type')

            # Phase events: used e.g. to show an indeterminate progress bar during OptimumTShift.
            if mtype == 'phase' and msg.get('phase') == 'optim_time_shift':
                state = msg.get('state')
                if state == 'start':
                    progress_note = 'Optimalizuji t_shift…'
                    try:
                        phase_spinner.visible = True
                    except Exception:
                        pass
                    try:
                        progress_bar.props('indeterminate')
                    except Exception:
                        pass
                    _update_progress_label()
                elif state == 'end':
                    try:
                        phase_spinner.visible = False
                    except Exception:
                        pass
                    try:
                        progress_bar.props(remove='indeterminate')
                    except Exception:
                        pass
                    try:
                        ts = msg.get('t_shift')
                        if ts is not None:
                            # Update the persisted auto t_shift and the UI label.
                            try:
                                auto_t_shift_value = float(ts)
                            except Exception:
                                auto_t_shift_value = None
                            progress_note = f"t_shift = {float(ts):.3g}"
                            try:
                                auto_t_shift_label.set_text(f"{float(ts):.6g}")
                                auto_t_shift_label.classes(remove='text-grey-6')
                            except Exception:
                                pass

                            # Persist the newly computed auto t_shift immediately so a crash/restart
                            # still keeps the value. This is best-effort and must not break UI.
                            try:
                                settings_repo.upsert(
                                    experiment_id=experiment_id,
                                    initialization=params.initialization.name,
                                    t_shift=float(params.t_shift),
                                    optim_time_shift=bool(params.optim_time_shift),
                                    models_to_compute=list(params.models_to_compute or []),
                                    t_max=float(getattr(params, 't_max', 400.0)),
                                    t_max_plot=float(getattr(params, 't_max_plot', 400.0)),
                                    last_auto_t_shift=float(auto_t_shift_value) if auto_t_shift_value is not None else None,
                                )
                                # Keep snapshot in sync.
                                nonlocal last_settings_snapshot
                                last_settings_snapshot = _settings_snapshot()
                            except Exception:
                                pass
                        else:
                            progress_note = ''
                    except Exception:
                        progress_note = ''
                    # keep done/total at 0/N until real model results arrive
                    progress_bar.set_value(min(1.0, max(0.0, (progress_done / progress_total) if progress_total else 0.0)))
                    _update_progress_label()
                continue

            # Model start events: update the note so the UI isn't "stuck" for long fits.
            if mtype == 'model_start':
                try:
                    progress_note = f"Počítám: {str(msg.get('name') or '')}"
                except Exception:
                    progress_note = 'Počítám…'
                try:
                    phase_spinner.visible = False
                except Exception:
                    pass
                try:
                    progress_bar.props(remove='indeterminate')
                except Exception:
                    pass
                _update_progress_label()
                continue

            if mtype == 'model_result':
                name = str(msg.get('name') or '')
                entry = msg.get('entry') or {}
                progress_done = int(msg.get('done') or progress_done)
                progress_total = int(msg.get('total') or progress_total)

                progress_note = f"Hotovo: {name}" if name else ''

                if progress_total > 0:
                    progress_bar.set_value(min(1.0, max(0.0, progress_done / progress_total)))
                    _update_progress_label()

                body = model_bodies.get(name)
                if body is not None:
                    _render_model_body(body, entry)
                else:
                    # unexpected model: append it
                    with results_container:
                        with ui.card().classes('w-full'):
                            ui.markdown(f"#### Výsledky pro {name}")
                            body2 = ui.column().classes('w-full')
                            model_bodies[name] = body2
                            _render_model_body(body2, entry)

    def load_excel_bytes_and_sheet() -> tuple[Optional[bytes], Optional[str]]:
        ef = file_repo.get_single_for_experiment(experiment_id)
        if not ef:
            return None, None
        if not st.current_excel_sheet:
            st.current_excel_sheet = ef.selected_sheet or ""
        try:
            xls = pd.ExcelFile(BytesIO(ef.content))
        except Exception as e:
            ui.notify(f"Nelze otevřít Excel: {e}", type="negative")
            return ef.content, None

        sheets = list(xls.sheet_names or [])
        sheet = st.current_excel_sheet or (sheets[0] if sheets else "")
        if sheet not in sheets and sheets:
            sheet = sheets[0]
        if not sheet:
            return ef.content, None
        return ef.content, sheet

    def render_results(result: dict[str, Any]) -> None:
        results_container.clear()
        with results_container:
            models = result.get("models") or {}
            if not models:
                StyledLabel("Kinetické modely nebyly spočítány.", "warning")
                return

            for name, entry in models.items():
                ui.markdown(f"#### Výsledky pro {name}")

                if entry.get("plot_error"):
                    StyledLabel(f"Chyba při vykreslování grafu: {entry['plot_error']}", "warning")
                else:
                    b64 = entry.get("plot_png_b64", "")
                    if b64:
                        ui.image(f"data:image/png;base64,{b64}").classes("max-w-4xl")

                consts = entry.get("constants") or []
                if consts:
                    ui.label("Konstanty:").classes("text-subtitle2")
                    for c in consts:
                        ui.markdown(
                            r"$$k_{" + f"{c['idx']}" + r"}" + f" = {c['latex']} = {c['value']:.8f}$$",
                            extras=["latex"],
                        )
                ui.separator()

    async def on_compute() -> None:
        nonlocal progress_queue, progress_total, progress_done, progress_note, model_bodies, manager
        if not params.models_to_compute:
            # allow running if there is at least one saved configurable computation
            try:
                from experimental_web.data.repositories import ExperimentComputationRepository

                if not ExperimentComputationRepository(DB_PATH).list_for_experiment(experiment_id):
                    status.set("Vyberte alespoň jeden model.", "warning")
                    return
            except Exception:
                status.set("Vyberte alespoň jeden model.", "warning")
                return

        excel_bytes, sheet = load_excel_bytes_and_sheet()
        if excel_bytes is None or sheet is None:
            status.set("Nejdřív nahrajte Excel a vyberte list v tabu „načtení dat“.", "warning")
            return

        fame_pick = _load_pick(pick_repo, experiment_id, "fame")
        epo_pick = _load_pick(pick_repo, experiment_id, "epo")

        payload = asdict(params)
        payload["initialization"] = params.initialization.name

        # Persist current settings immediately (so restart restores current UI state)
        try:
            settings_repo.upsert(
                experiment_id=experiment_id,
                initialization=str(payload.get('initialization') or 'TIME_SHIFT'),
                t_shift=float(payload.get('t_shift') or 0),
                optim_time_shift=bool(payload.get('optim_time_shift')),
                models_to_compute=list(payload.get('models_to_compute') or []),
                t_max=float(payload.get('t_max') or 400.0),
                t_max_plot=float(payload.get('t_max_plot') or 400.0),
                last_auto_t_shift=auto_t_shift_value,
            )
        except Exception:
            pass

        if log.isEnabledFor(10):
            log.debug(
                '[UI] compute payload: initialization=%s t_shift=%s optim_time_shift=%s t_max=%s t_max_plot=%s models=%s',
                payload.get('initialization'),
                payload.get('t_shift'),
                payload.get('optim_time_shift'),
                payload.get('t_max'),
                payload.get('t_max_plot'),
                list(payload.get('models_to_compute') or []),
            )

        # all configurable computations (optional)
        custom_payloads: list[dict[str, Any]] = []
        try:
            from experimental_web.data.repositories import ExperimentComputationRepository

            items = ExperimentComputationRepository(DB_PATH).list_for_experiment(experiment_id)
            if items:
                seen: dict[str, int] = {}
                for it in items:
                    base = str(it.name or '(bez názvu)')
                    if base not in seen:
                        seen[base] = 1
                        disp = base
                    else:
                        seen[base] += 1
                        disp = f"{base} ({seen[base]})"
                    custom_payloads.append(
                        {
                            "name": it.name,
                            "display_name": disp,
                            "table_name": it.table_name,
                            "used_heads": list(it.used_heads or []),
                            "nodes": list(getattr(it.graph_state, "nodes", []) or []),
                            "edge_modes": [
                                [a, b, int(m)]
                                for (a, b), m in dict(getattr(it.graph_state, "edge_modes", {}) or {}).items()
                            ],
                            "ode_text": it.ode_text,
                            "state_names": list(it.state_names or []),
                            "param_names": list(it.param_names or []),
                        }
                    )
        except Exception:
            custom_payloads = []

        expected_custom_names = [str(p.get('display_name') or p.get('name') or '(bez názvu)') for p in custom_payloads]
        expected_models = _expected_model_order(params.models_to_compute, expected_custom_names)

        # Replace old results with skeleton placeholders until new results arrive.
        results_container.clear()
        model_bodies = {}
        with results_container:
            for name in expected_models:
                with ui.card().classes('w-full'):
                    ui.markdown(f"#### Výsledky pro {name}")
                    body = ui.column().classes('w-full')
                    model_bodies[name] = body
                    with body:
                        ui.skeleton('rect').classes('w-full').style('height: 260px')
                        ui.skeleton('text').classes('w-64')
                        ui.skeleton('text').classes('w-48')
                        ui.skeleton('text').classes('w-72')

        progress_done = 0
        progress_total = max(1, len(expected_models))
        progress_note = 'Spouštím výpočet…'
        try:
            phase_spinner.visible = False
        except Exception:
            pass
        try:
            progress_bar.props(remove='indeterminate')
        except Exception:
            pass
        progress_bar.set_value(0)
        _update_progress_label()
        progress_dialog.open()

        # create a queue to stream progress from the subprocess
        try:
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
            manager = Manager()
            progress_queue = manager.Queue()
        except Exception:
            manager = None
            progress_queue = None

        status.set("Počítám… (může chvíli trvat)", "info")
        spinner.visible = True
        try:
            result = await run.cpu_bound(
                _compute_kinetics_job,
                excel_bytes=excel_bytes,
                sheet=sheet,
                fame_pick=fame_pick,
                epo_pick=epo_pick,
                params=payload,
                custom_computations=custom_payloads,
                progress_queue=progress_queue,
            )
        except Exception as e:
            status.set(f"Chyba při výpočtu: {e}", "error")
            spinner.visible = False
            progress_dialog.close()
            progress_queue = None
            if manager is not None:
                try:
                    manager.shutdown()
                except Exception:
                    pass
                manager = None
            return

        # Drain any remaining progress messages before closing (otherwise the last model can be missed).
        try:
            _poll_progress_queue()
        except Exception:
            pass

        # Ensure the dialog reaches 100% before disappearing.
        try:
            models_done = len(((result or {}) or {}).get('models') or {})
            if models_done > 0:
                progress_total = max(int(progress_total or 0), models_done)
            if progress_total > 0:
                progress_done = progress_total
                progress_note = 'Hotovo.'
                try:
                    progress_bar.props(remove='indeterminate')
                except Exception:
                    pass
                try:
                    phase_spinner.visible = False
                except Exception:
                    pass
                progress_bar.set_value(1.0)
                _update_progress_label()
        except Exception:
            pass

        spinner.visible = False
        progress_dialog.close()
        progress_queue = None
        if manager is not None:
            try:
                manager.shutdown()
            except Exception:
                pass
            manager = None

        status.set("Hotovo.", "ok")

        # Persist results (constants + plots) into DB so they can be shown after restart.
        try:
            used_t_shift = None
            if bool(params.optim_time_shift):
                used_t_shift = float(auto_t_shift_value) if auto_t_shift_value is not None else None
            else:
                used_t_shift = float(params.t_shift)

            # Persist also the data context used for this run (sheet + table picks),
            # so graphs can be reliably regenerated even if the user changes UI later.
            run_settings = dict(payload or {})
            try:
                from experimental_web.data.repositories import ExperimentFileRepository, TablePickRepository

                ef2 = ExperimentFileRepository(DB_PATH).get_single_for_experiment(experiment_id)
                if ef2 and ef2.selected_sheet:
                    run_settings['sheet'] = str(ef2.selected_sheet)
                if ef2 and getattr(ef2, 'sha256', None):
                    run_settings['file_sha256'] = str(ef2.sha256)
                if ef2 and getattr(ef2, 'filename', None):
                    run_settings['file_name'] = str(ef2.filename)

                pick_repo2 = TablePickRepository(DB_PATH)
                fp = pick_repo2.get(experiment_id, 'fame')
                ep = pick_repo2.get(experiment_id, 'epo')
                if fp:
                    run_settings['fame_pick'] = [fp['row_start'], fp['row_end'], fp['col_start'], fp['col_end']]
                if ep:
                    run_settings['epo_pick'] = [ep['row_start'], ep['row_end'], ep['col_start'], ep['col_end']]
            except Exception:
                pass

            run_id = results_repo.create_run(
                experiment_id=experiment_id,
                settings=run_settings,
                used_t_shift=used_t_shift,
                auto_t_shift=float(auto_t_shift_value) if auto_t_shift_value is not None else None,
            )
            import base64
            for name, entry in (((result or {}) or {}).get('models') or {}).items():
                b64 = entry.get('plot_png_b64') or ''
                png_bytes = base64.b64decode(b64) if b64 else None
                results_repo.add_model_result(
                    run_id=run_id,
                    model_name=str(name),
                    constants=list(entry.get('constants') or []),
                    plot_png=png_bytes,
                    plot_error=str(entry.get('plot_error')) if entry.get('plot_error') else None,
                )
        except Exception as e:
            log.debug('Failed to persist results: %s', e)

        # Notify other tabs (especially "grafy") that new results are available.
        # The experiment page builds all tab contents once; without an explicit refresh signal,
        # the graphs tab would only update after reopening the experiment.
        try:
            st.graphs_version = st.graphs_version + 1
        except Exception:
            pass

        # Ensure all models are rendered (in case streaming didn't deliver everything)
        models = (result or {}).get('models') or {}
        for name, entry in models.items():
            body = model_bodies.get(name)
            if body is not None:
                _render_model_body(body, entry)
            else:
                with results_container:
                    with ui.card().classes('w-full'):
                        ui.markdown(f"#### Výsledky pro {name}")
                        body2 = ui.column().classes('w-full')
                        model_bodies[name] = body2
                        _render_model_body(body2, entry)

        # If some expected placeholders did not receive a result (e.g. missing table/ODE), replace skeleton with a hint.
        for expected_name, body in list(model_bodies.items()):
            if expected_name in models:
                continue
            try:
                body.clear()
                with body:
                    StyledLabel(
                        "Tento model se nepodařilo spočítat (pravděpodobně chybí tabulka, vybrané sloupce nebo ODE rovnice).",
                        "warning",
                    )
            except Exception:
                pass

    with ui.expansion("Nastavení parametrů pro výpočet", icon="⚙️").classes("w-full rounded-borders shadow-1")\
        .props("dense").bind_value(params_gui, "expanded"):

        with ui.row().classes("q-gutter-xl items-start"):
            with ui.column().tooltip("Vyberte jen modely, které chcete spočítat. Více modelů snižuje rychlost výpočtu"):
                ui.label("Modely, které se budou počítat:")
                for item in get_possible_models():
                    ui.checkbox(
                        item,
                        value=item in params.models_to_compute,
                        on_change=wrap_ui_handler(
                            f'processing.models_to_compute.toggle:{item}',
                            lambda e, name=item: (
                                params.models_to_compute.append(name) if e.value and name not in params.models_to_compute else
                                (params.models_to_compute.remove(name) if (not e.value and name in params.models_to_compute) else None)
                            ),
                            level=10,
                            data=lambda e, name=item: {'value': bool(e.value), 'model': name, 'selected': list(params.models_to_compute)},
                        ),
                    )

            with ui.column():
                options = list(InitConditions)
                labels = [opt.name for opt in options]

                help_text = "### Nastavení volby\n\n"
                help_text += "\n".join([f"- **{opt.name.upper()}**: {opt.description}" for opt in options])
                help_text += "\n\nDoporučeno: Použít **TIME_SHIFT**"

                select = ui.select(options=labels, label="Zvolte způsob inicializace:", value=params.initialization.name)
                select.bind_value(params, "initialization", forward=lambda v: InitConditions[v], backward=lambda v: v.name).classes("w-80")
                select.on('update:model-value', wrap_ui_handler('processing.initialization.change', lambda e: None, level=10, data=lambda e: {'value': getattr(e, 'value', None)}))
                with ui.tooltip():
                    ui.markdown(help_text)

                ui.checkbox(
                    "Time shift automaticky",
                    on_change=wrap_ui_handler('processing.optim_time_shift.toggle', lambda e: None, level=10, data=lambda e: {'value': bool(getattr(e, 'value', False))}),
                ).bind_value(params, "optim_time_shift")

                # Show last computed auto t_shift near the toggle (only when auto is enabled).
                auto_info = ui.row().classes('items-center gap-2')
                auto_info.bind_visibility_from(params, 'optim_time_shift', lambda v: bool(v))
                with auto_info:
                    ui.label('Auto t_shift:').classes('text-caption text-grey-7')
                    auto_t_shift_label = ui.label('').classes('text-caption')
                    try:
                        if auto_t_shift_value is None:
                            auto_t_shift_label.set_text('(zatím nespočítáno)')
                            auto_t_shift_label.classes('text-grey-6')
                        else:
                            auto_t_shift_label.set_text(f"{float(auto_t_shift_value):.6g}")
                    except Exception:
                        pass

                manual = ui.column()
                manual.bind_visibility_from(params, "optim_time_shift", lambda v: not v)
                with manual:
                    num = ui.number(label=r"Manuální nastavení t_0", min=0.01, max=20.0, precision=2).bind_value(params, "t_shift").classes("w-80")
                    num.on('update:model-value', wrap_ui_handler('processing.t_shift.change', lambda e: None, level=10, data=lambda e: {'value': getattr(e, 'value', None)}))

                # t_max (persisted): clamp range from 1..max time in cached tables
                def _max_time_from_cache() -> float:
                    try:
                        from io import StringIO
                        tables = tables_cache_repo.load_latest_tables(experiment_id)
                        # Prefer transposed tables with explicit time column.
                        for key in ('IS_korig_t', 'C18_1', 'C18_2', 'C20_1', 'souhrn'):
                            if key in tables:
                                df_json, _ = tables[key]
                                df = pd.read_json(StringIO(df_json), orient='split')
                                if 'time' in df.columns:
                                    mx = float(pd.to_numeric(df['time'], errors='coerce').max())
                                    if mx > 0:
                                        return mx
                    except Exception:
                        pass

                    # Fallback: infer max time directly from Excel column headers (time points).
                    try:
                        excel_bytes, sheet = load_excel_bytes_and_sheet()
                        if excel_bytes is None or sheet is None:
                            return 400.0
                        xls = pd.ExcelFile(BytesIO(excel_bytes))
                        df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
                        fame_pick = _load_pick(pick_repo, experiment_id, 'fame')
                        epo_pick = _load_pick(pick_repo, experiment_id, 'epo')
                        r1, r2, c1, c2 = fame_pick
                        fame_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), 'FAME')
                        r1, r2, c1, c2 = epo_pick
                        epo_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), 'EPOXIDES')
                        times = []
                        for cols in (list(fame_df.columns[1:]), list(epo_df.columns[1:])):
                            for c in cols:
                                try:
                                    times.append(float(str(c).replace(',', '.')))
                                except Exception:
                                    continue
                        if times:
                            mx = max(times)
                            if mx > 0:
                                return float(mx)
                    except Exception:
                        pass
                    return 400.0

                tmax_max = _max_time_from_cache()
                tmax = ui.number(
                    label=r"t_{max} (čas pro simulaci)",
                    min=1.0,
                    max=max(1.0, tmax_max),
                    precision=1,
                ).bind_value(params, 't_max').classes('w-80')

                tmax_plot = ui.number(
                    label=r"t_{max,plot} (max xlim grafu)",
                    min=1.0,
                    max=max(1.0, tmax_max),
                    precision=1,
                ).bind_value(params, 't_max_plot').classes('w-80')
                tmax_plot.on('update:model-value', wrap_ui_handler('processing.t_max_plot.change', lambda e: None, level=10, data=lambda e: {'value': getattr(e, 'value', None)}))
                tmax.on('update:model-value', wrap_ui_handler('processing.t_max.change', lambda e: None, level=10, data=lambda e: {'value': getattr(e, 'value', None)}))

    from experimental_web.ui.experiment.computations_ui import computations_block
    computations_block(experiment_id, params=params)

    with ui.row().classes('items-center gap-2 q-mt-md'):
        ui.button(
            "Spočítat / přepočítat",
            icon="🔁",
            on_click=wrap_ui_handler(
                'processing.compute.click',
                on_compute,
                level=20,
                data=lambda: {
                    'experiment_id': experiment_id,
                    'models_to_compute': list(params.models_to_compute),
                    'initialization': params.initialization.name,
                    'optim_time_shift': bool(params.optim_time_shift),
                    't_shift': float(params.t_shift),
                    't_max': float(getattr(params, 't_max', 400.0)),
                    't_max_plot': float(getattr(params, 't_max_plot', 400.0)),
                },
            ),
        ).props("unelevated")

        # Visible when settings/computation graph changed since last compute.
        dirty_badge = ui.badge('Změněno – přepočítejte', color='orange').props('outline')
        dirty_badge.visible = False
    ui.separator()
    results_container = ui.column().classes("w-full gap-4")
    results_container

    # Per-model "changed" badges (results are stale compared to the latest settings/definitions)
    stale_badges: dict[str, Any] = {}
    last_seen_run_created_at: str | None = None

    # Initial render: load last stored results from DB (if any).
    def _render_saved_results_from_db() -> None:
        nonlocal auto_t_shift_value, last_seen_run_created_at
        try:
            saved_run = results_repo.get_latest_run(experiment_id)
        except Exception:
            saved_run = None
        if not saved_run:
            return

        try:
            _run, stale = compute_staleness(experiment_id)
        except Exception:
            _run, stale = (saved_run, None)

        last_seen_run_created_at = str(saved_run.get('created_at') or '')

        # If we have an auto-t_shift from last run, show it (when auto mode is enabled).
        try:
            if saved_run.get('auto_t_shift') is not None and auto_t_shift_value is None:
                auto_t_shift_value = float(saved_run.get('auto_t_shift'))
                if bool(params.optim_time_shift):
                    auto_t_shift_label.set_text(f"{auto_t_shift_value:.6g}")
        except Exception:
            pass

        # Build expected model order based on current UI selection + saved computations.
        custom_names: list[str] = []
        try:
            from experimental_web.data.repositories import ExperimentComputationRepository
            items = ExperimentComputationRepository(DB_PATH).list_for_experiment(experiment_id)
            if items:
                seen: dict[str, int] = {}
                for it in items:
                    base = str(it.name or '(bez názvu)')
                    if base not in seen:
                        seen[base] = 1
                        disp = base
                    else:
                        seen[base] += 1
                        disp = f"{base} ({seen[base]})"
                    custom_names.append(disp)
        except Exception:
            custom_names = []

        expected = _expected_model_order(params.models_to_compute, custom_names)
        models_db: dict[str, dict] = dict(saved_run.get('models') or {})

        results_container.clear()
        stale_badges.clear()
        with results_container:
            ui.label(f"Poslední výsledky: {saved_run.get('created_at')}").classes('text-caption text-grey-7')
            if saved_run.get('used_t_shift') is not None:
                ui.label(f"Použitý t_shift: {float(saved_run.get('used_t_shift')):.6g}").classes('text-caption text-grey-7')
            try:
                if stale is not None and stale.global_changed:
                    ui.badge('Změněno od posledního výpočtu', color='orange').props('outline')
            except Exception:
                pass
            ui.separator()

            rendered = set()
            for name in expected:
                with ui.card().classes('w-full'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.markdown(f"#### Výsledky pro {name}")
                        b = ui.badge('Změněno', color='orange').props('outline')
                        try:
                            b.visible = bool(stale is not None and is_model_stale(name, stale))
                        except Exception:
                            b.visible = False
                        stale_badges[name] = b
                    body = ui.column().classes('w-full')
                    model_bodies[name] = body
                    entry = models_db.get(name)
                    if entry:
                        rendered.add(name)
                        _render_model_body(body, {
                            'constants': entry.get('constants') or [],
                            'plot_error': entry.get('plot_error'),
                            'plot_png_bytes': entry.get('plot_png'),
                        })
                    else:
                        with body:
                            StyledLabel('Zatím nespočítáno.', 'info')

            # Any extra models from DB that are not expected
            extras = [k for k in models_db.keys() if k not in rendered and k not in set(expected)]
            for name in extras:
                entry = models_db.get(name) or {}
                with ui.card().classes('w-full'):
                    with ui.row().classes('w-full items-center justify-between'):
                        ui.markdown(f"#### Výsledky pro {name}")
                        b = ui.badge('Změněno', color='orange').props('outline')
                        try:
                            b.visible = bool(stale is not None and is_model_stale(name, stale))
                        except Exception:
                            b.visible = False
                        stale_badges[name] = b
                    body = ui.column().classes('w-full')
                    model_bodies[name] = body
                    _render_model_body(body, {
                        'constants': entry.get('constants') or [],
                        'plot_error': entry.get('plot_error'),
                        'plot_png_bytes': entry.get('plot_png'),
                    })

        status.set('Načteny poslední výsledky z databáze.', 'info')

        # Update top badge
        try:
            if stale is not None:
                dirty_badge.visible = bool(stale.global_changed or any(stale.custom_changed.values()))
        except Exception:
            pass


    def _refresh_staleness_indicators() -> None:
        """Refresh "changed" badges when user edits settings/computations."""
        nonlocal last_seen_run_created_at
        try:
            run, stale = compute_staleness(experiment_id)
        except Exception:
            return

        # If a new run appeared, rebuild to show latest timestamp and models.
        created_at = str((run or {}).get('created_at') or '') if run else ''
        if created_at and created_at != (last_seen_run_created_at or ''):
            try:
                _render_saved_results_from_db()
            except Exception:
                return

        # Update badge visibility
        try:
            dirty_badge.visible = bool(stale.global_changed or any(stale.custom_changed.values()))
        except Exception:
            pass

        for model_name, badge in list(stale_badges.items()):
            try:
                badge.visible = bool(is_model_stale(model_name, stale))
            except Exception:
                pass

    _render_saved_results_from_db()

    # Poll streamed results/progress from the subprocess.
    ui.timer(0.1, callback=_poll_progress_queue)

    # Persist settings periodically (debounced by snapshot comparison).
    ui.timer(0.5, callback=_persist_settings_if_changed)

    # Also poll for "stale" indicators so the user immediately sees changes.
    ui.timer(1.0, callback=_refresh_staleness_indicators)
