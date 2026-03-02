from __future__ import annotations

import asyncio

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Optional, Callable
import re

import pandas as pd
from nicegui import events, ui

from experimental_web.core.paths import DB_PATH
from experimental_web.core.state import get_state
from experimental_web.data.repositories import (
    ExperimentComputationRepository,
    ExperimentFileRepository,
    ExperimentGraphSettingsRepository,
    ExperimentProcessingResultsRepository,
    TablePickRepository,
)
from experimental_web.domain.advanced_plotting import AdvancedPlotter
from experimental_web.domain.fame_epo import DEFAULT_EPO, DEFAULT_FAME, extract_df
from experimental_web.domain.kinetic_model import InitConditions, KineticModel
from experimental_web.domain.ode_generation import generate_ode_model
from experimental_web.domain.ode_compiler import compile_ode_equations
from experimental_web.domain.processing import TableProcessor
from experimental_web.logging_setup import get_logger
from experimental_web.ui.utils.plots import export_all_figures_as_zip, update_plot_image
from experimental_web.ui.styled_elements.custom_color_picker import ColorPickerButton
from experimental_web.ui.widgets.styled_label import StyledLabel
from experimental_web.ui.utils.staleness import compute_staleness, is_model_stale
from experimental_web.ui.utils.tooltips import attach_tooltip


log = get_logger(__name__)

def _ensure_graphs_css() -> None:
    """Install small CSS helpers used by the graphs tab.

    - make the per-curve tab show a colored icon (without coloring the label)
    - allow many tabs without causing the whole page to scroll horizontally
    """
    # NOTE:
    # NiceGUI routes can trigger a full page reload in the browser while the
    # server process keeps this module cached. A module-level "already added"
    # flag would then incorrectly skip CSS injection after reopening an
    # experiment, causing the colored markers to disappear.
    #
    # Therefore we inject idempotently per *browser document* via JS.

    css = r"""
        /* Colored marker for curve tabs (without coloring the label text) */
        /* Quasar sometimes wraps the label; target both label and content for robustness */
        .curve-tab .q-tab__content,
        .curve-tab .q-tab__label {
            display: inline-flex;
            align-items: center;
        }
        .curve-tab .q-tab__content::before,
        .curve-tab .q-tab__label::before {
            content: '';
            width: 12px;
            height: 12px;
            background: var(--curve-color, #000);
            border: 1px solid rgba(0,0,0,0.25);
            border-radius: 2px;
            display: inline-block;
            margin-right: 6px;
            box-sizing: border-box;
            flex: 0 0 auto;
        }

        /* Keep the curve-tabs row inside its container.
           Allow horizontal scroll ONLY inside this wrapper (not the whole page). */
        .graphs-controls-tabs {
            width: 100%;
            max-width: 100%;
            min-width: 0;
            box-sizing: border-box;
            overflow-x: auto;
            overflow-y: hidden;
        }
        .graphs-controls-tabs .q-tabs {
            display: inline-flex;
            width: max-content !important;
            min-width: 100%;
        }
        .graphs-controls-tabs .q-tabs__content { flex-wrap: nowrap; }
        .graphs-controls-tabs::-webkit-scrollbar { height: 8px; }

        /* Make long labels compact (ellipsis) so each tab doesn't become extremely wide. */
        .graphs-controls-tabs .q-tab { max-width: 360px; }
        .graphs-controls-tabs .q-tab__label {
            max-width: 280px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        """

    ui.run_javascript(
        """
(() => {
  const id = 'graphs-tab-style';
  if (document.getElementById(id)) return;
  const style = document.createElement('style');
  style.id = id;
  style.textContent = %s;
  document.head.appendChild(style);
})();
""" % (repr(css),)
    )


def _short(v: Any, max_len: int = 200) -> str:
    """Safe short repr for debug logs."""
    try:
        s = repr(v)
    except Exception:
        try:
            s = str(v)
        except Exception:
            s = '<unrepr>'
    if len(s) > max_len:
        return s[: max_len - 3] + '...'
    return s


def _list_short(items: list[Any], max_items: int = 30) -> str:
    """Short list rendering for debug logs."""
    try:
        if len(items) <= max_items:
            return '[' + ', '.join(_short(x, 60) for x in items) + ']'
        head = items[: max_items]
        return '[' + ', '.join(_short(x, 60) for x in head) + f', ... (+{len(items) - max_items})]'
    except Exception:
        return _short(items)


_PREBUILT_ORDER = [
    'C18:1_simplified',
    'C18:1',
    'C18:2_simplified',
    'C18:2_eps_and_others',
    'C18:2',
    'C18:2_separated',
    'C18:2_with_k_uh',
    'C20:1_simplified',
]


_TAB10 = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


@dataclass
class CurveStyle:
    color: str = '#1f77b4'
    label: str = ''
    linestyle: str = 'solid'
    linewidth: float = 1.5
    marker: str = 'o'
    markersize: float = 6.0

    # Optional per-curve annotation (placed in data coordinates).
    additional_text_enabled: bool = False
    additional_text_text: str = ''
    additional_text_size: float = 14.0
    additional_text_x: float = 0.5
    additional_text_y: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return {
            'color': self.color,
            'label': self.label,
            'linestyle': self.linestyle,
            'linewidth': float(self.linewidth),
            'marker': self.marker,
            'markersize': float(self.markersize),
            'additional_text_enabled': bool(self.additional_text_enabled),
            'additional_text_text': str(self.additional_text_text or ''),
            'additional_text_size': float(self.additional_text_size),
            'additional_text_x': float(self.additional_text_x),
            'additional_text_y': float(self.additional_text_y),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> 'CurveStyle':
        cs = CurveStyle()
        cs.color = str(d.get('color', cs.color))
        cs.label = str(d.get('label', cs.label))
        cs.linestyle = str(d.get('linestyle', cs.linestyle))
        try:
            cs.linewidth = float(d.get('linewidth', cs.linewidth))
        except Exception:
            pass
        cs.marker = str(d.get('marker', cs.marker))
        try:
            cs.markersize = float(d.get('markersize', cs.markersize))
        except Exception:
            pass
        cs.additional_text_enabled = bool(d.get('additional_text_enabled', cs.additional_text_enabled))
        cs.additional_text_text = str(d.get('additional_text_text', cs.additional_text_text) or '')
        try:
            cs.additional_text_size = float(d.get('additional_text_size', cs.additional_text_size))
        except Exception:
            pass
        try:
            cs.additional_text_x = float(d.get('additional_text_x', cs.additional_text_x))
            cs.additional_text_y = float(d.get('additional_text_y', cs.additional_text_y))
        except Exception:
            pass
        return cs


@dataclass
class GridConfig:
    """Config for Matplotlib grid (major/minor).

    Stored in DB as plain dicts so the schema is forward/backward compatible.
    """

    # Show/hide this grid.
    visible: bool = True

    # Which axis to draw grid on: 'both' | 'x' | 'y'
    axis: str = 'both'

    # Quasar q-color uses the selected v-model format; we store it as a string (usually HEXA).
    # Default tries to match Matplotlib's default grid color.
    color: str = '#B0B0B0FF'

    # Line appearance.
    linestyle: str = '-'
    linewidth: float = 0.8

    # Major/minor (not editable in UI; controlled by which config this is).
    which: str = 'major'

    def to_dict(self) -> dict[str, Any]:
        return {
            'visible': bool(self.visible),
            'axis': str(self.axis or 'both'),
            'color': str(self.color or ''),
            'linestyle': str(self.linestyle or '-'),
            'linewidth': float(self.linewidth) if self.linewidth is not None else None,
            'which': str(self.which or 'major'),
        }

    @staticmethod
    def from_dict(d: Any, *, default: Optional['GridConfig'] = None) -> 'GridConfig':
        cfg = default if default is not None else GridConfig()
        if not isinstance(d, dict):
            return cfg
        try:
            cfg.visible = bool(d.get('visible', cfg.visible))
        except Exception:
            pass
        cfg.axis = str(d.get('axis', cfg.axis) or cfg.axis)
        cfg.color = str(d.get('color', cfg.color) or cfg.color)
        cfg.linestyle = str(d.get('linestyle', cfg.linestyle) or cfg.linestyle)
        try:
            lw = d.get('linewidth', cfg.linewidth)
            cfg.linewidth = float(lw) if lw is not None else cfg.linewidth
        except Exception:
            pass
        cfg.which = str(d.get('which', cfg.which) or cfg.which)
        return cfg

    def to_matplotlib_kwargs(self) -> dict[str, Any]:
        """Return kwargs for `Axes.grid(...)`.

        We keep keys compatible with the legacy project:
        - `visible` controls on/off
        - `axis` controls which axis
        - `which` is 'major'/'minor'
        """
        out: dict[str, Any] = {
            'visible': bool(self.visible),
            'axis': str(self.axis or 'both'),
            'which': str(self.which or 'major'),
        }
        if self.color:
            out['color'] = str(self.color)
        if self.linestyle:
            out['linestyle'] = str(self.linestyle)
        if self.linewidth is not None:
            try:
                out['linewidth'] = float(self.linewidth)
            except Exception:
                pass
        return out


def _default_grid_major() -> GridConfig:
    # Match Matplotlib defaults as closely as possible.
    return GridConfig(visible=True, axis='both', color='#B0B0B0FF', linestyle='-', linewidth=0.8, which='major')


def _default_grid_minor() -> GridConfig:
    # Minor grid is off by default (same as legacy behavior).
    return GridConfig(visible=False, axis='both', color='#B0B0B0FF', linestyle=':', linewidth=0.6, which='minor')

@dataclass
class GraphConfig:
    title: str = ''
    show_title: bool = True
    legend_mode: str = 'components_only'
    fig_width: float = 8.0
    fig_height: float = 6.0
    # Default axis labels (legacy project defaults)
    xlabel: str = 'čas [s]'
    ylabel: str = 'koncentrace [-]'

    # Matplotlib artist clipping.
    # True  -> current default behavior (artists are clipped to axes).
    # False -> allow drawing outside the axes (over spines/labels), useful for markers/annotations.
    clip_on: bool = True
    xlim_mode: str = 'manual'  # default/manual/auto/all_data/None
    xlim_min: Optional[float] = 0.0
    xlim_max: Optional[float] = 400.0
    ylim_mode: str = 'default'  # default/manual/None
    ylim_min: Optional[float] = 0.0
    ylim_max: Optional[float] = 1.0
    curve_styles: list[CurveStyle] = field(default_factory=list)

    grid_config: GridConfig = field(default_factory=_default_grid_major)
    grid_config_minor: GridConfig = field(default_factory=_default_grid_minor)

    def to_dict(self) -> dict[str, Any]:
        return {
            'title': self.title,
            'show_title': bool(self.show_title),
            'legend_mode': self.legend_mode,
            'fig_width': float(self.fig_width),
            'fig_height': float(self.fig_height),
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'clip_on': bool(self.clip_on),
            'xlim_mode': self.xlim_mode,
            'xlim_min': self.xlim_min,
            'xlim_max': self.xlim_max,
            'ylim_mode': self.ylim_mode,
            'ylim_min': self.ylim_min,
            'ylim_max': self.ylim_max,
            'curve_styles': [c.to_dict() for c in (self.curve_styles or [])],
            'grid_config': (self.grid_config.to_dict() if self.grid_config else _default_grid_major().to_dict()),
            'grid_config_minor': (self.grid_config_minor.to_dict() if self.grid_config_minor else _default_grid_minor().to_dict()),
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> 'GraphConfig':
        cfg = GraphConfig()
        cfg.title = str(d.get('title', cfg.title) or '')
        cfg.show_title = bool(d.get('show_title', cfg.show_title))
        cfg.legend_mode = str(d.get('legend_mode', cfg.legend_mode) or cfg.legend_mode)
        try:
            cfg.fig_width = float(d.get('fig_width', cfg.fig_width))
            cfg.fig_height = float(d.get('fig_height', cfg.fig_height))
        except Exception:
            pass
        cfg.xlabel = str(d.get('xlabel', cfg.xlabel) or cfg.xlabel)
        cfg.ylabel = str(d.get('ylabel', cfg.ylabel) or cfg.ylabel)
        cfg.clip_on = bool(d.get('clip_on', cfg.clip_on))
        cfg.xlim_mode = str(d.get('xlim_mode', cfg.xlim_mode) or cfg.xlim_mode)
        cfg.ylim_mode = str(d.get('ylim_mode', cfg.ylim_mode) or cfg.ylim_mode)
        cfg.xlim_min = d.get('xlim_min', cfg.xlim_min)
        cfg.xlim_max = d.get('xlim_max', cfg.xlim_max)
        cfg.ylim_min = d.get('ylim_min', cfg.ylim_min)
        cfg.ylim_max = d.get('ylim_max', cfg.ylim_max)
        styles_raw = d.get('curve_styles') or []
        out_styles: list[CurveStyle] = []
        if isinstance(styles_raw, list):
            for item in styles_raw:
                if isinstance(item, dict):
                    out_styles.append(CurveStyle.from_dict(item))
        cfg.curve_styles = out_styles
        cfg.grid_config = GridConfig.from_dict(d.get('grid_config') or {}, default=_default_grid_major())
        cfg.grid_config_minor = GridConfig.from_dict(d.get('grid_config_minor') or {}, default=_default_grid_minor())
        # Ensure correct `which` values even if an older config stored wrong strings.
        cfg.grid_config.which = 'major'
        cfg.grid_config_minor.which = 'minor'
        return cfg

    def to_kwargs(self) -> dict[str, Any]:
        return {
            'title': self.title,
            'show_title': self.show_title,
            'legend_mode': self.legend_mode,
            'fig_width': self.fig_width,
            'fig_height': self.fig_height,
            'xlabel': self.xlabel,
            'ylabel': self.ylabel,
            'clip_on': bool(self.clip_on),
            'xlim_mode': self.xlim_mode,
            'xlim_min': self.xlim_min,
            'xlim_max': self.xlim_max,
            'ylim_mode': self.ylim_mode,
            'ylim_min': self.ylim_min,
            'ylim_max': self.ylim_max,
            # AdvancedPlotter expects dicts
            'curve_styles': [
                {
                    **c.to_dict(),
                }
                for c in (self.curve_styles or [])
            ],
            'grid_config': (self.grid_config.to_matplotlib_kwargs() if self.grid_config else _default_grid_major().to_matplotlib_kwargs()),
            'grid_config_minor': (self.grid_config_minor.to_matplotlib_kwargs() if self.grid_config_minor else _default_grid_minor().to_matplotlib_kwargs()),
        }


def _load_pick(pick_repo: TablePickRepository, experiment_id: int, kind: str):
    saved = pick_repo.get(experiment_id, kind)
    if saved:
        return saved['row_start'], saved['row_end'], saved['col_start'], saved['col_end']
    c = DEFAULT_FAME if kind == 'fame' else DEFAULT_EPO
    return c.row_start, c.row_end, c.col_start, c.col_end


def _normalize_first_col(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df2 = df.copy()
    if df2.columns.size > 0:
        df2 = df2.rename(columns={df2.columns[0]: name})
    return df2


def _detect_time_column(columns: list[Any]) -> Optional[Any]:
    lowered = [str(c).lower() for c in columns]
    for key in ('čas', 'cas', 'time', 't'):
        for col, low in zip(columns, lowered):
            if key in low:
                return col
    return None


def _values_from_constants(constants: list[dict]) -> list[float]:
    """Extract ordered k-values from the persisted constants payload."""
    if not constants:
        return []
    try:
        # Prefer explicit idx ordering.
        pairs = []
        for c in constants:
            idx = int(c.get('idx') or 0)
            val = float(c.get('value'))
            pairs.append((idx, val))
        pairs.sort(key=lambda x: x[0])
        return [v for _, v in pairs]
    except Exception:
        out: list[float] = []
        for c in constants:
            try:
                out.append(float(c.get('value')))
            except Exception:
                pass
        return out


def _latex_for_graph_param(p: str) -> str:
    """Mirror KineticModel.get_constants_with_names latex mapping for graph-defined params.

    We intentionally duplicate the logic here so we can map persisted constants
    (stored as latex strings) back onto param_names_override when recreating
    models for the Graphs tab.
    """
    if not p:
        return r"k"
    if p.startswith("k_"):
        rest = p[2:]
        # k_um  -> U -> M
        if len(rest) == 2 and rest.isalpha():
            a, b = rest[0].upper(), rest[1].upper()
            return r"k_{\mathrm{" + a + r"}\rightarrow \mathrm{" + b + r"}}"
        parts = rest.split("_")
        if len(parts) == 2 and all(parts):
            a = parts[0].upper().replace("_", r"\_")
            b = parts[1].upper().replace("_", r"\_")
            return r"k_{\mathrm{" + a + r"}\rightarrow \mathrm{" + b + r"}}"
        return r"k_{\mathrm{" + rest.replace("_", r"\_") + r"}}"
    return r"k_{\mathrm{" + p.replace("_", r"\_") + r"}}"


def _build_latex_value_map(constants: list[dict]) -> dict[str, float]:
    """Build a latex->value map from persisted constants payload."""
    out: dict[str, float] = {}
    for c in list(constants or []):
        try:
            latex = str(c.get('latex') or '')
            if not latex:
                continue
            out[latex] = float(c.get('value'))
        except Exception:
            continue
    return out


def _apply_constants_to_model_for_plot(model: 'KineticModel', constants: list[dict]) -> None:
    """Apply persisted constants to a recreated model in a safe way.

    For graph-defined models we map by param_names_override -> latex.
    For legacy/prebuilt models we fall back to idx ordering.
    """
    if model is None:
        return
    try:
        pnames = list(getattr(model, 'param_names_override', None) or [])
    except Exception:
        pnames = []
    if pnames:
        m = _build_latex_value_map(constants)
        vals: list[float] = []
        for p in pnames:
            latex = _latex_for_graph_param(str(p))
            if latex in m:
                vals.append(float(m[latex]))
            else:
                vals.append(0.01)
        try:
            import numpy as np
            model.k_fit = np.array(vals, dtype=float)
        except Exception:
            model.k_fit = vals
        return

    vals = _values_from_constants(constants)
    if not vals:
        return
    try:
        n_expected = 0
        try:
            consts = model.get_constants_with_names() or []
            n_expected = len(list(consts))
        except Exception:
            n_expected = 0
        if n_expected > 0:
            vals = vals[:n_expected]
    except Exception:
        pass
    try:
        import numpy as np
        model.k_fit = np.array(vals, dtype=float)
    except Exception:
        model.k_fit = vals


class _MergedSimulationModel:
    """Proxy which merges simulated curves from main+control models for AdvancedPlotter."""

    def __init__(
        self,
        main: 'KineticModel',
        control: 'KineticModel',
        *,
        main_active_columns: set[str] | None,
        control_active_columns: set[str] | None,
    ) -> None:
        self._main = main
        self._control = control
        self._main_active = set(main_active_columns or [])
        self._ctrl_active = set(control_active_columns or [])

        # Attributes consumed by AdvancedPlotter.
        self.column_names = getattr(main, 'column_names', [])
        self.original_data = getattr(main, 'original_data', None)
        self.time_exp = getattr(main, 'time_exp', None)
        self.y_exp = getattr(main, 'y_exp', None)

    def get_constants_with_names(self):
        try:
            return self._main.get_constants_with_names()
        except Exception:
            return None

    def simulate(self, *, t_max=2000, time_points=2000):
        sim_t, sim_y = self._main.simulate(t_max=t_max, time_points=time_points)
        if self._control is None:
            return sim_t, sim_y
        try:
            sim_t2, sim_y2 = self._control.simulate(t_max=t_max, time_points=time_points)
        except Exception:
            return sim_t, sim_y

        try:
            import numpy as np
            sim_t_arr = np.asarray(sim_t, dtype=float)
            sim_y_arr = np.asarray(sim_y, dtype=float)
            sim_t2_arr = np.asarray(sim_t2, dtype=float)
            sim_y2_arr = np.asarray(sim_y2, dtype=float)

            if sim_t_arr.shape != sim_t2_arr.shape or np.max(np.abs(sim_t_arr - sim_t2_arr)) > 1e-9:
                y2i = []
                for i in range(sim_y2_arr.shape[0]):
                    y2i.append(np.interp(sim_t_arr, sim_t2_arr, sim_y2_arr[i]))
                sim_y2_arr = np.asarray(y2i, dtype=float)

            out = sim_y_arr.copy()
            main_cols = [str(c) for c in (getattr(self._main, 'column_names', []) or [])]
            ctrl_cols = [str(c) for c in (getattr(self._control, 'column_names', []) or [])]
            ctrl_idx = {c: i for i, c in enumerate(ctrl_cols)}

            for j, col in enumerate(main_cols):
                if self._main_active and col in self._main_active:
                    continue
                if self._ctrl_active and col in self._ctrl_active and col in ctrl_idx:
                    out[j, :] = sim_y2_arr[ctrl_idx[col], :]
            return sim_t_arr, out
        except Exception:
            return sim_t, sim_y


def _build_table_processor(
    *,
    excel_bytes: bytes,
    sheet: str,
    fame_pick: tuple[int, int, int, int],
    epo_pick: tuple[int, int, int, int],
) -> TableProcessor:
    log.debug(
        'graphs_tab: build_table_processor: sheet=%s fame_pick=%s epo_pick=%s excel_bytes=%d',
        sheet,
        fame_pick,
        epo_pick,
        len(excel_bytes or b''),
    )
    xls = pd.ExcelFile(BytesIO(excel_bytes))
    df_raw = pd.read_excel(xls, sheet_name=sheet, header=None)
    try:
        log.trace('graphs_tab: raw sheet %s shape=%s', sheet, getattr(df_raw, 'shape', None))
    except Exception:
        pass

    r1, r2, c1, c2 = fame_pick
    fame_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), 'FAME')
    r1, r2, c1, c2 = epo_pick
    epo_df = _normalize_first_col(extract_df(df_raw, r1, r2, c1, c2), 'EPOXIDES')

    for df in (fame_df, epo_df):
        for col in df.columns[1:]:
            # Pandas supports only errors={'raise','coerce'} for to_numeric.
            # We want a best-effort conversion for plotting/analysis,
            # therefore we coerce invalid values to NaN.
            df[col] = pd.to_numeric(df[col], errors='coerce')

    processor = TableProcessor(fame_df=fame_df, epo_df=epo_df)
    processor.process()
    try:
        log.debug('graphs_tab: processor.tables=%s', _list_short(list(processor.tables.keys()), 50))
    except Exception:
        pass
    return processor


def _create_prebuilt_models(
    *,
    processor: TableProcessor,
    settings: dict,
    used_t_shift: float,
    wanted_names: set[str],
) -> dict[str, KineticModel]:
    # In graphs tab we recreate only the models present in the persisted run.
    # (User might have changed selection since the run was computed.)
    models_to_compute = list(wanted_names)
    init_name = str(settings.get('initialization') or 'TIME_SHIFT')
    try:
        initialization = InitConditions[init_name]
    except Exception:
        initialization = InitConditions.TIME_SHIFT

    try:
        t_max = float(settings.get('t_max') or 400.0)
    except Exception:
        t_max = 400.0

    out: dict[str, KineticModel] = {}

    def add(name: str, table_name: str, columns: list[str], **kwargs: Any) -> None:
        if name not in models_to_compute:
            return
        try:
            if table_name not in processor.tables:
                log.debug(
                    'graphs_tab: prebuilt %s skipped (missing table %s). Available=%s',
                    name,
                    table_name,
                    _list_short(list(processor.tables.keys()), 30),
                )
                return
            missing_cols = [c for c in columns if c not in set(processor.tables[table_name].columns)]
            if missing_cols:
                log.debug(
                    'graphs_tab: prebuilt %s warning: missing columns=%s in table=%s (table columns=%s)',
                    name,
                    _list_short(missing_cols, 50),
                    table_name,
                    _list_short(list(processor.tables[table_name].columns), 40),
                )
            out[name] = processor.create_model(
                processor.tables[table_name],
                columns=columns,
                init_method=initialization,
                t_shift=float(used_t_shift),
                t_max=float(t_max),
                **kwargs,
            )
            log.trace('graphs_tab: prebuilt model created: %s (table=%s cols=%s)', name, table_name, _list_short(columns, 30))
        except Exception as e:
            log.debug('graphs_tab: failed to create prebuilt model %s: %s', name, e, exc_info=True)

    add(
        'C18:1_simplified',
        'C18_1',
        ['zastoupení C18:1', 'zastoupení C18:1 EPO + hydroxyly'],
    )
    add(
        'C18:1',
        'C18_1',
        ['zastoupení C18:1', 'zastoupení C18:1 EPO', 'zastoupení hydroxyly'],
        k_uh=False,
    )
    add(
        'C18:2_simplified',
        'C18_2',
        ['zastoupení C18:2', 'zastoupení Σ C18:2 vše EPO', 'zastoupení hydroxyly'],
    )
    add(
        'C18:2_eps_and_others',
        'C18_2',
        ['zastoupení C18:2', 'zastoupení Σ C18:2 vše EPO', 'zastoupení hydroxyly', 'zastoupení Σ C18:2 EPO + hydroxyly'],
        special_model='C18:2_eps_others',
    )
    add(
        'C18:2',
        'C18_2',
        ['zastoupení C18:2', 'zastoupení Σ C18:2 1-EPO', 'zastoupení C18:2 2-EPO', 'zastoupení hydroxyly'],
    )
    add(
        'C18:2_separated',
        'C18_2',
        ['zastoupení C18:2', 'zastoupení Σ C18:2 1-EPO', 'zastoupení C18:2 2-EPO', 'zastoupení hydroxyly', 'zastoupení Σ C18:2 EPO + hydroxyly'],
        special_model='C18:2_separated',
    )
    add(
        'C18:2_with_k_uh',
        'C18_2',
        ['zastoupení C18:2', 'zastoupení Σ C18:2 1-EPO', 'zastoupení C18:2 2-EPO', 'zastoupení hydroxyly'],
        k_uh=True,
    )
    add(
        'C20:1_simplified',
        'C20_1',
        ['zastoupení C20:1', 'zastoupení C20:1 EPO + hydroxyly'],
    )

    return out


def _create_custom_models(
    *,
    experiment_id: int,
    processor: TableProcessor,
    settings: dict,
    used_t_shift: float,
    wanted_names: set[str],
) -> dict[str, KineticModel]:
    """Recreate configurable graph-defined models from DB definitions."""
    init_name = str(settings.get('initialization') or 'TIME_SHIFT')
    try:
        initialization = InitConditions[init_name]
    except Exception:
        initialization = InitConditions.TIME_SHIFT

    try:
        t_max = float(settings.get('t_max') or 400.0)
    except Exception:
        t_max = 400.0

    out: dict[str, KineticModel] = {}
    repo = ExperimentComputationRepository(DB_PATH)
    items = repo.list_for_experiment(experiment_id)
    if not items:
        log.debug('graphs_tab: no custom computations in DB for experiment_id=%s', experiment_id)
        return out

    log.debug('graphs_tab: loaded %d custom computations for experiment_id=%s', len(items), experiment_id)

    # IMPORTANT: We intentionally do NOT rely on exact name matching.
    # The persisted run may contain slightly different display strings
    # (user renamed, added punctuation, duplicate suffixes, etc.). We therefore
    # create models for *all* computations and the caller maps them to run names.
    seen: dict[str, int] = {}
    for it in items:
        try:
            log.trace(
                'graphs_tab: computation id=%s name=%s table_name=%s used_heads=%s state_names=%s param_names=%s ode_len=%s',
                getattr(it, 'id', None),
                _short(getattr(it, 'name', None), 80),
                _short(getattr(it, 'table_name', None), 80),
                len(getattr(it, 'used_heads', None) or []),
                len(getattr(it, 'state_names', None) or []),
                len(getattr(it, 'param_names', None) or []),
                len(str(getattr(it, 'ode_text', '') or '')),
            )
        except Exception:
            pass

        base = str(it.name or '(bez názvu)')
        if base not in seen:
            seen[base] = 1
            disp = base
        else:
            seen[base] += 1
            disp = f"{base} ({seen[base]})"

        table = processor.tables.get(str(it.table_name or ''))
        if table is None:
            log.debug(
                'graphs_tab: computation %s skipped: table %s not found. Available=%s',
                disp,
                str(it.table_name or ''),
                _list_short(list(processor.tables.keys()), 40),
            )
            continue

        cols = list(it.used_heads or [])
        try:
            nodes = list(getattr(it.graph_state, 'nodes', []) or [])
            if nodes:
                cols = [c for c in nodes if c in set(cols)]
        except Exception:
            pass
        if not cols:
            log.debug('graphs_tab: computation %s skipped: empty used_heads after filtering', disp)
            continue

        missing = [c for c in cols if c not in set(table.columns)]
        if missing:
            log.debug(
                'graphs_tab: computation %s: some used_heads not in table=%s missing=%s',
                disp,
                str(it.table_name or ''),
                _list_short(missing, 60),
            )

        time_col = _detect_time_column(list(table.columns))
        if time_col is None:
            log.debug(
                'graphs_tab: computation %s: time column not detected, using synthetic index time',
                disp,
            )
            t = pd.Series(range(len(table)), name='time', dtype=float)
            df_sel = pd.concat([t, table[cols]], axis=1)
        else:
            df_sel = table[[time_col] + cols].copy()
            df_sel = df_sel.rename(columns={time_col: 'time'})
        for c in cols:
            if c in df_sel.columns:
                df_sel[c] = pd.to_numeric(df_sel[c], errors='coerce')
        df_sel['time'] = pd.to_numeric(df_sel['time'], errors='coerce')
        df_sel = df_sel.fillna(0.0)
        conc_data = df_sel[['time'] + cols].astype(float).values

        try:
            tmin = float(conc_data[:, 0].min()) if conc_data.size else None
            tmax = float(conc_data[:, 0].max()) if conc_data.size else None
            log.trace('graphs_tab: computation %s: conc_data shape=%s time=[%s,%s] cols=%s', disp, conc_data.shape, tmin, tmax, _list_short(cols, 30))
        except Exception:
            pass


        # Clamp t_max to available time (precompute tmax_eff early so the control model can use it too).
        try:
            max_time = float(conc_data[:, 0].max())
            if not (max_time > 0):
                max_time = 0.0
        except Exception:
            max_time = 0.0
        if max_time > 0:
            tmax_eff = min(max(1.0, t_max), max_time)
        else:
            tmax_eff = max(1.0, t_max)

        # Recreate ODE from the graph state so the graphs tab reflects the
        # *current* graph definition (including control edges).
        try:
            edge_modes = dict(getattr(it.graph_state, 'edge_modes', {}) or {})
        except Exception:
            edge_modes = {}

        ode_main = generate_ode_model(cols, edge_modes, include_modes=(1,))
        pnames = list(ode_main.param_names or [])
        state_names = list(ode_main.state_names or [])
        ode_text = str(ode_main.ode_text or '')
        try:
            odes = compile_ode_equations(
                ode_text,
                state_names=state_names,
                param_names=pnames,
                order=["d" + s for s in state_names],
            )
        except Exception as e:
            # Do not kill the whole tab because a single computation is inconsistent.
            log.debug(
                'graphs_tab: computation %s failed to compile MAIN ODE: %s (state_names=%s param_names=%s). First lines=%s',
                disp,
                e,
                _list_short(state_names, 30),
                _list_short(pnames, 30),
                _short('\n'.join(ode_text.splitlines()[:6]), 250),
                exc_info=True,
            )
            continue

        # Build control induced subgraph (mode=2). We keep it attached to the
        # main model so the plotting layer can merge simulated curves.
        #
        # IMPORTANT: keep node/edge labels in their original types.
        # Pandas column headers coming from Excel can be non-strings (numbers,
        # datetimes, ...). If we stringify edge endpoints here, generate_ode_model
        # will see edges whose endpoints do *not* match the node list and will
        # silently produce an empty ODE (all dX = 0), which then looks like
        # "merging doesn't work" on the Graphs tab.
        endpoints: set[Any] = set()
        edge_ctrl: dict[tuple[Any, Any], int] = {}
        try:
            for (a, b), m in edge_modes.items():
                try:
                    mv = int(m)
                except Exception:
                    mv = 0
                if mv == 2:
                    endpoints.add(a)
                    endpoints.add(b)
                    edge_ctrl[(a, b)] = 2
        except Exception:
            endpoints = set()
            edge_ctrl = {}

        nodes_ctrl = [n for n in cols if n in endpoints]
        model_ctrl = None
        ctrl_active_cols: set[str] = set()
        if nodes_ctrl and edge_ctrl:
            cols_ctrl = [c for c in cols if c in set(nodes_ctrl)]
            ctrl_active_cols = set(str(c) for c in cols_ctrl)
            try:
                conc_ctrl = df_sel[['time'] + cols_ctrl].astype(float).values
            except Exception:
                conc_ctrl = None
            if conc_ctrl is not None and len(cols_ctrl) > 0:

                # Clamp t_max for control data as well.
                try:
                    max_time2 = float(conc_ctrl[:, 0].max())
                    if not (max_time2 > 0):
                        max_time2 = 0.0
                except Exception:
                    max_time2 = 0.0
                if max_time2 > 0:
                    tmax_ctrl = min(max(1.0, t_max), max_time2)
                else:
                    tmax_ctrl = max(1.0, t_max)

                try:
                    ode_ctrl = generate_ode_model(cols_ctrl, edge_ctrl, include_modes=(2,))
                    pnames_ctrl = list(ode_ctrl.param_names or [])
                    state_names_ctrl = list(ode_ctrl.state_names or [])
                    odes_ctrl = compile_ode_equations(
                        str(ode_ctrl.ode_text or ''),
                        state_names=state_names_ctrl,
                        param_names=pnames_ctrl,
                        order=["d" + s for s in state_names_ctrl],
                    )
                    model_ctrl = KineticModel(
                        concentration_data=conc_ctrl,
                        column_names=list(cols_ctrl),
                        init_method=initialization,
                        t_shift=float(used_t_shift),
                        t_max=float(tmax_ctrl),
                        custom_odes=odes_ctrl,
                        param_names_override=pnames_ctrl,
                        k_init=[0.01] * (len(pnames_ctrl) if len(pnames_ctrl) > 0 else 1),
                        verbose=False,
                    )
                except Exception as e:
                    model_ctrl = None
                    log.debug('graphs_tab: computation %s failed to compile CONTROL ODE: %s', disp, e, exc_info=True)

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
            main_active_cols = {c for c in main_active_cols if c in set(str(x) for x in cols)}
        except Exception:
            main_active_cols = set(str(x) for x in cols)


        model = KineticModel(
            concentration_data=conc_data,
            column_names=cols,
            init_method=initialization,
            t_shift=float(used_t_shift),
            t_max=float(tmax_eff),
            custom_odes=odes,
            param_names_override=pnames,
            k_init=[0.01] * max(1, len(pnames)),
            verbose=False,
        )

        # Attach optional control model + active-column metadata for merged plotting.
        try:
            setattr(model, '_control_model', model_ctrl)
            setattr(model, '_main_active_columns', set(main_active_cols or []))
            setattr(model, '_control_active_columns', set(ctrl_active_cols or []))
        except Exception:
            pass
        # Keep the same key format as in the processing pipeline.
        base_key = f"Konfigurovatelný: {disp}"
        key = base_key
        # Avoid collisions.
        suffix = 2
        while key in out:
            key = f"{base_key} ({suffix})"
            suffix += 1
        out[key] = model

        log.trace(
            'graphs_tab: custom model created: %s (table=%s cols=%s params=%d)',
            key,
            str(it.table_name or ''),
            _list_short(cols, 30),
            len(pnames),
        )

    return out


def _norm_title(s: str) -> str:
    """Normalize a model/computation name for fuzzy matching."""
    s = (s or '').strip()
    s = re.sub(r'^Konfigurovatelný\s*:\s*', '', s, flags=re.IGNORECASE)
    s = re.sub(r'\s+', ' ', s)
    s = s.strip().rstrip(':').strip()
    return s.lower()


def _strip_dup_suffix(s: str) -> str:
    """Remove a trailing ' (N)' suffix."""
    return re.sub(r'\s*\(\d+\)\s*$', '', (s or '').strip())


def _pick_best_custom_key_for_run(
    run_name: str,
    custom_models: dict[str, KineticModel],
    run_constants: list[dict],
) -> Optional[str]:
    """Find the best matching custom model key for a run model name.

    Match by normalized title; if ambiguous, prefer matching parameter count.
    """
    if not custom_models:
        return None

    target = _norm_title(run_name)
    if not target:
        return None

    candidates = [k for k in custom_models.keys() if _norm_title(k) == target]
    if not candidates:
        target2 = _norm_title(_strip_dup_suffix(run_name))
        candidates = [k for k in custom_models.keys() if _norm_title(_strip_dup_suffix(k)) == target2]
    if not candidates:
        candidates = [k for k in custom_models.keys() if target in _norm_title(k)]

    if not candidates:
        log.trace('graphs_tab: no custom candidates for run_name=%s (target=%s)', _short(run_name, 120), target)
        return None

    try:
        n_const = len(list(run_constants or []))
    except Exception:
        n_const = 0

    if n_const > 0:
        # prefer candidate with same number of params
        best = None
        for k in candidates:
            best = best or k
            try:
                m = custom_models[k]
                n_param = len(getattr(m, 'param_names_override', None) or [])
                if n_param == n_const:
                    log.trace('graphs_tab: matched custom by param count: run=%s -> %s (n=%d)', _short(run_name, 120), _short(k, 120), n_const)
                    return k
            except Exception:
                pass
        log.trace('graphs_tab: matched custom fallback: run=%s -> %s (n_const=%d candidates=%s)', _short(run_name, 120), _short(best, 120), n_const, _list_short(candidates, 20))
        return best

    log.trace('graphs_tab: matched custom by title: run=%s -> %s', _short(run_name, 120), _short(candidates[0], 120))
    return candidates[0]


def _render_graphs_tab_body(experiment_id: int) -> None:
    """Inner implementation for the graphs tab.

    This is wrapped by a refreshable container to allow updating immediately
    after a new computation run is saved.
    """

    st = get_state()
    file_repo = ExperimentFileRepository(DB_PATH)
    pick_repo = TablePickRepository(DB_PATH)
    results_repo = ExperimentProcessingResultsRepository(DB_PATH)
    graph_settings_repo = ExperimentGraphSettingsRepository(DB_PATH)

    latest = results_repo.get_latest_run(experiment_id)
    if not latest:
        StyledLabel('Nejdřív spusťte výpočet v záložce „rychlosti“.', 'warning')
        return

    # Staleness: did the user change processing settings or computation graphs since the last run?
    try:
        _run2, stale = compute_staleness(experiment_id)
    except Exception:
        stale = None

    log.debug('graphs_tab: open experiment_id=%s latest_run_keys=%s', experiment_id, _list_short(list(latest.keys()), 50))

    settings = dict(latest.get('settings') or {})
    run_models: dict[str, dict] = dict(latest.get('models') or {})
    run_model_names = list(run_models.keys())
    wanted_names = set(run_model_names)

    try:
        log.debug(
            'graphs_tab: latest run has %d models: %s',
            len(run_model_names),
            _list_short(run_model_names, 60),
        )
        # log whether PNGs are present (helps distinguish regen vs fallback)
        png_info = {k: bool((run_models.get(k) or {}).get('plot_png')) for k in run_model_names}
        log.trace('graphs_tab: plot_png present: %s', _short(png_info, 400))
    except Exception:
        pass

    try:
        log.trace('graphs_tab: settings=%s', _short(settings, 1200))
    except Exception:
        pass

    # Determine which t_shift was used for the run (manual or auto).
    used_t_shift = latest.get('used_t_shift')
    try:
        used_t_shift_f = float(used_t_shift) if used_t_shift is not None else float(settings.get('t_shift') or 0.0)
    except Exception:
        used_t_shift_f = float(settings.get('t_shift') or 0.0)

    # Use processing setting t_max_plot as a sensible default xlim.
    try:
        default_xmax = float(settings.get('t_max_plot') or 400.0)
    except Exception:
        default_xmax = 400.0

    # Simulate at least until t_max_plot so re-rendering matches the plot window.
    sim_t_max = max(10.0, float(default_xmax))

    ef = file_repo.get_single_for_experiment(experiment_id)
    if not ef or not ef.content:
        StyledLabel('Nejdřív nahrajte Excel v tabu „načtení dat“.', 'warning')
        return

    sheet = str(settings.get('sheet') or ef.selected_sheet or st.current_excel_sheet or '')
    log.debug(
        'graphs_tab: resolved sheet=%s (settings.sheet=%s file.selected_sheet=%s state.current_excel_sheet=%s)',
        sheet,
        _short(settings.get('sheet'), 80),
        _short(getattr(ef, 'selected_sheet', None), 80),
        _short(getattr(st, 'current_excel_sheet', None), 80),
    )
    if not sheet:
        StyledLabel('Vyberte list v tabu „načtení dat“.', 'warning')
        return

    def _pick_from_settings(key: str) -> Optional[tuple[int, int, int, int]]:
        raw = settings.get(key)
        if isinstance(raw, (list, tuple)) and len(raw) == 4:
            try:
                return (int(raw[0]), int(raw[1]), int(raw[2]), int(raw[3]))
            except Exception:
                return None
        return None

    fame_pick = _pick_from_settings('fame_pick') or _load_pick(pick_repo, experiment_id, 'fame')
    epo_pick = _pick_from_settings('epo_pick') or _load_pick(pick_repo, experiment_id, 'epo')

    log.debug('graphs_tab: fame_pick=%s epo_pick=%s', fame_pick, epo_pick)

    # Build tables (fast) and recreate model objects (no fitting).
    # If this fails, we still show persisted PNGs, but the user cannot edit.
    processor: Optional[TableProcessor] = None
    recreated_models: dict[str, KineticModel] = {}
    recreate_report: dict[str, str] = {}
    try:
        processor = _build_table_processor(
            excel_bytes=ef.content,
            sheet=sheet,
            fame_pick=fame_pick,
            epo_pick=epo_pick,
        )

        prebuilt_models = _create_prebuilt_models(
            processor=processor,
            settings=settings,
            used_t_shift=used_t_shift_f,
            wanted_names=wanted_names,
        )
        custom_models_all = _create_custom_models(
            experiment_id=experiment_id,
            processor=processor,
            settings=settings,
            used_t_shift=used_t_shift_f,
            wanted_names=wanted_names,
        )

        log.debug(
            'graphs_tab: recreated model pools: prebuilt=%d custom=%d',
            len(prebuilt_models),
            len(custom_models_all),
        )
        log.trace('graphs_tab: prebuilt keys=%s', _list_short(list(prebuilt_models.keys()), 80))
        log.trace('graphs_tab: custom keys=%s', _list_short(list(custom_models_all.keys()), 80))

        # Map *run model names* to recreated models.
        # This is robust against minor naming differences (punctuation, duplicate suffixes).
        for run_name in run_model_names:
            if run_name in prebuilt_models:
                recreated_models[run_name] = prebuilt_models[run_name]
                recreate_report[run_name] = 'matched: prebuilt exact'
                continue
            if run_name in custom_models_all:
                recreated_models[run_name] = custom_models_all[run_name]
                recreate_report[run_name] = 'matched: custom exact'
                continue
            entry = run_models.get(run_name) or {}
            consts = list(entry.get('constants') or [])
            best_key = _pick_best_custom_key_for_run(run_name, custom_models_all, consts)
            if best_key and best_key in custom_models_all:
                recreated_models[run_name] = custom_models_all[best_key]
                recreate_report[run_name] = f'matched: custom fuzzy -> {best_key}'
            else:
                target = _norm_title(run_name)
                sample_custom = list(custom_models_all.keys())[:20]
                recreate_report[run_name] = f'NOT matched (target={target}); sample_custom={_list_short(sample_custom, 20)}'
    except Exception as e:
        log.debug('graphs_tab: failed to recreate models: %s', e, exc_info=True)

    log.debug('graphs_tab: recreated_models=%d / run_models=%d', len(recreated_models), len(run_models))
    missing = [n for n in run_model_names if n not in recreated_models]
    if missing:
        log.debug('graphs_tab: missing recreated models for: %s', _list_short(missing, 80))
        for n in missing[:30]:
            log.trace('graphs_tab: missing reason %s -> %s', _short(n, 120), _short(recreate_report.get(n), 400))

    # Choose order: prebuilt (in a stable order), then custom, then any extras.
    ordered: list[str] = []
    for m in _PREBUILT_ORDER:
        if m in run_models:
            ordered.append(m)
    for name in run_models.keys():
        if name not in ordered:
            ordered.append(name)

    # Create plotters & configs
    plotters: dict[str, AdvancedPlotter] = {}
    configs: dict[str, GraphConfig] = {}
    plot_images: dict[str, Any] = {}

    plot_error_labels: dict[str, StyledLabel] = {}

    for name in ordered:
        model = recreated_models.get(name)
        entry = run_models.get(name) or {}
        consts = list(entry.get('constants') or [])
        if model is None:
            continue

        # IMPORTANT: reinit_kinetic_model() resets k_fit -> we must apply constants *after* reinit.
        # We always try to re-init the model using the exact t_shift that was used in the saved run.
        # (For graph-defined models this is required to get the same shifted timeline.)
        try:
            model.reinit_kinetic_model(t_shift=used_t_shift_f)
            ctrl = getattr(model, '_control_model', None)
            if ctrl is not None:
                try:
                    ctrl.reinit_kinetic_model(t_shift=used_t_shift_f)
                except Exception:
                    pass
        except Exception as e:
            log.debug(
                'graphs_tab: model.reinit_kinetic_model failed for %s: %s (used_t_shift=%s init_method=%s)',
                name,
                e,
                _short(used_t_shift_f, 40),
                _short(getattr(model, 'init_method', None), 40),
                exc_info=True,
            )

        # Apply persisted constants (k-values) from the last run.
        if consts:
            _apply_constants_to_model_for_plot(model, consts)
            ctrl = getattr(model, '_control_model', None)
            if ctrl is not None:
                _apply_constants_to_model_for_plot(ctrl, consts)
            try:
                v = getattr(model, 'k_fit', None)
                v = list(v) if v is not None else []
                log.trace(
                    'graphs_tab: applied %d constants to %s (main_params=%d)',
                    len(consts),
                    name,
                    len(list(getattr(model, 'param_names_override', None) or [])),
                )
            except Exception:
                pass
        else:
            try:
                log.debug(
                    'graphs_tab: WARNING no constants found for %s; payload empty',
                    name,
                )
            except Exception:
                log.debug('graphs_tab: WARNING no constants found for %s', name)
        try:
            model_for_plot = model
            ctrl = getattr(model, '_control_model', None)
            if ctrl is not None:
                model_for_plot = _MergedSimulationModel(
                    model,
                    ctrl,
                    main_active_columns=getattr(model, '_main_active_columns', None),
                    control_active_columns=getattr(model, '_control_active_columns', None),
                )
            plotter = AdvancedPlotter(name, model_for_plot, t_max=sim_t_max, time_points=2000)
            plotters[name] = plotter
        except Exception as e:
            try:
                t_range = None
                try:
                    t_range = (
                        float(getattr(model, 'original_data', [])[:, 0].min()),
                        float(getattr(model, 'original_data', [])[:, 0].max()),
                    )
                except Exception:
                    t_range = None
                log.debug(
                    'graphs_tab: failed to create plotter for %s: %s (t_shift=%s t_max=%s sim_t_max=%s t_range=%s cols=%s)',
                    name,
                    e,
                    _short(used_t_shift_f, 40),
                    _short(getattr(model, 't_max', None), 40),
                    sim_t_max,
                    _short(t_range, 80),
                    _list_short(list(getattr(model, 'column_names', []) or []), 20),
                    exc_info=True,
                )
            except Exception:
                log.debug('graphs_tab: failed to create plotter for %s: %s', name, e, exc_info=True)

    try:
        ok = list(plotters.keys())
        miss = [n for n in ordered if n not in plotters]
        log.debug('graphs_tab: plotters created=%d missing=%d', len(ok), len(miss))
        if ok:
            log.trace('graphs_tab: plotter keys=%s', _list_short(ok, 80))
        if miss:
            log.debug('graphs_tab: models without plotter (will fallback to PNG if present): %s', _list_short(miss, 80))
    except Exception:
        pass

    def _default_config_for(name: str, plotter: Optional[AdvancedPlotter]) -> GraphConfig:
        n_curves = plotter.get_n_curves() if plotter is not None else 0
        styles = []
        labels: list[str] = []
        try:
            labels = list(plotter.data.get('column_names') or []) if plotter is not None else []
        except Exception:
            labels = []
        for i in range(n_curves):
            styles.append(CurveStyle(
                color=_TAB10[i % len(_TAB10)],
                label=str(labels[i]) if i < len(labels) else f'křivka {i + 1}',
            ))
        return GraphConfig(
            title=name,
            show_title=True,
            legend_mode='components_only',
            fig_width=8.0,
            fig_height=6.0,
            xlim_mode='manual',
            xlim_min=0.0,
            xlim_max=float(default_xmax),
            ylim_mode='default',
            curve_styles=styles,
        )

    def _merge_config_with_defaults(
        name: str,
        plotter: Optional[AdvancedPlotter],
        loaded: Optional[GraphConfig],
    ) -> GraphConfig:
        base = _default_config_for(name, plotter)
        if loaded is None:
            return base

        # If the user loaded an older config, ensure missing fields get sane defaults.
        cfg = loaded
        if not cfg.title:
            cfg.title = base.title
        # Bring curve styles length up to n_curves.
        n_curves = plotter.get_n_curves() if plotter is not None else 0
        if n_curves and len(cfg.curve_styles) < n_curves:
            for i in range(len(cfg.curve_styles), n_curves):
                cfg.curve_styles.append(base.curve_styles[i] if i < len(base.curve_styles) else CurveStyle())
        if n_curves and len(cfg.curve_styles) > n_curves:
            cfg.curve_styles = cfg.curve_styles[:n_curves]
        return cfg

    # Load persisted configs from DB (per experiment/model), fallback to defaults.
    for name in ordered:
        loaded_cfg: Optional[GraphConfig] = None
        try:
            row = graph_settings_repo.get(experiment_id, name)
            if row and isinstance(row.get('config'), dict):
                loaded_cfg = GraphConfig.from_dict(row['config'])
        except Exception:
            loaded_cfg = None
        configs[name] = _merge_config_with_defaults(name, plotters.get(name), loaded_cfg)

    # --- persistence helpers (debounced saves) ---
    save_tasks: dict[str, asyncio.Task] = {}

    def _schedule_save(model_name: str) -> None:
        # Debounce DB writes while user types.
        old = save_tasks.get(model_name)
        if old and not old.done():
            old.cancel()

        async def _do() -> None:
            try:
                await asyncio.sleep(0.4)
                cfg = configs.get(model_name)
                if cfg is None:
                    return
                graph_settings_repo.upsert(experiment_id, model_name, cfg.to_dict())
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.debug('graphs_tab: failed to persist graph settings for %s: %s', model_name, e)

        save_tasks[model_name] = asyncio.create_task(_do())

    # Pick mode: per-model which curve index should receive the next click.
    pending_pick: dict[str, int] = {}

    def _replot(model_name: str, *, persist: bool = True) -> None:
        plotter = plotters.get(model_name)
        cfg = configs.get(model_name)
        img = plot_images.get(model_name)
        if plotter is None or cfg is None or img is None:
            return
        try:
            fig = plotter.plot(ui=True, **cfg.to_kwargs())
            update_plot_image(fig, img)
            # clear inline error (if any)
            err = plot_error_labels.get(model_name)
            if err is not None:
                try:
                    err.set_visibility(False)
                except Exception:
                    pass

        except Exception as ex:
            # Do not spam ui.notify while the user types (e.g. unfinished LaTeX).
            # Instead show an inline error for this plot and keep it updated.
            err = plot_error_labels.get(model_name)
            if err is not None:
                msg = str(ex)
                # Shorten very long parse errors a bit for readability.
                if len(msg) > 500:
                    msg = msg[:500] + '…'
                err.set(f'Chyba při vykreslení: {msg}', 'error')
                try:
                    err.set_visibility(True)
                except Exception:
                    pass
            return
        if persist:
            _schedule_save(model_name)

    with ui.column().classes('w-full gap-2'):
        with ui.row().classes('w-full items-center justify-between'):
            lbl_title = ui.label('Grafy').classes('text-h6')
            attach_tooltip(
                lbl_title,
                'Grafy',
                'Interaktivní vykreslení výsledků posledního výpočtu.\n\n'
                'Nastavení křivek (barvy, popisky, osy…) se ukládá zvlášť a nevyžaduje přepočet modelů.',
            )

            lbl_tshift = ui.label(f"t_shift použité ve výpočtu: {used_t_shift_f:.6g}").classes('text-caption text-grey-7')
            attach_tooltip(
                lbl_tshift,
                't_shift z posledního běhu',
                'Časový posun, který byl skutečně použit při posledním výpočtu v záložce Zpracování.\n\n'
                'Používá se i při přegenerování grafů pro konzistenci.',
            )

        try:
            if stale is not None and (stale.global_changed or any(stale.custom_changed.values())):
                bchg = ui.badge('Změněno od posledního výpočtu – přepočítejte v záložce „rychlosti“', color='orange').props('outline')
                attach_tooltip(
                    bchg,
                    'Změněno',
                    'Od posledního běhu se změnila data, výběry tabulek nebo definice výpočtů.\n\n'
                    'Grafy stále zobrazují poslední uložené výsledky – pro nové výsledky spusťte přepočet v Zpracování.',
                )
        except Exception:
            pass

        if not ordered:
            StyledLabel('Nejsou žádné uložené výsledky pro vykreslení.', 'warning')
            return

        with ui.row().classes('w-full items-center gap-2'):
            btn_export = ui.button(
                'Export PNG/PDF/SVG',
                on_click=lambda: _open_export_dialog(plotters, configs),
            ).props('unelevated').classes('bg-secondary text-white')
            attach_tooltip(
                btn_export,
                'Export',
                'Vyexportuje všechny grafy do ZIPu ve zvoleném formátu (PNG/PDF/SVG).\n\n'
                'Použije aktuální nastavení každého grafu (osy, legenda, barvy, anotace).',
            )
            lbl_tip = ui.label('Tip: změny nastavení se promítají do exportu.').classes('text-caption text-grey-6')
            attach_tooltip(
                lbl_tip,
                'Ukládání nastavení',
                'Změny v nastavení grafů se ukládají automaticky do DB a použijí se i při exportu.',
            )

        ui.separator()

        def _on_image_mouse(e: events.MouseEventArguments, model_name: str) -> None:
            """When a curve is armed for picking, interpret the next mouse event as the label position."""
            if model_name not in pending_pick:
                return
            plotter = plotters.get(model_name)
            cfg = configs.get(model_name)
            if plotter is None or cfg is None:
                return
            idx = pending_pick.get(model_name)
            if idx is None or idx < 0 or idx >= len(cfg.curve_styles):
                pending_pick.pop(model_name, None)
                return
            axes = plotter.get_axes_values() or {}
            try:
                ix = float(getattr(e, 'image_x', 0.0))
                iy = float(getattr(e, 'image_y', 0.0))
                # Use relative coords to be robust against CSS scaling of the image element.
                iw = float(getattr(e, 'image_width', 0.0) or 0.0)
                ih = float(getattr(e, 'image_height', 0.0) or 0.0)
                if iw > 0 and ih > 0:
                    x, y = plotter.rel_to_data(ix / iw, iy / ih)
                else:
                    x, y = plotter.image_to_data(ix, iy)
            except Exception:
                return

            cs = cfg.curve_styles[idx]
            cs.additional_text_enabled = True
            cs.additional_text_x = float(x)
            cs.additional_text_y = float(y)
            pending_pick.pop(model_name, None)
            ui.notify(f'📍 Pozice textu nastavena ({model_name}): x={x:.3g}, y={y:.3g}', type='positive')
            _replot(model_name, persist=True)

        for name in ordered:
            entry = run_models.get(name) or {}
            plotter = plotters.get(name)
            cfg = configs[name]

            with ui.card().classes('w-full'):
                with ui.row().classes('w-full items-center justify-between'):
                    hdr = ui.markdown(f"### {name}")
                    attach_tooltip(
                        hdr,
                        f'Graf: {name}',
                        'Kliknutím otevřete nastavení grafu (vpravo).\n\n'
                        'Pokud se podaří zrekonstruovat model, graf se vykresluje znovu. Jinak se použije uložený PNG z posledního běhu.',
                    )
                    try:
                        b = ui.badge('Změněno', color='orange').props('outline')
                        b.visible = bool(stale is not None and is_model_stale(name, stale))
                        attach_tooltip(
                            b,
                            'Změněno',
                            'Nastavení nebo definice tohoto modelu se změnily od posledního přepočtu.\n\n'
                            'Zpracování (Rychlosti) uloží nové konstanty a grafy.',
                        )
                    except Exception:
                        pass

                consts = list(entry.get('constants') or [])
                if consts:
                    with ui.column().classes('w-full'):
                        lbl_c = ui.label('Konstanty:').classes('text-subtitle2')
                        attach_tooltip(
                            lbl_c,
                            'Konstanty (k)',
                            'Hodnoty konstant uložené z posledního fitu.\n\n'
                            'Tyto hodnoty se používají při přegenerování grafu bez nového fitování.',
                        )
                        for c in consts:
                            try:
                                ui.markdown(
                                    r"$$k_{" + f"{int(c.get('idx') or 0)}" + r"} = " +
                                    f"{c.get('latex', '')} = {float(c.get('value')):.8f}$$",
                                    extras=['latex'],
                                )
                            except Exception:
                                pass

                with ui.row().classes('w-full flex-wrap gap-6 items-start'):
                    with ui.column().classes('flex-1 min-w-[320px]'):
                        if plotter is not None:
                            img = ui.interactive_image(
                                on_mouse=lambda e, n=name: _on_image_mouse(e, n),
                                cross=False,
                            ).classes('w-full max-w-5xl [&>img]:w-full [&>svg]:w-full')
                            plot_images[name] = img
                            # inline error label for plotting problems (e.g. invalid LaTeX)
                            err_lbl = StyledLabel('', 'error')
                            try:
                                err_lbl.set_visibility(False)
                            except Exception:
                                pass
                            plot_error_labels[name] = err_lbl
                            attach_tooltip(
                                err_lbl,
                                'Chyba vykreslení',
                                'Pokud je text/nadpis ve formátu LaTeX nedokončený nebo neplatný,\n\n'
                                'zobrazí se zde chyba místo opakovaných notifikací.',
                            )

                            attach_tooltip(
                                img,
                                'Graf',
                                'Zobrazení simulace/modelu.\n\n'
                                'Pokud v nastavení křivky zvolíte „📍 Umístit kliknutím“, další klik do grafu nastaví pozici textu.',
                            )
                        else:
                            png = entry.get('plot_png')
                            if png:
                                import base64
                                b64 = base64.b64encode(png).decode('ascii')
                                ui.image(f"data:image/png;base64,{b64}").classes('w-full max-w-5xl [&>img]:w-full [&>svg]:w-full')
                            else:
                                StyledLabel('Graf pro tento model není uložený.', 'warning')

                    with ui.column().classes('flex-1 min-w-[320px]'):
                        if plotter is None:
                            ui.label('Tento graf nelze znovu přegenerovat (chybí model).').classes('text-caption text-grey-6')
                            # Helpful inline hint in UI; details go to debug logger.
                            reason = recreate_report.get(name)
                            if reason:
                                ui.label(f"Důvod (debug): {reason}").classes('text-caption text-grey-6')
                        else:
                            exp = ui.expansion('⚙️ Nastavení grafu', value=False)
                            attach_tooltip(
                                exp,
                                'Nastavení grafu',
                                'Úprava vzhledu grafu (osy, legenda, barvy, popisky, anotace).\n\n'
                                'Změny se ukládají automaticky a nevyžadují přepočet modelů.',
                            )
                            with exp:
                                _render_controls_for(
                                    name,
                                    cfg,
                                    plotter,
                                    plot_images,
                                    replot=lambda n=name: _replot(n, persist=True),
                                    arm_pick=lambda idx, n=name: (
                                        pending_pick.__setitem__(n, int(idx)),
                                        ui.notify('Klikněte do grafu pro umístění textu.', type='info'),
                                    ),
                                    is_pending=lambda idx, n=name: pending_pick.get(n) == int(idx),
                                )

                if plotter is not None:
                    try:
                        fig = plotter.plot(ui=True, **cfg.to_kwargs())
                        update_plot_image(fig, plot_images[name])
                        # Do not persist on initial render; only on user changes
                        
                    except Exception as e:
                        err = plot_error_labels.get(name)
                        if err is None:
                            err = StyledLabel('', 'error')
                            plot_error_labels[name] = err
                        msg = str(e)
                        if len(msg) > 500:
                            msg = msg[:500] + '…'
                        err.set(f'Chyba při vykreslení: {msg}', 'error')
                        try:
                            err.set_visibility(True)
                        except Exception:
                            pass


def render_graphs_tab(experiment_id: int) -> None:
    """Tab: grafy (pokročilé vykreslování pro všechny spočítané ODE modely).

    The experiment page builds all tabs once; therefore we need an explicit refresh
    mechanism so that a newly computed/updated run becomes visible immediately
    without reopening the experiment.
    """

    _ensure_graphs_css()

    st = get_state()
    last_version = st.graphs_version
    last_stale_token: str | None = None

    @ui.refreshable
    def _view() -> None:
        _render_graphs_tab_body(experiment_id)

    _view()

    def _maybe_refresh() -> None:
        nonlocal last_version
        try:
            current = get_state().graphs_version
        except Exception:
            current = last_version
        if current != last_version:
            last_version = current
            try:
                _view.refresh()
            except Exception:
                pass

    # Polling is cheap here (it only checks an int). Re-rendering happens only
    # when graphs_version changes (i.e., after a computation finishes).
    ui.timer(0.8, _maybe_refresh)

    def _maybe_refresh_staleness() -> None:
        """Refresh graphs tab when settings/computation definitions change.

        We keep this separate from graphs_version because changing settings should show
        a 'changed' indicator immediately, even if no recompute happened yet.
        """
        nonlocal last_stale_token
        try:
            _run, info = compute_staleness(experiment_id)
            token = f"{info.last_run_created_at}|{int(info.global_changed)}|{sum(1 for v in info.custom_changed.values() if v)}"
        except Exception:
            return
        if last_stale_token is None:
            last_stale_token = token
            return
        if token != last_stale_token:
            last_stale_token = token
            try:
                _view.refresh()
            except Exception:
                pass

    ui.timer(1.2, _maybe_refresh_staleness)


def _handle_click_for_future(e: events.MouseEventArguments, name: str) -> None:
    # Placeholder for future: coordinate picking for annotations.
    _ = e, name


def _render_controls_for(
    name: str,
    cfg: GraphConfig,
    plotter: AdvancedPlotter,
    plot_images: dict[str, Any],
    replot: Callable[[], None],
    arm_pick: Callable[[int], Any],
    is_pending: Callable[[int], bool],
) -> None:
    """Render per-graph controls and live-update plot on changes.

    `replot` must update the image *and* persist the configuration.
    """

    # The controls can become very large. We therefore split them into tabs:
    # - one "Graf" tab for global settings
    # - one tab per curve (Křivka 1, Křivka 2, ...), including the original column name

    colnames: list[str] = []
    try:
        colnames = list(plotter.data.get('column_names') or [])
    except Exception:
        colnames = []

    # Prevent a large number of curve-tabs from causing horizontal scrolling of the whole page.
    # The row should stay within the card width and scroll/arrow only inside itself.
    curve_tabs: list[Any] = []
    with ui.element('div').classes('w-full graphs-controls-tabs'):
        # Don't force the q-tabs element itself to full width; the wrapper provides width and scrolling.
        with ui.tabs().props('dense arrows outside-arrows mobile-arrows').classes('q-pa-none') as tabs:
            # In NiceGUI the tab's *name* is the identifier used by ui.tab_panels.
            # The `label` is what the user sees. We therefore use a stable name and a human label.
            t_graph = ui.tab(name='graph', label='Graf')
            attach_tooltip(
                t_graph,
                'Graf',
                'Globální nastavení grafu: nadpis, legenda, velikost, osy a rozsahy.\n\n'
                'Změny se projeví okamžitě a ukládají se automaticky.',
            )
            for idx, _cs in enumerate(cfg.curve_styles):
                orig = colnames[idx] if idx < len(colnames) else ''
                tab_title = f'Křivka {idx + 1}' + (f' - {orig}' if orig else '')
                t = ui.tab(name=f'curve_{idx}', label=tab_title)
                # Add a small colored marker (CSS ::before) for better orientation
                try:
                    t.classes('curve-tab')
                    t.style(f'--curve-color: {cfg.curve_styles[idx].color};')
                except Exception:
                    pass
                attach_tooltip(
                    t,
                    tab_title,
                    'Nastavení konkrétní křivky: barva, popisek, styl čáry, marker a anotace textem.\n\n'
                    'Původní název (ze sloupce tabulky) je uveden v záhlaví panelu.',
                )
                curve_tabs.append(t)

    with ui.tab_panels(tabs, value='graph').classes('w-full'):
        # --- Global graph settings ---
        with ui.tab_panel('graph'):
            def _set_enabled(el: Any, enabled: bool) -> None:
                """Best-effort enable/disable across NiceGUI versions."""
                try:
                    if hasattr(el, 'set_enabled'):
                        el.set_enabled(bool(enabled))
                        return
                except Exception:
                    pass
                try:
                    if bool(enabled):
                        el.enable()
                    else:
                        el.disable()
                    return
                except Exception:
                    pass
                try:
                    el.enabled = bool(enabled)
                except Exception:
                    pass

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                show_title = ui.switch('Nadpis', value=cfg.show_title,
                                       on_change=lambda e: (setattr(cfg, 'show_title', bool(e.value)), replot()))
                attach_tooltip(
                    show_title,
                    'Nadpis',
                    'Zapne/vypne zobrazení nadpisu grafu.\n\n'
                    'Text nadpisu nastavíte v poli vedle.',
                )
                title = ui.input('Text nadpisu', value=cfg.title,
                                 on_change=lambda e: (setattr(cfg, 'title', str(e.value)), replot())).classes('w-72')
                attach_tooltip(
                    title,
                    'Text nadpisu',
                    'Text, který se zobrazí nahoře v grafu.\n\n'
                    'Použije se také jako název při exportu (pokud exportní formát podporuje metadatový titulek).',
                )
                title.bind_enabled_from(show_title, 'value')

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                sw_clip = ui.switch(
                    'Ořez (clip_on)',
                    value=bool(getattr(cfg, 'clip_on', True)),
                    on_change=lambda e: (setattr(cfg, 'clip_on', bool(e.value)), replot()),
                )
                attach_tooltip(
                    sw_clip,
                    'Ořez (clip_on)',
                    'Určuje, zda se křivky/markery/anotace ořezávají na oblast os.\n\n'
                    '*Zapnuto* (default): standardní chování matplotlib – vše je oříznuté do os.\n'
                    '*Vypnuto*: prvky se mohou vykreslit i mimo osy (např. přes spiny), což se hodí pro vizuální zvýraznění.',
                )

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                sel_leg = ui.select(
                    label='Legenda',
                    options=['both', 'single', 'components_only', 'None'],
                    value=cfg.legend_mode,
                    on_change=lambda e: (setattr(cfg, 'legend_mode', str(e.value)), replot()),
                ).classes('w-64')
                attach_tooltip(
                    sel_leg,
                    'Legenda',
                    'Režim legendy v grafu.\n\n'
                    '*components_only* obvykle zobrazí jen složky (bez duplicitních názvů).',
                )
                num_w = ui.number('Šířka (inch)', value=cfg.fig_width, min=1, max=20, step=0.5,
                          on_change=lambda e: (setattr(cfg, 'fig_width', float(e.value)), replot())).classes('w-40')
                attach_tooltip(num_w, 'Šířka', 'Šířka výsledné figury (v palcích).\n\nPoužívá se i pro export.')
                num_h = ui.number('Výška (inch)', value=cfg.fig_height, min=1, max=20, step=0.5,
                          on_change=lambda e: (setattr(cfg, 'fig_height', float(e.value)), replot())).classes('w-40')
                attach_tooltip(num_h, 'Výška', 'Výška výsledné figury (v palcích).\n\nPoužívá se i pro export.')


            
            # --- Grid (major/minor) ---
            ui.label('Mřížka').classes('text-subtitle2 text-grey-8')

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                picker_g = ColorPickerButton(
                    icon='colorize',
                    color=cfg.grid_config.color,
                    color_type='hexa',
                    on_pick=lambda e: replot(),
                )
                picker_g.bind_color(cfg.grid_config, 'color')
                attach_tooltip(
                    picker_g,
                    'Barva hlavní mřížky',
                    'Barva hlavní (major) mřížky.\n\n'
                    'Můžete použít i průhlednost (alpha) – vybírá se přímo v pickeru.',
                )

                sw_g = ui.switch(
                    'Hlavní mřížka',
                    value=bool(cfg.grid_config.visible),
                    on_change=lambda e: (setattr(cfg.grid_config, 'visible', bool(e.value)), replot()),
                )
                attach_tooltip(
                    sw_g,
                    'Hlavní mřížka',
                    'Zapne/vypne hlavní (major) mřížku v grafu.',
                )

                sel_g_axis = ui.select(
                    label='Osy (major)',
                    options=['both', 'x', 'y'],
                    value=str(cfg.grid_config.axis or 'both'),
                    on_change=lambda e: (setattr(cfg.grid_config, 'axis', str(e.value)), replot()),
                ).classes('w-32')
                attach_tooltip(
                    sel_g_axis,
                    'Osy mřížky',
                    'Určuje, zda se mřížka vykreslí pro obě osy, nebo jen pro X/Y.',
                )

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                picker_m = ColorPickerButton(
                    icon='colorize',
                    color=cfg.grid_config_minor.color,
                    color_type='hexa',
                    on_pick=lambda e: replot(),
                )
                picker_m.bind_color(cfg.grid_config_minor, 'color')
                attach_tooltip(
                    picker_m,
                    'Barva vedlejší mřížky',
                    'Barva vedlejší (minor) mřížky.\n\n'
                    'Pozor: aby šla minor mřížka vykreslit, zapínají se i minor ticks.',
                )

                sw_m = ui.switch(
                    'Vedlejší mřížka',
                    value=bool(cfg.grid_config_minor.visible),
                    on_change=lambda e: (setattr(cfg.grid_config_minor, 'visible', bool(e.value)), replot()),
                )
                attach_tooltip(
                    sw_m,
                    'Vedlejší mřížka',
                    'Zapne/vypne vedlejší (minor) mřížku v grafu.',
                )

                sel_m_axis = ui.select(
                    label='Osy (minor)',
                    options=['both', 'x', 'y'],
                    value=str(cfg.grid_config_minor.axis or 'both'),
                    on_change=lambda e: (setattr(cfg.grid_config_minor, 'axis', str(e.value)), replot()),
                ).classes('w-32')
                attach_tooltip(
                    sel_m_axis,
                    'Osy mřížky',
                    'Určuje, zda se minor mřížka vykreslí pro obě osy, nebo jen pro X/Y.',
                )

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                inp_xl = ui.input('Popisek osy X', value=cfg.xlabel,
                         on_change=lambda e: (setattr(cfg, 'xlabel', str(e.value)), replot())).classes('w-56')
                attach_tooltip(inp_xl, 'Osa X', 'Popisek osy X (čas apod.).')
                inp_yl = ui.input('Popisek osy Y', value=cfg.ylabel,
                         on_change=lambda e: (setattr(cfg, 'ylabel', str(e.value)), replot())).classes('w-56')
                attach_tooltip(inp_yl, 'Osa Y', 'Popisek osy Y (koncentrace apod.).')

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                def _sync_x_manual() -> None:
                    manual = str(getattr(cfg, 'xlim_mode', '')) == 'manual'
                    try:
                        _set_enabled(num_xmin, manual)
                        _set_enabled(num_xmax, manual)
                    except Exception:
                        pass

                sel_x = ui.select(
                    label='Osa X',
                    options=['default', 'auto', 'all_data', 'manual', 'None'],
                    value=cfg.xlim_mode,
                    on_change=lambda e: (setattr(cfg, 'xlim_mode', str(e.value)), _sync_x_manual(), replot()),
                ).classes('w-40')
                attach_tooltip(
                    sel_x,
                    'Rozsah osy X',
                    'Zvolte, jak se určí rozsah osy X.\n\n'
                    '*manual* použije X min / X max, *all_data* se přizpůsobí datům.',
                )
                num_xmin = ui.number('X min', value=cfg.xlim_min or 0.0, min=0, step=1.0,
                          on_change=lambda e: (setattr(cfg, 'xlim_min', float(e.value)), replot())).classes('w-32')
                attach_tooltip(
                    num_xmin,
                    'X min',
                    'Spodní mez osy X.\n\n'
                    'Pole je aktivní pouze když je režim osy X nastaven na *manual*.\n'
                    'V ostatních režimech se rozsah určuje automaticky a ruční hodnoty se ignorují.',
                )
                num_xmax = ui.number('X max', value=cfg.xlim_max or 0.0, min=0, step=1.0,
                          on_change=lambda e: (setattr(cfg, 'xlim_max', float(e.value)), replot())).classes('w-32')
                attach_tooltip(
                    num_xmax,
                    'X max',
                    'Horní mez osy X.\n\n'
                    'Pole je aktivní pouze když je režim osy X nastaven na *manual*.\n'
                    'V ostatních režimech se rozsah určuje automaticky a ruční hodnoty se ignorují.',
                )

                # Ensure the disabled/enabled state matches the current mode immediately.
                _sync_x_manual()

            with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                def _sync_y_manual() -> None:
                    manual = str(getattr(cfg, 'ylim_mode', '')) == 'manual'
                    try:
                        _set_enabled(num_ymin, manual)
                        _set_enabled(num_ymax, manual)
                    except Exception:
                        pass

                sel_y = ui.select(
                    label='Osa Y',
                    options=['default', 'manual', 'None'],
                    value=cfg.ylim_mode,
                    on_change=lambda e: (setattr(cfg, 'ylim_mode', str(e.value)), _sync_y_manual(), replot()),
                ).classes('w-40')
                attach_tooltip(
                    sel_y,
                    'Rozsah osy Y',
                    'Zvolte, jak se určí rozsah osy Y.\n\n'
                    '*manual* použije Y min / Y max.',
                )
                num_ymin = ui.number('Y min', value=cfg.ylim_min or 0.0, min=0, step=0.05,
                          on_change=lambda e: (setattr(cfg, 'ylim_min', float(e.value)), replot())).classes('w-32')
                attach_tooltip(
                    num_ymin,
                    'Y min',
                    'Spodní mez osy Y.\n\n'
                    'Pole je aktivní pouze když je režim osy Y nastaven na *manual*.\n'
                    'V ostatních režimech se rozsah určuje automaticky a ruční hodnoty se ignorují.',
                )
                num_ymax = ui.number('Y max', value=cfg.ylim_max or 1.0, min=0, step=0.05,
                          on_change=lambda e: (setattr(cfg, 'ylim_max', float(e.value)), replot())).classes('w-32')
                attach_tooltip(
                    num_ymax,
                    'Y max',
                    'Horní mez osy Y.\n\n'
                    'Pole je aktivní pouze když je režim osy Y nastaven na *manual*.\n'
                    'V ostatních režimech se rozsah určuje automaticky a ruční hodnoty se ignorují.',
                )

                _sync_y_manual()

        # --- Per-curve settings ---
        for idx, cs in enumerate(cfg.curve_styles):
            with ui.tab_panel(f'curve_{idx}'):
                orig = colnames[idx] if idx < len(colnames) else None
                if orig:
                    ui.label(f'Původní název: {orig}').classes('text-caption text-grey-7')

                with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                    def _set_curve_color(color: str, _cs=cs, _idx=idx) -> None:
                        _cs.color = str(color)
                        # Update the colored marker in the curve-tab header immediately.
                        try:
                            if 0 <= _idx < len(curve_tabs):
                                curve_tabs[_idx].style(f'--curve-color: {str(color)};')
                        except Exception:
                            pass
                        replot()

                    picker = ColorPickerButton(
                        icon='colorize',
                        color=cs.color,
                        color_type='hexa',
                        # IMPORTANT: bind the function object as a default arg.
                        # Otherwise Python's late-binding inside the for-loop would make
                        # every picker call the *last* loop iteration's handler.
                        on_pick=(
                            lambda e, _f=_set_curve_color, _fallback=str(cs.color):
                                _f(str(getattr(e, 'color', _fallback)))
                        ),
                    )
                    picker.bind_color(cs, 'color')
                    attach_tooltip(
                        picker,
                        'Barva křivky',
                        'Změní barvu této křivky.\n\n'
                        'Barevný marker v záhlaví tabu se aktualizuje okamžitě.',
                    )
                    inp_lbl = ui.input('Popisek', value=cs.label,
                             on_change=lambda e, _cs=cs: (setattr(_cs, 'label', str(e.value)), replot())).classes('w-56')
                    attach_tooltip(inp_lbl, 'Popisek', 'Text v legendě pro tuto křivku (volitelné).')
                    sel_ls = ui.select(
                        label='Čára',
                        options=['solid', 'dashed', 'dashdot', 'dotted'],
                        value=cs.linestyle,
                        on_change=lambda e, _cs=cs: (setattr(_cs, 'linestyle', str(e.value)), replot()),
                    ).classes('w-36')
                    attach_tooltip(sel_ls, 'Styl čáry', 'Zvolte typ čáry pro tuto křivku.')
                    num_lw = ui.number('Tloušťka', value=cs.linewidth, min=0.1, step=0.1,
                              on_change=lambda e, _cs=cs: (setattr(_cs, 'linewidth', float(e.value)), replot())).classes('w-28')
                    attach_tooltip(num_lw, 'Tloušťka', 'Tloušťka čáry (v bodech).')
                    sel_m = ui.select(
                        label='Marker',
                        options=['o', 's', '^', 'v', 'x', '+', '*', 'None'],
                        value=cs.marker,
                        on_change=lambda e, _cs=cs: (setattr(_cs, 'marker', str(e.value)), replot()),
                    ).classes('w-28')
                    attach_tooltip(sel_m, 'Marker', 'Tvar značek bodů. Hodnota *None* značky vypne.')
                    num_ms = ui.number('Velikost', value=cs.markersize, min=1, step=0.5,
                              on_change=lambda e, _cs=cs: (setattr(_cs, 'markersize', float(e.value)), replot())).classes('w-28')
                    attach_tooltip(num_ms, 'Velikost markeru', 'Velikost značek bodů.')

                ui.separator().classes('q-mt-sm q-mb-sm')
                lbl_ann = ui.label('Anotace (text v grafu)').classes('text-caption text-grey-7')
                attach_tooltip(
                    lbl_ann,
                    'Anotace',
                    'Volitelný text přímo v grafu (např. poznámka, označení bodu).\n\n'
                    'Pozici můžete zadat ručně nebo ji „chytit“ kliknutím do grafu.',
                )
                with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                    sw_txt = ui.switch('Zobrazit text', value=cs.additional_text_enabled,
                              on_change=lambda e, _cs=cs: (setattr(_cs, 'additional_text_enabled', bool(e.value)), replot()))
                    attach_tooltip(sw_txt, 'Zobrazit text', 'Zapne/vypne zobrazení anotace.')
                    num_fs = ui.number('Velikost písma', value=cs.additional_text_size, min=6, max=40, step=1,
                              on_change=lambda e, _cs=cs: (setattr(_cs, 'additional_text_size', float(e.value)), replot())).classes('w-40')
                    attach_tooltip(num_fs, 'Velikost písma', 'Velikost písma anotace v bodech.')
                    btn_pick = ui.button('📍 Umístit kliknutím', on_click=lambda _=None, i=idx: arm_pick(i)).props('outline')
                    attach_tooltip(
                        btn_pick,
                        'Umístit kliknutím',
                        'Po stisknutí klikněte do grafu vlevo – pozice textu se nastaví podle místa kliknutí.',
                    )
                    if is_pending(idx):
                        ui.label('čekám na klik do grafu…').classes('text-caption text-orange-9')

                # For compactness we use a single-line input. Use "\\n" to insert line breaks.
                inp_txt = ui.input('Text (\\n pro nový řádek)', value=cs.additional_text_text,
                         on_change=lambda e, _cs=cs: (setattr(_cs, 'additional_text_text', str(e.value)), replot()))\
                    .classes('w-full')
                attach_tooltip(
                    inp_txt,
                    'Text anotace',
                    'Text, který se vykreslí do grafu.\n\n'
                    'Použijte „\\n“ pro nový řádek (víceřádkový text).',
                )

                with ui.row().classes('w-full flex-wrap gap-3 items-center'):
                    num_ax = ui.number('X', value=cs.additional_text_x, step=1.0,
                              on_change=lambda e, _cs=cs: (setattr(_cs, 'additional_text_x', float(e.value)), replot())).classes('w-32')
                    attach_tooltip(num_ax, 'X', 'Souřadnice X anotace v datových souřadnicích.')
                    num_ay = ui.number('Y', value=cs.additional_text_y, step=0.05,
                              on_change=lambda e, _cs=cs: (setattr(_cs, 'additional_text_y', float(e.value)), replot())).classes('w-32')
                    attach_tooltip(num_ay, 'Y', 'Souřadnice Y anotace v datových souřadnicích.')


def _open_export_dialog(plotters: dict[str, AdvancedPlotter], configs: dict[str, GraphConfig]) -> None:
    if not plotters:
        ui.notify('Nemám co exportovat (grafy nejsou připravené).', type='warning')
        return

    async def do_export(fmt: str):
        await export_all_figures_as_zip(plotters, configs, save_format=fmt)

    with ui.dialog() as dialog, ui.card():
        lbl = ui.label('Export grafů').classes('text-h6')
        attach_tooltip(
            lbl,
            'Export grafů',
            'Vygeneruje všechny grafy z aktuálních nastavení a zabalí je do ZIPu.\n\n'
            'Volba formátu ovlivní kvalitu a použitelnost (PDF/SVG jsou vektorové).',
        )
        lbl2 = ui.label('Vyberte formát:')
        attach_tooltip(lbl2, 'Formát', 'Zvolte výstupní formát pro všechny grafy v exportu.')
        with ui.row().classes('gap-2'):
            async def export_png():
                dialog.close()
                await do_export('png')

            async def export_pdf():
                dialog.close()
                await do_export('pdf')

            async def export_svg():
                dialog.close()
                await do_export('svg')

            b_png = ui.button('PNG', on_click=export_png)
            attach_tooltip(b_png, 'PNG', 'Rastrový obrázek (dobré pro prezentace, rychlé sdílení).')
            b_pdf = ui.button('PDF', on_click=export_pdf)
            attach_tooltip(b_pdf, 'PDF', 'Vektorový výstup (vhodné pro tisk, publikace).')
            b_svg = ui.button('SVG', on_click=export_svg)
            attach_tooltip(b_svg, 'SVG', 'Vektorový obrázek (dobré pro editaci v grafických nástrojích).')
        ui.button('Zavřít', on_click=dialog.close).props('flat')
        dialog.open()
