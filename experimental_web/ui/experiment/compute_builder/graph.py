from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

from nicegui import ui

from experimental_web.logging_setup import get_logger, TRACE

log = get_logger(__name__)

Edge = Tuple[str, str]
EdgeMode = int  # 0=disabled, 1=main, 2=control


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------


@dataclass
class GraphState:
    nodes: List[str] = field(default_factory=list)
    node_enabled: Dict[str, bool] = field(default_factory=dict)  # legacy (kept for compatibility)
    edge_enabled: Dict[Edge, Union[bool, EdgeMode]] = field(default_factory=dict)

    @property
    def edge_modes(self) -> Dict[Edge, EdgeMode]:
        """Return normalized edge modes (0/1/2) filtered to valid, forward edges.

        The interactive widget only renders edges from earlier nodes to later
        nodes (according to `nodes`). If the user reorders nodes, old edges can
        become "backward" and would otherwise stay hidden yet still be saved and
        used for ODE generation.
        """
        idx = {n: i for i, n in enumerate(self.nodes)}
        out: Dict[Edge, EdgeMode] = {}
        for (a, b), v in (self.edge_enabled or {}).items():
            if a not in idx or b not in idx:
                continue
            if idx[a] >= idx[b]:
                continue
            out[(a, b)] = _normalize_edge_mode(v)
        return out


class GraphController:
    def __init__(self, state: GraphState, dom_id: str) -> None:
        self.state = state
        self.dom_id = dom_id
        self._refresh: Optional[Callable[[], None]] = None

    def refresh(self) -> None:
        if self._refresh is not None:
            self._refresh()

    def get_state(self) -> GraphState:
        return self.state


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _normalize_edge_mode(v: Union[bool, int, None]) -> EdgeMode:
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        iv = int(v)
    except Exception:
        return 0
    return 0 if iv < 0 else (2 if iv > 2 else iv)


def _set_edge_mode(state: GraphState, e: Edge, mode: EdgeMode) -> None:
    state.edge_enabled[e] = int(mode)


def _get_edge_mode(state: GraphState, e: Edge) -> EdgeMode:
    return _normalize_edge_mode(state.edge_enabled.get(e))


def _cycle_edge_mode(state: GraphState, e: Edge) -> EdgeMode:
    cur = _get_edge_mode(state, e)
    nxt = (cur + 1) % 3
    _set_edge_mode(state, e, nxt)
    return nxt


def normalize_graph_state(state: GraphState) -> None:
    """Normalize and prune graph state in-place.

    - Drops edges whose endpoints are missing.
    - Drops backward/self edges (not representable in the UI).
    - Normalizes all modes to integers 0/1/2.
    """
    idx = {n: i for i, n in enumerate(state.nodes)}
    before = dict(state.edge_enabled or {})
    cleaned: Dict[Edge, EdgeMode] = {}
    for (a, b), v in before.items():
        if a not in idx or b not in idx:
            continue
        if idx[a] >= idx[b]:
            continue
        cleaned[(a, b)] = _normalize_edge_mode(v)
    state.edge_enabled = cleaned

    if log.isEnabledFor(TRACE):
        # TRACE: show pruning details (kept small)
        removed = [e for e in before.keys() if e not in cleaned]
        if removed:
            log.trace('[GRAPH] normalized edges: removed=%s kept=%s example_removed=%s', len(removed), len(cleaned), removed[:5])


def _ensure_defaults_for_missing_edges(state: GraphState) -> None:
    normalize_graph_state(state)
    nodes = list(state.nodes)
    default_path = {(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            e = (nodes[i], nodes[j])
            if e not in state.edge_enabled:
                _set_edge_mode(state, e, 1 if e in default_path else 0)
            else:
                _set_edge_mode(state, e, _get_edge_mode(state, e))


def _mid(n: str) -> str:
    return f"n_{abs(hash(n))}"


def _node_number_map(state: GraphState) -> Dict[str, int]:
    return {name: i + 1 for i, name in enumerate(state.nodes)}


def _node_display_label(state: GraphState, node_name: str) -> str:
    """Return a UI-only node label with a 1..N order prefix.

    Important: This is *only* for mermaid rendering. Node names used elsewhere
    (tables, generated plots, ODE generation) stay unchanged.
    """
    num = _node_number_map(state).get(node_name, 0)
    safe = node_name.replace('"', "\\\"")
    return f"{num}. {safe}" if num else safe


def _edge_color(mode: EdgeMode) -> str:
    return {0: '#94a3b8', 1: '#16a34a', 2: '#dc2626'}.get(mode, '#94a3b8')


def _edge_opacity(mode: EdgeMode) -> float:
    return 0.12 if mode == 0 else 1.0


# -----------------------------------------------------------------------------
# Compact rendering (cards)
# -----------------------------------------------------------------------------


def build_compact_mermaid(state: GraphState) -> str:
    """Compact graph for cards (only enabled edges)."""
    _ensure_defaults_for_missing_edges(state)
    nodes = list(state.nodes)
    num = _node_number_map(state)

    shown_edges: List[Edge] = []
    shown_modes: List[EdgeMode] = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            e = (nodes[i], nodes[j])
            mode = _get_edge_mode(state, e)
            if mode != 0:
                shown_edges.append(e)
                shown_modes.append(mode)

    lines = ['flowchart LR']
    for n in nodes:
        lines.append(f'{_mid(n)}["{_node_display_label(state, n)}"]')

    for idx, (a, b) in enumerate(shown_edges):
        mode = shown_modes[idx]
        lines.append(f"{_mid(a)} -->|k_{num[a]}->{num[b]}| {_mid(b)}")
        lines.append(
            f"linkStyle {idx} stroke:{_edge_color(mode)},stroke-width:4px,stroke-opacity:{_edge_opacity(mode)}"
        )

    return '\n'.join(lines)


# -----------------------------------------------------------------------------
# Interactive widget
# -----------------------------------------------------------------------------


_EDGE_EVENT_NAME = 'compute_builder_edge_click'
_EDGE_EVENT_REGISTERED = False
_EDGE_CALLBACKS: Dict[str, Callable[[int], None]] = {}


def _ensure_edge_event_registered() -> None:
    global _EDGE_EVENT_REGISTERED
    if _EDGE_EVENT_REGISTERED:
        return

    def _dispatch(e) -> None:  # e: GenericEventArguments
        args = getattr(e, 'args', None) or {}
        dom_id = args.get('dom_id')
        idx = args.get('edge_index')
        if dom_id is None or idx is None:
            return
        cb = _EDGE_CALLBACKS.get(dom_id)
        if cb is None:
            return
        try:
            cb(int(idx))
        except Exception:
            return

    ui.on(_EDGE_EVENT_NAME, _dispatch)
    _EDGE_EVENT_REGISTERED = True


def create_graph_widget(state: GraphState, key: str, title: Optional[str] = None) -> GraphController:
    """Interactive mermaid graph.

    Clicking an edge cycles: disabled → main → control → ...

    Implementation notes:
    - Uses a single global event to avoid "Event listeners changed" warnings.
    - Preserves scroll position and prevents "jump" by pinning min-height during re-render.
    """

    _ensure_edge_event_registered()

    dom_id = f"mermaid_graph_{key}".replace(':', '_').replace('/', '_')
    controller = GraphController(state, dom_id)

    current_edges: List[Edge] = []
    mermaid_el: Optional[ui.mermaid] = None

    def build_mermaid_full() -> str:
        nonlocal current_edges
        _ensure_defaults_for_missing_edges(state)
        nodes = list(state.nodes)
        num = _node_number_map(state)

        current_edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                current_edges.append((nodes[i], nodes[j]))

        lines = ['flowchart LR']
        for n in nodes:
            lines.append(f'{_mid(n)}["{_node_display_label(state, n)}"]')

        for idx, (a, b) in enumerate(current_edges):
            lines.append(f"{_mid(a)} -->|k_{num[a]}->{num[b]}| {_mid(b)}")
            mode = _get_edge_mode(state, (a, b))
            lines.append(
                f"linkStyle {idx} stroke:{_edge_color(mode)},stroke-width:4px,stroke-opacity:{_edge_opacity(mode)}"
            )

        return '\n'.join(lines)

    def wire_js_click_handlers() -> None:
        # Use JS polling (no Python timers) to avoid "parent slot deleted" errors.
        ui.run_javascript(
            f"""
            (function(){{
              const domId = {dom_id!r};
              const intervalKey = '__compute_builder_wire_interval__' + domId;
              if (window[intervalKey]) {{ clearInterval(window[intervalKey]); }}

              window[intervalKey] = setInterval(() => {{
                const root = document.getElementById(domId);
                if (!root) return;
                const svg = root.querySelector('svg');
                if (!svg) return;

                const paths = root.querySelectorAll('g.edgePaths path.path');
                const labels = root.querySelectorAll('g.edgeLabels g.edgeLabel');

                paths.forEach((p, idx) => {{
                  p.style.cursor = 'pointer';
                  p.onclick = () => emitEvent({_EDGE_EVENT_NAME!r}, {{ dom_id: domId, edge_index: idx }});
                }});
                labels.forEach((l, idx) => {{
                  l.style.cursor = 'pointer';
                  l.onclick = () => emitEvent({_EDGE_EVENT_NAME!r}, {{ dom_id: domId, edge_index: idx }});
                }});

                clearInterval(window[intervalKey]);
                window[intervalKey] = null;
              }}, 60);
            }})();
            """
        )

    def refresh_all() -> None:
        nonlocal mermaid_el
        if mermaid_el is None:
            return
        mermaid_el.content = build_mermaid_full()
        mermaid_el.update()
        wire_js_click_handlers()

    def refresh_all_preserve_scroll() -> None:
        ui.run_javascript(
            f"""
            (function(){{
              const el = document.getElementById({dom_id!r});
              window.__compute_builder_scrollY = window.scrollY;
              if (el) {{
                window.__compute_builder_prevH = el.offsetHeight;
                el.style.minHeight = (window.__compute_builder_prevH || 0) + 'px';
              }}
            }})();
            """
        )

        refresh_all()

        ui.run_javascript(
            f"""
            (function(){{
              const domId = {dom_id!r};
              const key = '__compute_builder_restore_interval__' + domId;
              if (window[key]) {{ clearInterval(window[key]); }}
              window[key] = setInterval(() => {{
                const el = document.getElementById(domId);
                const svg = el ? el.querySelector('svg') : null;
                if (!svg) return;

                if (el) el.style.minHeight = '';

                const y = window.__compute_builder_scrollY;
                if (typeof y === 'number') window.scrollTo({{ top: y, behavior: 'auto' }});

                clearInterval(window[key]);
                window[key] = null;
              }}, 60);
            }})();
            """
        )

    def on_edge_click(edge_index: int) -> None:
        if 0 <= edge_index < len(current_edges):
            _cycle_edge_mode(state, current_edges[edge_index])
            refresh_all_preserve_scroll()

    # register callback for this DOM id
    _EDGE_CALLBACKS[dom_id] = on_edge_click

    if title:
        ui.label(title).classes('text-h6')

    ui.label('Klik na hrany: vypnuto → hlavní → kontrolní → ...').classes('text-body2 text-grey-7')
    ui.separator()

    mermaid_el = ui.mermaid(build_mermaid_full(), config={'securityLevel': 'loose'}).props(f'id={dom_id}').classes(
        'w-full'
    )

    wire_js_click_handlers()

    controller._refresh = refresh_all_preserve_scroll
    return controller
