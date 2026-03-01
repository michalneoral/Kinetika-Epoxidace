from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import json

from nicegui import ui

Edge = Tuple[str, str]
EdgeMode = int  # 0=disabled, 1=main, 2=control


@dataclass
class GraphState:
    nodes: List[str] = field(default_factory=list)                 # pořadí uzlů (názvy) = řízené drag&dropem
    node_enabled: Dict[str, bool] = field(default_factory=dict)    # ponecháno kvůli zpětné kompatibilitě (už se nepoužívá)
    edge_enabled: Dict[Edge, Union[bool, EdgeMode]] = field(default_factory=dict)
    # ^ zpětná kompatibilita: dřív bool, teď int 0/1/2


class GraphController:
    def __init__(self, state: GraphState) -> None:
        self.state = state
        self._refresh: Optional[callable] = None

    def refresh(self) -> None:
        if self._refresh is not None:
            self._refresh()

    def get_state(self) -> GraphState:
        return self.state


# ----------------------------
# Helpers
# ----------------------------
def _normalize_edge_mode(v: Union[bool, int, None]) -> EdgeMode:
    """Zpětná kompatibilita:
    - None -> 0
    - False -> 0
    - True -> 1
    - int -> clamp 0..2
    """
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


def _ensure_defaults_for_missing_edges(state: GraphState) -> None:
    """Doplní hrany pro aktuální state.nodes:
    default: sousední hrany => MAIN (1), ostatní => DISABLED (0)
    """
    nodes = list(state.nodes)
    default_path = {(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            e = (nodes[i], nodes[j])
            if e not in state.edge_enabled:
                _set_edge_mode(state, e, 1 if e in default_path else 0)
            else:
                # normalizuj staré bool/int
                _set_edge_mode(state, e, _get_edge_mode(state, e))


def _mid(n: str) -> str:
    return f"n_{abs(hash(n))}"


def _node_number_map(state: GraphState) -> Dict[str, int]:
    return {name: i + 1 for i, name in enumerate(state.nodes)}


def _edge_color(mode: EdgeMode) -> str:
    return {0: "#94a3b8", 1: "#16a34a", 2: "#dc2626"}.get(mode, "#94a3b8")


def _edge_opacity(mode: EdgeMode) -> float:
    return 0.12 if mode == 0 else 1.0


# ----------------------------
# Public: compact mermaid
# ----------------------------
def build_compact_mermaid(state: GraphState) -> str:
    """Kompaktní Mermaid na hlavní stránku:
    - uzly: všechny v state.nodes
    - hrany: jen MAIN + CONTROL (disabled se neukazuje)
    """
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

    lines = ["flowchart LR"]
    for n in nodes:
        label = n.replace('"', '\\"')
        lines.append(f'{_mid(n)}["{label}"]')

    for idx, (a, b) in enumerate(shown_edges):
        mode = shown_modes[idx]
        lines.append(f"{_mid(a)} -->|k_{num[a]}->{num[b]}| {_mid(b)}")
        lines.append(
            f"linkStyle {idx} stroke:{_edge_color(mode)},stroke-width:4px,stroke-opacity:{_edge_opacity(mode)}"
        )

    return "\n".join(lines)


# ----------------------------
# Public: interactive widget
# ----------------------------
def create_graph_widget(state: GraphState, key: str, title: Optional[str] = None) -> GraphController:
    """Interaktivní Mermaid graf:
    - uzly = state.nodes (řízení drag&drop)
    - klik na hrany: cyklus disabled->main->control->disabled
    - legenda barev
    """
    controller = GraphController(state)

    event_edge_click = f"edge_click_{key}"
    dom_id = f"mermaid_graph_{key}"

    current_edges: List[Edge] = []
    mermaid_edit = None

    def build_mermaid_edit() -> str:
        nonlocal current_edges

        _ensure_defaults_for_missing_edges(state)
        nodes = list(state.nodes)
        num = _node_number_map(state)

        # všechny hrany i<j (kvůli klikání)
        current_edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                current_edges.append((nodes[i], nodes[j]))

        lines = ["flowchart LR"]
        for n in nodes:
            label = n.replace('"', '\\"')
            lines.append(f'{_mid(n)}["{label}"]')

        for idx, (a, b) in enumerate(current_edges):
            lines.append(f"{_mid(a)} -->|k_{num[a]}->{num[b]}| {_mid(b)}")

        for idx, e in enumerate(current_edges):
            mode = _get_edge_mode(state, e)
            lines.append(
                f"linkStyle {idx} stroke:{_edge_color(mode)},stroke-width:4px,stroke-opacity:{_edge_opacity(mode)}"
            )

        return "\n".join(lines)

    def wire_js_for_edge_click() -> None:
        modes = [_get_edge_mode(state, e) for e in current_edges]
        modes_json = json.dumps(modes)

        js = f"""
        (() => {{
          const root = document.getElementById('{dom_id}');
          if (!root) return;
          const modes = {modes_json};

          const apply = () => {{
            const svg = root.querySelector('svg');
            if (!svg) return false;

            const edgePaths = Array.from(svg.querySelectorAll('g.edgePath'));
            const edgeLabels = Array.from(svg.querySelectorAll('g.edgeLabel'));

            const opacity = (m) => m===0 ? '0.45' : '1';

            edgePaths.forEach((g, idx) => {{
              g.style.cursor = 'pointer';
              g.onclick = () => emitEvent('{event_edge_click}', idx);
              const m = modes[idx] ?? 0;
              g.style.opacity = opacity(m);
            }});

            edgeLabels.forEach((g, idx) => {{
              g.style.cursor = 'pointer';
              g.onclick = () => emitEvent('{event_edge_click}', idx);
              const m = modes[idx] ?? 0;
              g.style.opacity = opacity(m);
            }});

            return true;
          }};

          let attempts = 0;
          const timer = setInterval(() => {{
            attempts++;
            if (apply() || attempts > 30) clearInterval(timer);
          }}, 50);
        }})();
        """
        ui.run_javascript(js)

    def refresh_all() -> None:
        mermaid_edit.content = build_mermaid_edit()
        ui.timer(0.1, wire_js_for_edge_click, once=True)

    def on_edge_click(e) -> None:
        idx = int(e.args)
        if 0 <= idx < len(current_edges):
            edge = current_edges[idx]
            _cycle_edge_mode(state, edge)
            refresh_all()

    # UI
    if title:
        ui.label(title).classes("text-h6")

    ui.label("Klik na hrany: vypnuto → hlavní → kontrolní → ...").classes("text-body2 text-grey-7")

    # Legenda
    with ui.row().classes("items-center gap-4 q-mt-sm"):
        with ui.row().classes("items-center gap-2"):
            ui.element("div").classes("w-4 h-4 rounded").style("background:#94a3b8")
            ui.label("vypnuto").classes("text-body2 text-grey-8")
        with ui.row().classes("items-center gap-2"):
            ui.element("div").classes("w-4 h-4 rounded").style("background:#16a34a")
            ui.label("hlavní výpočet").classes("text-body2 text-grey-8")
        with ui.row().classes("items-center gap-2"):
            ui.element("div").classes("w-4 h-4 rounded").style("background:#dc2626")
            ui.label("kontrolní výpočet").classes("text-body2 text-grey-8")

    ui.separator()
    mermaid_edit = ui.mermaid(build_mermaid_edit(), config={"securityLevel": "loose"}).props(f"id={dom_id}").classes("w-full")

    ui.on(event_edge_click, on_edge_click)
    ui.timer(0.1, wire_js_for_edge_click, once=True)

    controller._refresh = refresh_all
    return controller
