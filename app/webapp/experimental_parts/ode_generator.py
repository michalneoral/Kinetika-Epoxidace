from __future__ import annotations

from typing import Dict, List, Tuple, Union
from nody_v_grafu import GraphState

Edge = Tuple[str, str]


def _mode(v: Union[bool, int, None]) -> int:
    """Zpětná kompatibilita: bool -> 0/1, int -> clamp 0..2."""
    if v is None:
        return 0
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        iv = int(v)
    except Exception:
        return 0
    return 0 if iv < 0 else (2 if iv > 2 else iv)


def _collect_edges(gs: GraphState) -> Dict[int, List[Edge]]:
    """Vrátí {1: [edges_main], 2: [edges_control]} (0 ignoruje)."""
    out: Dict[int, List[Edge]] = {1: [], 2: []}
    for (a, b), v in gs.edge_enabled.items():
        m = _mode(v)
        if m in (1, 2):
            out[m].append((a, b))
    return out


def _format_sum(terms: List[str]) -> str:
    if not terms:
        return "0"
    # poskládáme s +, znaménka jsou už v termu
    expr = " + ".join(terms)
    # kosmetika: "+ -X" -> "- X"
    expr = expr.replace("+ -", "- ")
    return expr


def _k_name(i_from: int, i_to: int) -> str:
    return f"k_{i_from}->{i_to}"


def generate_odes_text(gs: GraphState, *, variable_prefix: str = "y") -> str:
    """
    Vygeneruje text ODE pro GraphState.
    Předpoklad: každá hrana i->j je 1. řád: v = k_{i->j} * y_i.
    Uzly jsou gs.nodes v pořadí => indexy 1..N.
    Výstup obsahuje rovnice pro MAIN i CONTROL zvlášť.
    """
    nodes = list(gs.nodes)
    n = len(nodes)
    idx = {name: i + 1 for i, name in enumerate(nodes)}  # 1-based

    edges_by_mode = _collect_edges(gs)
    edges_main = edges_by_mode[1]
    edges_ctrl = edges_by_mode[2]

    def build_system(edges: List[Edge], label: str) -> List[str]:
        inflow: Dict[str, List[str]] = {name: [] for name in nodes}
        outflow: Dict[str, List[str]] = {name: [] for name in nodes}

        for a, b in edges:
            ia, ib = idx[a], idx[b]
            k = _k_name(ia, ib)
            ya = f"{variable_prefix}{ia}"
            term = f"{k}*{ya}"
            outflow[a].append(term)
            inflow[b].append(term)

        lines: List[str] = []
        lines.append(f"# {label}")
        lines.append("# mapping uzlů:")
        for name in nodes:
            lines.append(f"#   {variable_prefix}{idx[name]} = {name}")

        lines.append("")
        for name in nodes:
            i = idx[name]
            rhs_terms: List[str] = []
            rhs_terms += inflow[name]
            rhs_terms += [f"-{t}" for t in outflow[name]]
            rhs = _format_sum(rhs_terms)
            lines.append(f"d{variable_prefix}{i}/dt = {rhs}")
        return lines

    out_lines: List[str] = []
    out_lines.append("## ODE systém (1. řád: v = k * y_odkud)")
    out_lines.append("")

    if edges_main:
        out_lines += build_system(edges_main, "MAIN (zelené hrany)")
        out_lines.append("")
    else:
        out_lines.append("# MAIN (zelené hrany) – žádné aktivní hrany")
        out_lines.append("")

    if edges_ctrl:
        out_lines += build_system(edges_ctrl, "CONTROL (červené hrany)")
        out_lines.append("")
    else:
        out_lines.append("# CONTROL (červené hrany) – žádné aktivní hrany")
        out_lines.append("")

    return "\n".join(out_lines).rstrip()
