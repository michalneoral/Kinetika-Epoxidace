from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import re
import unicodedata

Edge = Tuple[str, str]


@dataclass(frozen=True)
class ODEModel:
    """Pure-text representation of an ODE system generated from a directed graph."""
    ode_text: str
    state_names: List[str]
    param_names: List[str]
    node_id_map: Dict[str, str]          # original node label -> safe python identifier
    param_id_map: Dict[Edge, str]        # (from_label, to_label) -> parameter identifier


_NON_IDENT = re.compile(r"[^0-9A-Za-z_]+")


def sanitize_identifier(label: str, *, prefix: str = "x") -> str:
    r"""Convert an arbitrary label to a safe Python identifier (ASCII, [A-Za-z_]\w*)."""
    s = (label or "").strip()
    # remove diacritics (č -> c, ě -> e, ...)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _NON_IDENT.sub("_", s).strip("_")
    if not s:
        s = prefix
    if s[0].isdigit():
        s = f"{prefix}_{s}"
    return s


def _make_unique(names: Sequence[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for n in names:
        base = n
        if base not in seen:
            seen[base] = 1
            out.append(base)
            continue
        seen[base] += 1
        out.append(f"{base}_{seen[base]}")
    return out


def param_name(from_state: str, to_state: str) -> str:
    """Create a parameter identifier from two state identifiers.

    For single-letter states (e.g. U->M) it yields k_um.
    Otherwise it yields k_<from>_<to> (lowercased), e.g. k_c18_1_c18_2.
    """
    a = from_state.lower()
    b = to_state.lower()
    if len(a) == 1 and len(b) == 1:
        return f"k_{a}{b}"
    return f"k_{a}_{b}"


def generate_ode_model(
    nodes: Sequence[str],
    edge_modes: Mapping[Edge, int],
    *,
    include_modes: Iterable[int] = (1, 2),
) -> ODEModel:
    """Generate ODE equations (mass-transfer first-order terms) from a directed graph.

    Each enabled edge A->B contributes:
      dA += -k_ab * A
      dB +=  k_ab * A

    The output is compatible with `load_odes_from_txt` provided by the user:
      dU = ...
      dM = ...
    """
    # build stable state identifiers in node order
    raw_ids = [sanitize_identifier(n, prefix="s") for n in nodes]
    ids = _make_unique(raw_ids)
    node_id_map = {n: i for n, i in zip(nodes, ids)}
    state_names = ids[:]  # order matters

    include_modes_set = set(include_modes)

    # determine enabled edges, keep deterministic order by node order
    node_index = {n: idx for idx, n in enumerate(nodes)}
    enabled_edges: List[Edge] = [
        e for e, mode in edge_modes.items()
        if (
            mode in include_modes_set
            and e[0] in node_index
            and e[1] in node_index
            and node_index[e[0]] < node_index[e[1]]  # only forward edges (UI cannot represent backward ones)
        )
    ]
    enabled_edges.sort(key=lambda e: (node_index[e[0]], node_index[e[1]]))

    # map edges -> parameter ids
    param_id_map: Dict[Edge, str] = {}
    raw_params: List[str] = []
    for a, b in enabled_edges:
        p = param_name(node_id_map[a], node_id_map[b])
        raw_params.append(p)
        param_id_map[(a, b)] = p
    param_names = _make_unique(raw_params)

    # if there were duplicates, update map to unique ones based on position
    for (a, b), p_unique in zip(enabled_edges, param_names):
        param_id_map[(a, b)] = p_unique

    # prepare incoming/outgoing lists
    incoming: Dict[str, List[Tuple[str, str]]] = {n: [] for n in nodes}  # node -> [(src, param)]
    outgoing: Dict[str, List[Tuple[str, str]]] = {n: [] for n in nodes}  # node -> [(dst, param)]
    for a, b in enabled_edges:
        p = param_id_map[(a, b)]
        outgoing[a].append((b, p))
        incoming[b].append((a, p))

    def _join_terms(terms: List[str]) -> str:
        if not terms:
            return "0"
        expr = " + ".join(terms)
        expr = expr.replace("+ -", "- ")
        return expr

    lines: List[str] = []
    lines.append("# Generated from computation graph")
    lines.append("# state_names: " + ", ".join(state_names))
    lines.append("# param_names: " + ", ".join(param_names))
    lines.append("")

    for node in nodes:
        sid = node_id_map[node]
        terms: List[str] = []
        # outflow
        for _dst, p in outgoing[node]:
            terms.append(f"-{p}*{sid}")
        # inflow
        for src, p in incoming[node]:
            src_id = node_id_map[src]
            terms.append(f"{p}*{src_id}")
        lines.append(f"d{sid} = {_join_terms(terms)}")

    ode_text = "\n".join(lines).strip() + "\n"
    return ODEModel(
        ode_text=ode_text,
        state_names=state_names,
        param_names=param_names,
        node_id_map=node_id_map,
        param_id_map=param_id_map,
    )
