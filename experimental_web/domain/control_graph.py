from __future__ import annotations

"""Utilities for splitting a graph computation into main/control parts.

Edge modes:
  0 = disabled
  1 = main
  2 = control

The "control" computation is defined as the induced subgraph formed by all edges
with mode=2 and the nodes they connect.
"""

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

Edge = Tuple[str, str]


def parse_edge_modes_payload(edge_modes: object) -> List[Tuple[str, str, int]]:
    """Parse edge mode payload (list of [a,b,mode]) into a normalized list."""
    out: List[Tuple[str, str, int]] = []
    if not isinstance(edge_modes, list):
        return out
    for item in edge_modes:
        if not isinstance(item, list) or len(item) != 3:
            continue
        a, b, m = item
        try:
            mm = int(m)
        except Exception:
            mm = 0
        out.append((str(a), str(b), 0 if mm < 0 else (2 if mm > 2 else mm)))
    return out


def control_subgraph(
    nodes: Sequence[str],
    edge_modes_payload: object,
    *,
    mode: int = 2,
) -> Tuple[List[str], Dict[Edge, int]]:
    """Return nodes + edge_modes mapping for the induced control subgraph.

    Nodes are kept in the original order.
    Only forward edges (a before b) are kept.
    """
    edges = parse_edge_modes_payload(edge_modes_payload)

    # collect nodes participating in control edges
    node_set: set[str] = set()
    for a, b, m in edges:
        if m == mode:
            node_set.add(a)
            node_set.add(b)

    if not node_set:
        return [], {}

    nodes_control = [str(n) for n in nodes if str(n) in node_set]

    idx = {n: i for i, n in enumerate(nodes_control)}
    em: Dict[Edge, int] = {}
    for a, b, m in edges:
        if m != mode:
            continue
        if a not in idx or b not in idx:
            continue
        if idx[a] >= idx[b]:
            continue
        em[(a, b)] = mode

    return nodes_control, em
