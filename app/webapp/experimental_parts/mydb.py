from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import json
import sqlite3

from nody_v_grafu import GraphState

DB_PATH = Path(__file__).with_name("computations.sqlite3")


@dataclass
class DbComputation:
    id: int
    name: str
    table_name: str
    used_heads: List[str]
    graph_state: GraphState


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db() -> None:
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS computations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                table_name TEXT NOT NULL,
                used_heads_json TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                edge_modes_json TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )
        con.commit()


def _clamp_mode(v: Any) -> int:
    # zpětná kompatibilita: bool -> 0/1
    if isinstance(v, bool):
        return 1 if v else 0
    try:
        iv = int(v)
    except Exception:
        return 0
    return 0 if iv < 0 else (2 if iv > 2 else iv)


def _graphstate_to_payload(gs: GraphState) -> Tuple[str, str]:
    nodes_json = json.dumps(gs.nodes, ensure_ascii=False)

    # edges: [[a,b,mode], ...]
    edges_out: List[list] = []
    for (a, b), v in gs.edge_enabled.items():
        edges_out.append([a, b, _clamp_mode(v)])

    edge_modes_json = json.dumps(edges_out, ensure_ascii=False)
    return nodes_json, edge_modes_json


def _graphstate_from_payload(nodes_json: str, edge_modes_json: str) -> GraphState:
    nodes = json.loads(nodes_json) if nodes_json else []
    edges_raw = json.loads(edge_modes_json) if edge_modes_json else []

    gs = GraphState(nodes=list(nodes))
    for item in edges_raw:
        if not isinstance(item, list) or len(item) != 3:
            continue
        a, b, mode = item
        gs.edge_enabled[(a, b)] = _clamp_mode(mode)
    return gs


def insert_computation(name: str, table_name: str, used_heads: List[str], gs: GraphState) -> int:
    used_heads_json = json.dumps(used_heads, ensure_ascii=False)
    nodes_json, edge_modes_json = _graphstate_to_payload(gs)

    with _connect() as con:
        cur = con.execute(
            """
            INSERT INTO computations(name, table_name, used_heads_json, nodes_json, edge_modes_json, updated_at)
            VALUES(?,?,?,?,?, datetime('now'))
            """,
            (name, table_name, used_heads_json, nodes_json, edge_modes_json),
        )
        con.commit()
        return int(cur.lastrowid)


def update_computation(cid: int, name: str, table_name: str, used_heads: List[str], gs: GraphState) -> None:
    used_heads_json = json.dumps(used_heads, ensure_ascii=False)
    nodes_json, edge_modes_json = _graphstate_to_payload(gs)

    with _connect() as con:
        con.execute(
            """
            UPDATE computations
            SET name=?, table_name=?, used_heads_json=?, nodes_json=?, edge_modes_json=?, updated_at=datetime('now')
            WHERE id=?
            """,
            (name, table_name, used_heads_json, nodes_json, edge_modes_json, cid),
        )
        con.commit()


def delete_computation(cid: int) -> None:
    with _connect() as con:
        con.execute("DELETE FROM computations WHERE id=?", (cid,))
        con.commit()


def load_all() -> List[DbComputation]:
    with _connect() as con:
        rows = con.execute(
            "SELECT id, name, table_name, used_heads_json, nodes_json, edge_modes_json FROM computations ORDER BY id"
        ).fetchall()

    out: List[DbComputation] = []
    for r in rows:
        used_heads = json.loads(r["used_heads_json"]) if r["used_heads_json"] else []
        gs = _graphstate_from_payload(r["nodes_json"], r["edge_modes_json"])
        out.append(
            DbComputation(
                id=int(r["id"]),
                name=str(r["name"]),
                table_name=str(r["table_name"]),
                used_heads=used_heads,
                graph_state=gs,
            )
        )
    return out
