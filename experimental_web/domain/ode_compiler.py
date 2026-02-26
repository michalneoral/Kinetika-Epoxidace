from __future__ import annotations

import ast
from io import StringIO
from math import cos, exp, log, sin, sqrt
from typing import Callable, Dict, Iterable, List, Optional, Sequence

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.USub, ast.UAdd, ast.Load, ast.Name, ast.Constant, ast.Call
)

_ALLOWED_FUNCS = {
    "exp": exp,
    "log": log,
    "sin": sin,
    "cos": cos,
    "sqrt": sqrt,
}


def _compile_expr(expr: str, allowed_names: set):
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Nepovolená konstrukce ve výrazu: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Nepovolené jméno ve výrazu: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Povoleny jsou jen funkce: " + ", ".join(_ALLOWED_FUNCS))
    code = compile(tree, "<ode_expr>", "eval")
    return code


def load_odes_from_txt(path: str, state_names, param_names, order=None):
    """Compatibility function: same signature as user's snippet (kept as-is, but safe)."""
    with open(path, "r", encoding="utf-8") as f:
        return load_odes_from_text(f.read(), state_names, param_names, order=order)


def load_odes_from_text(text: str, state_names, param_names, order=None):
    """
    state_names: např. ["U","M","D","T","H"]
    param_names: např. ["k_um","k_md","k_dt","k_mh","k_dh","k_th"]
    order: pořadí derivací v návratu; default = ["d"+s for s in state_names]
    """
    if order is None:
        order = ["d" + s for s in state_names]

    # načíst dX = expr
    equations: Dict[str, str] = {}
    f = StringIO(text)
    for raw in f:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Řádek bez '=': {line}")
        lhs, rhs = [x.strip() for x in line.split("=", 1)]
        equations[lhs] = rhs

    # validace
    for key in order:
        if key not in equations:
            raise ValueError(f"Chybí rovnice pro {key}")

    allowed_names = set(state_names) | set(param_names) | set(_ALLOWED_FUNCS.keys()) | {"t"}

    compiled = {k: _compile_expr(equations[k], allowed_names) for k in order}

    def odes(t, conc, k):
        env = {}
        env["t"] = t
        # stavy
        for name, val in zip(state_names, conc):
            env[name] = val
        # parametry
        for name, val in zip(param_names, k):
            env[name] = val
        # funkce
        env.update(_ALLOWED_FUNCS)

        return [eval(compiled[key], {"__builtins__": {}}, env) for key in order]

    return odes


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def compile_ode_equations(ode_text: str, state_names, param_names, order=None):
    """Backwards-compatible alias.

    Some parts of the app (e.g. the graph-based ODE fitting) historically
    imported `compile_ode_equations`. The refactor introduced
    `load_odes_from_text` instead. Keep this wrapper so older imports keep
    working.
    """
    return load_odes_from_text(ode_text, state_names=state_names, param_names=param_names, order=order)
