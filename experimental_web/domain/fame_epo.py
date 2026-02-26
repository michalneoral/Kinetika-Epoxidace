from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import pandas as pd

# Correct required values (from your message / old project)
# - For FAME some items have acceptable aliases -> list[str]
REQUIRED_FAME: List[Union[str, List[str]]] = [
    ["C17", "C17:0"],
    ["C18", "C18:0"],
    "C18:1 I",
    "C18:1 II",
    "C18:2",
    "C18:3",
    "C20:1",
    "C22:1",
]

REQUIRED_EPO: List[str] = [
    "C18:1 EPO",
    "C18:2 1-EPO I",
    "C18:2 1-EPO II",
    "C18:3 1-EPO I",
    "C18:3 1-EPO II",
    "C18:3 1-EPO III",
    "C20:1 1-EPO",
    "C18:2 2-EPO",
    "C18:3 2-EPO I",
]


@dataclass
class TablePickConfig:
    row_start: int
    row_end: int
    col_start: int
    col_end: int


# Defaults (1-indexed), same as your old project config
DEFAULT_FAME = TablePickConfig(row_start=3, row_end=16, col_start=2, col_end=15)
DEFAULT_EPO = TablePickConfig(row_start=28, row_end=37, col_start=2, col_end=15)


def extract_df(df: pd.DataFrame, row_start: int, row_end: int, col_start: int, col_end: int) -> pd.DataFrame:
    """Extract a rectangular region (1-indexed) and treat first row as header."""
    table = df.iloc[row_start - 1:row_end, col_start - 1:col_end].copy()
    if table.empty:
        return table
    table.columns = table.iloc[0]
    return table[1:]


def extract_df_dict(df: pd.DataFrame, row_start: int, row_end: int, col_start: int, col_end: int) -> Dict:
    return extract_df(df, row_start, row_end, col_start, col_end).to_dict(orient="list")


def _norm(x: object) -> str:
    return str(x).strip().lower()


def validate_table(table_dict: Dict, required_values: List[Union[str, List[str]]]) -> Tuple[bool, List[str]]:
    """Validate that all required items exist in the first column (case-insensitive).

    required_values entries can be:
    - str: exact required label
    - list[str]: any of the aliases is acceptable (OR)
    Returns (ok, missing_list) where missing_list contains human-friendly names.
    """
    if not table_dict:
        return False, [r[0] if isinstance(r, list) and r else str(r) for r in required_values]

    first_col = next(iter(table_dict.keys()), None)
    if not first_col:
        return False, [r[0] if isinstance(r, list) and r else str(r) for r in required_values]

    present = {_norm(v) for v in table_dict.get(first_col, []) if _norm(v)}

    missing: List[str] = []
    for r in required_values:
        if isinstance(r, list):
            aliases = [_norm(a) for a in r]
            if not any(a in present for a in aliases):
                # show the first alias as "main" label
                missing.append(r[0])
        else:
            if _norm(r) not in present:
                missing.append(str(r))

    return (len(missing) == 0), missing
