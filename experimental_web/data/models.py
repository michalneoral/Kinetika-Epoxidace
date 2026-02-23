from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Experiment:
    id: int
    name: str
    created_at: str
    updated_at: str
    folder: Optional[str] = None
