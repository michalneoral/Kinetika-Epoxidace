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


@dataclass(frozen=True)
class ExperimentFile:
    id: int
    experiment_id: int
    filename: str
    content: bytes
    sha256: Optional[str]
    size_bytes: int
    uploaded_at: str
    selected_sheet: Optional[str] = None
