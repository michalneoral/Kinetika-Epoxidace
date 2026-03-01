from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class KineticsResult:
    model_name: str
    params: Dict[str, float]
    debug_df: pd.DataFrame


class TableProcessor:
    """
    Dummy processor:
      - drží FAME/EPO data
      - process(): vygeneruje pár "zpracovaných tabulek"
      - compute_all_kinetics(): vygeneruje "výsledky modelů"
    """

    def __init__(self) -> None:
        self.fame_df: Optional[pd.DataFrame] = None
        self.epo_df: Optional[pd.DataFrame] = None
        self.processed: Dict[str, pd.DataFrame] = {}
        self.kinetics: Dict[str, KineticsResult] = {}

    def has_data(self) -> bool:
        return self.fame_df is not None and self.epo_df is not None

    def add_fame(self, df: pd.DataFrame) -> None:
        self.fame_df = df.copy()

    def add_epo(self, df: pd.DataFrame) -> None:
        self.epo_df = df.copy()

    def process(self) -> None:
        if not self.has_data():
            self.processed = {}
            return

        # Dummy "processing"
        fame = self.fame_df.copy()
        epo = self.epo_df.copy()

        self.processed = {
            "FAME (raw cut)": fame,
            "EPO (raw cut)": epo,
            "FAME summary": fame.describe(include="all").fillna("").head(10),
            "EPO summary": epo.describe(include="all").fillna("").head(10),
        }

    def compute_all_kinetics(
        self,
        models_to_compute: List[str],
        init_mode: str,
        optim_time_shift: bool,
        t_shift: float,
    ) -> None:
        if not self.has_data():
            self.kinetics = {}
            return

        results: Dict[str, KineticsResult] = {}
        for m in models_to_compute:
            params = {
                "k": round(random.uniform(0.01, 3.0), 4),
                "t0": round(random.uniform(0.0, 5.0), 4),
                "rss": round(random.uniform(0.0, 1.0), 6),
            }
            n = 60
            debug_df = pd.DataFrame(
                {
                    "t": [i / 10 for i in range(n)],
                    "y_obs": [random.random() for _ in range(n)],
                    "y_fit": [random.random() for _ in range(n)],
                }
            )
            results[m] = KineticsResult(model_name=m, params=params, debug_df=debug_df)

        self.kinetics = results
