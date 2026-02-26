from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from experimental_web.domain.kinetic_model import InitConditions


@dataclass
class ProcessingConfig:
    initialization: InitConditions = InitConditions.TIME_SHIFT
    t_shift: float = 6.0
    optim_time_shift: bool = False

    # Maximum simulated time for fitting/simulation (clamped to available max time in data).
    t_max: float = 400.0

    # Maximum x-axis limit for plots (xlim max). This does NOT change fitting/simulation,
    # it only affects rendering. Clamped to available max time in data.
    t_max_plot: float = 400.0

    models_to_compute: List[str] = field(default_factory=list)
