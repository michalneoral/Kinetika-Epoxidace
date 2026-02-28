from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from experimental_web.core.paths import DB_PATH
from experimental_web.core.time import parse_datetime_any
from experimental_web.data.repositories import (
    ExperimentComputationRepository,
    ExperimentGraphSettingsRepository,
    ExperimentProcessingResultsRepository,
    ExperimentProcessingSettingsRepository,
)


@dataclass(frozen=True)
class StalenessInfo:
    """Information whether results are stale compared to latest settings/definitions."""

    has_run: bool
    last_run_created_at: str | None
    global_changed: bool
    # key is computation *name* (without the "Konfigurovatelný:" prefix)
    custom_changed: Dict[str, bool]
    # key is model_name as used in graphs (e.g. "Konfigurovatelný: Výpočet 1")
    graph_changed: Dict[str, bool]


def compute_staleness(experiment_id: int) -> Tuple[Optional[dict], StalenessInfo]:
    """Compute staleness compared to the latest stored run.

    We mark results as "changed" when:
    - processing settings were modified after the last run
    - a custom computation definition was modified after the last run
    """

    results_repo = ExperimentProcessingResultsRepository(DB_PATH)
    run = results_repo.get_latest_run(experiment_id)
    if not run:
        # no run -> everything is effectively "not computed"
        comps = ExperimentComputationRepository(DB_PATH).list_for_experiment(experiment_id)
        return None, StalenessInfo(
            has_run=False,
            last_run_created_at=None,
            global_changed=True,
            custom_changed={c.name: True for c in comps},
            graph_changed={},
        )

    run_dt = parse_datetime_any(run.get('created_at'))
    settings = ExperimentProcessingSettingsRepository(DB_PATH).get(experiment_id) or {}
    settings_dt = parse_datetime_any(settings.get('updated_at'))

    global_changed = bool(run_dt and settings_dt and settings_dt > run_dt)

    comps = ExperimentComputationRepository(DB_PATH).list_for_experiment(experiment_id)
    custom_changed: Dict[str, bool] = {}
    for c in comps:
        c_dt = parse_datetime_any(getattr(c, 'updated_at', None))
        custom_changed[c.name] = bool(global_changed or (run_dt and c_dt and c_dt > run_dt))

    graphs = ExperimentGraphSettingsRepository(DB_PATH).list_for_experiment(experiment_id)
    graph_changed: Dict[str, bool] = {}
    for g in graphs:
        g_dt = parse_datetime_any(g.get('updated_at') or g.get('created_at'))
        name = g.get('model_name')
        if name:
            graph_changed[name] = bool(run_dt and g_dt and g_dt > run_dt)

    return run, StalenessInfo(
        has_run=True,
        last_run_created_at=run.get('created_at'),
        global_changed=global_changed,
        custom_changed=custom_changed,
        graph_changed=graph_changed,
    )


def is_model_stale(model_name: str, info: StalenessInfo) -> bool:
    """Return whether a plotted/processed model should be considered stale."""
    if not info.has_run:
        return True
    if info.global_changed:
        return True
    if info.graph_changed.get(model_name):
        return True
    prefix = 'Konfigurovatelný:'
    if model_name.startswith(prefix):
        base = model_name.split(':', 1)[1].strip()
        return bool(info.custom_changed.get(base, True))
    return False


def get_custom_base_name(model_name: str) -> Optional[str]:
    prefix = 'Konfigurovatelný:'
    if not model_name.startswith(prefix):
        return None
    return model_name.split(':', 1)[1].strip()
