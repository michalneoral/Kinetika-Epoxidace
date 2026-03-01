from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from experimental_web.core.paths import DB_PATH
from experimental_web.core.time import parse_datetime_any
from experimental_web.data.repositories import (
    ExperimentComputationRepository,
    ExperimentFileRepository,
    ExperimentGraphSettingsRepository,
    ExperimentProcessingResultsRepository,
    ExperimentProcessingSettingsRepository,
    TablePickRepository,
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

    # --- data context (uploaded file + selected sheet + table picks) ---
    # These changes invalidate the processing results even when the user didn't touch
    # the processing-tab parameters.
    data_changed = False
    try:
        run_settings = (run.get('settings') or {}) if isinstance(run, dict) else {}
        file_repo = ExperimentFileRepository(DB_PATH)
        ef = file_repo.get_single_for_experiment(experiment_id)

        # File change: compare sha if we have it; otherwise fall back to timestamp.
        if ef is None:
            # If a run exists but the file is now missing, results cannot be trusted.
            data_changed = True
        else:
            # Prefer explicit SHA stored in run settings (newer runs).
            run_sha = run_settings.get('file_sha256')
            if run_sha and ef.sha256 and str(run_sha) != str(ef.sha256):
                data_changed = True
            else:
                ef_dt = parse_datetime_any(getattr(ef, 'uploaded_at', None))
                if run_dt and ef_dt and ef_dt > run_dt:
                    data_changed = True

            # Sheet change: selected_sheet doesn't bump file timestamps, so compare value.
            run_sheet = run_settings.get('sheet')
            cur_sheet = getattr(ef, 'selected_sheet', None) or ''
            if run_sheet is not None and str(run_sheet) != str(cur_sheet):
                data_changed = True

        # Pick changes: compare value (if stored) and/or updated_at timestamp.
        pick_repo = TablePickRepository(DB_PATH)
        fp = pick_repo.get(experiment_id, 'fame')
        ep = pick_repo.get(experiment_id, 'epo')

        def _pick_tuple(p: dict | None) -> tuple[int, int, int, int] | None:
            if not p:
                return None
            try:
                return (int(p['row_start']), int(p['row_end']), int(p['col_start']), int(p['col_end']))
            except Exception:
                return None

        run_fp = run_settings.get('fame_pick')
        run_ep = run_settings.get('epo_pick')
        cur_fp = _pick_tuple(fp)
        cur_ep = _pick_tuple(ep)

        if run_fp is not None and cur_fp is not None:
            try:
                if tuple(int(x) for x in run_fp) != cur_fp:
                    data_changed = True
            except Exception:
                pass
        if run_ep is not None and cur_ep is not None:
            try:
                if tuple(int(x) for x in run_ep) != cur_ep:
                    data_changed = True
            except Exception:
                pass

        # Timestamp fallback (covers older runs without stored pick ranges).
        # If the run already stored the explicit pick values, we MUST NOT use
        # updated_at as a signal because it can be bumped by harmless UI re-renders.
        if run_fp is None and fp:
            p_dt = parse_datetime_any(fp.get('updated_at') or fp.get('created_at'))
            if run_dt and p_dt and p_dt > run_dt:
                data_changed = True
        if run_ep is None and ep:
            p_dt = parse_datetime_any(ep.get('updated_at') or ep.get('created_at'))
            if run_dt and p_dt and p_dt > run_dt:
                data_changed = True
    except Exception:
        # staleness must never crash the UI
        data_changed = data_changed or False

    settings_changed = bool(run_dt and settings_dt and settings_dt > run_dt)
    global_changed = bool(settings_changed or data_changed)

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
