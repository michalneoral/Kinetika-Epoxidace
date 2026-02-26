from __future__ import annotations

import inspect
import logging
from functools import wraps
from typing import Any, Callable, Optional

from experimental_web.logging_setup import get_logger


ui_log = get_logger('experimental_web.ui')


def _safe_dict(d: Any) -> dict:
    if d is None:
        return {}
    if isinstance(d, dict):
        return d
    try:
        return dict(d)
    except Exception:
        return {"value": repr(d)}


def wrap_ui_handler(
    action: str,
    handler: Callable[..., Any],
    *,
    level: int = logging.DEBUG,
    data: Optional[Callable[..., Any]] = None,
) -> Callable[..., Any]:
    """Wrap a NiceGUI callback to log the UI action.

    Keeps overhead low:
    - If the desired log level is disabled, it just calls the original handler.
    - Uses lazy formatting.
    """
    if handler is None:
        return handler

    is_async = inspect.iscoroutinefunction(handler)

    def _call(*args, **kwargs):
        try:
            return handler(*args, **kwargs)
        except TypeError:
            # Some NiceGUI events pass an argument while many handlers are written with no params.
            return handler()

    def _log(*args, **kwargs) -> None:
        if not ui_log.isEnabledFor(level):
            return
        extra = {}
        if data is not None:
            try:
                extra = _safe_dict(data(*args, **kwargs))
            except Exception as e:
                extra = {"_data_error": repr(e)}
        if extra:
            ui_log.log(level, '[UI] %s %s', action, extra)
        else:
            ui_log.log(level, '[UI] %s', action)

    if is_async:
        @wraps(handler)
        async def _wrapped(*args, **kwargs):
            _log(*args, **kwargs)
            res = _call(*args, **kwargs)
            if inspect.isawaitable(res):
                return await res
            return res

        return _wrapped

    @wraps(handler)
    def _wrapped(*args, **kwargs):
        _log(*args, **kwargs)
        res = _call(*args, **kwargs)
        return res

    return _wrapped


def log_ui(action: str, *, level: int = logging.INFO, **fields) -> None:
    """Direct UI log call."""
    if not ui_log.isEnabledFor(level):
        return
    ui_log.log(level, '[UI] %s %s', action, fields if fields else '')
