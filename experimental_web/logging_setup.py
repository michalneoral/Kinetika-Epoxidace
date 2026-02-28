"""Project-wide logging setup.

Goals:
- Enable a single CLI flag: --debug [LEVEL]
- Use Rich for readable console output (when available)
- Include caller context (module/function/line)
- Keep overhead low via standard logging best practices
"""

from __future__ import annotations

import contextvars
import itertools
import logging
import sys
import threading
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple


# Custom TRACE level (more verbose than DEBUG)
TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def trace(self: logging.Logger, msg, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, msg, args, **kwargs)


logging.Logger.trace = trace  # type: ignore[attr-defined]


try:
    from rich.logging import RichHandler  # type: ignore
except Exception:  # pragma: no cover
    RichHandler = None


if RichHandler is not None:
    class SafeRichHandler(RichHandler):
        """A RichHandler variant that must never crash the app.

        Rich traceback rendering can sometimes hit recursion issues on complex exception chains.
        We fall back to a plain stderr line in that case.
        """

        def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
            try:
                super().emit(record)
            except RecursionError:
                try:
                    msg = self.format(record)
                except Exception:
                    msg = record.getMessage()
                try:
                    sys.stderr.write(msg + "\n")
                except Exception:
                    pass
            except Exception:
                # As a last resort, try to avoid breaking the app.
                try:
                    sys.stderr.write(record.getMessage() + "\n")
                except Exception:
                    pass


@dataclass(frozen=True)
class DebugConfig:
    """Debug verbosity.

    - None: default (INFO)
    - -1: all (TRACE)
    - 0..4: ERROR, WARNING, INFO, DEBUG, TRACE
    """

    level: Optional[int] = None


def verbosity_to_level(v: Optional[int]) -> int:
    if v is None:
        return logging.INFO
    if v < 0:
        return TRACE
    mapping = {
        0: logging.ERROR,
        1: logging.WARNING,
        2: logging.INFO,
        3: logging.DEBUG,
        4: TRACE,
    }
    return mapping.get(int(v), TRACE)


class IndentFormatter(logging.Formatter):
    """Indent messages by severity/verbosity level.

    Higher verbosity => deeper indent.
    """

    def format(self, record: logging.LogRecord) -> str:
        scope_indent = int(_SCOPE_INDENT.get())
        if record.levelno >= logging.ERROR:
            indent = 0
        elif record.levelno >= logging.WARNING:
            indent = 1
        elif record.levelno >= logging.INFO:
            indent = 2
        elif record.levelno >= logging.DEBUG:
            indent = 3
        else:
            indent = 4

        indent += max(0, scope_indent)

        original_msg = record.msg
        try:
            record.msg = ("  " * indent) + str(record.msg)
            return super().format(record)
        finally:
            record.msg = original_msg


# -----------------------------------------------------------------------------
# Scope indentation (very low overhead)
# -----------------------------------------------------------------------------

_SCOPE_INDENT: contextvars.ContextVar[int] = contextvars.ContextVar('experimental_web_log_scope_indent', default=0)


@contextmanager
def log_scope(title: str, *, logger: Optional[logging.Logger] = None, level: int = logging.DEBUG, **fields):
    """Create a lightweight scoped logging context.

    - Adds indentation to all log messages inside the scope (using a contextvar).
    - Logs an entry/exit line (at DEBUG by default).

    Designed to be cheap: no heavy formatting unless the logger is enabled.
    """
    log = logger or logging.getLogger(__name__)
    enabled = log.isEnabledFor(level)
    if enabled:
        if fields:
            log.log(level, "▶ %s %s", title, fields)
        else:
            log.log(level, "▶ %s", title)
    token = _SCOPE_INDENT.set(int(_SCOPE_INDENT.get()) + 1)
    try:
        yield
    finally:
        _SCOPE_INDENT.reset(token)
        if enabled:
            log.log(level, "◀ %s", title)


# -----------------------------------------------------------------------------
# In-memory ring buffer (for UI debug panel)
# -----------------------------------------------------------------------------


class RingBufferHandler(logging.Handler):
    """Keep last N formatted log lines in memory.

    The buffer stores (id, line). The id is monotonically increasing.
    """

    def __init__(self, capacity: int = 2000):
        super().__init__(level=logging.NOTSET)
        self._capacity = int(capacity)
        self._lock = threading.Lock()
        self._counter = itertools.count(1)
        self._buffer: Deque[Tuple[int, str]] = deque(maxlen=self._capacity)

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover
        try:
            line = self.format(record)
        except Exception:
            self.handleError(record)
            return
        with self._lock:
            self._buffer.append((next(self._counter), line))

    def get_since(self, last_id: int = 0) -> List[Tuple[int, str]]:
        """Return lines with id > last_id."""
        with self._lock:
            return [(i, s) for (i, s) in self._buffer if i > last_id]

    def snapshot(self) -> List[Tuple[int, str]]:
        with self._lock:
            return list(self._buffer)


_buffer_handler: Optional[RingBufferHandler] = None


def setup_logging(debug: Optional[int]) -> None:
    """Configure root logger.

    Safe to call in subprocesses.
    """

    level = verbosity_to_level(debug)
    root = logging.getLogger()
    root.setLevel(level)

    # Remove existing handlers to avoid duplicates (e.g., on reload).
    for h in list(root.handlers):
        root.removeHandler(h)

    if RichHandler is not None:
        handler = SafeRichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_level=True,
            show_path=True,  # file:line
            markup=True,
        )
    else:
        handler = logging.StreamHandler()

    # RichHandler already renders time/level/path; we add logger name + func:line for quick pinpointing.
    fmt = "%(name)s | %(funcName)s:%(lineno)d | %(message)s"
    handler.setFormatter(IndentFormatter(fmt=fmt))
    root.addHandler(handler)

    # In-memory buffer for UI debugging (cheap, bounded).
    global _buffer_handler
    _buffer_handler = RingBufferHandler(capacity=2500)
    _buffer_handler.setLevel(level)
    _buffer_handler.setFormatter(IndentFormatter(fmt=fmt))
    root.addHandler(_buffer_handler)

    # Keep 3rd-party noise under control.
    noisy = [
        "matplotlib",
        "asyncio",
        "uvicorn",
        "uvicorn.error",
        "uvicorn.access",
    ]
    for n in noisy:
        logging.getLogger(n).setLevel(max(level, logging.WARNING))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def get_log_buffer_handler() -> Optional[RingBufferHandler]:
    return _buffer_handler


def is_debug_enabled() -> bool:
    """True if root logger is verbose enough to justify showing debug UI."""
    return logging.getLogger().isEnabledFor(logging.DEBUG) or logging.getLogger().isEnabledFor(TRACE)
