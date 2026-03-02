"""PyInstaller entrypoint (duplicate kept in /build for robustness).

Some PyInstaller/.spec setups resolve the entry script relative to the build folder.
Keeping a copy here prevents fragile path issues.

See ../pack_entry.py for details.
"""

import os
import sys
from multiprocessing import freeze_support


_DEVNULL_OUT = None
_DEVNULL_IN = None


def _ensure_stdio() -> None:
    """Ensure sys.stdin/stdout/stderr exist in frozen windowed builds."""

    global _DEVNULL_OUT, _DEVNULL_IN

    if sys.stdout is None:
        _DEVNULL_OUT = _DEVNULL_OUT or open(os.devnull, 'w', encoding='utf-8')
        sys.stdout = _DEVNULL_OUT
    if sys.stderr is None:
        _DEVNULL_OUT = _DEVNULL_OUT or open(os.devnull, 'w', encoding='utf-8')
        sys.stderr = _DEVNULL_OUT
    if sys.stdin is None:
        _DEVNULL_IN = _DEVNULL_IN or open(os.devnull, 'r', encoding='utf-8')
        sys.stdin = _DEVNULL_IN


def main() -> None:
    _ensure_stdio()

    args = set(a.strip().lower() for a in sys.argv[1:])

    if '--quit' in args or '--shutdown' in args:
        try:
            from experimental_web.core.config import APP_NAME
            from experimental_web.core.paths import APP_DIR
            from experimental_web.core.runtime_control import (
                default_port,
                is_our_server,
                ping,
                read_saved_port,
                request_shutdown,
            )

            port = read_saved_port(APP_DIR) or default_port()
            if is_our_server(ping(port), app_name=APP_NAME):
                request_shutdown(port)
        finally:
            return

    if '--check-update' in args or '--check-updates' in args:
        try:
            from experimental_web.core.updater import check_for_updates

            check_for_updates(interactive=True)
        finally:
            return

    from experimental_web.app import main as app_main

    app_main(reload=False)


if __name__ in ('__main__', '__mp_main__'):
    _ensure_stdio()
    freeze_support()
    main()
