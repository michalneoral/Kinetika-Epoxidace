from __future__ import annotations

"""NiceGUI application entrypoint.

Important for packaging (PyInstaller):
- Do NOT execute ui.run() at import time.
- Use a proper __main__ guard + multiprocessing.freeze_support().

This module can still be executed directly:
    python -m experimental_web.app

It also supports an additional flag:
    --debug [LEVEL]

The flag is stripped from sys.argv so it doesn't interfere with NiceGUI/uvicorn args.

Packaged runtime quality-of-life:
- The app runs as a local HTTP server (browser UI). A windowed EXE has no taskbar window,
  so we implement:
  - "single-instance" behavior: launching again opens a new browser tab to the already
    running instance instead of silently failing due to port conflicts.
  - a shutdown endpoint used by "--quit" and a Settings button.
"""

import argparse
import os
import sys
import threading
import time


# In PyInstaller "windowed" builds on Windows, stdio streams may be None.
# Uvicorn's default logging formatter calls sys.stderr.isatty(), which would crash.
_DEVNULL_OUT = None
_DEVNULL_IN = None


def _ensure_stdio() -> None:
    """Ensure sys.stdin/stdout/stderr are usable in frozen windowed builds."""

    global _DEVNULL_OUT, _DEVNULL_IN

    # NOTE: Using os.devnull is sufficient; it provides an isatty() method.
    if sys.stdout is None:
        _DEVNULL_OUT = _DEVNULL_OUT or open(os.devnull, 'w', encoding='utf-8')
        sys.stdout = _DEVNULL_OUT
    if sys.stderr is None:
        _DEVNULL_OUT = _DEVNULL_OUT or open(os.devnull, 'w', encoding='utf-8')
        sys.stderr = _DEVNULL_OUT
    if sys.stdin is None:
        _DEVNULL_IN = _DEVNULL_IN or open(os.devnull, 'r', encoding='utf-8')
        sys.stdin = _DEVNULL_IN


def _resource_path(*parts: str) -> str:
    """Return an absolute path to a bundled resource (works for source and PyInstaller)."""
    from pathlib import Path
    if bool(getattr(sys, 'frozen', False)) and hasattr(sys, '_MEIPASS'):
        base = Path(getattr(sys, '_MEIPASS'))  # type: ignore[attr-defined]
    else:
        # experimental_web/app.py -> repo root
        base = Path(__file__).resolve().parents[1]
    return str(base.joinpath(*parts))


def _parse_debug_flag() -> int | None:
    """Parse --debug [LEVEL] without breaking NiceGUI/uvicorn args."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--debug',
        nargs='?',
        const=-1,  # --debug without number => all
        default=None,
        type=int,
        help='Debug verbosity: (no arg)=all, 0=ERROR,1=WARNING,2=INFO,3=DEBUG,4=TRACE',
    )
    args, unknown = parser.parse_known_args(sys.argv[1:])
    # Keep other args intact (if any).
    sys.argv = [sys.argv[0]] + unknown
    return args.debug


def main(*, reload: bool = False) -> None:
    """Start the NiceGUI application."""

    _ensure_stdio()

    from experimental_web.core.config import APP_NAME, APP_DISPLAY_NAME
    from experimental_web.core.paths import APP_DIR
    from experimental_web.core.runtime_control import (
        default_port,
        find_free_port,
        is_our_server,
        open_in_browser,
        ping,
        read_saved_port,
        save_port,
    )

    # Store NiceGUI's persistence files (e.g. storage-*.json) next to the database.
    # Users can still override this by explicitly setting NICEGUI_STORAGE_PATH.
    os.environ.setdefault('NICEGUI_STORAGE_PATH', str(APP_DIR))

    debug_level = _parse_debug_flag()
    if debug_level is None:
        os.environ.pop('EXPERIMENTAL_WEB_DEBUG', None)
    else:
        os.environ['EXPERIMENTAL_WEB_DEBUG'] = str(int(debug_level))

    # --- Single-instance behavior (browser UI) ---
    # If another instance is already running, just open a new tab and exit.
    candidate_port = read_saved_port(APP_DIR) or default_port()
    pj = ping(candidate_port)
    if is_our_server(pj, app_name=APP_NAME):
        open_in_browser(candidate_port)
        return

    # Choose a port (try last-saved / default first; otherwise pick a free ephemeral port)
    port = find_free_port(candidate_port)
    save_port(APP_DIR, port)
    os.environ['EXPERIMENTAL_WEB_ACTIVE_PORT'] = str(port)

    from experimental_web.logging_setup import setup_logging
    from experimental_web.core.app_init import init_app
    from experimental_web.core.secret import get_storage_secret
    from experimental_web.core.version import __version__

    setup_logging(debug_level)
    init_app()

    # NOTE: Updates are *not* installed automatically. The UI will prompt the user
    # if a newer version is available.

    # register routes by importing pages
    from experimental_web.pages import home, experiment  # noqa: F401

    # import NiceGUI only after env vars are set
    from nicegui import app, ui

    # --- internal control endpoints (used by --quit and for detecting running instance) ---
    @app.get('/__ping__')
    def __ping__():
        return {
            'app': APP_NAME,
            'version': __version__,
            'pid': os.getpid(),
            'port': port,
        }

    @app.post('/__shutdown__')
    def __shutdown__():
        # Exiting the process is the most reliable across uvicorn/nicegui versions.
        # We delay slightly so the HTTP response can be sent.
        def _exit() -> None:
            time.sleep(0.25)
            os._exit(0)

        threading.Thread(target=_exit, daemon=True).start()
        return {'ok': True}


    # --- tray icon (Windows, frozen build) ---
    tray_icon = None  # keep reference to avoid GC
    if os.name == 'nt' and bool(getattr(sys, 'frozen', False)) and os.getenv('EXPERIMENTAL_WEB_DISABLE_TRAY', '').strip() != '1':
        try:
            from experimental_web.core.tray import start_tray
            tray_icon = start_tray(port=port, app_id=APP_NAME, display_name=APP_DISPLAY_NAME)
        except Exception:
            tray_icon = None

    # --- favicon ---
    # Browsers (especially Chrome) can be picky/caching about favicons. We therefore:
    # 1) ensure /favicon.ico is served explicitly
    # 2) add <link rel="icon"> tags with a version cache-buster
    # 3) pass favicon path to NiceGUI as well
    from experimental_web.core.version import __version__
    favicon_path: str | None = None
    try:
        from pathlib import Path
        from fastapi.responses import FileResponse

        candidate = _resource_path('static', 'favicon.ico')
        if Path(candidate).exists():
            favicon_path = candidate

            # Serve static directory (so /static/favicon.ico also works)
            try:
                app.add_static_files('/static', str(Path(candidate).parent))
            except Exception:
                pass

            # Explicit route for /favicon.ico (overrides NiceGUI default)
            def _favicon_response() -> FileResponse:
                return FileResponse(
                    candidate,
                    media_type='image/x-icon',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate'},
                )

            # Avoid duplicate routes if reloaded in dev.
            if not any(getattr(r, 'path', None) == '/favicon.ico' for r in app.routes):
                app.add_api_route('/favicon.ico', _favicon_response, include_in_schema=False)

            # Add explicit <link> tags with cache-busting query.
            ui.add_head_html(f'<link rel="icon" type="image/x-icon" href="/favicon.ico?v={__version__}">', shared=True)
            ui.add_head_html(f'<link rel="shortcut icon" href="/favicon.ico?v={__version__}">', shared=True)
    except Exception:
        favicon_path = None

    run_kwargs = dict(
        title=APP_DISPLAY_NAME,
        reload=reload,
        storage_secret=get_storage_secret(),
        host='127.0.0.1',
        port=port,
        show=True,
    )
    if favicon_path:
        run_kwargs['favicon'] = favicon_path
    ui.run(**run_kwargs)


if __name__ in ('__main__', '__mp_main__'):
    # Required for Windows multiprocessing when frozen (PyInstaller).
    from multiprocessing import freeze_support
    from experimental_web.core.version import __version__

    freeze_support()
    _ensure_stdio()
    # Useful for CLI runs; in the packaged app there is no console by default.
    try:
        from experimental_web.core.config import APP_DISPLAY_NAME
        name = APP_DISPLAY_NAME
    except Exception:
        name = 'FAME_EPO_Manager'
    print(f'Starting {name} v{__version__}…')
    main(reload=False)
