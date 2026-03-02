from __future__ import annotations

"""System tray integration (Windows).

The packaged NiceGUI app is a background HTTP server. In a windowed EXE it has no
taskbar window. A tray icon provides discoverability and a way to:
- Open the UI in the default browser
- Check for updates
- Quit the running instance

The tray is started only for frozen builds by default.
"""

import os
import sys
import threading
from pathlib import Path
from typing import Any


def _resource_path(rel: str) -> str | None:
    """Return an absolute path to a bundled resource if it exists."""
    try:
        if hasattr(sys, '_MEIPASS'):
            p = Path(getattr(sys, '_MEIPASS')) / rel
        else:
            # dev run: relative to project root (two levels up from this file)
            p = Path(__file__).resolve().parents[2] / rel
        return str(p) if p.exists() else None
    except Exception:
        return None


def _load_icon_image() -> Any | None:
    """Load icon image for pystray (PIL Image)."""
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    # Prefer the same icon as used for the EXE
    ico = _resource_path('build/icon.ico')
    if ico:
        try:
            return Image.open(ico)
        except Exception:
            pass

    # Fallback: simple solid-color square (UPCE-ish)
    try:
        return Image.new('RGBA', (64, 64), (152, 25, 78, 255))
    except Exception:
        return None


def start_tray(
    *,
    port: int,
    app_id: str,
    display_name: str | None = None,
    on_open: callable | None = None,
    on_quit: callable | None = None,
    on_check_updates: callable | None = None,
) -> Any | None:
    """Start a system tray icon (best-effort). Returns the pystray.Icon or None."""

    if os.name != 'nt':
        return None
    if os.getenv('EXPERIMENTAL_WEB_DISABLE_TRAY', '').strip() == '1':
        return None

    try:
        import pystray  # type: ignore
    except Exception:
        return None

    image = _load_icon_image()
    if image is None:
        return None

    # Lazy imports to avoid side effects unless tray actually starts.
    def _default_open(*_args: Any, **_kwargs: Any) -> None:
        try:
            from experimental_web.core.runtime_control import open_in_browser
            open_in_browser(port)
        except Exception:
            return

    def _default_check_updates(*_args: Any, **_kwargs: Any) -> None:
        def _run() -> None:
            try:
                from experimental_web.core.updater import check_for_updates
                check_for_updates(interactive=True)
            except Exception:
                return

        threading.Thread(target=_run, daemon=True).start()

    def _default_quit(*_args: Any, **_kwargs: Any) -> None:
        # Ask server to shut down; if that fails, hard-exit.
        def _run() -> None:
            try:
                from experimental_web.core.runtime_control import request_shutdown
                ok = request_shutdown(port)
                if not ok:
                    os._exit(0)
            except Exception:
                os._exit(0)

        threading.Thread(target=_run, daemon=True).start()

    open_cb = on_open or _default_open
    quit_cb = on_quit or _default_quit
    upd_cb = on_check_updates or _default_check_updates

    # pystray menu callbacks receive (icon, item) but extra args are fine
    menu = pystray.Menu(
        pystray.MenuItem('Otevřít', open_cb, default=True),
        pystray.MenuItem('Zkontrolovat aktualizace', upd_cb),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem('Ukončit', lambda icon, item: (quit_cb(icon, item), icon.stop())),
    )

    title = display_name or app_id.replace('_', ' ')
    # Use a stable ascii-ish identifier for the tray icon "name".
    icon = pystray.Icon(app_id, image, title=title, menu=menu)

    # On Windows, running in a background thread is OK. Prefer run_detached if available.
    try:
        if hasattr(icon, 'run_detached'):
            icon.run_detached()
        else:
            threading.Thread(target=icon.run, daemon=True).start()
    except Exception:
        return None

    return icon
