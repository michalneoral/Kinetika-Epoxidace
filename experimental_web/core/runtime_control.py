from __future__ import annotations

"""Runtime control helpers for the packaged (PyInstaller) app.

Goals:
- Make the app effectively "single-instance" for end users.
  If another instance is already serving the UI, a new launch should just
  open a new browser tab pointing to the existing server.
- Provide a way to terminate the background server without Task Manager
  ("Ukončit" shortcut uses --quit).

Implementation notes:
- We persist the chosen port in the app data directory (platformdirs) so we can
  reliably re-open the running instance.
- A lightweight /__ping__ endpoint lets us verify we are talking to the *same*
  app (not some other service on that port).
"""

import os
import socket
import webbrowser
from pathlib import Path
from typing import Any

import requests

PING_PATH = '/__ping__'
SHUTDOWN_PATH = '/__shutdown__'


def port_file(app_dir: Path) -> Path:
    return app_dir / 'server_port.txt'


def port_backup_file(app_dir: Path) -> Path:
    """Backup of the last known-good port.

    This helps recover from a corrupted/partial write of server_port.txt
    after a crash or power loss.
    """

    return app_dir / 'server_port.bak'


def _parse_port(text: str) -> int | None:
    try:
        p = int(str(text).strip())
        if 1 <= p <= 65535:
            return p
    except Exception:
        return None
    return None


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a text file atomically (best-effort).

    On Windows, os.replace is atomic for same-volume operations.
    """

    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    os.replace(tmp, path)


def read_saved_port(app_dir: Path) -> int | None:
    # 1) Try primary file
    try:
        pf = port_file(app_dir)
        if pf.exists():
            p = _parse_port(pf.read_text(encoding='utf-8'))
            if p is not None:
                return p
    except Exception:
        pass

    # 2) Try backup and restore primary (best-effort)
    try:
        bf = port_backup_file(app_dir)
        if bf.exists():
            p = _parse_port(bf.read_text(encoding='utf-8'))
            if p is not None:
                try:
                    _atomic_write_text(port_file(app_dir), str(p))
                except Exception:
                    pass
                return p
    except Exception:
        pass

    return None


def save_port(app_dir: Path, port: int) -> None:
    # Best-effort, but keep a backup of the last value to survive partial writes.
    try:
        app_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # still try to write; may fail
        pass

    pf = port_file(app_dir)
    bf = port_backup_file(app_dir)

    try:
        new_value = str(int(port))

        # Backup existing value if it looks valid and differs
        try:
            if pf.exists():
                old_value = pf.read_text(encoding='utf-8')
                old_port = _parse_port(old_value)
                if old_port is not None and str(old_port) != new_value:
                    try:
                        _atomic_write_text(bf, str(old_port))
                    except Exception:
                        # Not fatal; continue
                        pass
        except Exception:
            pass

        _atomic_write_text(pf, new_value)
    except Exception:
        # Not fatal; the app can still run.
        pass


def base_url(port: int) -> str:
    return f'http://127.0.0.1:{int(port)}'


def ping(port: int, *, timeout: float = 0.35) -> dict[str, Any] | None:
    """Return ping JSON if the server responds, otherwise None."""

    try:
        r = requests.get(base_url(port) + PING_PATH, timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def is_our_server(ping_json: dict[str, Any] | None, *, app_name: str) -> bool:
    try:
        return bool(ping_json) and str(ping_json.get('app', '')) == app_name
    except Exception:
        return False


def open_in_browser(port: int) -> None:
    """Open the UI in the default browser (new tab preferred)."""

    webbrowser.open_new_tab(base_url(port) + '/')


def request_shutdown(port: int, *, timeout: float = 0.6) -> bool:
    """Ask the running app to shut down. Returns True if request was accepted."""

    try:
        r = requests.post(base_url(port) + SHUTDOWN_PATH, timeout=timeout)
        return r.status_code in (200, 202)
    except Exception:
        return False


def _is_port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(('127.0.0.1', int(port)))
            return True
        except OSError:
            return False


def find_free_port(preferred: int) -> int:
    """Find a free localhost port; try preferred first, otherwise pick an ephemeral one."""

    if 1 <= preferred <= 65535 and _is_port_free(preferred):
        return preferred

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def default_port() -> int:
    """Default port if nothing is stored yet (can be overridden by env)."""

    env = os.getenv('EXPERIMENTAL_WEB_PORT', '').strip()
    if env.isdigit():
        p = int(env)
        if 1 <= p <= 65535:
            return p
    return 8080
