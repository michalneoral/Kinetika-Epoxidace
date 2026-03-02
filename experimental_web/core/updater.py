"""Self-updater (Windows) via GitHub Releases.

This mirrors the approach from the older project (download latest Setup.exe from
GitHub Releases and run it silently in the background). It is intentionally
best-effort and should never break app startup.

Notes:
- If the installer targets Program Files, Windows will still show a UAC prompt.
  For fully silent updates without admin prompts, use a per-user install
  directory (e.g. {localappdata}) and PrivilegesRequired=lowest in Inno Setup.
- Users can disable update checks by setting EXPERIMENTAL_WEB_DISABLE_UPDATE=1.
- Repo/asset naming can be overridden by env vars (see defaults below).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import time
import sys
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore


from experimental_web.core.version import __version__
from experimental_web.core.config import UPDATE_OWNER, UPDATE_REPO



TIMEOUT_S = 10


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


# GitHub updater target (hardcoded for this project)
GITHUB_OWNER = UPDATE_OWNER
GITHUB_REPO = UPDATE_REPO

# Expect an asset named like: FAME_EPO_Manager_Setup-<version>.exe
ASSET_PREFIX = _env('EXPERIMENTAL_WEB_UPDATE_ASSET_PREFIX', 'FAME_EPO_Manager_Setup-')


@dataclass(frozen=True)
class _Asset:
    name: str
    url: str


def _http_get_json(url: str) -> Any:
    req = Request(url, headers={'User-Agent': f'FAME_EPO_Manager/{__version__}'})
    with urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read().decode('utf-8', errors='ignore'))


def _download(url: str, dst_path: str) -> None:
    req = Request(url, headers={'User-Agent': f'FAME_EPO_Manager/{__version__}'})
    with urlopen(req, timeout=30) as r:
        with open(dst_path, 'wb') as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest().lower()


def _parse_checksums_file(text: str) -> dict[str, str]:
    """Parse a checksums file.

    Supported formats:
    - "SHA256  <hash>  <filename>" (like in the old project)
    - "<hash>  <filename>" (common sha256sum format)
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        if parts[0].upper().startswith('SHA256') and len(parts) >= 3:
            digest, filename = parts[1], ' '.join(parts[2:])
        else:
            digest, filename = parts[0], ' '.join(parts[1:])
        out[filename] = digest.lower()
    return out


def _get_latest_release() -> dict[str, Any]:
    url = f'https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest'
    return _http_get_json(url)


def _pick_windows_installer_asset(release: dict[str, Any]) -> tuple[_Asset | None, _Asset | None]:
    """Pick installer exe and optional checksums asset."""
    exe: _Asset | None = None
    checksums: _Asset | None = None
    for a in release.get('assets', []) or []:
        name = str(a.get('name', ''))
        url = str(a.get('browser_download_url', ''))
        if not name or not url:
            continue
        if name.lower().endswith('.exe') and name.startswith(ASSET_PREFIX):
            exe = _Asset(name=name, url=url)
        elif name.lower().startswith('checksums') and name.lower().endswith('.txt'):
            checksums = _Asset(name=name, url=url)
    return exe, checksums


def _version_obj(v: str):
    if Version is None:
        try:
            return tuple(int(x) for x in v.split('.'))
        except Exception:
            return v
    return Version(v)


def _run_installer_silently(installer_path: str) -> bool:
    """Launch Inno Setup installer in silent mode."""
    args = [
        installer_path,
        '/CURRENTUSER',
        '/VERYSILENT',
        '/SUPPRESSMSGBOXES',
        '/CLOSEAPPLICATIONS',
        '/RESTARTAPPLICATIONS',
        '/NORESTART',
    ]
    creationflags = 0x00000008  # CREATE_NEW_CONSOLE
    try:
        subprocess.Popen(args, creationflags=creationflags)
        return True
    except Exception:
        return False


def _message_box(title: str, text: str) -> None:
    """Best-effort message box for interactive update checks (Windows)."""

    # Avoid importing tkinter in frozen builds; use WinAPI directly.
    if os.name != 'nt':
        return
    try:
        import ctypes  # local import

        MB_OK = 0x0000
        MB_ICONINFORMATION = 0x0040
        ctypes.windll.user32.MessageBoxW(None, text, title, MB_OK | MB_ICONINFORMATION)
    except Exception:
        return


def check_for_updates(interactive: bool = True) -> bool:
    """Check GitHub Releases and run the newest installer if available.

    Returns True if an installer was launched.
    """

    if os.getenv('EXPERIMENTAL_WEB_DISABLE_UPDATE', '').strip() == '1':
        if interactive:
            _message_box('Aktualizace', 'Kontrola aktualizací je vypnutá (EXPERIMENTAL_WEB_DISABLE_UPDATE=1).')
        return False

    try:
        release = _get_latest_release()
        tag = str(release.get('tag_name', '')).lstrip('v').strip()
        if not tag:
            if interactive:
                _message_box('Aktualizace', 'Nepodařilo se zjistit nejnovější verzi (chybí tag).')
            return False

        current = _version_obj(__version__)
        latest = _version_obj(tag)
        if latest <= current:
            if interactive:
                _message_box('Aktualizace', f'Máte nejnovější verzi ({__version__}).')
            return False

        exe_asset, checks_asset = _pick_windows_installer_asset(release)
        if not exe_asset:
            if interactive:
                _message_box('Aktualizace', 'V releasu nebyl nalezen instalační soubor (Setup.exe).')
            return False

        with tempfile.TemporaryDirectory() as td:
            exe_path = os.path.join(td, exe_asset.name)
            _download(exe_asset.url, exe_path)

            # optional checksum verification
            if checks_asset:
                checks_path = os.path.join(td, checks_asset.name)
                _download(checks_asset.url, checks_path)
                checks_txt = open(checks_path, 'r', encoding='utf-8', errors='ignore').read()
                checks = _parse_checksums_file(checks_txt)
                expected = checks.get(exe_asset.name)
                if expected:
                    got = _sha256_file(exe_path)
                    if got != expected:
                        if interactive:
                            _message_box('Aktualizace', 'Kontrola integrity selhala (SHA256 nesouhlasí).')
                        return False

            ok = _run_installer_silently(exe_path)
            if not ok:
                if interactive:
                    _message_box('Aktualizace', 'Nepodařilo se spustit instalátor aktualizace.')
                return False

            if interactive:
                _message_box('Aktualizace', f'Nalezena nová verze {tag}. Spouštím instalátor…')
            return True
    except Exception:
        if interactive:
            _message_box('Aktualizace', 'Kontrola aktualizací selhala (síť / GitHub API).')
        return False


def maybe_update_in_background() -> None:
    """Best-effort update check. Safe to run in a daemon thread."""
    if os.getenv('EXPERIMENTAL_WEB_DISABLE_UPDATE', '').strip() == '1':
        return

    try:
        release = _get_latest_release()
        tag = str(release.get('tag_name', '')).lstrip('v').strip()
        if not tag:
            return

        current = _version_obj(__version__)
        latest = _version_obj(tag)
        if latest <= current:
            return

        exe_asset, checks_asset = _pick_windows_installer_asset(release)
        if not exe_asset:
            return

        with tempfile.TemporaryDirectory() as td:
            exe_path = os.path.join(td, exe_asset.name)
            _download(exe_asset.url, exe_path)

            # optional checksum verification
            if checks_asset:
                checks_path = os.path.join(td, checks_asset.name)
                _download(checks_asset.url, checks_path)
                checks_txt = open(checks_path, 'r', encoding='utf-8', errors='ignore').read()
                checks = _parse_checksums_file(checks_txt)
                expected = checks.get(exe_asset.name)
                if expected:
                    got = _sha256_file(exe_path)
                    if got != expected:
                        return

            ok = _run_installer_silently(exe_path)
            if ok:
                time.sleep(1.0)
                os._exit(0)
    except Exception:
        return
