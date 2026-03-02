"""Self-updater (Windows) via GitHub Releases.

Key behavior (by design):
- The app may *detect* a newer version on startup, but it **must not install
  automatically**.
- The user is asked for confirmation in the UI (NiceGUI dialog) or via an
  interactive check (tray / Start menu).

This module therefore provides small building blocks:
- get_update_info(): check latest GitHub release and return UpdateInfo
- download_installer(): download installer (and optionally verify checksums)
- launch_installer(): run the Inno Setup installer in silent mode
- check_for_updates(): interactive (MessageBox Yes/No) flow used by tray/shortcut

The updater is best-effort: failures are logged and must never prevent the app
from starting.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import traceback
from dataclasses import dataclass
from typing import Any
from urllib.request import Request, urlopen


try:
    from packaging.version import Version
except Exception:  # pragma: no cover
    Version = None  # type: ignore


from experimental_web.core.config import UPDATE_OWNER, UPDATE_REPO
from experimental_web.core.paths import APP_DIR
from experimental_web.core.version import __version__


TIMEOUT_S = 10


def _log(msg: str) -> None:
    """Append a short line to the updater log in the user data directory."""

    try:
        log_path = APP_DIR / 'update.log'
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, 'a', encoding='utf-8', errors='ignore') as f:
            f.write(f'[{ts}] {msg}\n')
    except Exception:
        return


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


# GitHub updater target (hardcoded for this project via config)
GITHUB_OWNER = UPDATE_OWNER
GITHUB_REPO = UPDATE_REPO

# Expect an asset named like: FAME_EPO_Manager_Setup-<version>.exe
ASSET_PREFIX = _env('EXPERIMENTAL_WEB_UPDATE_ASSET_PREFIX', 'FAME_EPO_Manager_Setup-')


@dataclass(frozen=True)
class _Asset:
    name: str
    url: str


@dataclass(frozen=True)
class UpdateInfo:
    current_version: str
    latest_version: str
    release_tag: str
    release_url: str
    installer: _Asset
    checksums: _Asset | None = None


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
    - "SHA256  <hash>  <filename>" (legacy)
    - "<hash>  <filename>" (sha256sum)
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


def _download_dir() -> str:
    """A persistent download directory (avoids temp cleanup races)."""

    d = APP_DIR / 'updates'
    d.mkdir(parents=True, exist_ok=True)
    return str(d)


def _pick_windows_installer_asset(release: dict[str, Any]) -> tuple[_Asset | None, _Asset | None]:
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
    """Best-effort message box (Windows)."""

    if os.name != 'nt':
        return
    try:
        import ctypes  # local import

        MB_OK = 0x0000
        MB_ICONINFORMATION = 0x0040
        ctypes.windll.user32.MessageBoxW(None, text, title, MB_OK | MB_ICONINFORMATION)
    except Exception:
        return


def _message_box_yes_no(title: str, text: str) -> bool | None:
    """Return True/False for Yes/No, or None if not supported."""

    if os.name != 'nt':
        return None
    try:
        import ctypes  # local import

        MB_YESNO = 0x0004
        MB_ICONQUESTION = 0x0020
        IDYES = 6
        r = ctypes.windll.user32.MessageBoxW(None, text, title, MB_YESNO | MB_ICONQUESTION)
        return bool(r == IDYES)
    except Exception:
        return None


def get_update_info() -> UpdateInfo | None:
    """Return UpdateInfo if a newer version is available, else None.

    Never raises; errors are logged and treated as "no update".
    """

    if os.getenv('EXPERIMENTAL_WEB_DISABLE_UPDATE', '').strip() == '1':
        return None

    try:
        _log(f'Update check started (current={__version__}).')
        release = _get_latest_release()
        if isinstance(release, dict) and release.get('message'):
            _log(f"GitHub API message: {release.get('message')}")

        release_tag = str(release.get('tag_name', '')).strip()
        tag = release_tag.lstrip('v').strip()
        if not tag:
            _log('Update check: missing tag_name.')
            return None

        current = _version_obj(__version__)
        latest = _version_obj(tag)
        if latest <= current:
            return None

        exe_asset, checks_asset = _pick_windows_installer_asset(release)
        if not exe_asset:
            _log('Update check: missing installer asset in latest release.')
            return None

        release_url = str(
            release.get('html_url', '')
            or f'https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}/releases/latest'
        )

        return UpdateInfo(
            current_version=__version__,
            latest_version=tag,
            release_tag=release_tag or f'v{tag}',
            release_url=release_url,
            installer=exe_asset,
            checksums=checks_asset,
        )
    except Exception:
        _log('Update check failed: ' + traceback.format_exc().strip())
        return None


def download_installer(info: UpdateInfo) -> str:
    """Download installer (and verify checksums if available). Returns local path."""

    td = _download_dir()
    exe_path = os.path.join(td, info.installer.name)
    _log(f'Downloading installer: {info.installer.url} -> {exe_path}')
    _download(info.installer.url, exe_path)

    # optional checksum verification
    if info.checksums:
        checks_path = os.path.join(td, info.checksums.name)
        _log(f'Downloading checksums: {info.checksums.url} -> {checks_path}')
        _download(info.checksums.url, checks_path)
        checks_txt = open(checks_path, 'r', encoding='utf-8', errors='ignore').read()
        checks = _parse_checksums_file(checks_txt)
        expected = checks.get(info.installer.name)
        if expected:
            got = _sha256_file(exe_path)
            if got != expected:
                _log('Checksum mismatch for installer download.')
                raise ValueError('SHA256 mismatch')

    return exe_path


def launch_installer(installer_path: str) -> bool:
    return _run_installer_silently(installer_path)


def check_for_updates(interactive: bool = True) -> bool:
    """Interactive update check used by tray / Start menu shortcut.

    If a newer version exists, asks the user and only then downloads & runs the installer.
    Returns True if installer was launched.
    """

    if os.getenv('EXPERIMENTAL_WEB_DISABLE_UPDATE', '').strip() == '1':
        if interactive:
            _message_box('Aktualizace', 'Kontrola aktualizací je vypnutá (EXPERIMENTAL_WEB_DISABLE_UPDATE=1).')
        return False

    try:
        info = get_update_info()
        if not info:
            if interactive:
                _message_box('Aktualizace', f'Máte nejnovější verzi ({__version__}).')
            return False

        if not interactive:
            return False

        answer = _message_box_yes_no(
            'Aktualizace',
            f'Nalezena nová verze {info.latest_version}.\n'
            f'Nainstalovaná verze: {info.current_version}\n\n'
            'Chcete aktualizovat teď?',
        )
        if answer is False:
            return False
        if answer is None:
            _message_box(
                'Aktualizace',
                f'Nalezena nová verze {info.latest_version}, ale nelze zobrazit potvrzení.',
            )
            return False

        try:
            exe_path = download_installer(info)
        except Exception:
            _log('Installer download/verify failed: ' + traceback.format_exc().strip())
            _message_box('Aktualizace', 'Stažení aktualizace selhalo (síť / integrita).')
            return False

        ok = launch_installer(exe_path)
        if not ok:
            _log('Failed to launch installer.')
            _message_box('Aktualizace', 'Nepodařilo se spustit instalátor aktualizace.')
            return False

        _log(f'Installer launched for update to {info.latest_version}.')
        _message_box('Aktualizace', f'Spouštím instalátor verze {info.latest_version}…')
        time.sleep(0.5)
        os._exit(0)
    except Exception:
        _log('Interactive update check failed: ' + traceback.format_exc().strip())
        if interactive:
            _message_box('Aktualizace', 'Kontrola aktualizací selhala (síť / GitHub API).')
        return False


def maybe_update_in_background() -> None:
    """Deprecated: kept for backward compatibility (does nothing)."""

    return
