from __future__ import annotations

import os
import secrets
from pathlib import Path

from experimental_web.core.paths import SECRETS_DIR


SECRET_FILE: Path = SECRETS_DIR / "storage_secret.txt"


def get_storage_secret() -> str:
    """Return secret for NiceGUI user storage.

    Priority:
    1) env var NICEGUI_STORAGE_SECRET (useful for deployments)
    2) persistent secret stored in AppData (generated once)
    """
    env = os.getenv("NICEGUI_STORAGE_SECRET")
    if env and env.strip():
        return env.strip()

    if SECRET_FILE.exists():
        s = SECRET_FILE.read_text(encoding="utf-8").strip()
        if s:
            return s

    s = secrets.token_urlsafe(48)
    SECRET_FILE.write_text(s, encoding="utf-8")
    try:
        os.chmod(SECRET_FILE, 0o600)
    except Exception:
        # On Windows chmod may not behave as expected; ignore.
        pass
    return s
