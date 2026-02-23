from __future__ import annotations

from pathlib import Path
from platformdirs import user_data_dir

from .config import APP_NAME, APP_AUTHOR


def get_app_dir() -> Path:
    """Cross-platform user data directory (Windows AppData / Linux ~/.local/share)."""
    base = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    base.mkdir(parents=True, exist_ok=True)
    return base


APP_DIR: Path = get_app_dir()
DB_PATH: Path = APP_DIR / "experiments.sqlite3"
EXPERIMENTS_DIR: Path = APP_DIR / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
SECRETS_DIR: Path = APP_DIR / "secrets"
SECRETS_DIR.mkdir(parents=True, exist_ok=True)
