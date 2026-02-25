from __future__ import annotations

from experimental_web.data.database import Database
from experimental_web.core.paths import DB_PATH


def init_app() -> None:
    """Initialize non-UI parts (DB schema etc.).

    IMPORTANT: Do not create any UI here when using @ui.page.
    """
    Database(DB_PATH).ensure_schema()

    # Pandas: avoid FutureWarning about silent downcasting on fillna/ffill/bfill
    try:
        import pandas as pd  # type: ignore
        pd.set_option('future.no_silent_downcasting', True)
    except Exception:
        pass
