from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp as ISO string.""" 
    return datetime.now(timezone.utc).isoformat()


def parse_datetime_any(value: object | None) -> datetime | None:
    """Parse timestamps coming from both our app (ISO) and SQLite (datetime('now')).

    We store some timestamps as ISO strings ("2026-02-26T21:11:00+00:00")
    and some are produced by SQLite ("2026-02-26 21:11:00").
    This helper understands both so we can reliably compare "updated_at" vs "created_at".
    """
    if value is None:
        return None

    # sqlite3 can return Python datetime objects depending on connection settings.
    # Normalize everything to timezone-aware UTC so all comparisons are safe.
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None

    # ISO 8601 (our code)
    # NOTE: datetime.fromisoformat returns *naive* datetimes when the string
    # has no timezone offset. We normalize everything to UTC-aware so all
    # comparisons are safe (no naive vs aware TypeError).
    try:
        dt = datetime.fromisoformat(s.replace('Z', '+00:00'))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass

    # SQLite datetime('now')
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None
