from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """Timezone-aware UTC timestamp as ISO string.""" 
    return datetime.now(timezone.utc).isoformat()
