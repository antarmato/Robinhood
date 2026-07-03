"""
Timezone helpers — all market/position math runs in US-Eastern time.

Positions created before this module existed stored naive local timestamps
(server TZ was set to ET on Railway), so parsing attaches ET to naive strings
rather than rejecting them.
"""

from datetime import datetime

try:
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
except ImportError:  # pragma: no cover
    ET = None


def now_et() -> datetime:
    return datetime.now(ET) if ET else datetime.now()


def parse_iso_et(ts: str) -> datetime:
    """Parse an ISO timestamp; naive values are assumed to be Eastern."""
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None and ET:
        dt = dt.replace(tzinfo=ET)
    return dt


def hours_since(ts: str) -> float:
    return (now_et() - parse_iso_et(ts)).total_seconds() / 3600


def days_since(ts: str) -> int:
    return max(0, (now_et() - parse_iso_et(ts)).days)
