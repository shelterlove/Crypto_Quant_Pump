from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta

TIMEFRAME_MS = {
    "15m": 15 * 60 * 1000,
    "1h": 60 * 60 * 1000,
    "4h": 4 * 60 * 60 * 1000,
    "1d": 24 * 60 * 60 * 1000,
}


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def parse_utc_datetime(value: str | None, default: datetime) -> datetime:
    if value is None:
        return ensure_utc(default)
    parsed: datetime
    if len(value) == 10:
        parsed = datetime.combine(date.fromisoformat(value), time.min, tzinfo=UTC)
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return ensure_utc(parsed)


def timeframe_delta(timeframe: str) -> timedelta:
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"unsupported timeframe: {timeframe}")
    return timedelta(milliseconds=TIMEFRAME_MS[timeframe])


def previous_closed_open_time(now: datetime, timeframe: str) -> datetime:
    now = ensure_utc(now)
    delta = timeframe_delta(timeframe)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    elapsed = now - epoch
    bucket = elapsed // delta
    current_open = epoch + bucket * delta
    return current_open - delta


def default_one_year_window(now: datetime | None = None) -> tuple[datetime, datetime]:
    end = previous_closed_open_time(now or datetime.now(UTC), "1h")
    start = end - timedelta(days=365)
    return start, end


def monday_utc(value: datetime) -> datetime:
    value = ensure_utc(value)
    start = datetime.combine(value.date(), time.min, tzinfo=UTC)
    return start - timedelta(days=start.weekday())
