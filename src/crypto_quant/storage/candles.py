from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import Select, distinct, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session

from crypto_quant.storage.models import Candle
from crypto_quant.utils.time import ensure_utc, previous_closed_open_time

UPSERT_BATCH_SIZE = 5_000


@dataclass(frozen=True)
class CandleCoverage:
    symbol: str
    timeframe: str
    rows: int
    start: datetime | None
    end: datetime | None
    missing_intervals: int


def upsert_candles(session: Session, exchange: str, symbol: str, timeframe: str, frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    records = []
    for row in frame.itertuples(index=False):
        records.append(
            {
                "exchange": exchange,
                "symbol": symbol,
                "timeframe": timeframe,
                "open_time": ensure_utc(pd.Timestamp(row.open_time).to_pydatetime()),
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "volume": float(row.volume),
                "quote_volume": float(row.quote_volume) if hasattr(row, "quote_volume") else float(row.close) * float(row.volume),
            }
        )
    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        max_id = session.execute(select(func.max(Candle.id))).scalar() or 0
        for offset, record in enumerate(records, start=1):
            record["id"] = int(max_id) + offset
    for batch_start in range(0, len(records), UPSERT_BATCH_SIZE):
        batch = records[batch_start : batch_start + UPSERT_BATCH_SIZE]
        _upsert_candle_batch(session, dialect, batch)
    return len(records)


def _upsert_candle_batch(session: Session, dialect: str, records: list[dict[str, Any]]) -> None:
    upsert: Any
    if dialect == "sqlite":
        sqlite_stmt = sqlite_insert(Candle).values(records)
        sqlite_update_cols = {
            "open": sqlite_stmt.excluded.open,
            "high": sqlite_stmt.excluded.high,
            "low": sqlite_stmt.excluded.low,
            "close": sqlite_stmt.excluded.close,
            "volume": sqlite_stmt.excluded.volume,
            "quote_volume": sqlite_stmt.excluded.quote_volume,
        }
        upsert = sqlite_stmt.on_conflict_do_update(
            index_elements=["exchange", "symbol", "timeframe", "open_time"],
            set_=sqlite_update_cols,
        )
    else:
        pg_stmt = pg_insert(Candle).values(records)
        pg_update_cols = {
            "open": pg_stmt.excluded.open,
            "high": pg_stmt.excluded.high,
            "low": pg_stmt.excluded.low,
            "close": pg_stmt.excluded.close,
            "volume": pg_stmt.excluded.volume,
            "quote_volume": pg_stmt.excluded.quote_volume,
        }
        upsert = pg_stmt.on_conflict_do_update(
            constraint="uq_candles_key",
            set_=pg_update_cols,
        )
    session.execute(upsert)


def load_candles(
    session: Session,
    exchange: str,
    symbols: list[str],
    timeframe: str,
    start: datetime,
    end: datetime,
) -> dict[str, pd.DataFrame]:
    if not symbols:
        return {}
    query: Select[tuple[str, datetime, float, float, float, float, float, float | None]] = (
        select(
            Candle.symbol,
            Candle.open_time,
            Candle.open,
            Candle.high,
            Candle.low,
            Candle.close,
            Candle.volume,
            Candle.quote_volume,
        )
        .where(Candle.exchange == exchange)
        .where(Candle.symbol.in_(symbols))
        .where(Candle.timeframe == timeframe)
        .where(Candle.open_time >= ensure_utc(start))
        .where(Candle.open_time <= ensure_utc(end))
        .order_by(Candle.symbol, Candle.open_time)
    )
    rows = session.execute(query).all()
    by_symbol: dict[str, list[dict[str, object]]] = {symbol: [] for symbol in symbols}
    for symbol, open_time, open_price, high, low, close, volume, quote_volume in rows:
        by_symbol.setdefault(symbol, []).append(
            {
                "open_time": open_time,
                "open": float(open_price),
                "high": float(high),
                "low": float(low),
                "close": float(close),
                "volume": float(volume),
                "quote_volume": float(quote_volume or 0),
            }
        )
    return {symbol: pd.DataFrame(items) for symbol, items in by_symbol.items()}


def distinct_candle_symbols(session: Session, exchange: str, timeframe: str) -> list[str]:
    rows = session.execute(
        select(distinct(Candle.symbol))
        .where(Candle.exchange == exchange)
        .where(Candle.timeframe == timeframe)
        .order_by(Candle.symbol)
    ).scalars()
    return [str(row) for row in rows]


def has_complete_candle_coverage(
    session: Session,
    exchange: str,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> bool:
    start_bound, end_bound = _aligned_candle_bounds(start, end, timeframe)
    closed_bound = previous_closed_open_time(datetime.now(tz=ensure_utc(end).tzinfo), timeframe)
    end_bound = min(end_bound, ensure_utc(closed_bound))
    if start_bound > end_bound:
        return True
    row = session.execute(
        select(func.count(Candle.id), func.min(Candle.open_time), func.max(Candle.open_time))
        .where(Candle.exchange == exchange)
        .where(Candle.symbol == symbol)
        .where(Candle.timeframe == timeframe)
        .where(Candle.open_time >= start_bound)
        .where(Candle.open_time <= end_bound)
    ).one()
    row_count = int(row[0])
    first = row[1]
    last = row[2]
    if row_count == 0 or first is None or last is None:
        return False
    step_seconds = {"1d": 86_400, "4h": 14_400, "1h": 3_600, "15m": 900}[timeframe]
    coverage_start = max(ensure_utc(first), start_bound)
    expected = int((end_bound - coverage_start).total_seconds() // step_seconds) + 1
    return ensure_utc(last) >= end_bound and row_count >= expected


def _aligned_candle_bounds(start: datetime, end: datetime, timeframe: str) -> tuple[datetime, datetime]:
    freq = {"1d": "1D", "4h": "4h", "1h": "1h", "15m": "15min"}[timeframe]
    start_ts = pd.Timestamp(ensure_utc(start))
    end_ts = pd.Timestamp(ensure_utc(end))
    lower = start_ts.ceil(freq)
    upper = end_ts.floor(freq)
    return lower.to_pydatetime(), upper.to_pydatetime()


def coverage(frame: pd.DataFrame, symbol: str, timeframe: str) -> CandleCoverage:
    if frame.empty:
        return CandleCoverage(symbol, timeframe, 0, None, None, 0)
    ordered = frame.sort_values("open_time")
    freq = {"1d": "1D", "4h": "4h", "1h": "1h", "15m": "15min"}[timeframe]
    actual = pd.DatetimeIndex(pd.to_datetime(ordered["open_time"], utc=True))
    expected = pd.date_range(actual[0], actual[-1], freq=freq)
    missing = len(expected.difference(actual))
    return CandleCoverage(
        symbol=symbol,
        timeframe=timeframe,
        rows=len(ordered),
        start=actual[0].to_pydatetime(),
        end=actual[-1].to_pydatetime(),
        missing_intervals=missing,
    )
