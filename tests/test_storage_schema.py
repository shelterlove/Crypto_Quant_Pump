from datetime import UTC, datetime
from typing import Any, cast

import pandas as pd

from crypto_quant.storage import candles
from crypto_quant.storage.models import Candle, OrderRecord, PositionRecord, RejectedSignalRecord, SignalRecord


def test_candle_unique_key_is_declared() -> None:
    constraint_names = {constraint.name for constraint in cast(Any, Candle.__table__).constraints}
    assert "uq_candles_key" in constraint_names


def test_signal_tables_bind_strategy_runs() -> None:
    assert "strategy_run_id" in SignalRecord.__table__.columns
    assert "strategy_run_id" in RejectedSignalRecord.__table__.columns


def test_orders_and_positions_have_required_risk_fields() -> None:
    order_columns = set(OrderRecord.__table__.columns.keys())
    assert {
        "expected_price",
        "limit_price",
        "filled_price",
        "fee",
        "slippage",
        "reason",
        "mechanism",
        "trigger",
        "details",
    } <= order_columns
    position_columns = set(PositionRecord.__table__.columns.keys())
    assert {"atr", "stop_risk_exposure", "volatility_risk_exposure", "current_risk_exposure_tag", "state"} <= position_columns


def test_large_candle_upserts_are_chunked(monkeypatch: Any) -> None:
    calls: list[int] = []

    class Dialect:
        name = "postgresql"

    class Bind:
        dialect = Dialect()

    class FakeSession:
        def get_bind(self) -> Bind:
            return Bind()

    def record_batch(_session: Any, _dialect: str, records: list[dict[str, Any]]) -> None:
        calls.append(len(records))

    monkeypatch.setattr(candles, "_upsert_candle_batch", record_batch)
    rows = candles.UPSERT_BATCH_SIZE + 3
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC"),
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 1.5,
            "volume": 10,
            "quote_volume": 15,
        }
    )
    assert candles.upsert_candles(cast(Any, FakeSession()), "binanceus", "BTC/USDT", "1h", frame) == rows
    assert calls == [candles.UPSERT_BATCH_SIZE, 3]


def test_candle_coverage_handles_timezone_aware_datetimes() -> None:
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"),
            "open": [1, 2, 3],
            "high": [1, 2, 3],
            "low": [1, 2, 3],
            "close": [1, 2, 3],
            "volume": [10, 10, 10],
            "quote_volume": [10, 20, 30],
        }
    )

    result = candles.coverage(frame, "BTC/USDT", "1h")

    assert result.rows == 3
    assert result.missing_intervals == 0
    assert result.start is not None
    assert result.start.tzinfo is not None


def test_complete_candle_coverage_detects_full_range(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 1, 3, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range(start, periods=4, freq="h", tz="UTC"),
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 1.5,
            "volume": 10,
            "quote_volume": 15,
        }
    )
    candles.upsert_candles(sqlite_session, "binance", "BTC/USDT", "1h", frame)
    sqlite_session.commit()

    assert candles.has_complete_candle_coverage(sqlite_session, "binance", "BTC/USDT", "1h", start, end)
    assert not candles.has_complete_candle_coverage(
        sqlite_session,
        "binance",
        "BTC/USDT",
        "1h",
        start,
        datetime(2024, 1, 1, 4, tzinfo=UTC),
    )


def test_complete_candle_coverage_aligns_non_boundary_daily_windows(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-02", periods=3, freq="D", tz="UTC"),
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 1.5,
            "volume": 10,
            "quote_volume": 15,
        }
    )
    candles.upsert_candles(sqlite_session, "binance", "BTC/USDT", "1d", frame)
    sqlite_session.commit()

    assert candles.has_complete_candle_coverage(
        sqlite_session,
        "binance",
        "BTC/USDT",
        "1d",
        datetime(2024, 1, 1, 6, tzinfo=UTC),
        datetime(2024, 1, 4, 6, tzinfo=UTC),
    )


def test_complete_candle_coverage_allows_symbols_listed_after_window_start(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-03", periods=2, freq="D", tz="UTC"),
            "open": 1,
            "high": 2,
            "low": 1,
            "close": 1.5,
            "volume": 10,
            "quote_volume": 15,
        }
    )
    candles.upsert_candles(sqlite_session, "binance", "NEW/USDT", "1d", frame)
    sqlite_session.commit()

    assert candles.has_complete_candle_coverage(
        sqlite_session,
        "binance",
        "NEW/USDT",
        "1d",
        datetime(2024, 1, 1, 6, tzinfo=UTC),
        datetime(2024, 1, 4, 6, tzinfo=UTC),
    )
