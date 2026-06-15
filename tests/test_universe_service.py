from datetime import UTC, datetime, timedelta

import pandas as pd

from crypto_quant.config.settings import load_config
from crypto_quant.storage.candles import upsert_candles
from crypto_quant.storage.models import Base
from crypto_quant.universe.service import WeeklyUniverseService


def test_weekly_universe_uses_prior_30_daily_candles(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    cfg = load_config("configs/mvp.yaml").model_copy(update={"database_url": "sqlite://"})
    effective = datetime(2024, 2, 5, tzinfo=UTC)
    days = pd.date_range(effective - timedelta(days=30), periods=30, freq="D", tz="UTC")
    for symbol, volume in [("AAA/USDT", 100_000_000), ("BBB/USDT", 90_000_000), ("USDC/USDT", 200_000_000)]:
        frame = pd.DataFrame(
            {
                "open_time": days,
                "open": 10,
                "high": 11,
                "low": 9,
                "close": 10,
                "volume": volume / 10,
                "quote_volume": volume,
            }
        )
        upsert_candles(sqlite_session, cfg.exchange_id, symbol, "1d", frame)
    sqlite_session.commit()
    result = WeeklyUniverseService(cfg).build(
        sqlite_session,
        ["AAA/USDT", "BBB/USDT", "USDC/USDT"],
        effective,
        effective,
        persist=False,
    )
    assert result.snapshots[0].symbols == ["AAA/USDT", "BBB/USDT"]


def test_sqlite_session_fixture_has_schema(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    assert Base.metadata.tables
