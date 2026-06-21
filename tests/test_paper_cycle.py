from datetime import UTC, datetime

import pandas as pd
import pytest

from crypto_quant.config.settings import AppConfig
from crypto_quant.paper.cycle import FileLock, PaperCycleDataStale, PaperCycleLocked, PaperCycleRunner
from crypto_quant.storage.candles import upsert_candles


def test_file_lock_rejects_concurrent_acquire(tmp_path) -> None:  # type: ignore[no-untyped-def]
    lock_path = tmp_path / "paper.lock"
    with FileLock(lock_path):
        with pytest.raises(PaperCycleLocked):
            with FileLock(lock_path):
                pass


def test_paper_cycle_rejects_stale_btc_candles(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01 00:00:00+00:00", periods=1, freq="h"),
            "open": [1],
            "high": [1],
            "low": [1],
            "close": [1],
            "volume": [10],
            "quote_volume": [10],
        }
    )
    upsert_candles(sqlite_session, "binance", "BTC/USDT", "1h", frame)
    sqlite_session.commit()

    runner = PaperCycleRunner(
        AppConfig(database_url="sqlite://"),
        state_path=tmp_path / "state.json",
        lock_path=tmp_path / "paper.lock",
        skip_sync=True,
        max_staleness_hours=1,
    )

    with pytest.raises(PaperCycleDataStale, match="latest BTC candle is stale"):
        runner.run(sqlite_session, now=datetime(2024, 1, 1, 6, 5, tzinfo=UTC))
