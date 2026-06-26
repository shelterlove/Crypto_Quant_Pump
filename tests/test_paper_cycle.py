from datetime import UTC, datetime

import pandas as pd
import pytest

from crypto_quant.config.settings import AppConfig
from crypto_quant.paper.cycle import FileLock, PaperCycleDataStale, PaperCycleLocked, PaperCycleResult, PaperCycleRunner
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


def test_paper_cycle_writes_latest_status_files(tmp_path) -> None:  # type: ignore[no-untyped-def]
    state_path = tmp_path / "paper_state" / "main.json"
    runner = PaperCycleRunner(AppConfig(database_url="sqlite://"), state_path=state_path)
    result = PaperCycleResult(
        status="completed",
        run_id=42,
        orders=1,
        buys=1,
        sells=0,
        candidates=3,
        open_positions=1,
        equity=1010,
        pnl=10,
        return_pct=0.01,
        signal_time=datetime(2024, 1, 1, tzinfo=UTC),
        fill_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        report_path="/tmp/reports/42/report.html",
        state_path=str(state_path),
        working_dir="/tmp/project",
        synced_rows=10,
        sync_errors=0,
        sync_error_symbols=[],
        latest_candle_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        expected_candle_time=datetime(2024, 1, 1, 1, tzinfo=UTC),
        lag_seconds=0,
        market_phase="expanding",
        market_entry_mode="normal",
        pump_regime="HOT",
    )

    runner._write_latest_status(result)

    text = (state_path.parent / "latest_status.txt").read_text(encoding="utf-8")
    data = (state_path.parent / "latest_status.json").read_text(encoding="utf-8")
    assert "pnl=10.00" in text
    assert "candidates=3" in text
    assert '"run_id": 42' in data
