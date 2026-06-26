from datetime import UTC, datetime

import pandas as pd

from crypto_quant.analysis.futures import write_futures_coverage_report
from crypto_quant.config.settings import load_config
from crypto_quant.storage.candles import upsert_candles


def test_futures_coverage_report_writes_summary_and_candidates(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    start = datetime(2024, 1, 5, 4, tzinfo=UTC)
    end = datetime(2024, 1, 5, 12, tzinfo=UTC)
    times = pd.date_range("2024-01-01", periods=180, freq="h", tz="UTC")
    close = pd.Series([1.02**i for i in range(len(times))], dtype=float)
    frame = pd.DataFrame(
        {
            "open_time": times,
            "open": close,
            "high": close * 1.05,
            "low": close * 0.98,
            "close": close,
            "volume": [1_000_000] * len(times),
            "quote_volume": [1_000_000] * len(times),
        }
    )
    upsert_candles(sqlite_session, "binance_usdm", "NEW/USDT", "1h", frame)
    sqlite_session.commit()
    cfg = load_config("configs/futures_1x.yaml")

    paths = write_futures_coverage_report(
        sqlite_session,
        cfg,
        start,
        end,
        tmp_path,
        spot_symbols=["BTC/USDT"],
        futures_symbols=["BTC/USDT", "NEW/USDT"],
    )

    summary = pd.read_csv(paths.summary_csv)
    candidates = pd.read_csv(paths.candidates_csv)
    assert paths.html.exists()
    assert int(summary.loc[summary["metric"] == "futures_only_symbols", "value"].iloc[0]) == 1
    assert not candidates.empty
    assert set(candidates["symbol"]) == {"NEW/USDT"}
