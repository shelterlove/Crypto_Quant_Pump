from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.data.binance import BinanceSpotDataProvider, BinanceUsdmDataProvider
from crypto_quant.storage.candles import distinct_candle_symbols, load_candles
from crypto_quant.utils.time import ensure_utc


@dataclass(frozen=True)
class FuturesCoveragePaths:
    directory: Path
    summary_csv: Path
    candidates_csv: Path
    html: Path


def write_futures_coverage_report(
    session: Session,
    cfg: AppConfig,
    start: datetime,
    end: datetime,
    out_dir: Path = Path("reports/futures_diagnostics"),
    spot_symbols: list[str] | None = None,
    futures_symbols: list[str] | None = None,
) -> FuturesCoveragePaths:
    start = ensure_utc(start)
    end = ensure_utc(end)
    spot_set = set(spot_symbols if spot_symbols is not None else BinanceSpotDataProvider("binance").fetch_usdt_spot_symbols())
    futures_set = set(
        futures_symbols if futures_symbols is not None else BinanceUsdmDataProvider("binance_usdm").fetch_usdt_perp_symbols()
    )
    common = spot_set & futures_set
    spot_only = spot_set - futures_set
    futures_only = futures_set - spot_set

    futures_db_symbols = set(distinct_candle_symbols(session, "binance_usdm", "1h"))
    spot_db_symbols = set(distinct_candle_symbols(session, "binance", "1h"))
    futures_only_with_data = sorted(futures_only & futures_db_symbols)
    shared_with_data = sorted(common & futures_db_symbols)
    candidates = _futures_only_candidates(session, cfg, futures_only_with_data, start, end)
    shared_candidates = _candidate_count(session, cfg, shared_with_data, start, end)

    summary = pd.DataFrame(
        [
            {"metric": "spot_symbols", "value": len(spot_set)},
            {"metric": "futures_symbols", "value": len(futures_set)},
            {"metric": "common_symbols", "value": len(common)},
            {"metric": "spot_only_symbols", "value": len(spot_only)},
            {"metric": "futures_only_symbols", "value": len(futures_only)},
            {"metric": "spot_symbols_with_local_data", "value": len(spot_db_symbols)},
            {"metric": "futures_symbols_with_local_data", "value": len(futures_db_symbols)},
            {"metric": "futures_only_symbols_with_local_data", "value": len(futures_only_with_data)},
            {"metric": "futures_only_pump_candidates", "value": len(candidates)},
            {"metric": "shared_futures_pump_candidates", "value": shared_candidates},
            {
                "metric": "futures_only_candidate_overlap_rate",
                "value": 0.0 if len(candidates) == 0 else 1.0 - (len(candidates) / max(len(candidates) + shared_candidates, 1)),
            },
            {"metric": "futures_only_candidate_median_mfe_24h", "value": _safe_median(candidates, "mfe_24h")},
            {"metric": "futures_only_candidate_median_return_24h", "value": _safe_median(candidates, "return_24h_forward")},
            {"metric": "futures_only_candidate_winners_24h", "value": _winner_count(candidates, "return_24h_forward")},
        ]
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "summary.csv"
    candidates_csv = out_dir / "futures_only_candidates.csv"
    html = out_dir / "index.html"
    summary.to_csv(summary_csv, index=False)
    candidates.to_csv(candidates_csv, index=False)
    html.write_text(_html(summary, candidates, start, end), encoding="utf-8")
    return FuturesCoveragePaths(out_dir, summary_csv, candidates_csv, html)


def _futures_only_candidates(
    session: Session,
    cfg: AppConfig,
    symbols: list[str],
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    if not symbols:
        return _empty_candidate_frame()
    warmup_start = start - timedelta(hours=100)
    lookahead_end = end + timedelta(hours=72)
    candles = load_candles(session, "binance_usdm", symbols, "1h", warmup_start, lookahead_end)
    rows: list[dict[str, object]] = []
    for symbol, frame in candles.items():
        if frame.empty:
            continue
        data = frame.sort_values("open_time").reset_index(drop=True).copy()
        data["open_time"] = pd.to_datetime(data["open_time"], utc=True)
        data["ret_6h"] = data["close"].pct_change(6)
        data["ret_24h"] = data["close"].pct_change(24)
        data["ret_72h"] = data["close"].pct_change(72)
        data["quote_volume_24h"] = data["quote_volume"].rolling(24).sum()
        qv_6h = data["quote_volume"].rolling(6).sum()
        qv_30_avg = data["quote_volume"].rolling(24 * 30, min_periods=24).sum() / 30
        data["volume_ratio"] = qv_6h / qv_30_avg.replace(0, pd.NA)
        window = data[(data["open_time"] >= pd.Timestamp(start)) & (data["open_time"] <= pd.Timestamp(end))]
        matches = window[
            (window["ret_24h"] >= cfg.pump_mode.min_24h_return)
            & (window["ret_72h"] >= cfg.pump_mode.min_72h_return)
            & (window["ret_6h"] >= cfg.pump_mode.early_6h_return)
            & (window["quote_volume_24h"] >= cfg.pump_mode.min_quote_volume_24h)
            & (window["volume_ratio"] >= cfg.pump_mode.volume_ratio_min)
        ]
        for match in matches.itertuples(index=True):
            index = int(match.Index)
            entry = float(match.close)
            future_24 = data.iloc[index + 1 : index + 25]
            future_72 = data.iloc[index + 1 : index + 73]
            rows.append(
                {
                    "symbol": symbol,
                    "time": match.open_time,
                    "ret_6h": float(match.ret_6h),
                    "ret_24h": float(match.ret_24h),
                    "ret_72h": float(match.ret_72h),
                    "volume_ratio": float(match.volume_ratio),
                    "quote_volume_24h": float(match.quote_volume_24h),
                    "mfe_24h": _mfe(future_24, entry),
                    "return_24h_forward": _forward_return(future_24, entry),
                    "mfe_72h": _mfe(future_72, entry),
                    "return_72h_forward": _forward_return(future_72, entry),
                }
            )
    if not rows:
        return _empty_candidate_frame()
    return pd.DataFrame(rows).sort_values(["time", "symbol"]).reset_index(drop=True)


def _empty_candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "time",
            "ret_6h",
            "ret_24h",
            "ret_72h",
            "volume_ratio",
            "quote_volume_24h",
            "mfe_24h",
            "return_24h_forward",
            "mfe_72h",
            "return_72h_forward",
        ]
    )


def _candidate_count(session: Session, cfg: AppConfig, symbols: list[str], start: datetime, end: datetime) -> int:
    return len(_futures_only_candidates(session, cfg, symbols, start, end))


def _mfe(frame: pd.DataFrame, entry: float) -> float:
    if frame.empty or entry <= 0:
        return 0.0
    return float(frame["high"].max()) / entry - 1


def _forward_return(frame: pd.DataFrame, entry: float) -> float:
    if frame.empty or entry <= 0:
        return 0.0
    return float(frame["close"].iloc[-1]) / entry - 1


def _safe_median(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].median())


def _winner_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame:
        return 0
    return int((frame[column] > 0).sum())


def _html(summary: pd.DataFrame, candidates: pd.DataFrame, start: datetime, end: datetime) -> str:
    summary_table = summary.to_html(index=False)
    sample = candidates.head(100).to_html(index=False) if not candidates.empty else "<p>No futures-only candidates found in local data.</p>"
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Futures Coverage Diagnostics</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 6px; }}
  </style>
</head>
<body>
  <h1>Binance USD-M Futures Coverage Diagnostics</h1>
  <p>Window: {start.isoformat()} to {end.isoformat()}</p>
  <h2>Summary</h2>
  {summary_table}
  <h2>Futures-Only Pump Candidates</h2>
  {sample}
</body>
</html>
"""
