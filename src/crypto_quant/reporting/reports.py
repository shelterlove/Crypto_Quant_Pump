from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from crypto_quant.storage.models import (
    Candle,
    EquityCurveRecord,
    FactorScoreRecord,
    OrderRecord,
    PositionRecord,
    RejectedSignalRecord,
    SignalRecord,
    StrategyRun,
)


@dataclass(frozen=True)
class ReportPaths:
    directory: Path
    html: Path
    csv: Path | None = None


class BacktestReportWriter:
    def __init__(self, base_dir: Path = Path("reports")) -> None:
        self.base_dir = base_dir

    def write(self, session: Session, strategy_run_id: int) -> ReportPaths:
        out_dir = self.base_dir / str(strategy_run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        run = session.get(StrategyRun, strategy_run_id)
        if run is None:
            raise ValueError(f"strategy_run_id not found: {strategy_run_id}")
        frames = {
            "orders": self._table(session, OrderRecord, strategy_run_id),
            "positions": self._table(session, PositionRecord, strategy_run_id),
            "equity_curve": self._table(session, EquityCurveRecord, strategy_run_id),
            "signals": self._table(session, SignalRecord, strategy_run_id),
            "rejected_signals": self._table(session, RejectedSignalRecord, strategy_run_id),
            "factor_scores_top": self._table(session, FactorScoreRecord, strategy_run_id),
        }
        signal_quality = self._signal_quality_frames(session, run, frames)
        frames.update(signal_quality)
        for name, frame in frames.items():
            frame.to_csv(out_dir / f"{name}.csv", index=False)
        html = out_dir / "report.html"
        html.write_text(self._html(run, frames), encoding="utf-8")
        return ReportPaths(out_dir, html)

    def write_overview(
        self,
        run_ids: list[int],
        rows: list[dict[str, object]],
        warnings: list[str] | None = None,
    ) -> ReportPaths:
        out_dir = self.base_dir / "overview"
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        csv = out_dir / "slippage_pressure_overview.csv"
        frame.to_csv(csv, index=False)
        html = out_dir / "index.html"
        html.write_text(self._overview_html(run_ids, frame, warnings or []), encoding="utf-8")
        return ReportPaths(out_dir, html, csv)

    def _table(self, session: Session, model: Any, strategy_run_id: int) -> pd.DataFrame:
        rows = session.execute(select(model).where(model.strategy_run_id == strategy_run_id)).scalars().all()
        data: list[dict[str, object]] = []
        for row in rows:
            data.append(
                {
                    key: value
                    for key, value in vars(row).items()
                    if not key.startswith("_") and key not in {"metadata_"}
                }
            )
        return pd.DataFrame(data)

    def _html(self, run: StrategyRun, frames: dict[str, pd.DataFrame]) -> str:
        equity = frames["equity_curve"]
        orders = frames["orders"]
        rejected = frames["rejected_signals"]
        forward_returns = frames.get("signal_quality_forward_returns", pd.DataFrame())
        rejected_returns = frames.get("rejected_signal_forward_returns", pd.DataFrame())
        false_breakouts = frames.get("false_breakout_diagnostics", pd.DataFrame())
        exit_breakdown = self._exit_mechanism_breakdown(orders)
        start_equity = float(equity["equity"].iloc[0]) if not equity.empty else float(run.config["backtest"]["initial_equity"])
        final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else start_equity
        total_return = final_equity / start_equity - 1 if start_equity else 0
        max_drawdown = float(equity["drawdown"].min()) if not equity.empty else 0
        buys = orders[orders["side"] == "buy"] if not orders.empty else pd.DataFrame()
        sells = orders[orders["side"] == "sell"] if not orders.empty else pd.DataFrame()
        total_fees = float(orders["fee"].sum()) if not orders.empty else 0
        trade_stats = self._trade_stats(orders)
        exchange = run.config.get("exchange_id", "unknown")
        rejection_stats = (
            rejected["reason"].value_counts().to_frame("count").reset_index().to_html(index=False)
            if not rejected.empty
            else "<p>No rejected signals.</p>"
        )
        forward_table = (
            forward_returns.to_html(index=False) if not forward_returns.empty else "<p>No forward-return diagnostics.</p>"
        )
        rejected_table = (
            rejected_returns.to_html(index=False) if not rejected_returns.empty else "<p>No rejected-signal diagnostics.</p>"
        )
        breakout_table = (
            false_breakouts.head(50).to_html(index=False)
            if not false_breakouts.empty
            else "<p>No false-breakout diagnostics.</p>"
        )
        exit_table = exit_breakdown.to_html(index=False) if not exit_breakdown.empty else "<p>No exit-mechanism diagnostics.</p>"
        equity_rows = self._svg_polyline(equity, "equity") if not equity.empty else ""
        drawdown_rows = self._svg_polyline(equity, "drawdown") if not equity.empty else ""
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Crypto Quant Backtest {run.id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 12px; }}
    .metric {{ border: 1px solid #d8dee4; border-radius: 6px; padding: 12px; }}
    .metric span {{ display: block; color: #667085; font-size: 12px; }}
    svg {{ width: 100%; height: 260px; border: 1px solid #d8dee4; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 6px; }}
    .warning {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>MVP Backtest Report</h1>
  <p>Run ID: {run.id} | Exchange: {exchange} | Strategy: {run.strategy_version} | Status: {run.status}</p>
  <div class="warning">
    Survivorship bias risk: true. This first pass uses current Binance spot symbols
    and does not fully include historical delisted coins.
  </div>
  <div class="grid">
    <div class="metric"><span>Initial Equity</span>{start_equity:,.2f}</div>
    <div class="metric"><span>Final Equity</span>{final_equity:,.2f}</div>
    <div class="metric"><span>Total Return</span>{total_return:.2%}</div>
    <div class="metric"><span>Max Drawdown</span>{max_drawdown:.2%}</div>
    <div class="metric"><span>Orders</span>{len(orders)}</div>
    <div class="metric"><span>Entries</span>{len(buys)}</div>
    <div class="metric"><span>Exits</span>{len(sells)}</div>
    <div class="metric"><span>Total Fees</span>{total_fees:,.2f}</div>
    <div class="metric"><span>Closed Trade Win Rate</span>{trade_stats["closed_trade_win_rate"]:.2%}</div>
    <div class="metric"><span>Net Win Rate</span>{trade_stats["net_win_rate"]:.2%}</div>
    <div class="metric"><span>ATR Stop Count</span>{trade_stats["atr_stop_count"]}</div>
  </div>
  <h2>Equity Curve</h2>
  <svg viewBox="0 0 800 260">{equity_rows}</svg>
  <h2>Drawdown</h2>
  <svg viewBox="0 0 800 260">{drawdown_rows}</svg>
  <h2>Rejected Signal Reasons</h2>
  {rejection_stats}
  <h2>Exit Mechanism Breakdown</h2>
  {exit_table}
  <h2>Signal Quality</h2>
  <h3>Top Signal Forward Returns</h3>
  {forward_table}
  <h3>Rejected Signal Forward Returns</h3>
  {rejected_table}
  <h3>False Breakout Diagnostics</h3>
  {breakout_table}
  <h2>Whitepaper Coverage Notes</h2>
  <ul>
    <li>
      Implemented: 1H/4H closed candles, weekly universe, BTC 4H MA50 filter,
      breadth, raw momentum Top3, 1H ATR sizing, basic ATR stop, fees and slippage.
    </li>
    <li>
      Not implemented in this pass: 15m fast-risk condition, order book depth filter,
      full volume/trend factors, volume-stall filter,
      delisted-symbol backfill.
    </li>
  </ul>
</body>
</html>
"""

    def _signal_quality_frames(
        self,
        session: Session,
        run: StrategyRun,
        frames: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        factors = frames["factor_scores_top"]
        rejected = frames["rejected_signals"]
        exchange = str(run.config.get("exchange_id", "binance"))
        start = self._run_start(factors, rejected)
        end = self._run_end(factors, rejected)
        if start is None or end is None:
            return {
                "signal_quality_forward_returns": pd.DataFrame(),
                "rejected_signal_forward_returns": pd.DataFrame(),
                "false_breakout_diagnostics": pd.DataFrame(),
            }
        horizons = [1, 4, 12, 24, 48, 72]
        symbols = sorted(
            set(factors.get("symbol", pd.Series(dtype=str)).astype(str))
            | set(rejected.get("symbol", pd.Series(dtype=str)).astype(str))
        )
        candles = self._load_candle_frame(session, exchange, symbols, start, end + pd.Timedelta(hours=max(horizons)))
        top_returns = self._top_forward_returns(factors, candles, horizons)
        rejected_returns = self._rejected_forward_returns(rejected, candles, horizons)
        breakouts = self._false_breakout_diagnostics(factors, candles)
        return {
            "signal_quality_forward_returns": top_returns,
            "rejected_signal_forward_returns": rejected_returns,
            "false_breakout_diagnostics": breakouts,
        }

    def _run_start(self, *frames: pd.DataFrame) -> pd.Timestamp | None:
        times = [pd.to_datetime(frame["time"], utc=True).min() for frame in frames if not frame.empty and "time" in frame]
        return min(times) if times else None

    def _run_end(self, *frames: pd.DataFrame) -> pd.Timestamp | None:
        times = [pd.to_datetime(frame["time"], utc=True).max() for frame in frames if not frame.empty and "time" in frame]
        return max(times) if times else None

    def _load_candle_frame(
        self,
        session: Session,
        exchange: str,
        symbols: list[str],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        if not symbols:
            return pd.DataFrame()
        rows = session.execute(
            select(Candle.symbol, Candle.open_time, Candle.open, Candle.high, Candle.low, Candle.close)
            .where(Candle.exchange == exchange)
            .where(Candle.symbol.in_(symbols))
            .where(Candle.timeframe == "1h")
            .where(Candle.open_time >= start.to_pydatetime())
            .where(Candle.open_time <= end.to_pydatetime())
            .order_by(Candle.symbol, Candle.open_time)
        ).all()
        return pd.DataFrame(
            [
                {
                    "symbol": symbol,
                    "open_time": open_time,
                    "open": float(open_price),
                    "high": float(high),
                    "low": float(low),
                    "close": float(close),
                }
                for symbol, open_time, open_price, high, low, close in rows
            ]
        )

    def _top_forward_returns(self, factors: pd.DataFrame, candles: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        if factors.empty or candles.empty:
            return pd.DataFrame()
        ranked = factors.copy()
        ranked["time"] = pd.to_datetime(ranked["time"], utc=True)
        ranked["rank"] = ranked.groupby("time")["final_score"].rank(method="first", ascending=False)
        rows: list[dict[str, object]] = []
        for bucket in [3, 5, 10]:
            selected = ranked[ranked["rank"] <= bucket]
            for horizon in horizons:
                returns = [
                    item
                    for item in (
                        self._forward_return(candles, str(row.symbol), row.time, horizon)
                        for row in selected.itertuples(index=False)
                    )
                    if item is not None
                ]
                rows.append(
                    {
                        "bucket": f"top{bucket}",
                        "horizon_hours": horizon,
                        "samples": len(returns),
                        "mean_return_pct": sum(returns) / len(returns) * 100 if returns else 0,
                        "win_rate": sum(1 for value in returns if value > 0) / len(returns) if returns else 0,
                    }
                )
        return pd.DataFrame(rows)

    def _rejected_forward_returns(self, rejected: pd.DataFrame, candles: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
        if rejected.empty or candles.empty:
            return pd.DataFrame()
        rejected = rejected.copy()
        rejected["time"] = pd.to_datetime(rejected["time"], utc=True)
        rows: list[dict[str, object]] = []
        for reason, group in rejected.groupby("reason"):
            for horizon in horizons:
                returns = [
                    item
                    for item in (
                        self._forward_return(candles, str(row.symbol), row.time, horizon)
                        for row in group.itertuples(index=False)
                    )
                    if item is not None
                ]
                rows.append(
                    {
                        "reason": reason,
                        "horizon_hours": horizon,
                        "samples": len(returns),
                        "mean_return_pct": sum(returns) / len(returns) * 100 if returns else 0,
                        "win_rate": sum(1 for value in returns if value > 0) / len(returns) if returns else 0,
                    }
                )
        return pd.DataFrame(rows)

    def _false_breakout_diagnostics(self, factors: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
        if factors.empty or candles.empty:
            return pd.DataFrame()
        ranked = factors.copy()
        ranked["time"] = pd.to_datetime(ranked["time"], utc=True)
        ranked["rank"] = ranked.groupby("time")["final_score"].rank(method="first", ascending=False)
        rows: list[dict[str, object]] = []
        for row in ranked[ranked["rank"] <= 10].itertuples(index=False):
            window = self._future_window(candles, str(row.symbol), row.time, 24)
            if len(window) < 2:
                continue
            base_close = float(window["close"].iloc[0])
            final_close = float(window["close"].iloc[-1])
            mfe = float(window["high"].max()) / base_close - 1 if base_close else 0
            mae = float(window["low"].min()) / base_close - 1 if base_close else 0
            final_return = final_close / base_close - 1 if base_close else 0
            rows.append(
                {
                    "time": row.time,
                    "symbol": row.symbol,
                    "rank": int(row.rank),
                    "mfe_24h_pct": mfe * 100,
                    "mae_24h_pct": mae * 100,
                    "return_24h_pct": final_return * 100,
                    "surged_then_reverted": bool(mfe > 0 and final_return <= max(0, mfe * 0.25)),
                }
            )
        return pd.DataFrame(rows)

    def _forward_return(self, candles: pd.DataFrame, symbol: str, time: pd.Timestamp, horizon_hours: int) -> float | None:
        window = self._future_window(candles, symbol, time, horizon_hours)
        if len(window) < 2:
            return None
        start_close = float(window["close"].iloc[0])
        end_close = float(window["close"].iloc[-1])
        return end_close / start_close - 1 if start_close else None

    def _future_window(self, candles: pd.DataFrame, symbol: str, time: pd.Timestamp, horizon_hours: int) -> pd.DataFrame:
        symbol_candles = candles[candles["symbol"] == symbol].copy()
        if symbol_candles.empty:
            return symbol_candles
        symbol_candles["open_time"] = pd.to_datetime(symbol_candles["open_time"], utc=True)
        end = time + pd.Timedelta(hours=horizon_hours)
        return symbol_candles[(symbol_candles["open_time"] >= time) & (symbol_candles["open_time"] <= end)].sort_values("open_time")

    def _trade_stats(self, orders: pd.DataFrame) -> dict[str, float | int]:
        if orders.empty:
            return {"closed_trade_win_rate": 0.0, "net_win_rate": 0.0, "atr_stop_count": 0}
        open_entries: dict[str, list[float]] = {}
        trade_returns: list[float] = []
        for row in orders.sort_values("time").itertuples(index=False):
            symbol = str(row.symbol)
            price = float(row.filled_price or 0)
            if row.side == "buy":
                open_entries.setdefault(symbol, []).append(price)
            elif row.side == "sell" and open_entries.get(symbol):
                entry = open_entries[symbol].pop(0)
                if entry > 0:
                    trade_returns.append(price / entry - 1)
        closed_win_rate = sum(1 for value in trade_returns if value > 0) / len(trade_returns) if trade_returns else 0.0
        atr_stop_count = int((orders["reason"] == "atr_stop").sum()) if "reason" in orders else 0
        sells = orders[orders["side"] == "sell"]
        net_win_rate = (
            float((sells["filled_price"].astype(float) > sells["expected_price"].astype(float)).mean())
            if not sells.empty
            else 0.0
        )
        return {"closed_trade_win_rate": closed_win_rate, "net_win_rate": net_win_rate, "atr_stop_count": atr_stop_count}

    def _exit_mechanism_breakdown(self, orders: pd.DataFrame) -> pd.DataFrame:
        if orders.empty:
            return pd.DataFrame()
        sells = orders[orders["side"] == "sell"].copy()
        if sells.empty:
            return pd.DataFrame()
        if "mechanism" not in sells:
            sells["mechanism"] = sells["reason"]
        sells["mechanism"] = sells["mechanism"].fillna(sells["reason"])
        return (
            sells.groupby("mechanism", dropna=False)
            .agg(
                exits=("symbol", "count"),
                total_fees=("fee", lambda values: float(pd.Series(values).astype(float).sum())),
                avg_slippage=("slippage", lambda values: float(pd.Series(values).astype(float).mean())),
            )
            .reset_index()
        )

    def _svg_polyline(self, frame: pd.DataFrame, column: str) -> str:
        values = frame[column].astype(float).tolist()
        if len(values) < 2:
            return ""
        low = min(values)
        high = max(values)
        scale = high - low or 1
        points = []
        for index, value in enumerate(values):
            x = index * 800 / (len(values) - 1)
            y = 240 - ((value - low) / scale * 220) + 10
            points.append(f"{x:.2f},{y:.2f}")
        return f'<polyline fill="none" stroke="#2563eb" stroke-width="2" points="{" ".join(points)}" />'

    def _overview_html(self, run_ids: list[int], frame: pd.DataFrame, warnings: list[str]) -> str:
        table = frame.to_html(index=False) if not frame.empty else "<p>No runs completed.</p>"
        warning_items = "".join(f"<li>{warning}</li>" for warning in warnings) or "<li>No data-quality warnings.</li>"
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Crypto Quant MVP Pipeline Overview</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border-bottom: 1px solid #e5e7eb; text-align: left; padding: 6px; }}
    .warning {{ background: #fff7ed; border: 1px solid #fed7aa; padding: 12px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>One-Year MVP Pipeline Overview</h1>
  <p>Run IDs: {", ".join(str(run_id) for run_id in run_ids)}</p>
  <div class="warning">
    Survivorship bias risk: true. This first pass uses current Binance spot symbols
    and does not fully include historical delisted coins.
  </div>
  <h2>Slippage Pressure Test</h2>
  {table}
  <h2>Data Coverage Warnings</h2>
  <ul>{warning_items}</ul>
  <h2>Whitepaper Coverage Notes</h2>
  <ul>
    <li>
      Implemented: 1H/4H closed candles, weekly universe, BTC 4H MA50 filter,
      breadth, raw momentum Top3, 1H ATR sizing, basic ATR stop, fees and slippage.
    </li>
    <li>
      Not implemented in this pass: 15m fast-risk condition, order book depth filter,
      full volume/trend factors, volume-stall filter, trailing take-profit,
      delisted-symbol backfill.
    </li>
  </ul>
</body>
</html>
"""
