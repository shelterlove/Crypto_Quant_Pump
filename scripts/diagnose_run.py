from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text

HORIZONS = [1, 4, 12, 24, 48, 72]
TOP_BUCKETS = [3, 5, 10]


@dataclass(frozen=True)
class RunContext:
    run_id: int
    exchange: str
    slippage_bps: float | None
    initial_equity: float


def load_run_context(engine: Any, run_id: int) -> RunContext:
    run = pd.read_sql(
        text("select id, config from strategy_runs where id = :run_id"),
        engine,
        params={"run_id": run_id},
    )
    if run.empty:
        raise ValueError(f"strategy_run_id not found: {run_id}")
    config = run["config"].iloc[0]
    if isinstance(config, str):
        config = json.loads(config)
    return RunContext(
        run_id=run_id,
        exchange=str(config.get("exchange_id", "binance")),
        slippage_bps=_optional_float(config.get("backtest", {}).get("slippage_bps")),
        initial_equity=float(config.get("backtest", {}).get("initial_equity", 100_000)),
    )


def load_run_frames(engine: Any, run_id: int) -> dict[str, pd.DataFrame]:
    return {
        "factors": pd.read_sql(
            text(
                """
                select time, symbol, final_score
                from factor_scores
                where strategy_run_id = :run_id
                order by time, final_score desc
                """
            ),
            engine,
            params={"run_id": run_id},
        ),
        "rejected": pd.read_sql(
            text(
                """
                select time, symbol, reason
                from rejected_signals
                where strategy_run_id = :run_id
                order by time
                """
            ),
            engine,
            params={"run_id": run_id},
        ),
        "orders": pd.read_sql(
            text(
                """
                select
                    time, symbol, side, quantity, expected_price, filled_price, fee, slippage, status,
                    reason, mechanism, trigger, details
                from orders
                where strategy_run_id = :run_id
                order by time, id
                """
            ),
            engine,
            params={"run_id": run_id},
        ),
        "positions": pd.read_sql(
            text(
                """
                select symbol, state, entry_price, current_price, atr, stop_price, opened_at, closed_at
                from positions
                where strategy_run_id = :run_id
                order by opened_at, id
                """
            ),
            engine,
            params={"run_id": run_id},
        ),
        "equity": pd.read_sql(
            text(
                """
                select time, equity, cash, gross_exposure, drawdown
                from equity_curve
                where strategy_run_id = :run_id
                order by time
                """
            ),
            engine,
            params={"run_id": run_id},
        ),
    }


def load_candles(engine: Any, exchange: str, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not symbols:
        return pd.DataFrame(columns=["symbol", "open_time", "open", "high", "low", "close"])
    rows = pd.read_sql(
        text(
            """
            select symbol, open_time, open, high, low, close
            from candles
            where exchange = :exchange
              and timeframe = '1h'
              and symbol = any(:symbols)
              and open_time >= :start
              and open_time <= :end
            order by symbol, open_time
            """
        ),
        engine,
        params={
            "exchange": exchange,
            "symbols": symbols,
            "start": start.to_pydatetime(),
            "end": end.to_pydatetime(),
        },
    )
    for column in ["open", "high", "low", "close"]:
        rows[column] = rows[column].astype(float)
    rows["open_time"] = pd.to_datetime(rows["open_time"], utc=True)
    return rows


def build_candle_lookup(candles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    lookup: dict[str, pd.DataFrame] = {}
    if candles.empty:
        return lookup
    for symbol, frame in candles.groupby("symbol", sort=False):
        lookup[str(symbol)] = frame.sort_values("open_time").set_index("open_time")
    return lookup


def top_forward_returns(factors: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if factors.empty or not candles:
        return pd.DataFrame()
    ranked = factors.copy()
    ranked["time"] = pd.to_datetime(ranked["time"], utc=True)
    ranked["rank"] = ranked.groupby("time")["final_score"].rank(method="first", ascending=False)
    rows: list[dict[str, object]] = []
    for bucket in TOP_BUCKETS:
        selected = ranked[ranked["rank"] <= bucket]
        for horizon in HORIZONS:
            returns = [
                value
                for value in (
                    forward_return(candles, str(row.symbol), row.time, horizon)
                    for row in selected.itertuples(index=False)
                )
                if value is not None
            ]
            rows.append(return_summary({"bucket": f"top{bucket}", "horizon_hours": horizon}, returns))
    return pd.DataFrame(rows)


def rejected_forward_returns(rejected: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if rejected.empty or not candles:
        return pd.DataFrame()
    rejected = rejected.copy()
    rejected["time"] = pd.to_datetime(rejected["time"], utc=True)
    rows: list[dict[str, object]] = []
    for reason, group in rejected.groupby("reason", sort=True):
        for horizon in HORIZONS:
            returns = [
                value
                for value in (
                    forward_return(candles, str(row.symbol), row.time, horizon)
                    for row in group.itertuples(index=False)
                )
                if value is not None
            ]
            rows.append(return_summary({"reason": reason, "horizon_hours": horizon}, returns))
    return pd.DataFrame(rows)


def false_breakout_diagnostics(factors: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if factors.empty or not candles:
        return pd.DataFrame()
    ranked = factors.copy()
    ranked["time"] = pd.to_datetime(ranked["time"], utc=True)
    ranked["rank"] = ranked.groupby("time")["final_score"].rank(method="first", ascending=False)
    rows: list[dict[str, object]] = []
    for row in ranked[ranked["rank"] <= 10].itertuples(index=False):
        window = future_window(candles, str(row.symbol), row.time, 24)
        if len(window) < 2:
            continue
        base_close = float(window["close"].iloc[0])
        if base_close <= 0:
            continue
        final_return = float(window["close"].iloc[-1]) / base_close - 1
        mfe = float(window["high"].max()) / base_close - 1
        mae = float(window["low"].min()) / base_close - 1
        rows.append(
            {
                "time": row.time,
                "symbol": row.symbol,
                "rank": int(row.rank),
                "final_score": float(row.final_score),
                "return_24h_pct": final_return * 100,
                "mfe_24h_pct": mfe * 100,
                "mae_24h_pct": mae * 100,
                "surged_then_reverted": bool(mfe > 0 and final_return <= max(0, mfe * 0.25)),
            }
        )
    return pd.DataFrame(rows)


def false_breakout_summary(breakouts: pd.DataFrame) -> pd.DataFrame:
    if breakouts.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "samples": len(breakouts),
                "surged_then_reverted_rate": float(breakouts["surged_then_reverted"].mean()),
                "mean_return_24h_pct": float(breakouts["return_24h_pct"].mean()),
                "mean_mfe_24h_pct": float(breakouts["mfe_24h_pct"].mean()),
                "mean_mae_24h_pct": float(breakouts["mae_24h_pct"].mean()),
            }
        ]
    )


def actual_trade_excursions(positions: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if positions.empty or not candles:
        return pd.DataFrame()
    opened = positions[positions["state"] == "open"].copy()
    closed = positions[positions["state"] == "closed"].copy()
    if opened.empty or closed.empty:
        return pd.DataFrame()
    opened["opened_at"] = pd.to_datetime(opened["opened_at"], utc=True)
    closed["opened_at"] = pd.to_datetime(closed["opened_at"], utc=True)
    closed["closed_at"] = pd.to_datetime(closed["closed_at"], utc=True)
    rows: list[dict[str, object]] = []
    for row in opened.itertuples(index=False):
        matches = closed[(closed["symbol"] == row.symbol) & (closed["opened_at"] == row.opened_at)]
        if matches.empty:
            continue
        close_row = matches.iloc[0]
        entry_price = float(row.entry_price)
        atr = float(row.atr)
        if entry_price <= 0 or atr <= 0:
            continue
        window = future_window(candles, str(row.symbol), row.opened_at, int((close_row.closed_at - row.opened_at).total_seconds() / 3600))
        if window.empty:
            continue
        high = float(window["high"].max())
        low = float(window["low"].min())
        close_price = float(close_row.current_price)
        mfe_price = high - entry_price
        mae_price = low - entry_price
        rows.append(
            {
                "symbol": row.symbol,
                "opened_at": row.opened_at,
                "closed_at": close_row.closed_at,
                "hold_hours": (close_row.closed_at - row.opened_at).total_seconds() / 3600,
                "entry_price": entry_price,
                "atr": atr,
                "mfe_atr": mfe_price / atr,
                "mae_atr": mae_price / atr,
                "mfe_pct": mfe_price / entry_price * 100,
                "mae_pct": mae_price / entry_price * 100,
                "exit_return_pct": (close_price / entry_price - 1) * 100,
                "hit_1atr": mfe_price >= atr,
                "hit_3atr": mfe_price >= atr * 3,
            }
        )
    return pd.DataFrame(rows)


def actual_trade_excursion_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "closed_trades": len(trades),
                "hit_1atr_count": int(trades["hit_1atr"].sum()),
                "hit_1atr_rate": float(trades["hit_1atr"].mean()),
                "hit_3atr_count": int(trades["hit_3atr"].sum()),
                "hit_3atr_rate": float(trades["hit_3atr"].mean()),
                "median_mfe_atr": float(trades["mfe_atr"].median()),
                "mean_mfe_atr": float(trades["mfe_atr"].mean()),
                "median_hold_hours": float(trades["hold_hours"].median()),
            }
        ]
    )


def equity_order_summary(context: RunContext, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    equity = frames["equity"]
    orders = frames["orders"]
    final_equity = float(equity["equity"].iloc[-1]) if not equity.empty else context.initial_equity
    max_drawdown = float(equity["drawdown"].min()) if not equity.empty else 0.0
    fees = float(orders["fee"].astype(float).sum()) if not orders.empty else 0.0
    sells = orders[orders["side"] == "sell"] if not orders.empty else pd.DataFrame()
    buys = orders[orders["side"] == "buy"] if not orders.empty else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "run_id": context.run_id,
                "exchange": context.exchange,
                "slippage_bps": context.slippage_bps,
                "initial_equity": context.initial_equity,
                "final_equity": final_equity,
                "return_pct": (final_equity / context.initial_equity - 1) * 100 if context.initial_equity else 0.0,
                "max_drawdown_pct": max_drawdown * 100,
                "orders": len(orders),
                "buy_orders": len(buys),
                "sell_orders": len(sells),
                "fees": fees,
                "atr_stop_count": int((sells["reason"] == "atr_stop").sum()) if not sells.empty else 0,
                "trailing_stop_count": int((sells["reason"] == "trailing_stop").sum()) if not sells.empty else 0,
                "breakeven_stop_count": (
                    int((sells["mechanism"] == "breakeven_stop").sum()) if not sells.empty and "mechanism" in sells else 0
                ),
                "defensive_exit_count": (
                    int((sells["mechanism"] == "defensive_exit").sum()) if not sells.empty and "mechanism" in sells else 0
                ),
            }
        ]
    )


def reason_counts(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    rejected = frames["rejected"]
    orders = frames["orders"]
    return {
        "rejected_reason_counts": value_counts_frame(rejected, "reason"),
        "order_reason_counts": value_counts_frame(orders, "reason"),
        "exit_mechanism_counts": exit_mechanism_counts(orders),
    }


def value_counts_frame(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if frame.empty or column not in frame:
        return pd.DataFrame(columns=[column, "count"])
    return frame[column].value_counts().rename_axis(column).reset_index(name="count")


def exit_mechanism_counts(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame(columns=["mechanism", "count"])
    sells = orders[orders["side"] == "sell"].copy()
    if sells.empty:
        return pd.DataFrame(columns=["mechanism", "count"])
    if "mechanism" not in sells:
        sells["mechanism"] = sells["reason"]
    sells["mechanism"] = sells["mechanism"].fillna(sells["reason"])
    return sells["mechanism"].value_counts().rename_axis("mechanism").reset_index(name="count")


def return_summary(prefix: dict[str, object], returns: list[float]) -> dict[str, object]:
    row = dict(prefix)
    row.update(
        {
            "samples": len(returns),
            "mean_return_pct": sum(returns) / len(returns) * 100 if returns else 0.0,
            "median_return_pct": float(pd.Series(returns).median() * 100) if returns else 0.0,
            "win_rate": sum(1 for value in returns if value > 0) / len(returns) if returns else 0.0,
        }
    )
    return row


def forward_return(candles: dict[str, pd.DataFrame], symbol: str, time: pd.Timestamp, horizon_hours: int) -> float | None:
    frame = candles.get(symbol)
    if frame is None or frame.empty:
        return None
    start = pd.Timestamp(time)
    end = start + pd.Timedelta(hours=horizon_hours)
    left = int(frame.index.searchsorted(start, side="left"))
    right = int(frame.index.searchsorted(end, side="right"))
    if right - left < 2:
        return None
    start_close = float(frame["close"].iloc[left])
    end_close = float(frame["close"].iloc[right - 1])
    if start_close <= 0:
        return None
    return end_close / start_close - 1


def future_window(candles: dict[str, pd.DataFrame], symbol: str, time: pd.Timestamp, horizon_hours: int) -> pd.DataFrame:
    frame = candles.get(symbol)
    if frame is None or frame.empty:
        return pd.DataFrame()
    start = pd.Timestamp(time)
    end = start + pd.Timedelta(hours=horizon_hours)
    left = int(frame.index.searchsorted(start, side="left"))
    right = int(frame.index.searchsorted(end, side="right"))
    return frame.iloc[left:right]


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def write_markdown_summary(out_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    overview = frames["overview"].iloc[0].to_dict() if not frames["overview"].empty else {}
    top_24 = frames["signal_quality_forward_returns"]
    top_24 = top_24[top_24["horizon_hours"] == 24] if not top_24.empty else top_24
    rejected_24 = frames["rejected_signal_forward_returns"]
    rejected_24 = rejected_24[rejected_24["horizon_hours"] == 24] if not rejected_24.empty else rejected_24
    breakout = frames["false_breakout_summary"].iloc[0].to_dict() if not frames["false_breakout_summary"].empty else {}
    trade_excursions = (
        frames["actual_trade_excursion_summary"].iloc[0].to_dict()
        if not frames["actual_trade_excursion_summary"].empty
        else {}
    )
    lines = [
        "# Run Diagnostic Summary",
        "",
        f"- run_id: {overview.get('run_id', '')}",
        f"- exchange: {overview.get('exchange', '')}",
        f"- final_equity: {overview.get('final_equity', 0):,.2f}",
        f"- return_pct: {overview.get('return_pct', 0):.2f}",
        f"- max_drawdown_pct: {overview.get('max_drawdown_pct', 0):.2f}",
        f"- orders: {overview.get('orders', 0)}",
        f"- fees: {overview.get('fees', 0):,.2f}",
        f"- atr_stop_count: {overview.get('atr_stop_count', 0)}",
        f"- trailing_stop_count: {overview.get('trailing_stop_count', 0)}",
        f"- breakeven_stop_count: {overview.get('breakeven_stop_count', 0)}",
        f"- defensive_exit_count: {overview.get('defensive_exit_count', 0)}",
        "",
        "## Top Signal 24h Forward Returns",
        markdown_table(top_24) if not top_24.empty else "No data.",
        "",
        "## Rejected Signal 24h Forward Returns",
        markdown_table(rejected_24) if not rejected_24.empty else "No data.",
        "",
        "## False Breakout Summary",
        "\n".join(f"- {key}: {value}" for key, value in breakout.items()) if breakout else "No data.",
        "",
        "## Actual Trade Excursion Summary",
        "\n".join(f"- {key}: {value}" for key, value in trade_excursions.items()) if trade_excursions else "No data.",
        "",
        "## Exit Mechanism Counts",
        markdown_table(frames["exit_mechanism_counts"]) if not frames["exit_mechanism_counts"].empty else "No data.",
    ]
    (out_dir / "diagnostic_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [columns, ["---" for _ in columns]]
    for row in frame.itertuples(index=False):
        rows.append([format_table_value(value) for value in row])
    return "\n".join("| " + " | ".join(values) + " |" for values in rows)


def format_table_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def run(database_url: str, run_id: int, output_dir: Path) -> dict[str, pd.DataFrame]:
    engine = create_engine(database_url, future=True)
    context = load_run_context(engine, run_id)
    frames = load_run_frames(engine, run_id)
    time_frames = [frames["factors"], frames["rejected"]]
    start = min(pd.to_datetime(frame["time"], utc=True).min() for frame in time_frames if not frame.empty)
    end = max(pd.to_datetime(frame["time"], utc=True).max() for frame in time_frames if not frame.empty)
    symbols = sorted(set(frames["factors"]["symbol"].astype(str)) | set(frames["rejected"]["symbol"].astype(str)))
    candles = build_candle_lookup(load_candles(engine, context.exchange, symbols, start, end + pd.Timedelta(hours=max(HORIZONS))))
    output = {
        "overview": equity_order_summary(context, frames),
        "signal_quality_forward_returns": top_forward_returns(frames["factors"], candles),
        "rejected_signal_forward_returns": rejected_forward_returns(frames["rejected"], candles),
        "false_breakout_diagnostics": false_breakout_diagnostics(frames["factors"], candles),
    }
    output["false_breakout_summary"] = false_breakout_summary(output["false_breakout_diagnostics"])
    output["actual_trade_excursions"] = actual_trade_excursions(frames["positions"], candles)
    output["actual_trade_excursion_summary"] = actual_trade_excursion_summary(output["actual_trade_excursions"])
    output.update(reason_counts(frames))
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in output.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    write_markdown_summary(output_dir, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lightweight signal-quality diagnostics for a strategy run.")
    parser.add_argument("--database-url", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = run(args.database_url, args.run_id, args.output_dir)
    overview = output["overview"].iloc[0]
    print(
        f"run_id={int(overview.run_id)} exchange={overview.exchange} "
        f"return={float(overview.return_pct):.2f}% max_dd={float(overview.max_drawdown_pct):.2f}% "
        f"orders={int(overview.orders)} atr_stops={int(overview.atr_stop_count)} "
        f"out={args.output_dir}"
    )


if __name__ == "__main__":
    main()
