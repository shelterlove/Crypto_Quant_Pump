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
        return pd.DataFrame(columns=["symbol", "open_time", "open", "high", "low", "close", "volume", "quote_volume"])
    rows = pd.read_sql(
        text(
            """
            select symbol, open_time, open, high, low, close, volume, quote_volume
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
    for column in ["open", "high", "low", "close", "volume", "quote_volume"]:
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


def pump_trade_diagnostics(orders: pd.DataFrame, candles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if orders.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    orders = orders.copy()
    orders["time"] = pd.to_datetime(orders["time"], utc=True)
    buys = orders[orders["side"] == "buy"].copy()
    sells = orders[orders["side"] == "sell"].copy()
    for sell in sells.itertuples(index=False):
        details = parse_details(getattr(sell, "details", None))
        avg_entry = details_float(details, "avg_entry_price")
        exit_price = _optional_float(getattr(sell, "filled_price", None))
        final_ret = details_float(details, "final_trade_ret_pct")
        opened_at = details_timestamp(details, "opened_at")
        if avg_entry is None or exit_price is None or final_ret is None or opened_at is None:
            continue

        symbol = str(sell.symbol)
        trade_buys = buys[
            (buys["symbol"].astype(str) == symbol)
            & (buys["time"] >= opened_at)
            & (buys["time"] <= pd.Timestamp(sell.time))
        ]
        entry_rows = trade_buys[trade_buys["mechanism"].fillna("") == "pump_entry"]
        entry_details = parse_details(entry_rows["details"].iloc[0]) if not entry_rows.empty else {}
        entry_reason = str(entry_rows["reason"].iloc[0]) if not entry_rows.empty else ""
        entry_trigger = str(entry_rows["trigger"].iloc[0]) if not entry_rows.empty else ""
        probe_entry = details_float(details, "probe_entry_price") or details_float(entry_details, "probe_entry_price") or avg_entry
        confirm_entry = details_float(details, "confirm_entry_price")
        stop_anchor = details_float(details, "stop_anchor_price") or avg_entry
        active_stop = details_float(details, "active_stop_price")
        mfe_trade = details_float(details, "mfe_trade_level")
        highest_price = details_float(details, "highest_price")
        quantity = float(sell.quantity)
        sell_fee = float(sell.fee)
        entry_fees = float(trade_buys["fee"].astype(float).sum()) if not trade_buys.empty else 0.0
        gross_pnl = (exit_price - avg_entry) * quantity
        net_pnl = gross_pnl - sell_fee - entry_fees
        ema = ema_metrics(candles, symbol, opened_at)
        classification = classify_entry(entry_reason, entry_trigger)
        rows.append(
            {
                "time": pd.Timestamp(sell.time),
                "symbol": symbol,
                "exit_mechanism": str(sell.mechanism or sell.reason),
                "exit_reason": str(sell.reason),
                "opened_at": opened_at,
                "hold_hours": (pd.Timestamp(sell.time) - opened_at).total_seconds() / 3600,
                "quantity": quantity,
                "avg_entry_price": avg_entry,
                "probe_entry_price": probe_entry,
                "confirm_entry_price": confirm_entry or 0.0,
                "stop_anchor_price": stop_anchor,
                "active_stop_price": active_stop or 0.0,
                "exit_price": exit_price,
                "final_trade_ret_pct": final_ret * 100,
                "mfe_trade_pct": (mfe_trade * 100) if mfe_trade is not None else 0.0,
                "mae_atr": details_float(details, "mae_atr") or 0.0,
                "mfe_atr": details_float(details, "mfe_atr") or 0.0,
                "highest_price": highest_price or 0.0,
                "gross_pnl": gross_pnl,
                "net_pnl": net_pnl,
                "entry_fees": entry_fees,
                "sell_fee": sell_fee,
                "entry_reason": entry_reason,
                "entry_trigger": entry_trigger,
                "regime": classification["regime"],
                "tier": classification["tier"],
                "signal_type": classification["signal_type"],
                "confirmed_group": classification["confirmed_group"],
                "ret_6h": details_float(entry_details, "ret_6h") or 0.0,
                "ret_24h": details_float(entry_details, "ret_24h") or 0.0,
                "ret_72h": details_float(entry_details, "ret_72h") or 0.0,
                "volume_ratio": details_float(entry_details, "volume_ratio") or 0.0,
                "quote_volume_24h": details_float(entry_details, "quote_volume_24h") or 0.0,
                "risk_multiplier": details_float(entry_details, "risk_multiplier") or 0.0,
                "probe_pct": details_float(entry_details, "probe_pct") or 0.0,
                "anchor_gap_pct": (avg_entry / probe_entry - 1) * 100 if probe_entry > 0 else 0.0,
                "stop_vs_avg_pct": (stop_anchor / avg_entry - 1) * 100 if avg_entry > 0 else 0.0,
                "ema20": ema.get("ema20", 0.0),
                "ema20_dev_pct": ema.get("ema20_dev_pct", 0.0),
                "ema20_dev_rank_90h": ema.get("ema20_dev_rank_90h", 0.0),
                "ema20_dev_rank_2160h": ema.get("ema20_dev_rank_2160h", 0.0),
                "ema20_slope_3_pct": ema.get("ema20_slope_3_pct", 0.0),
                "ema20_slope_6_pct": ema.get("ema20_slope_6_pct", 0.0),
                "price_above_ema20": ema.get("price_above_ema20", False),
            }
        )
    return pd.DataFrame(rows)


def grouped_trade_summary(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, group in trades.groupby(group_columns, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        wins = group[group["net_pnl"] > 0]
        losses = group[group["net_pnl"] < 0]
        gross_profit = float(wins["net_pnl"].sum())
        gross_loss = float(losses["net_pnl"].sum())
        row = {column: value for column, value in zip(group_columns, key_tuple, strict=True)}
        row.update(
            {
                "trades": len(group),
                "net_pnl": float(group["net_pnl"].sum()),
                "win_rate": float((group["net_pnl"] > 0).mean()),
                "profit_factor": gross_profit / abs(gross_loss) if gross_loss < 0 else 0.0,
                "avg_ret_pct": float(group["final_trade_ret_pct"].mean()),
                "median_ret_pct": float(group["final_trade_ret_pct"].median()),
                "trailing_count": int((group["exit_mechanism"] == "pump_trailing_stop").sum()),
                "trailing_pnl": float(group.loc[group["exit_mechanism"] == "pump_trailing_stop", "net_pnl"].sum()),
                "three_h_down_count": int((group["exit_mechanism"] == "pump_3h_down").sum()),
                "initial_stop_count": int((group["exit_mechanism"] == "pump_initial_stop").sum()),
                "median_mfe_pct": float(group["mfe_trade_pct"].median()),
                "median_anchor_gap_pct": float(group["anchor_gap_pct"].median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False).reset_index(drop=True)


def binned_trade_summary(trades: pd.DataFrame, column: str, bins: list[float], labels: list[str]) -> pd.DataFrame:
    if trades.empty or column not in trades:
        return pd.DataFrame()
    frame = trades.copy()
    frame[f"{column}_bin"] = pd.cut(frame[column], bins=bins, labels=labels, include_lowest=True)
    return grouped_trade_summary(frame, [f"{column}_bin"])


def focused_exit_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    mechanisms = ["pump_trailing_stop", "pump_3h_down", "pump_initial_stop", "pump_lock_2pct", "pump_breakeven"]
    return grouped_trade_summary(trades[trades["exit_mechanism"].isin(mechanisms)], ["exit_mechanism"])


def lock_breakeven_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    subset = trades[trades["exit_mechanism"].isin(["pump_lock_2pct", "pump_breakeven"])].copy()
    if subset.empty:
        return pd.DataFrame()
    rows = []
    for mechanism, group in subset.groupby("exit_mechanism", sort=True):
        negatives = group[group["final_trade_ret_pct"] < 0]
        rows.append(
            {
                "exit_mechanism": mechanism,
                "trades": len(group),
                "net_pnl": float(group["net_pnl"].sum()),
                "negative_trades": len(negatives),
                "negative_pnl": float(negatives["net_pnl"].sum()),
                "min_true_ret_pct": float(group["final_trade_ret_pct"].min()),
                "median_true_ret_pct": float(group["final_trade_ret_pct"].median()),
                "median_stop_vs_avg_pct": float(group["stop_vs_avg_pct"].median()),
                "median_anchor_gap_pct": float(group["anchor_gap_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def bad_exit_diagnostics(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    subset = trades[trades["exit_mechanism"].isin(["pump_3h_down", "pump_initial_stop", "pump_probe_kill", "pump_stagnation_exit"])].copy()
    if subset.empty:
        return pd.DataFrame()
    subset["had_mfe_ge_8"] = subset["mfe_trade_pct"] >= 8
    subset["had_mfe_ge_10"] = subset["mfe_trade_pct"] >= 10
    return grouped_trade_summary(subset, ["exit_mechanism", "tier", "confirmed_group"])


def add_trade_bins(output: dict[str, pd.DataFrame], trades: pd.DataFrame) -> None:
    output["pump_exit_summary"] = grouped_trade_summary(trades, ["exit_mechanism"])
    output["pump_entry_group_summary"] = grouped_trade_summary(trades, ["regime", "tier", "confirmed_group"])
    output["pump_signal_type_summary"] = grouped_trade_summary(trades, ["signal_type"])
    output["pump_anchor_gap_bins"] = binned_trade_summary(
        trades,
        "anchor_gap_pct",
        [-1_000, 0, 2, 5, 10, 20, 1_000],
        ["<=0", "0-2", "2-5", "5-10", "10-20", ">20"],
    )
    output["pump_mfe_bins"] = binned_trade_summary(
        trades,
        "mfe_trade_pct",
        [-1_000, 5, 10, 15, 20, 40, 1_000],
        ["<5", "5-10", "10-15", "15-20", "20-40", ">40"],
    )
    output["pump_r72_bins"] = binned_trade_summary(
        trades,
        "ret_72h",
        [-1_000, 0.45, 0.86, 1.2, 2.2, 3.5, 1_000],
        ["<45%", "45-86%", "86-120%", "120-220%", "220-350%", ">350%"],
    )
    output["pump_r6_bins"] = binned_trade_summary(
        trades,
        "ret_6h",
        [-1_000, 0.12, 0.25, 0.50, 1.0, 1_000],
        ["<12%", "12-25%", "25-50%", "50-100%", ">100%"],
    )
    output["pump_vr_bins"] = binned_trade_summary(
        trades,
        "volume_ratio",
        [-1_000, 2, 5, 10, 15, 30, 1_000],
        ["<2", "2-5", "5-10", "10-15", "15-30", ">30"],
    )
    output["pump_ema20_dev_rank_90h_bins"] = binned_trade_summary(
        trades,
        "ema20_dev_rank_90h",
        [-0.01, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01],
        ["0-20%", "20-40%", "40-60%", "60-80%", "80-90%", "90-95%", "95-100%"],
    )
    output["pump_ema20_dev_rank_2160h_bins"] = binned_trade_summary(
        trades,
        "ema20_dev_rank_2160h",
        [-0.01, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01],
        ["0-20%", "20-40%", "40-60%", "60-80%", "80-90%", "90-95%", "95-100%"],
    )
    output["pump_focused_exit_summary"] = focused_exit_summary(trades)
    output["pump_lock_breakeven_diagnostics"] = lock_breakeven_diagnostics(trades)
    output["pump_bad_exit_diagnostics"] = bad_exit_diagnostics(trades)


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


def parse_details(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    if isinstance(value, float) and pd.isna(value):
        return {}
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def details_float(details: dict[str, object], key: str) -> float | None:
    value = details.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def details_timestamp(details: dict[str, object], key: str) -> pd.Timestamp | None:
    value = details.get(key)
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def classify_entry(reason: str, trigger: str) -> dict[str, str]:
    text = f"{reason} {trigger}"
    regime = "HOT" if "HOT" in text else ("WARM" if "WARM" in text else "unknown")
    tier = "A" if "_A_" in text else ("B" if "_B_" in text else "unknown")
    if "early_confirmed" in text:
        signal_type = "early_confirmed"
        confirmed_group = "early_confirmed"
    elif "confirmed" in text:
        signal_type = "confirmed"
        confirmed_group = "confirmed"
    elif "early" in text:
        signal_type = "early"
        confirmed_group = "unconfirmed"
    else:
        signal_type = "unknown"
        confirmed_group = "unknown"
    return {
        "regime": regime,
        "tier": tier,
        "signal_type": signal_type,
        "confirmed_group": confirmed_group,
    }


def ema_metrics(candles: dict[str, pd.DataFrame], symbol: str, time: pd.Timestamp) -> dict[str, object]:
    frame = candles.get(symbol)
    if frame is None or frame.empty:
        return {}
    idx = int(frame.index.searchsorted(pd.Timestamp(time), side="right")) - 1
    if idx < 0:
        return {}
    close = frame["close"].astype(float)
    ema20 = close.ewm(span=20, adjust=False).mean()
    dev = close / ema20 - 1
    current_close = float(close.iloc[idx])
    current_ema = float(ema20.iloc[idx])
    current_dev = float(dev.iloc[idx])
    rank_90h = rolling_rank(dev, idx, 90)
    rank_2160h = rolling_rank(dev, idx, 2160)
    slope_3 = float(ema20.iloc[idx] / ema20.iloc[idx - 3] - 1) if idx >= 3 and float(ema20.iloc[idx - 3]) > 0 else 0.0
    slope_6 = float(ema20.iloc[idx] / ema20.iloc[idx - 6] - 1) if idx >= 6 and float(ema20.iloc[idx - 6]) > 0 else 0.0
    return {
        "ema20": current_ema,
        "ema20_dev_pct": current_dev * 100,
        "ema20_dev_rank_90h": rank_90h,
        "ema20_dev_rank_2160h": rank_2160h,
        "ema20_slope_3_pct": slope_3 * 100,
        "ema20_slope_6_pct": slope_6 * 100,
        "price_above_ema20": current_close > current_ema,
    }


def rolling_rank(series: pd.Series, idx: int, window: int) -> float:
    start = max(0, idx - window + 1)
    values = series.iloc[start : idx + 1].dropna()
    if values.empty:
        return 0.0
    current = float(series.iloc[idx])
    return float((values <= current).mean())


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
        "",
        "## Pump Focused Exit Summary",
        markdown_table(frames["pump_focused_exit_summary"]) if not frames.get("pump_focused_exit_summary", pd.DataFrame()).empty else "No data.",
        "",
        "## Pump Entry Group Summary",
        markdown_table(frames["pump_entry_group_summary"]) if not frames.get("pump_entry_group_summary", pd.DataFrame()).empty else "No data.",
        "",
        "## Pump Lock/Breakeven True Return Diagnostics",
        markdown_table(frames["pump_lock_breakeven_diagnostics"])
        if not frames.get("pump_lock_breakeven_diagnostics", pd.DataFrame()).empty
        else "No data.",
        "",
        "## Pump EMA20 2160h Deviation Rank Bins",
        markdown_table(frames["pump_ema20_dev_rank_2160h_bins"])
        if not frames.get("pump_ema20_dev_rank_2160h_bins", pd.DataFrame()).empty
        else "No data.",
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
    time_frames = [frame for frame in [frames["factors"], frames["rejected"], frames["orders"]] if not frame.empty and "time" in frame]
    if not time_frames:
        raise ValueError(f"strategy_run_id has no time-indexed diagnostics: {run_id}")
    start = min(pd.to_datetime(frame["time"], utc=True).min() for frame in time_frames)
    end = max(pd.to_datetime(frame["time"], utc=True).max() for frame in time_frames)
    symbols = sorted(
        set(frames["factors"]["symbol"].astype(str) if not frames["factors"].empty else [])
        | set(frames["rejected"]["symbol"].astype(str) if not frames["rejected"].empty else [])
        | set(frames["orders"]["symbol"].astype(str) if not frames["orders"].empty else [])
    )
    candles = build_candle_lookup(
        load_candles(
            engine,
            context.exchange,
            symbols,
            start - pd.Timedelta(hours=2160),
            end + pd.Timedelta(hours=max(HORIZONS)),
        )
    )
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
    output["pump_trade_diagnostics"] = pump_trade_diagnostics(frames["orders"], candles)
    add_trade_bins(output, output["pump_trade_diagnostics"])
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
