from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from crypto_quant.backtest.persistence import OrderMetadata
from crypto_quant.config.settings import AppConfig
from crypto_quant.execution.broker import Order


class StrategyExitExecutor:
    def __init__(
        self,
        config: AppConfig,
        write_orders: Callable[[Session, int, datetime, list[Order], list[OrderMetadata] | None], None],
        write_position: Callable[[Session, int, datetime, Any, str, float, float | None], None],
        slice_frame: Callable[[pd.DataFrame, datetime], pd.DataFrame],
        next_open: Callable[[dict[str, pd.DataFrame], str, datetime], float | None],
        trade_entry_price: Callable[[Any], float],
        exit_details: Callable[[Any, pd.DataFrame, Any, float, float, datetime], dict[str, object]],
        compute_post_entry_path: Callable[[dict[str, pd.DataFrame], str, datetime, float, datetime], dict[str, object] | None],
    ) -> None:
        self.config = config
        self.write_orders = write_orders
        self.write_position = write_position
        self.slice_frame = slice_frame
        self.next_open = next_open
        self.trade_entry_price = trade_entry_price
        self.exit_details = exit_details
        self.compute_post_entry_path = compute_post_entry_path

    def partial_add_exit(
        self,
        session: Session,
        run_id: int,
        exit_time: datetime,
        symbol: str,
        position: Any,
        portfolio: Any,
        candles_1h: dict[str, pd.DataFrame],
        broker: Any,
        orders: list[Order],
        market_state: Any,
        stop_before: float,
        stop_mechanism_before: str,
        stop_trigger_before: str,
        trigger_time: datetime,
    ) -> None:
        add_qty = min(position.add_qty, position.quantity)
        if add_qty <= 0:
            return
        frame = candles_1h.get(symbol, pd.DataFrame())
        current = self.slice_frame(frame, trigger_time) if not frame.empty else pd.DataFrame()
        exit_price = self.next_open(candles_1h, symbol, exit_time)
        if exit_price is None:
            exit_price = float(current["close"].iloc[-1]) if not current.empty else position.add_entry_price
        if exit_price <= 0:
            return

        order = broker.execute_market(symbol, "sell", add_qty, exit_price, "pump_add_tranche_exit")
        if order.filled_price <= 0:
            return
        orders.append(order)
        portfolio.cash += order.quantity * order.filled_price - order.fee

        add_entry = position.add_entry_price or position.confirm_entry_price or self.trade_entry_price(position)
        add_pnl = (order.filled_price - add_entry) * order.quantity - order.fee
        position.quantity = max(position.quantity - order.quantity, 0.0)
        position.add_qty = max(position.add_qty - order.quantity, 0.0)
        if position.add_qty <= 1e-12:
            position.add_qty = 0.0
            position.add_opened_at = None
            position.add_highest_price = 0.0
            position.add_entry_price = 0.0

        remaining_notional = max(position.entry_notional - order.quantity * add_entry, 0.0)
        position.entry_notional = remaining_notional
        position.avg_entry_price = remaining_notional / position.quantity if position.quantity > 0 else position.entry_price
        position.stop_mechanism = "pump_add_tranche_exit"
        position.stop_trigger = "pump_add_failed_follow_through"

        details = self.exit_details(position, current, market_state, stop_before, order.filled_price, trigger_time)
        details.update(
            {
                "partial_exit": True,
                "add_exit_qty": order.quantity,
                "add_entry_price": add_entry,
                "add_exit_pnl": add_pnl,
                "remaining_qty": position.quantity,
                "remaining_avg_entry_price": position.avg_entry_price,
            }
        )
        partial_mechanism = position.stop_mechanism
        partial_trigger = position.stop_trigger
        self.write_orders(
            session,
            run_id,
            exit_time,
            [order],
            [OrderMetadata(mechanism=partial_mechanism, trigger=partial_trigger, details=details)],
        )
        if position.quantity <= 1e-12:
            del portfolio.positions[symbol]
        else:
            position.stop_mechanism = stop_mechanism_before
            position.stop_trigger = stop_trigger_before

    def full_exit(
        self,
        session: Session,
        run_id: int,
        exit_time: datetime,
        symbol: str,
        position: Any,
        portfolio: Any,
        candles_1h: dict[str, pd.DataFrame],
        broker: Any,
        orders: list[Order],
        market_state: Any,
        stop_before: float,
        reason: str,
        trigger_time: datetime,
        recent_exits: list[str],
        symbol_last_exit: dict[str, tuple[datetime, str]],
        symbol_cooldowns: dict[str, datetime],
        post_entry_store: list[dict[str, object]],
    ) -> None:
        frame = candles_1h.get(symbol, pd.DataFrame())
        current = self.slice_frame(frame, trigger_time) if not frame.empty else pd.DataFrame()
        exit_price = self.next_open(candles_1h, symbol, exit_time)
        if (
            self.config.pump_mode.strict_loss_stop_enabled
            and reason in {"pump_initial_stop", "pump_probe_kill"}
            and position.stop_price > 0
        ):
            exit_price = position.stop_price
        if exit_price is None:
            exit_price = float(current["close"].iloc[-1]) if not current.empty else position.entry_price
        if exit_price <= 0:
            return
        order = broker.execute_market(symbol, "sell", position.quantity, exit_price, reason)
        if order.filled_price <= 0:
            return
        orders.append(order)
        entry_anchor = self.trade_entry_price(position)
        pnl = (order.filled_price - entry_anchor) * position.quantity - order.fee
        portfolio.cash += order.quantity * order.filled_price - order.fee
        won = order.filled_price > entry_anchor
        portfolio.record_trade(pnl, won)
        cfg = self.config.pump_mode
        recent_exits.append(reason or "?")
        symbol_last_exit[symbol] = (exit_time, reason or "?")
        if cfg.failed_reentry_cooldown_enabled and reason in set(cfg.failed_reentry_exit_reasons):
            symbol_cooldowns[symbol] = exit_time + timedelta(hours=cfg.failed_reentry_cooldown_hours)
        lookback = cfg.adaptive_risk_lookback
        if len(recent_exits) > lookback * 2:
            del recent_exits[:-lookback * 2]
        post = self.compute_post_entry_path(candles_1h, symbol, position.opened_at, entry_anchor, trigger_time)
        if post:
            post["symbol"] = symbol
            post["exit_reason"] = reason
            post["pnl"] = pnl
            post_entry_store.append(post)
        del portfolio.positions[symbol]
        details = self.exit_details(position, current, market_state, stop_before, order.filled_price, trigger_time)
        details["pnl"] = pnl
        self.write_orders(
            session,
            run_id,
            exit_time,
            [order],
            [OrderMetadata(mechanism=position.stop_mechanism, trigger=position.stop_trigger, details=details)],
        )
        self.write_position(session, run_id, exit_time, position, "closed", order.filled_price)
