from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.execution.broker import Order
from crypto_quant.storage.models import OrderRecord, PositionRecord, RejectedSignalRecord, StrategyRun


@dataclass(frozen=True)
class OrderMetadata:
    mechanism: str | None = None
    trigger: str | None = None
    details: dict[str, object] | None = None


class StrategyPersistence:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def create_run(self, session: Session | None, name: str) -> int | None:
        if session is None:
            return None
        run = StrategyRun(
            run_name=f"{name}-{datetime.now(UTC).isoformat()}",
            strategy_version=self.config.strategy_version,
            config=self.config.model_dump(),
            config_hash=self.config.stable_hash(),
            started_at=datetime.now(UTC),
            status="running",
        )
        session.add(run)
        session.flush()
        return run.id

    def finish_run(self, session: Session, run_id: int, status: str) -> None:
        run = session.get(StrategyRun, run_id)
        if run is not None:
            run.status = status
            run.finished_at = datetime.now(UTC)

    def write_orders(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        orders: list[Order],
        metadata: list[OrderMetadata] | None = None,
    ) -> None:
        metadata = metadata or [OrderMetadata() for _ in orders]
        for order, extra in zip(orders, metadata, strict=True):
            session.add(
                OrderRecord(
                    strategy_run_id=run_id,
                    time=now,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    expected_price=order.expected_price,
                    limit_price=order.expected_price * (1.003 if order.side == "buy" else 0.997),
                    filled_price=order.filled_price,
                    fee=order.fee,
                    slippage=order.slippage,
                    status=order.status,
                    reason=order.reason,
                    mechanism=extra.mechanism,
                    trigger=extra.trigger,
                    details=extra.details,
                )
            )

    def write_position(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        position: Any,
        state: str,
        current_price: float,
        entry_anchor: float,
        equity: float | None = None,
    ) -> None:
        stop_distance = max(entry_anchor - position.stop_price, 0)
        denom = max(equity or self.config.backtest.initial_equity, 1)
        exposure = position.quantity * stop_distance / denom
        session.add(
            PositionRecord(
                strategy_run_id=run_id,
                symbol=position.symbol,
                state=state,
                quantity=position.quantity,
                entry_price=entry_anchor,
                current_price=current_price,
                atr=position.atr,
                stop_price=position.stop_price,
                stop_risk_exposure=exposure,
                volatility_risk_exposure=exposure,
                current_risk_exposure_tag="normal" if exposure <= 0.01 else "elevated",
                opened_at=position.opened_at,
                closed_at=now if state == "closed" else None,
            )
        )

    def reject(self, session: Session, run_id: int, now: datetime, symbol: str, reason: str) -> None:
        session.add(RejectedSignalRecord(strategy_run_id=run_id, time=now, symbol=symbol, reason=reason, details=None))
