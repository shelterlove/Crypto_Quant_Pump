from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy import case, desc, func, select
from sqlalchemy.orm import Session

from crypto_quant.storage.models import EquityCurveRecord, OrderRecord, PositionRecord, RejectedSignalRecord, SignalRecord, StrategyRun
from crypto_quant.utils.time import ensure_utc


@dataclass(frozen=True)
class RunSummary:
    run_id: int
    run_name: str
    started_at: datetime
    finished_at: datetime | None
    status: str
    orders: int
    buys: int
    sells: int
    signals: int
    rejected_signals: int
    equity: float | None
    report_path: str
    report_exists: bool


@dataclass(frozen=True)
class PositionSummary:
    symbol: str
    opened_at: datetime | None
    quantity: float
    entry_price: float | None
    current_price: float | None
    stop_price: float | None
    state: str
    risk_tag: str | None
    market_value: float | None
    unrealized_pnl: float | None
    unrealized_return_pct: float | None


@dataclass(frozen=True)
class ClosedTradeSummary:
    time: datetime
    symbol: str
    quantity: float
    filled_price: float | None
    reason: str
    mechanism: str | None
    pnl: float | None
    return_pct: float | None
    run_id: int


class RunSummaryBuilder:
    def __init__(self, report_dir: Path = Path("reports")) -> None:
        self.report_dir = report_dir

    def build_many(self, session: Session, run_ids: list[int]) -> dict[int, RunSummary]:
        if not run_ids:
            return {}
        runs = session.execute(select(StrategyRun).where(StrategyRun.id.in_(run_ids))).scalars().all()
        run_map = {run.id: run for run in runs}

        order_stats = {
            int(row.run_id): {
                "orders": int(row.orders or 0),
                "buys": int(row.buys or 0),
                "sells": int(row.sells or 0),
            }
            for row in session.execute(
                select(
                    OrderRecord.strategy_run_id.label("run_id"),
                    func.count().label("orders"),
                    func.sum(case((OrderRecord.side == "buy", 1), else_=0)).label("buys"),
                    func.sum(case((OrderRecord.side == "sell", 1), else_=0)).label("sells"),
                )
                .where(OrderRecord.strategy_run_id.in_(run_ids))
                .group_by(OrderRecord.strategy_run_id)
            )
        }
        signal_counts = {
            int(row.run_id): int(row.count or 0)
            for row in session.execute(
                select(
                    SignalRecord.strategy_run_id.label("run_id"),
                    func.count().label("count"),
                )
                .where(SignalRecord.strategy_run_id.in_(run_ids))
                .group_by(SignalRecord.strategy_run_id)
            )
        }
        rejected_counts = {
            int(row.run_id): int(row.count or 0)
            for row in session.execute(
                select(
                    RejectedSignalRecord.strategy_run_id.label("run_id"),
                    func.count().label("count"),
                )
                .where(RejectedSignalRecord.strategy_run_id.in_(run_ids))
                .group_by(RejectedSignalRecord.strategy_run_id)
            )
        }
        latest_equity_ids = [
            int(value)
            for value in session.execute(
                select(func.max(EquityCurveRecord.id))
                .where(EquityCurveRecord.strategy_run_id.in_(run_ids))
                .group_by(EquityCurveRecord.strategy_run_id)
            ).scalars()
            if value is not None
        ]
        equity_map = {
            int(row.strategy_run_id): float(row.equity)
            for row in session.execute(
                select(EquityCurveRecord.strategy_run_id, EquityCurveRecord.equity)
                .where(EquityCurveRecord.id.in_(latest_equity_ids))
            )
        }

        summaries: dict[int, RunSummary] = {}
        for run_id in run_ids:
            run = run_map.get(run_id)
            if run is None:
                continue
            order = order_stats.get(run_id, {"orders": 0, "buys": 0, "sells": 0})
            report_file = self.report_file(run_id)
            summaries[run_id] = RunSummary(
                run_id=run.id,
                run_name=run.run_name,
                started_at=ensure_utc(run.started_at),
                finished_at=ensure_utc(run.finished_at) if run.finished_at is not None else None,
                status=run.status,
                orders=order["orders"],
                buys=order["buys"],
                sells=order["sells"],
                signals=signal_counts.get(run_id, 0),
                rejected_signals=rejected_counts.get(run_id, 0),
                equity=equity_map.get(run_id),
                report_path=str(report_file),
                report_exists=report_file.exists(),
            )
        return summaries

    def build_one(self, session: Session, run_id: int) -> RunSummary | None:
        return self.build_many(session, [run_id]).get(run_id)

    def latest_runs(self, session: Session, prefix: str, limit: int) -> list[RunSummary]:
        run_ids = [
            int(row.id)
            for row in session.execute(
                select(StrategyRun.id)
                .where(StrategyRun.run_name.like(f"{prefix}-%"))
                .order_by(desc(StrategyRun.id))
                .limit(limit)
            )
        ]
        summaries = self.build_many(session, run_ids)
        return [summaries[run_id] for run_id in run_ids if run_id in summaries]

    def latest_run(self, session: Session, prefix: str) -> RunSummary | None:
        runs = self.latest_runs(session, prefix, 1)
        return runs[0] if runs else None

    def latest_completed_run(self, session: Session, prefix: str) -> RunSummary | None:
        run_id = session.execute(
            select(StrategyRun.id)
            .where(StrategyRun.run_name.like(f"{prefix}-%"))
            .where(StrategyRun.status == "completed")
            .order_by(desc(StrategyRun.id))
            .limit(1)
        ).scalar_one_or_none()
        if run_id is None:
            return None
        return self.build_one(session, int(run_id))

    def report_file(self, run_id: int) -> Path:
        return (self.report_dir / str(run_id) / "report.html").resolve()

    def open_positions_for_run(self, session: Session, run_id: int) -> list[PositionSummary]:
        rows = session.execute(
            select(PositionRecord)
            .where(PositionRecord.strategy_run_id == run_id)
            .where(PositionRecord.state == "paper_open")
            .order_by(PositionRecord.symbol.asc())
        ).scalars().all()
        positions: list[PositionSummary] = []
        for row in rows:
            entry_price = float(row.entry_price) if row.entry_price is not None else None
            current_price = float(row.current_price) if row.current_price is not None else None
            quantity = float(row.quantity)
            market_value = quantity * current_price if current_price is not None else None
            unrealized_pnl = (
                quantity * (current_price - entry_price)
                if current_price is not None and entry_price is not None
                else None
            )
            unrealized_return_pct = (
                current_price / entry_price - 1
                if current_price is not None and entry_price not in (None, 0)
                else None
            )
            positions.append(
                PositionSummary(
                    symbol=row.symbol,
                    opened_at=ensure_utc(row.opened_at) if row.opened_at is not None else None,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    stop_price=float(row.stop_price) if row.stop_price is not None else None,
                    state=row.state,
                    risk_tag=row.current_risk_exposure_tag,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_return_pct=unrealized_return_pct,
                )
            )
        return positions

    def recent_closed_trades(self, session: Session, prefix: str, limit: int) -> list[ClosedTradeSummary]:
        rows = session.execute(
            select(OrderRecord, StrategyRun.run_name)
            .join(StrategyRun, StrategyRun.id == OrderRecord.strategy_run_id)
            .where(StrategyRun.run_name.like(f"{prefix}-%"))
            .where(OrderRecord.side == "sell")
            .order_by(OrderRecord.time.desc(), OrderRecord.id.desc())
            .limit(limit)
        ).all()
        trades: list[ClosedTradeSummary] = []
        for order, _run_name in rows:
            details = order.details or {}
            trades.append(
                ClosedTradeSummary(
                    time=ensure_utc(order.time),
                    symbol=order.symbol,
                    quantity=float(order.quantity),
                    filled_price=float(order.filled_price) if order.filled_price is not None else None,
                    reason=order.reason,
                    mechanism=order.mechanism,
                    pnl=float(details["pnl"]) if details.get("pnl") is not None else None,
                    return_pct=float(details["final_trade_ret_pct"]) if details.get("final_trade_ret_pct") is not None else None,
                    run_id=int(order.strategy_run_id),
                )
            )
        return trades
