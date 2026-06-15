from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, BigInteger, Boolean, DateTime, ForeignKey, Integer, Numeric, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class MarketSymbol(Base):
    __tablename__ = "market_symbols"
    __table_args__ = (UniqueConstraint("exchange", "symbol", name="uq_market_symbols_exchange_symbol"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(64))
    base_asset: Mapped[str] = mapped_column(String(32))
    quote_asset: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    is_spot_trading_allowed: Mapped[bool] = mapped_column(Boolean, default=True)
    first_seen_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    metadata_: Mapped[dict[str, Any] | None] = mapped_column("metadata", JSON)


class Candle(Base):
    __tablename__ = "candles"
    __table_args__ = (
        UniqueConstraint("exchange", "symbol", "timeframe", "open_time", name="uq_candles_key"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(32))
    symbol: Mapped[str] = mapped_column(String(64))
    timeframe: Mapped[str] = mapped_column(String(16))
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[float] = mapped_column(Numeric(28, 12))
    high: Mapped[float] = mapped_column(Numeric(28, 12))
    low: Mapped[float] = mapped_column(Numeric(28, 12))
    close: Mapped[float] = mapped_column(Numeric(28, 12))
    volume: Mapped[float] = mapped_column(Numeric(28, 12))
    quote_volume: Mapped[float | None] = mapped_column(Numeric(28, 12))


class UniverseSnapshot(Base):
    __tablename__ = "universe_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    exchange: Mapped[str] = mapped_column(String(32))
    snapshot_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    effective_from: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    config_hash: Mapped[str] = mapped_column(String(64))
    members: Mapped[list[UniverseMember]] = relationship(back_populates="snapshot")


class UniverseMember(Base):
    __tablename__ = "universe_members"
    __table_args__ = (UniqueConstraint("snapshot_id", "symbol", name="uq_universe_members_snapshot_symbol"),)

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    snapshot_id: Mapped[int] = mapped_column(ForeignKey("universe_snapshots.id"))
    symbol: Mapped[str] = mapped_column(String(64))
    liquidity_rank: Mapped[int] = mapped_column(Integer)
    quote_volume_30d: Mapped[float] = mapped_column(Numeric(28, 12))
    eligible: Mapped[bool] = mapped_column(Boolean, default=True)
    reason: Mapped[str | None] = mapped_column(String(255))
    snapshot: Mapped[UniverseSnapshot] = relationship(back_populates="members")


class StrategyRun(Base):
    __tablename__ = "strategy_runs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    run_name: Mapped[str] = mapped_column(String(128))
    strategy_version: Mapped[str] = mapped_column(String(32))
    config: Mapped[dict[str, Any]] = mapped_column(JSON)
    config_hash: Mapped[str] = mapped_column(String(64))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32))


class MarketStateRecord(Base):
    __tablename__ = "market_state"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int | None] = mapped_column(ForeignKey("strategy_runs.id"))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    btc_close: Mapped[float | None] = mapped_column(Numeric(28, 12))
    btc_ma50: Mapped[float | None] = mapped_column(Numeric(28, 12))
    ma50_slope_4: Mapped[float | None] = mapped_column(Numeric(18, 10))
    breadth: Mapped[float | None] = mapped_column(Numeric(18, 10))
    state: Mapped[str] = mapped_column(String(32))
    fast_risk_valve: Mapped[bool] = mapped_column(Boolean, default=False)
    reasons: Mapped[list[str] | None] = mapped_column(JSON)


class FactorScoreRecord(Base):
    __tablename__ = "factor_scores"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int] = mapped_column(ForeignKey("strategy_runs.id"))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(64))
    momentum_score: Mapped[float] = mapped_column(Numeric(18, 10))
    volume_score: Mapped[float | None] = mapped_column(Numeric(18, 10))
    trend_score: Mapped[float | None] = mapped_column(Numeric(18, 10))
    final_score: Mapped[float] = mapped_column(Numeric(18, 10))
    raw_factors: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class SignalRecord(Base):
    __tablename__ = "signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int] = mapped_column(ForeignKey("strategy_runs.id"))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(64))
    side: Mapped[str] = mapped_column(String(16))
    rank: Mapped[int] = mapped_column(Integer)
    target_weight: Mapped[float] = mapped_column(Numeric(18, 10))
    reason: Mapped[str] = mapped_column(String(255))


class RejectedSignalRecord(Base):
    __tablename__ = "rejected_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int] = mapped_column(ForeignKey("strategy_runs.id"))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(64))
    reason: Mapped[str] = mapped_column(String(255))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class PositionRecord(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int] = mapped_column(ForeignKey("strategy_runs.id"))
    symbol: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(32))
    quantity: Mapped[float] = mapped_column(Numeric(28, 12))
    entry_price: Mapped[float | None] = mapped_column(Numeric(28, 12))
    current_price: Mapped[float | None] = mapped_column(Numeric(28, 12))
    atr: Mapped[float | None] = mapped_column(Numeric(28, 12))
    stop_price: Mapped[float | None] = mapped_column(Numeric(28, 12))
    stop_risk_exposure: Mapped[float | None] = mapped_column(Numeric(18, 10))
    volatility_risk_exposure: Mapped[float | None] = mapped_column(Numeric(18, 10))
    current_risk_exposure_tag: Mapped[str | None] = mapped_column(String(32))
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class OrderRecord(Base):
    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int] = mapped_column(ForeignKey("strategy_runs.id"))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    symbol: Mapped[str] = mapped_column(String(64))
    side: Mapped[str] = mapped_column(String(16))
    quantity: Mapped[float] = mapped_column(Numeric(28, 12))
    expected_price: Mapped[float] = mapped_column(Numeric(28, 12))
    limit_price: Mapped[float | None] = mapped_column(Numeric(28, 12))
    filled_price: Mapped[float | None] = mapped_column(Numeric(28, 12))
    fee: Mapped[float] = mapped_column(Numeric(28, 12), default=0)
    slippage: Mapped[float] = mapped_column(Numeric(28, 12), default=0)
    status: Mapped[str] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(255))
    mechanism: Mapped[str | None] = mapped_column(String(64))
    trigger: Mapped[str | None] = mapped_column(String(128))
    details: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class EquityCurveRecord(Base):
    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    strategy_run_id: Mapped[int] = mapped_column(ForeignKey("strategy_runs.id"))
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    equity: Mapped[float] = mapped_column(Numeric(28, 12))
    cash: Mapped[float] = mapped_column(Numeric(28, 12))
    gross_exposure: Mapped[float] = mapped_column(Numeric(18, 10))
    drawdown: Mapped[float] = mapped_column(Numeric(18, 10))
