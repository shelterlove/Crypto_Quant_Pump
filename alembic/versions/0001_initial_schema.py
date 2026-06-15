"""initial schema

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-06-04
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "market_symbols",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("base_asset", sa.String(32), nullable=False),
        sa.Column("quote_asset", sa.String(32), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("is_spot_trading_allowed", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("first_seen_at", sa.DateTime(timezone=True)),
        sa.Column("metadata", sa.JSON()),
        sa.UniqueConstraint("exchange", "symbol", name="uq_market_symbols_exchange_symbol"),
    )
    op.create_table(
        "candles",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("timeframe", sa.String(16), nullable=False),
        sa.Column("open_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open", sa.Numeric(28, 12), nullable=False),
        sa.Column("high", sa.Numeric(28, 12), nullable=False),
        sa.Column("low", sa.Numeric(28, 12), nullable=False),
        sa.Column("close", sa.Numeric(28, 12), nullable=False),
        sa.Column("volume", sa.Numeric(28, 12), nullable=False),
        sa.Column("quote_volume", sa.Numeric(28, 12)),
        sa.UniqueConstraint("exchange", "symbol", "timeframe", "open_time", name="uq_candles_key"),
    )
    op.create_table(
        "universe_snapshots",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("exchange", sa.String(32), nullable=False),
        sa.Column("snapshot_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("effective_from", sa.DateTime(timezone=True), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
    )
    op.create_table(
        "universe_members",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("snapshot_id", sa.BigInteger(), sa.ForeignKey("universe_snapshots.id"), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("liquidity_rank", sa.Integer(), nullable=False),
        sa.Column("quote_volume_30d", sa.Numeric(28, 12), nullable=False),
        sa.Column("eligible", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("reason", sa.String(255)),
        sa.UniqueConstraint("snapshot_id", "symbol", name="uq_universe_members_snapshot_symbol"),
    )
    op.create_table(
        "strategy_runs",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("run_name", sa.String(128), nullable=False),
        sa.Column("strategy_version", sa.String(32), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("status", sa.String(32), nullable=False),
    )
    op.create_table(
        "market_state",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id")),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("btc_close", sa.Numeric(28, 12)),
        sa.Column("btc_ma50", sa.Numeric(28, 12)),
        sa.Column("ma50_slope_4", sa.Numeric(18, 10)),
        sa.Column("breadth", sa.Numeric(18, 10)),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("fast_risk_valve", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reasons", sa.JSON()),
    )
    op.create_table(
        "factor_scores",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("momentum_score", sa.Numeric(18, 10), nullable=False),
        sa.Column("volume_score", sa.Numeric(18, 10)),
        sa.Column("trend_score", sa.Numeric(18, 10)),
        sa.Column("final_score", sa.Numeric(18, 10), nullable=False),
        sa.Column("raw_factors", sa.JSON()),
    )
    op.create_table(
        "signals",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("target_weight", sa.Numeric(18, 10), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
    )
    op.create_table(
        "rejected_signals",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.Column("details", sa.JSON()),
    )
    op.create_table(
        "positions",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("quantity", sa.Numeric(28, 12), nullable=False),
        sa.Column("entry_price", sa.Numeric(28, 12)),
        sa.Column("current_price", sa.Numeric(28, 12)),
        sa.Column("atr", sa.Numeric(28, 12)),
        sa.Column("stop_price", sa.Numeric(28, 12)),
        sa.Column("stop_risk_exposure", sa.Numeric(18, 10)),
        sa.Column("volatility_risk_exposure", sa.Numeric(18, 10)),
        sa.Column("current_risk_exposure_tag", sa.String(32)),
        sa.Column("opened_at", sa.DateTime(timezone=True)),
        sa.Column("closed_at", sa.DateTime(timezone=True)),
    )
    op.create_table(
        "orders",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("symbol", sa.String(64), nullable=False),
        sa.Column("side", sa.String(16), nullable=False),
        sa.Column("quantity", sa.Numeric(28, 12), nullable=False),
        sa.Column("expected_price", sa.Numeric(28, 12), nullable=False),
        sa.Column("limit_price", sa.Numeric(28, 12)),
        sa.Column("filled_price", sa.Numeric(28, 12)),
        sa.Column("fee", sa.Numeric(28, 12), nullable=False, server_default="0"),
        sa.Column("slippage", sa.Numeric(28, 12), nullable=False, server_default="0"),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
    )
    op.create_table(
        "equity_curve",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column("strategy_run_id", sa.BigInteger(), sa.ForeignKey("strategy_runs.id"), nullable=False),
        sa.Column("time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("equity", sa.Numeric(28, 12), nullable=False),
        sa.Column("cash", sa.Numeric(28, 12), nullable=False),
        sa.Column("gross_exposure", sa.Numeric(18, 10), nullable=False),
        sa.Column("drawdown", sa.Numeric(18, 10), nullable=False),
    )


def downgrade() -> None:
    tables: Sequence[str] = (
        "equity_curve",
        "orders",
        "positions",
        "rejected_signals",
        "signals",
        "factor_scores",
        "market_state",
        "strategy_runs",
        "universe_members",
        "universe_snapshots",
        "candles",
        "market_symbols",
    )
    for table in tables:
        op.drop_table(table)
