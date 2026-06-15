# My First Crypto Quant

Research-first Python framework for an hourly high-liquidity crypto spot momentum rotation strategy.

The first version keeps strategy logic independent from Freqtrade. It provides typed strategy/risk interfaces, PostgreSQL schema migrations, dry-run data and universe commands, and a minimal research backtest that writes reproducible `strategy_runs`.

## Quick Start

```bash
uv sync --extra dev
cp .env.example .env
docker compose up -d postgres
uv run crypto-quant db upgrade
uv run crypto-quant data sync-candles --dry-run
uv run crypto-quant universe build --dry-run
uv run crypto-quant backtest run --config configs/mvp.yaml
uv run pytest
```

Freqtrade integration is intentionally limited to adapter interfaces in this phase.
