# My First Crypto Quant

Pump-only crypto spot research framework.

The active strategy is `pump-v1-main` in `configs/main.yaml`. It:

- loads Binance spot `1h` candles from PostgreSQL
- detects cross-sectional pump regimes and 4h market context
- selects pump candidates from snapshot features
- enters small probe positions and uses two-stage confirmation scaling
- manages pump exits and writes reproducible runs, orders, positions, and equity curves

## Quick Start

```bash
uv sync --extra dev
cp .env.example .env
docker compose up -d postgres
uv run crypto-quant db upgrade
uv run crypto-quant data sync-candles --config configs/main.yaml --timeframe 1h --all-usdt-spot
uv run crypto-quant backtest run --config configs/main.yaml
uv run crypto-quant paper cycle --config configs/main.yaml
uv run pytest
```

Use `configs/main.yaml` for the current strategy. `configs/legacy_baseline_hot_vr5_warm_a_ema30.yaml` keeps the prior baseline for comparison.

## Useful Commands

```bash
# 2022 stress window
uv run crypto-quant backtest run --config configs/main.yaml --start 2022-01-01 --end 2022-12-31

# Combined OOS window
uv run crypto-quant backtest run --config configs/main.yaml --start 2023-01-01 --end 2025-05-31

# INS validation window
uv run crypto-quant backtest run --config configs/main.yaml --start 2025-06-01 --end 2026-06-01

# VPS/local paper cycle: sync recent candles, check freshness, run one paper step
uv run crypto-quant paper cycle --config configs/main.yaml --json-output
```

## Deployment

See [docs/vps_deployment.md](docs/vps_deployment.md) for VPS setup, PostgreSQL, data sync, validation, and systemd timer examples.
