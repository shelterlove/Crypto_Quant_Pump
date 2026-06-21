# VPS Deployment

This document deploys the research/backtest and signal-validation environment on a VPS. The current repository does not yet expose a live exchange execution loop in the CLI, so treat this as the production data and dry-run foundation before adding real orders.

## 1. Server Baseline

Recommended minimum:

- Ubuntu 22.04 or 24.04
- 2 vCPU
- 4 GB RAM minimum, 8 GB preferred
- 40 GB disk minimum
- Docker and Docker Compose
- Python 3.11+
- `uv`

Install system packages:

```bash
sudo apt update
sudo apt install -y git curl ca-certificates docker.io docker-compose-plugin python3.11 python3.11-venv
sudo systemctl enable --now docker
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## 2. Clone And Configure

```bash
cd /opt
sudo git clone <YOUR_REPO_URL> crypto-quant
sudo chown -R "$USER:$USER" /opt/crypto-quant
cd /opt/crypto-quant

uv sync --extra dev
cp .env.example .env
```

For the included Docker PostgreSQL service, set `.env` to use port `5439`:

```bash
CRYPTO_QUANT_DATABASE_URL=postgresql+psycopg://crypto_quant:crypto_quant@localhost:5439/crypto_quant
CRYPTO_QUANT_EXCHANGE_ID=binance
CRYPTO_QUANT_LOG_LEVEL=INFO
```

## 3. Start PostgreSQL

```bash
docker compose up -d postgres
docker compose ps
uv run crypto-quant db upgrade
```

Check connectivity:

```bash
uv run python - <<'PY'
from crypto_quant.config.settings import load_config
from crypto_quant.storage.database import get_session_factory
from sqlalchemy import text
cfg = load_config("configs/main.yaml")
with get_session_factory(cfg.database_url)() as session:
    print(session.execute(text("select 1")).scalar_one())
PY
```

## 4. Initial Data Sync

For a fresh VPS, sync all Binance USDT spot `1h` candles. This can take a while.

```bash
uv run crypto-quant data sync-candles --config configs/main.yaml --timeframe 1h --all-usdt-spot --start 2022-01-01
```

Run a quick validation:

```bash
uv run crypto-quant backtest run --config configs/main.yaml --start 2025-06-01 --end 2026-06-01
uv run pytest
```

Expected recent validation reference:

- INS baseline old strategy: final around `182,572`, max DD around `-29.95%`
- `configs/main.yaml`: final around `192,252`, max DD around `-24.61%`

Exact values may shift if the data vendor revises historical candles or the symbol universe changes.

## 5. Run The Paper Cycle

`paper cycle` is the first-stage runtime command. Each invocation:

- acquires a file lock so overlapping runs do not double-process a candle
- syncs recent `1h` candles for symbols already present in the database
- checks BTC candle freshness before strategy execution
- runs one paper step using `configs/main.yaml`
- writes `paper_state/main.json` and a report when a new strategy run is created
- prints a concise summary, or JSON when `--json-output` is set

Run it manually first:

```bash
uv run crypto-quant paper cycle --config configs/main.yaml --json-output
```

If you want the hourly cycle to also discover newly listed USDT spot symbols, add `--all-usdt-spot`. For routine operation after the initial sync, the default is lighter because it only syncs symbols already in the local database.

Create a systemd service:

```bash
sudo tee /etc/systemd/system/crypto-quant-paper.service >/dev/null <<'EOF'
[Unit]
Description=Crypto Quant paper cycle
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=/opt/crypto-quant
EnvironmentFile=/opt/crypto-quant/.env
ExecStart=/opt/crypto-quant/.venv/bin/crypto-quant paper cycle --config configs/main.yaml --state-path paper_state/main.json --lock-path paper_state/paper.lock --json-output
EOF
```

Create a timer:

```bash
sudo tee /etc/systemd/system/crypto-quant-paper.timer >/dev/null <<'EOF'
[Unit]
Description=Run Crypto Quant paper cycle hourly

[Timer]
OnCalendar=*:05
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now crypto-quant-paper.timer
systemctl list-timers crypto-quant-paper.timer
```

Check logs:

```bash
journalctl -u crypto-quant-paper.service -n 200 --no-pager
journalctl -u crypto-quant-paper.service -f
```

Exit codes used by the paper cycle:

- `0`: completed or intentionally skipped an already processed candle
- `70`: data is stale and `--allow-stale` was not set
- `75`: another paper cycle already holds the lock

For a quick local background trial without systemd:

```bash
cd /opt/crypto-quant
mkdir -p logs paper_state
nohup bash -lc 'while true; do date -Is; uv run crypto-quant paper cycle --config configs/main.yaml --json-output; sleep 300; done' >> logs/paper.log 2>&1 &
echo $! > paper_state/paper.pid
tail -f logs/paper.log
```

Stop it with:

```bash
kill "$(cat paper_state/paper.pid)"
```

## 6. Regular Validation Run

Use a separate service for a daily dry validation backtest over the most recent year:

```bash
sudo tee /etc/systemd/system/crypto-quant-validate.service >/dev/null <<'EOF'
[Unit]
Description=Crypto Quant validation backtest
After=crypto-quant-paper.service

[Service]
Type=oneshot
WorkingDirectory=/opt/crypto-quant
EnvironmentFile=/opt/crypto-quant/.env
ExecStart=/opt/crypto-quant/.venv/bin/crypto-quant backtest run --config configs/main.yaml --start 2025-06-01 --end 2026-06-01
EOF
```

Timer:

```bash
sudo tee /etc/systemd/system/crypto-quant-validate.timer >/dev/null <<'EOF'
[Unit]
Description=Run Crypto Quant validation daily

[Timer]
OnCalendar=03:30
Persistent=true

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now crypto-quant-validate.timer
```

## 7. Operational Checks

Before connecting any real exchange execution:

- Confirm `configs/main.yaml` is the only production config used by services.
- Confirm `crypto-quant-paper.timer` runs successfully for at least several days.
- Confirm reports are generated under `reports/<run_id>/`.
- Confirm latest candles are no more than 1-2 hours stale.
- Confirm lock conflicts show exit code `75` rather than creating duplicate runs.
- Confirm the strategy records `pump_entry`, `pump_probe_confirm`, and exit mechanisms as expected.
- Keep API keys out of git and only in `.env` or a VPS secret manager.

## 8. Notes On Simulated Trading

The repository now has a first-stage local paper cycle, but it still does not send orders to Binance. Before real exchange execution, add a broker that:

- validates Binance filters such as `LOT_SIZE`, `MIN_NOTIONAL`, and `PRICE_FILTER`
- handles order rejects, partial fills, timeouts, and cancellations
- reconciles local positions against exchange balances and user-data events
- records exchange order IDs and exact fill fees
- never sends real exchange orders until paper logs have been reviewed

Do not point this code at live exchange keys until the paper runner, order reconciliation, and kill switches are implemented and tested.
