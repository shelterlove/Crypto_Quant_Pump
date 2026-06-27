from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.data.binance import BinanceSpotDataProvider
from crypto_quant.reporting import RunSummaryBuilder
from crypto_quant.utils.time import ensure_utc


@dataclass(frozen=True)
class LivePosition:
    symbol: str
    opened_at: str | None
    quantity: float
    entry_price: float
    current_price: float
    stop_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_return_pct: float
    stage: str
    stop_hit: bool


class PaperLiveStateWriter:
    def __init__(
        self,
        config: AppConfig,
        state_path: Path = Path("paper_state/main.json"),
        out_path: Path = Path("paper_state/live_status.json"),
        report_dir: Path = Path("reports"),
    ) -> None:
        self.config = config
        self.state_path = state_path
        self.out_path = out_path
        self.summary_builder = RunSummaryBuilder(report_dir)
        self.provider = BinanceSpotDataProvider(config.exchange_id)

    def write(self, session: Session) -> Path:
        payload = self.build(session)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return self.out_path

    def build(self, session: Session) -> dict[str, Any]:
        state = self._load_state()
        positions = state.get("positions", [])
        symbols = [str(item.get("symbol")) for item in positions if item.get("symbol")]
        prices = self.provider.fetch_last_prices(symbols)

        live_positions: list[LivePosition] = []
        unrealized_total = 0.0
        market_value_total = 0.0
        for item in positions:
            symbol = str(item.get("symbol"))
            quantity = float(item.get("quantity", 0.0))
            entry_price = float(item.get("avg_entry_price") or item.get("entry_price") or 0.0)
            current_price = float(prices.get(symbol, entry_price))
            stop_price = float(item.get("stop_price") or 0.0)
            market_value = quantity * current_price
            unrealized_pnl = quantity * (current_price - entry_price)
            unrealized_return_pct = current_price / entry_price - 1 if entry_price > 0 else 0.0
            live_positions.append(
                LivePosition(
                    symbol=symbol,
                    opened_at=str(item.get("opened_at")) if item.get("opened_at") else None,
                    quantity=quantity,
                    entry_price=entry_price,
                    current_price=current_price,
                    stop_price=stop_price,
                    market_value=market_value,
                    unrealized_pnl=unrealized_pnl,
                    unrealized_return_pct=unrealized_return_pct,
                    stage=self._position_stage(item),
                    stop_hit=current_price <= stop_price if stop_price > 0 else False,
                )
            )
            unrealized_total += unrealized_pnl
            market_value_total += market_value

        cash = float(state.get("cash", self.config.backtest.initial_equity))
        live_equity = cash + market_value_total
        realized_pnl = live_equity - self.config.backtest.initial_equity - unrealized_total
        stats = self._trade_stats(session)
        latest_completed = self.summary_builder.latest_completed_run(session, "paper")
        return {
            "generated_at": ensure_utc(datetime.now(UTC)).isoformat(),
            "initial_equity": self.config.backtest.initial_equity,
            "cash": cash,
            "live_equity": live_equity,
            "total_return_pct": live_equity / self.config.backtest.initial_equity - 1 if self.config.backtest.initial_equity > 0 else 0.0,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_total,
            "positions_count": len(live_positions),
            "positions": [position.__dict__ for position in live_positions],
            "closed_trade_count": stats["closed_trade_count"],
            "win_rate": stats["win_rate"],
            "latest_trade_time": stats["latest_trade_time"],
            "runtime_started_at": stats["runtime_started_at"],
            "last_completed_run_id": latest_completed.run_id if latest_completed is not None else None,
        }

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "cash": self.config.backtest.initial_equity,
                "positions": [],
            }
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _trade_stats(self, session: Session) -> dict[str, Any]:
        trades = self.summary_builder.recent_closed_trades(session, "paper", 500)
        closed_count = len(trades)
        wins = sum(1 for trade in trades if trade.pnl is not None and trade.pnl > 0)
        latest_trade = trades[0].time.isoformat() if trades else None
        first_run = self.summary_builder.latest_runs(session, "paper", 500)
        runtime_started_at = first_run[-1].started_at.isoformat() if first_run else None
        return {
            "closed_trade_count": closed_count,
            "win_rate": wins / closed_count if closed_count > 0 else 0.0,
            "latest_trade_time": latest_trade,
            "runtime_started_at": runtime_started_at,
        }

    def _position_stage(self, item: dict[str, Any]) -> str:
        add_qty = float(item.get("add_qty") or 0.0)
        probe_confirmed = bool(item.get("probe_confirmed"))
        is_probe = bool(item.get("is_probe"))
        if add_qty > 0:
            return "add"
        if probe_confirmed:
            return "confirmed"
        if is_probe:
            return "probe"
        return "normal"
