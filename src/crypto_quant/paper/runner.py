from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from crypto_quant.backtest.runner import OpenPosition, Portfolio, ResearchBacktester
from crypto_quant.config.settings import AppConfig
from crypto_quant.data.binance import BinanceBaseDataProvider, binance_provider_for_exchange
from crypto_quant.execution.broker import BacktestBroker, Order
from crypto_quant.risk.market_state import MarketState, fast_risk_valve_triggered
from crypto_quant.storage.candles import distinct_candle_symbols, load_candles
from crypto_quant.storage.models import Candle, EquityCurveRecord
from crypto_quant.utils.time import ensure_utc


@dataclass(frozen=True)
class PaperRunResult:
    strategy_run_id: int | None
    processed_signal_time: datetime | None
    fill_time: datetime | None
    orders: list[Order]
    equity: float
    state_path: Path
    candidate_count: int = 0
    open_positions: int = 0
    market_phase: str | None = None
    market_entry_mode: str | None = None
    pump_regime: str | None = None
    skipped: bool = False
    reason: str | None = None


class PaperRunner:
    """Single-cycle local paper runner.

    It evaluates the second latest available 1h candle and fills paper orders at the latest candle open.
    This intentionally avoids exchange APIs while exercising the same portfolio state machine as backtests.
    """

    def __init__(
        self,
        config: AppConfig,
        state_path: Path = Path("paper_state/main.json"),
        lookback_days: int = 120,
        provider: BinanceBaseDataProvider | None = None,
    ) -> None:
        self.config = config
        self.state_path = state_path
        self.lookback_days = lookback_days
        self.backtester = ResearchBacktester(config)
        self.provider = provider or binance_provider_for_exchange(config.exchange_id)

    def run_once(self, session: Session) -> PaperRunResult:
        latest = self._latest_open_time(session)
        if latest is None:
            return self._skipped("no local 1h candles")

        signal_time = ensure_utc(latest)
        fill_time = signal_time + timedelta(hours=1)
        state = self._load_state()
        last_fill_time = self._parse_dt(state.get("last_fill_time"))
        if last_fill_time is not None and fill_time <= last_fill_time:
            return PaperRunResult(
                strategy_run_id=None,
                processed_signal_time=self._parse_dt(state.get("last_signal_time")),
                fill_time=fill_time,
                orders=[],
                equity=float(state.get("last_equity", self.config.backtest.initial_equity)),
                state_path=self.state_path,
                skipped=True,
                reason="already processed latest fill candle",
            )

        start = signal_time - timedelta(days=self.lookback_days)
        all_symbols = sorted(
            set(distinct_candle_symbols(session, self.config.exchange_id, "1h"))
            | {self.config.market_state.btc_symbol}
        )
        candles = self.backtester._prepare_candles(
            load_candles(session, self.config.exchange_id, all_symbols, "1h", start, signal_time)
        )
        timeline = self.backtester._timeline(candles, start, signal_time)
        if len(timeline) < 1:
            return self._skipped("not enough local 1h candles")

        now = timeline[-1]
        next_time = fill_time

        if self.config.market_state.btc_symbol not in candles or candles[self.config.market_state.btc_symbol].empty:
            return self._skipped("missing BTC/USDT candles")

        self.backtester._pos_cache_1h = self.backtester._build_position_cache(candles, timeline)
        self.backtester._precompute_indicators(candles)
        self.backtester._snapshot_cache_1h = self.backtester._build_snapshot_value_cache(candles)
        self._warm_market_context(candles, timeline)

        portfolio = self._portfolio_from_state(state)
        self.backtester._equity_high = float(state.get("equity_high", self.config.backtest.initial_equity))
        run_id = self.backtester._create_run(session, "paper")
        if run_id is None:
            raise RuntimeError("paper runner requires a database session")

        orders: list[Order] = []
        try:
            mega_caps = set(self.config.universe.mega_cap_exclude)
            keywords = tuple(self.config.universe.exclude_keywords)
            pump_symbols = [
                symbol
                for symbol in all_symbols
                if symbol != self.config.market_state.btc_symbol
                and symbol.split("/")[0].upper() not in mega_caps
                and not any(keyword in symbol.split("/")[0].upper() for keyword in keywords)
            ]

            current_1h = self.backtester._slice(candles, all_symbols, now, cache=self.backtester._pos_cache_1h)
            prices_all = self.backtester._last_prices(current_1h)
            current_btc = current_1h.get(self.config.market_state.btc_symbol, pd.DataFrame())
            fast_valve, fast_reasons = fast_risk_valve_triggered(btc_1h=current_btc)
            market = self.backtester._market_context
            if fast_valve and not market.fast_risk_valve:
                market = MarketState(
                    "risk_off",
                    fast_risk_valve=True,
                    reasons=fast_reasons or ["btc_fast_valve"],
                    phase="risk_off",
                    transition="deteriorating",
                    risk_multiplier=0.0,
                    entry_mode="none",
                    exit_profile="normal",
                    metrics=market.metrics,
                )

            broker = BacktestBroker(self.config.backtest.fee_bps, self.backtester._slippage_bps())
            held_symbols = list(portfolio.positions)
            fill_open_prices = self._fetch_fill_open_prices(held_symbols, next_time)
            missing_held = sorted(set(held_symbols) - set(fill_open_prices))
            if missing_held:
                raise RuntimeError(f"missing current hour open prices for held symbols: {', '.join(missing_held)}")
            self.backtester._paper_fill_time = next_time
            self.backtester._paper_fill_open_prices = fill_open_prices
            self.backtester._process_stops(session, run_id, now, next_time, portfolio, candles, broker, orders, market)

            pump_snapshot = self.backtester._pump_snapshot(candles, pump_symbols, now)
            pump_prices_all = self.backtester._pump_latest_prices(candles, pump_symbols, now)
            pump_candidates = []
            pump_prices: dict[str, float] = {}
            if (
                not market.fast_risk_valve
                and market.entry_mode != "none"
                and (self.backtester._pump_regime in {"HOT", "WARM"} or self.config.pump_mode.cold_squeeze_enabled)
            ):
                pump_prices = pump_prices_all
                pump_equity = portfolio.equity({**prices_all, **pump_prices})
                pump_candidates = self.backtester._pump_candidates_from_snapshot(pump_snapshot, portfolio, pump_equity, now)
                candidate_opens = self._fetch_fill_open_prices([candidate.symbol for candidate in pump_candidates], next_time)
                self.backtester._paper_fill_open_prices.update(candidate_opens)

            self.backtester._enter_pump_positions(
                session,
                run_id,
                now,
                next_time,
                pump_candidates,
                portfolio,
                candles,
                broker,
                orders,
                pump_prices,
                market,
            )

            final_prices = dict(prices_all)
            final_prices.update(self.backtester._paper_fill_open_prices)
            equity = portfolio.equity(final_prices)
            self.backtester._equity_high = max(self.backtester._equity_high, equity)
            drawdown = equity / self.backtester._equity_high - 1 if self.backtester._equity_high > 0 else 0.0
            session.add(
                EquityCurveRecord(
                    strategy_run_id=run_id,
                    time=next_time,
                    equity=equity,
                    cash=portfolio.cash,
                    gross_exposure=portfolio.gross_exposure(equity, final_prices),
                    drawdown=drawdown,
                )
            )
            for position in portfolio.positions.values():
                price = final_prices.get(position.symbol, position.avg_entry_price or position.entry_price)
                self.backtester._write_position(session, run_id, next_time, position, "paper_open", price, equity)

            self._save_state(
                {
                    "cash": portfolio.cash,
                    "positions": [self._position_to_dict(position) for position in portfolio.positions.values()],
                    "equity_high": self.backtester._equity_high,
                    "last_equity": equity,
                    "last_signal_time": ensure_utc(now).isoformat(),
                    "last_fill_time": ensure_utc(next_time).isoformat(),
                    "last_run_id": run_id,
                }
            )
            self.backtester._finish_run(session, run_id, "completed")
            return PaperRunResult(
                run_id,
                now,
                next_time,
                orders,
                equity,
                self.state_path,
                candidate_count=len(pump_candidates),
                open_positions=len(portfolio.positions),
                market_phase=market.phase,
                market_entry_mode=market.entry_mode,
                pump_regime=self.backtester._pump_regime,
            )
        except Exception:
            self.backtester._finish_run(session, run_id, "failed")
            raise
        finally:
            self.backtester._paper_fill_time = None
            self.backtester._paper_fill_open_prices = {}

    def _warm_market_context(self, candles: dict[str, pd.DataFrame], timeline: list[datetime]) -> None:
        btc = candles.get(self.config.market_state.btc_symbol, pd.DataFrame())
        if btc.empty:
            return
        mega_caps = set(self.config.universe.mega_cap_exclude)
        keywords = tuple(self.config.universe.exclude_keywords)
        all_symbols = sorted(candles)
        pump_symbols = [
            symbol
            for symbol in all_symbols
            if symbol != self.config.market_state.btc_symbol
            and symbol.split("/")[0].upper() not in mega_caps
            and not any(keyword in symbol.split("/")[0].upper() for keyword in keywords)
        ]
        for now in timeline[-720:]:
            if (now - self.backtester._last_regime_check).total_seconds() < 14400:
                continue
            snapshot = self.backtester._pump_snapshot(candles, pump_symbols, now)
            current_btc = self.backtester._slice(
                candles,
                [self.config.market_state.btc_symbol],
                now,
                cache=self.backtester._pos_cache_1h,
            ).get(self.config.market_state.btc_symbol, pd.DataFrame())
            fast_valve, reasons = fast_risk_valve_triggered(btc_1h=current_btc)
            self.backtester._pump_regime = self.backtester._detect_pump_regime_snapshot(snapshot)
            self.backtester._market_context = self.backtester._detect_market_context(snapshot, current_btc, fast_valve, reasons)
            self.backtester._last_regime_check = now

    def _latest_open_time(self, session: Session) -> datetime | None:
        return session.execute(
            select(func.max(Candle.open_time))
            .where(Candle.exchange == self.config.exchange_id)
            .where(Candle.timeframe == "1h")
            .where(Candle.symbol == self.config.market_state.btc_symbol)
        ).scalar_one_or_none()

    def _fetch_fill_open_prices(self, symbols: list[str], fill_time: datetime) -> dict[str, float]:
        requested = sorted(set(symbols))
        if not requested:
            return {}
        try:
            return self.provider.fetch_open_prices_at(requested, "1h", fill_time)
        except Exception:
            return {}

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "cash": self.config.backtest.initial_equity,
                "positions": [],
                "equity_high": self.config.backtest.initial_equity,
                "last_equity": self.config.backtest.initial_equity,
            }
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _save_state(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.state_path)

    def _portfolio_from_state(self, state: dict[str, Any]) -> Portfolio:
        portfolio = Portfolio(
            cash=float(state.get("cash", self.config.backtest.initial_equity)),
            initial_equity=self.config.backtest.initial_equity,
        )
        for item in state.get("positions", []):
            position = self._position_from_dict(item)
            portfolio.positions[position.symbol] = position
        return portfolio

    def _position_to_dict(self, position: OpenPosition) -> dict[str, Any]:
        data = asdict(position)
        data["opened_at"] = ensure_utc(position.opened_at).isoformat()
        if position.add_opened_at is not None:
            data["add_opened_at"] = ensure_utc(position.add_opened_at).isoformat()
        return data

    def _position_from_dict(self, data: dict[str, Any]) -> OpenPosition:
        item = dict(data)
        item["opened_at"] = self._parse_dt(item["opened_at"]) or datetime.now(UTC)
        if item.get("add_opened_at"):
            item["add_opened_at"] = self._parse_dt(item["add_opened_at"])
        return OpenPosition(**item)

    def _parse_dt(self, value: Any) -> datetime | None:
        if not value:
            return None
        if isinstance(value, datetime):
            return ensure_utc(value)
        return ensure_utc(datetime.fromisoformat(str(value)))

    def _skipped(self, reason: str) -> PaperRunResult:
        return PaperRunResult(
            strategy_run_id=None,
            processed_signal_time=None,
            fill_time=None,
            orders=[],
            equity=self.config.backtest.initial_equity,
            state_path=self.state_path,
            skipped=True,
            reason=reason,
        )
