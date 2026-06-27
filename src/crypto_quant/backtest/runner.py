from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from crypto_quant.backtest.candidates import CandidateEngine, PumpCandidate
from crypto_quant.backtest.exits import StrategyExitExecutor
from crypto_quant.backtest.market import MarketContextEngine
from crypto_quant.backtest.persistence import OrderMetadata, StrategyPersistence
from crypto_quant.config.settings import AppConfig
from crypto_quant.execution.broker import BacktestBroker, Order
from crypto_quant.risk.market_state import MarketState, fast_risk_valve_triggered
from crypto_quant.storage.candles import distinct_candle_symbols, load_candles
from crypto_quant.storage.models import (
    EquityCurveRecord,
)
from crypto_quant.utils.time import ensure_utc

# Max bars needed by the pump indicators plus a small buffer.
_SLICE_TAIL_BARS = 200
# Batch DB flushes: only flush every N iterations to reduce round-trips
_DB_FLUSH_EVERY = 100


@dataclass(frozen=True)
class BacktestResult:
    strategy_run_id: int | None
    orders: list[Order] = field(default_factory=list)
    final_equity: float = 0.0
    report_dir: Path | None = None


@dataclass
class OpenPosition:
    symbol: str
    quantity: float
    entry_price: float
    stop_price: float
    atr: float
    opened_at: datetime
    trailing_active: bool = False
    highest_price: float | None = None
    stop_mechanism: str = "initial_atr_stop"
    stop_trigger: str = "low_below_initial_atr_stop"
    last_stop_update: dict[str, object] = field(default_factory=dict)
    max_favorable_pct: float = 0.0
    is_probe: bool = False
    probe_full_qty: float = 0.0
    probe_tier: str = ""
    probe_confirmed: bool = False
    probe_stage: int = 0
    probe_add_qty: float = 0.0
    probe_entry_price: float = 0.0
    confirm_entry_price: float = 0.0
    avg_entry_price: float = 0.0
    entry_notional: float = 0.0
    core_qty: float = 0.0
    add_qty: float = 0.0
    add_entry_price: float = 0.0
    add_opened_at: datetime | None = None
    add_highest_price: float = 0.0
    v_bounce: bool = False  # h2>h1 after entry
    ema20_dev_pct: float = 15.0  # signal-bar EMA deviation for exit tiering
    signal_wick_ratio: float = 0.0
    market_phase: str = "normal"
    exit_profile: str = "normal"


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    daily_realized_loss: float = 0.0
    recent_trade_results: list[bool] = field(default_factory=list)
    pump_trade_results: list[bool] = field(default_factory=list)
    last_date: object = None  # date object, reset daily loss on date change
    initial_equity: float = 100_000.0

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(position.quantity * prices.get(symbol, position.entry_price) for symbol, position in self.positions.items())

    def gross_exposure(self, equity: float, prices: dict[str, float]) -> float:
        if equity <= 0:
            return 0.0
        return sum(position.quantity * prices.get(symbol, position.entry_price) for symbol, position in self.positions.items()) / equity

    def reset_daily_loss(self, now: datetime) -> None:
        """Reset daily realized loss at UTC midnight."""
        today = now.date()
        if self.last_date is None or today != self.last_date:
            self.daily_realized_loss = 0.0
            self.last_date = today

    def record_trade(self, pnl: float, won: bool) -> None:
        """Record a closed trade's PnL and outcome."""
        if pnl < 0:
            self.daily_realized_loss += pnl
        self.pump_trade_results.append(won)
        if len(self.pump_trade_results) > 20:
            self.pump_trade_results = self.pump_trade_results[-20:]
        self.recent_trade_results.append(won)
        if len(self.recent_trade_results) > 20:
            self.recent_trade_results = self.recent_trade_results[-20:]


@dataclass
class ResearchBacktester:
    config: AppConfig
    _pos_cache_1h: dict[str, dict[pd.Timestamp, int]] = field(default_factory=dict)
    _pump_cooldown_until: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))
    _pump_regime: str = "COLD"  # HOT | WARM | COLD
    _last_regime_check: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))
    _pump_consecutive_losses: int = 0
    _pump_symbol_cooldowns: dict[str, datetime] = field(default_factory=dict)  # per-symbol re-entry cooldown
    _pump_symbol_last_exit: dict[str, tuple[datetime, str]] = field(default_factory=dict)
    _pump_post_entry: list[dict[str, object]] = field(default_factory=list)  # post-entry price paths
    _pump_recent_exits: list[str] = field(default_factory=list)
    _candles_ref: dict[str, pd.DataFrame] = field(default_factory=dict)
    _snapshot_cache_1h: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    _equity_high: float = 0.0
    _market_context: MarketState = field(default_factory=lambda: MarketState("risk_on"))
    _persistence: StrategyPersistence = field(init=False)
    _exit_executor: StrategyExitExecutor = field(init=False)
    _market_engine: MarketContextEngine = field(init=False)
    _candidate_engine: CandidateEngine = field(init=False)
    _paper_fill_time: datetime | None = None
    _paper_fill_open_prices: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._persistence = StrategyPersistence(self.config)
        self._market_engine = MarketContextEngine(self.config)
        self._candidate_engine = CandidateEngine(self.config)
        self._exit_executor = StrategyExitExecutor(
            self.config,
            self._write_orders,
            self._write_position,
            self._slice_frame,
            self._next_open,
            self._trade_entry_price,
            self._exit_details,
            self._compute_post_entry_path,
        )

    def run_real(
        self,
        session: Session,
        start: datetime,
        end: datetime,
        report_dir: Path | None = None,
    ) -> BacktestResult:
        start = ensure_utc(start)
        end = ensure_utc(end)
        run_id = self._create_run(session, "pump")
        if run_id is None:
            raise RuntimeError("real backtest requires a database session")
        orders: list[Order] = []
        try:
            if not self.config.pump_mode.enabled:
                raise RuntimeError("pump-only backtester requires pump_mode.enabled=true")

            all_symbols = sorted(
                set(distinct_candle_symbols(session, self.config.exchange_id, "1h"))
                | {self.config.market_state.btc_symbol}
            )
            candles_1h = self._prepare_candles(load_candles(session, self.config.exchange_id, all_symbols, "1h", start, end))
            self._candles_ref = candles_1h  # for on-demand enrichment in _pump_candidates
            btc_1h = candles_1h.get(self.config.market_state.btc_symbol, pd.DataFrame())
            if btc_1h.empty:
                raise RuntimeError("missing required BTC/USDT 1h candles for backtest")
            timeline = self._timeline(candles_1h, start, end)
            if not timeline:
                raise RuntimeError("no 1h candles available for backtest timeline")
            self._pos_cache_1h = self._build_position_cache(candles_1h, timeline)
            self._precompute_indicators(candles_1h)
            self._snapshot_cache_1h = self._build_snapshot_value_cache(candles_1h)
            portfolio = Portfolio(cash=self.config.backtest.initial_equity, initial_equity=self.config.backtest.initial_equity)
            broker = BacktestBroker(self.config.backtest.fee_bps, self._slippage_bps())
            self._equity_high = self.config.backtest.initial_equity
            progress_every = int(os.environ.get("CQ_BACKTEST_PROGRESS_EVERY", "0") or 0)
            progress_started = time.perf_counter()
            progress_total = max(len(timeline) - 1, 1)

            for i, now in enumerate(timeline[:-1]):
                if progress_every and (i == 0 or i % progress_every == 0):
                    elapsed = time.perf_counter() - progress_started
                    rate = (i + 1) / elapsed if elapsed > 0 else 0.0
                    remaining = (progress_total - i - 1) / rate if rate > 0 else 0.0
                    print(
                        f"[backtest] {i + 1}/{progress_total} "
                        f"now={now.isoformat()} "
                        f"positions={len(portfolio.positions)} orders={len(orders)} "
                        f"regime={self._pump_regime} elapsed={elapsed / 60:.1f}m eta={remaining / 60:.1f}m",
                        flush=True,
                    )
                next_time = timeline[i + 1]
                portfolio.reset_daily_loss(now)

                mega_caps = {c.upper() for c in self.config.universe.mega_cap_exclude}
                keywords = [kw.upper() for kw in self.config.universe.exclude_keywords]
                held_symbols = list(portfolio.positions.keys())
                all_relevant = sorted(set(held_symbols))

                if all_relevant:
                    current_1h = self._slice(candles_1h, all_relevant, now, cache=self._pos_cache_1h)
                else:
                    current_1h = {}

                pump_symbols = [
                    s for s in all_symbols
                    if s != self.config.market_state.btc_symbol
                    and s.split("/")[0].upper() not in mega_caps
                    and not any(kw in s.split("/")[0].upper() for kw in keywords)
                ]
                pump_snapshot = self._pump_snapshot(candles_1h, pump_symbols, now)
                pump_prices_all = self._pump_latest_prices(candles_1h, pump_symbols, now)

                current_btc_1h = self._slice(
                    candles_1h,
                    [self.config.market_state.btc_symbol],
                    now,
                    cache=self._pos_cache_1h,
                ).get(self.config.market_state.btc_symbol, pd.DataFrame())
                fast_valve, fast_reasons = fast_risk_valve_triggered(btc_1h=current_btc_1h)
                if (now - self._last_regime_check).total_seconds() >= 14400:
                    self._pump_regime = self._detect_pump_regime_snapshot(pump_snapshot)
                    self._market_context = self._detect_market_context(pump_snapshot, current_btc_1h, fast_valve, fast_reasons)
                    self._last_regime_check = now
                market = self._market_context
                if fast_valve and not market.fast_risk_valve:
                    market = MarketState(
                        "risk_off",
                        fast_risk_valve=True,
                        reasons=fast_reasons or ["btc_fast_valve"],
                        phase="risk_off",
                        transition="deteriorating",
                        risk_multiplier=0.0,
                        entry_mode="none",
                        exit_profile="aggressive_tighten" if self.config.pump_mode.market_context_exit_tightening_enabled else "normal",
                        metrics=market.metrics,
                    )
                prices_all = self._last_prices(current_1h)

                self._process_stops(session, run_id, now, next_time, portfolio, candles_1h, broker, orders, market)

                pump_candidates: list = []
                pump_prices: dict[str, float] = {}
                if (
                    (not market.fast_risk_valve)
                    and market.entry_mode != "none"
                    and (not pump_snapshot.empty or pump_prices_all)
                    and now >= self._pump_cooldown_until
                ):
                    if self._pump_regime in ("HOT", "WARM") or (
                        self._pump_regime == "COLD" and self.config.pump_mode.cold_squeeze_enabled
                    ):
                        pump_prices = pump_prices_all
                        pump_equity = portfolio.equity({**prices_all, **pump_prices})
                        pump_candidates = self._pump_candidates_from_snapshot(pump_snapshot, portfolio, pump_equity, now)
                self._enter_pump_positions(
                    session,
                    run_id,
                    now,
                    next_time,
                    pump_candidates,
                    portfolio,
                    candles_1h,
                    broker,
                    orders,
                    {**prices_all, **pump_prices},
                    market,
                )

                self._apply_funding_cost(portfolio, {**prices_all, **pump_prices}, now, next_time)
                equity_now = portfolio.equity(prices_all)
                self._equity_high = max(self._equity_high, equity_now)
                drawdown = equity_now / self._equity_high - 1 if self._equity_high else 0
                session.add(
                    EquityCurveRecord(
                        strategy_run_id=run_id,
                        time=now,
                        equity=equity_now,
                        cash=portfolio.cash,
                        gross_exposure=portfolio.gross_exposure(equity_now, prices_all),
                        drawdown=drawdown,
                    )
                )
                # Batch flush: only flush to DB every N iterations
                if i % _DB_FLUSH_EVERY == 0:
                    session.flush()
            final_prices = self._last_prices(self._slice(candles_1h, all_symbols, timeline[-1])) if timeline else {}
            final_equity = portfolio.equity(final_prices)
            self._finish_run(session, run_id, "completed")
            return BacktestResult(run_id, orders, final_equity, report_dir)
        except Exception:
            self._finish_run(session, run_id, "failed")
            raise

    def _slippage_bps(self) -> float:
        return (
            self.config.backtest.pessimistic_slippage_bps
            if self.config.backtest.cost_mode == "pessimistic"
            else self.config.backtest.slippage_bps
        )

    def _apply_funding_cost(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        start: datetime,
        end: datetime,
    ) -> float:
        funding_bps = self.config.backtest.funding_bps_per_hour
        if funding_bps <= 0 or not portfolio.positions:
            return 0.0
        hours = max((ensure_utc(end) - ensure_utc(start)).total_seconds() / 3600, 0.0)
        if hours <= 0:
            return 0.0
        notional = sum(
            position.quantity * prices.get(symbol, position.entry_price)
            for symbol, position in portfolio.positions.items()
        )
        cost = notional * (funding_bps / 10_000) * hours
        portfolio.cash -= cost
        return cost

    def _create_run(self, session: Session | None, name: str) -> int | None:
        return self._persistence.create_run(session, name)

    def _finish_run(self, session: Session, run_id: int, status: str) -> None:
        self._persistence.finish_run(session, run_id, status)

    def _timeline(self, candles: dict[str, pd.DataFrame], start: datetime, end: datetime) -> list[datetime]:
        btc = candles.get(self.config.market_state.btc_symbol)
        source = (
            btc
            if btc is not None and not btc.empty
            else next((frame for frame in candles.values() if not frame.empty), pd.DataFrame())
        )
        if source.empty:
            return []
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        if isinstance(source.index, pd.DatetimeIndex):
            times = source.index[(source.index >= start_ts) & (source.index <= end_ts)]
        else:
            times = pd.to_datetime(source["open_time"], utc=True)
            times = times[(times >= start_ts) & (times <= end_ts)]
        return [t.to_pydatetime() for t in times]

    def _prepare_candles(self, candles: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        prepared: dict[str, pd.DataFrame] = {}
        for symbol, frame in candles.items():
            if frame.empty:
                prepared[symbol] = frame
                continue
            ordered = frame.sort_values("open_time").reset_index(drop=True).copy()
            # Set open_time as DatetimeIndex for O(1) .loc slicing
            ordered["open_time"] = pd.to_datetime(ordered["open_time"], utc=True)
            ordered = ordered.set_index("open_time", drop=False)
            prepared[symbol] = ordered
        return prepared

    def _slice(
        self,
        candles: dict[str, pd.DataFrame],
        symbols: list[str],
        end: datetime,
        cache: dict[str, dict[pd.Timestamp, int]] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Slice candles up to `end` timestamp. Uses pre-computed position cache for speed."""
        sliced: dict[str, pd.DataFrame] = {}
        end_ts = pd.Timestamp(ensure_utc(end))
        for symbol in symbols:
            frame = candles.get(symbol)
            if frame is None or frame.empty:
                continue
            if cache is not None and symbol in cache:
                pos = cache[symbol].get(end_ts)
                if pos is not None:
                    p = int(pos)
                    sliced[symbol] = frame.iloc[max(0, p - _SLICE_TAIL_BARS):p]
                    continue
            try:
                sliced[symbol] = frame.loc[:end_ts].iloc[-_SLICE_TAIL_BARS:]
            except KeyError:
                continue
        return sliced

    def _snapshot(self, candles: dict[str, pd.DataFrame], symbols: list[str], end: datetime, full: bool = True) -> pd.DataFrame:
        """Build DataFrame with latest precomputed values.
        If full=False: only lightweight columns (price, ret_24h) for pre-filtering."""
        rows = []
        end_ts = pd.Timestamp(ensure_utc(end))
        cache = self._pos_cache_1h
        for sym in symbols:
            frame = candles.get(sym)
            if frame is None or frame.empty:
                continue
            if sym in cache:
                pos = cache[sym].get(end_ts)
                if pos is None or int(pos) < 73:
                    continue
                idx = int(pos) - 1
            else:
                try:
                    idx = frame.index.get_loc(end_ts) - 1
                    if idx < 72:
                        continue
                except (KeyError, IndexError):
                    continue
            row = {'symbol': sym, 'price': float(frame['close'].iloc[idx]),
                   'ret_24h': float(frame['ret_24h'].iloc[idx])}
            if full:
                row['ret_6h'] = float(frame['ret_6h'].iloc[idx])
                row['ret_72h'] = float(frame['ret_72h'].iloc[idx])
                row['above_ma20'] = bool(frame['above_ma20'].iloc[idx]) if 'above_ma20' in frame.columns else True
                if 'qv_6h_sum' in frame.columns:
                    row['qv_6h'] = float(frame['qv_6h_sum'].iloc[idx])
                    row['qv_24h'] = float(frame['qv_24h_sum'].iloc[idx])
                    row['qv_30_avg'] = float(frame['qv_30_avg'].iloc[idx])
                if 'wick_ratio' in frame.columns:
                    row['wick_ratio'] = float(frame['wick_ratio'].iloc[idx])
                    row['new_12h_high'] = bool(frame['new_12h_high'].iloc[idx])
                if 'atr14' in frame.columns:
                    row['atr'] = float(frame['atr14'].iloc[idx])
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_snapshot_value_cache(self, candles_1h: dict[str, pd.DataFrame]) -> dict[str, dict[str, np.ndarray]]:
        columns = [
            "close", "volume", "ret_24h", "ret_6h", "ret_72h", "above_ma20",
            "qv_6h_sum", "qv_24h_sum", "qv_30_avg", "wick_ratio",
            "new_12h_high", "regime_vol_expansion", "atr14", "ema20_dev_rank_2160h",
            "ema20_dev", "r1", "r2", "r3", "pos24h", "vol_trend6",
        ]
        cache: dict[str, dict[str, np.ndarray]] = {}
        for symbol, frame in candles_1h.items():
            if frame.empty or not set(columns).issubset(frame.columns):
                continue
            cache[symbol] = {column: frame[column].to_numpy(copy=False) for column in columns}
        return cache

    def _cached_position(self, symbol: str, end: datetime) -> int | None:
        symbol_cache = self._pos_cache_1h.get(symbol)
        if symbol_cache is None:
            return None
        return symbol_cache.get(pd.Timestamp(ensure_utc(end)))

    def _pump_latest_prices(self, candles: dict[str, pd.DataFrame], symbols: list[str], end: datetime) -> dict[str, float]:
        return self._candidate_engine.latest_prices(candles, symbols, end, self._cached_position, self._snapshot_cache_1h)

    def _pump_snapshot(self, candles: dict[str, pd.DataFrame], symbols: list[str], end: datetime) -> pd.DataFrame:
        return self._candidate_engine.snapshot(candles, symbols, end, self._cached_position, self._snapshot_cache_1h)

    def _last_prices(self, candles: dict[str, pd.DataFrame]) -> dict[str, float]:
        return {symbol: float(frame["close"].iloc[-1]) for symbol, frame in candles.items() if not frame.empty}

    @staticmethod
    def _trade_entry_price(position: OpenPosition) -> float:
        return position.avg_entry_price if position.avg_entry_price > 0 else position.entry_price

    @staticmethod
    def _trade_entry_notional(position: OpenPosition) -> float:
        return position.entry_notional if position.entry_notional > 0 else position.quantity * position.entry_price

    def _pump_stop_anchor_price(self, position: OpenPosition) -> float:
        if (
            self.config.pump_mode.probe_anchor_breathing_enabled
            and position.probe_confirmed
            and position.probe_entry_price > 0
        ):
            return position.probe_entry_price
        return self._trade_entry_price(position)

    def _process_stops(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        next_time: datetime,
        portfolio: Portfolio,
        candles_1h: dict[str, pd.DataFrame],
        broker: BacktestBroker,
        orders: list[Order],
        market_state: MarketState,
    ) -> None:
        cfg = self.config.pump_mode
        for symbol, position in list(portfolio.positions.items()):
            frame = candles_1h.get(symbol, pd.DataFrame())
            current = self._slice_frame(frame, now) if not frame.empty else pd.DataFrame()
            if current.empty:
                continue

            stop_before = position.stop_price
            stop_mechanism_before = position.stop_mechanism
            stop_trigger_before = position.stop_trigger
            forced_reason = self._update_pump_stop(position, current, now, market_state.exit_profile)
            if forced_reason == "probe_confirm":
                add_qty = position.probe_add_qty
                next_open = self._next_open(candles_1h, symbol, next_time)
                if add_qty > 0 and next_open is not None and next_open > 0:
                    risk_prices = self._pump_latest_prices(candles_1h, list(portfolio.positions), now)
                    equity = portfolio.equity(risk_prices)
                    active_pct = self._active_capital_pct(equity)
                    exposure_cap = cfg.max_total_exposure_pct * active_pct
                    pump_exposure = portfolio.gross_exposure(equity, risk_prices)
                    add_qty = min(
                        add_qty,
                        equity * max(exposure_cap - pump_exposure, 0) / next_open,
                        max(
                            equity * active_pct * cfg.max_symbol_position_pct
                            - position.quantity * risk_prices.get(symbol, position.entry_price),
                            0,
                        )
                        / next_open,
                    )
                    if cfg.portfolio_open_risk_enabled:
                        add_qty = self._risk_capped_quantity(
                            portfolio,
                            risk_prices,
                            equity,
                            add_qty,
                            max(next_open - position.stop_price, 0.0),
                            cfg.max_portfolio_open_risk_pct,
                        )
                    est_cost = add_qty * next_open * 1.002
                    if add_qty > 0 and est_cost <= portfolio.cash:
                        scale_order = broker.execute_market(symbol, "buy", add_qty, next_open, f"probe_confirm_{position.probe_tier}")
                        portfolio.cash -= scale_order.quantity * scale_order.filled_price + scale_order.fee
                        entry_notional = self._trade_entry_notional(position) + scale_order.quantity * scale_order.filled_price
                        position.quantity += scale_order.quantity
                        position.entry_notional = entry_notional
                        position.avg_entry_price = entry_notional / position.quantity if position.quantity > 0 else position.entry_price
                        position.confirm_entry_price = scale_order.filled_price
                        position.core_qty = max(position.core_qty or 0.0, position.quantity - scale_order.quantity)
                        position.add_qty += scale_order.quantity
                        position.add_entry_price = scale_order.filled_price
                        position.add_opened_at = next_time
                        position.add_highest_price = scale_order.filled_price
                        orders.append(scale_order)
                        self._write_orders(session, run_id, next_time, [scale_order],
                            [OrderMetadata(
                                mechanism="pump_probe_confirm",
                                trigger=f"probe_{position.probe_tier}_scale_in",
                                details={
                                    "probe_entry_price": position.probe_entry_price or position.entry_price,
                                    "confirm_entry_price": position.confirm_entry_price,
                                    "avg_entry_price": position.avg_entry_price,
                                    "entry_notional": position.entry_notional,
                                    "core_qty": position.core_qty,
                                    "add_qty": position.add_qty,
                                    "probe_stage": position.probe_stage,
                                },
                            )])
            elif forced_reason == "add_tranche_exit":
                next_open = self._next_open(candles_1h, symbol, next_time)
                if next_open is not None:
                    self._partial_add_exit(
                        session,
                        run_id,
                        next_time,
                        symbol,
                        position,
                        portfolio,
                        candles_1h,
                        broker,
                        orders,
                        market_state,
                        stop_before,
                        stop_mechanism_before,
                        stop_trigger_before,
                        now,
                    )
                continue
            elif forced_reason is not None:
                next_open = self._next_open(candles_1h, symbol, next_time)
                if next_open is not None:
                    self._full_exit(
                        session,
                        run_id,
                        next_time,
                        symbol,
                        position,
                        portfolio,
                        candles_1h,
                        broker,
                        orders,
                        market_state,
                        stop_before,
                        forced_reason,
                        now,
                    )
                continue
            # normal stop check
            if float(current["low"].iloc[-1]) > position.stop_price:
                continue

            next_open = self._next_open(candles_1h, symbol, next_time)
            if next_open is None:
                continue
            mechanism = position.stop_mechanism
            reason = mechanism if mechanism.startswith("pump_") else "pump_stop"
            self._full_exit(
                session,
                run_id,
                next_time,
                symbol,
                position,
                portfolio,
                candles_1h,
                broker,
                orders,
                market_state,
                stop_before,
                reason,
                now,
            )

    def _partial_add_exit(
        self,
        session: Session,
        run_id: int,
        exit_time: datetime,
        symbol: str,
        position: OpenPosition,
        portfolio: Portfolio,
        candles_1h: dict[str, pd.DataFrame],
        broker: BacktestBroker,
        orders: list[Order],
        market_state: MarketState,
        stop_before: float,
        stop_mechanism_before: str,
        stop_trigger_before: str,
        trigger_time: datetime,
    ) -> None:
        self._exit_executor.partial_add_exit(
            session,
            run_id,
            exit_time,
            symbol,
            position,
            portfolio,
            candles_1h,
            broker,
            orders,
            market_state,
            stop_before,
            stop_mechanism_before,
            stop_trigger_before,
            trigger_time,
        )

    def _full_exit(
        self,
        session: Session,
        run_id: int,
        exit_time: datetime,
        symbol: str,
        position: OpenPosition,
        portfolio: Portfolio,
        candles_1h: dict[str, pd.DataFrame],
        broker: BacktestBroker,
        orders: list[Order],
        market_state: MarketState,
        stop_before: float,
        reason: str,
        trigger_time: datetime,
    ) -> None:
        self._exit_executor.full_exit(
            session,
            run_id,
            exit_time,
            symbol,
            position,
            portfolio,
            candles_1h,
            broker,
            orders,
            market_state,
            stop_before,
            reason,
            trigger_time,
            self._pump_recent_exits,
            self._pump_symbol_last_exit,
            self._pump_symbol_cooldowns,
            self._pump_post_entry,
        )

    def _update_pump_stop(
        self,
        position: OpenPosition,
        current: pd.DataFrame,
        now: datetime,
        current_exit_profile: str = "normal",
    ) -> str | None:
        cfg = self.config.pump_mode
        if position.atr <= 0:
            return None
        current = self._position_history(current, position.opened_at)
        if current.empty:
            return None

        high = float(current["high"].max())
        close = float(current["close"].iloc[-1])
        anchor_price = self._pump_stop_anchor_price(position)
        trade_entry_price = self._trade_entry_price(position)
        position.highest_price = max(position.highest_price or anchor_price, high)
        mfe_pct = position.highest_price / anchor_price - 1
        trade_mfe_pct = position.highest_price / trade_entry_price - 1 if trade_entry_price > 0 else mfe_pct
        position.max_favorable_pct = max(position.max_favorable_pct, mfe_pct)

        held_hours = (ensure_utc(now) - ensure_utc(position.opened_at)).total_seconds() / 3600
        exit_profile = self._effective_exit_profile(position.exit_profile, current_exit_profile)
        tighten = exit_profile in {"light_tighten", "tighten", "aggressive_tighten"}
        aggressive = exit_profile == "aggressive_tighten"
        cold_squeeze = exit_profile == "cold_squeeze"

        # Experimental, default-off: use signal-bar metadata captured at entry.
        # Do not derive this from post-entry candles; that shifts the rule into a different time axis.
        if getattr(cfg, "exit_confidence_enabled", False):
            if position.signal_wick_ratio > cfg.exit_confidence_wick_threshold:
                position.stop_mechanism = "pump_wick_kill"
                position.stop_trigger = "pump_signal_wick_dead"
                return "pump_wick_kill"
            if position.ema20_dev_pct < cfg.exit_confidence_low_ema_threshold and position.probe_tier == "B":
                position.stop_mechanism = "pump_lowconf_kill"
                position.stop_trigger = "pump_signal_low_ema_b_tier"
                return "pump_lowconf_kill"

        if (
            cfg.early_probe_fail_enabled
            and position.is_probe
            and not position.probe_confirmed
            and held_hours >= cfg.early_probe_fail_hours
        ):
            early_ret = close / anchor_price - 1
            if early_ret <= cfg.early_probe_fail_ret_pct and mfe_pct < cfg.early_probe_fail_mfe_max_pct:
                position.stop_mechanism = "pump_early_probe_fail"
                position.stop_trigger = "pump_probe_no_mfe_early_loss"
                return "pump_early_probe_fail"

        if cold_squeeze and position.is_probe and not position.probe_confirmed:
            if held_hours >= cfg.cold_squeeze_fail_hours:
                early_ret = close / anchor_price - 1
                if mfe_pct < cfg.cold_squeeze_fail_mfe_pct or early_ret <= cfg.cold_squeeze_fail_ret_pct:
                    position.stop_mechanism = "pump_cold_squeeze_fail"
                    position.stop_trigger = "pump_cold_no_fast_follow_through"
                    return "pump_cold_squeeze_fail"
            if (
                cfg.cold_squeeze_confirm_enabled
                and held_hours >= 2.5
                and close >= anchor_price
                and mfe_pct >= cfg.cold_squeeze_confirm_mfe_pct
            ):
                target_qty = position.probe_full_qty * cfg.cold_squeeze_confirm_target_pct
                remaining_qty = target_qty - position.quantity
                position.probe_confirmed = True
                position.probe_add_qty = max(remaining_qty, 0.0)
                return "probe_confirm" if position.probe_add_qty > 0 else None

        if cfg.two_stage_confirm_enabled and not cold_squeeze and position.is_probe:
            ret_4h_val = close / anchor_price - 1
            if position.probe_stage <= 0 and held_hours >= 3.5:
                if ret_4h_val >= 0:
                    target_qty = position.probe_full_qty * cfg.two_stage_weak_target_pct
                    remaining_qty = target_qty - position.quantity
                    position.probe_confirmed = True
                    position.probe_stage = 1
                    position.probe_add_qty = max(remaining_qty, 0.0)
                    return "probe_confirm" if position.probe_add_qty > 0 else None
                if ret_4h_val <= -0.02:
                    tight_stop = close - position.atr * 0.3
                    if tight_stop > position.stop_price:
                        position.stop_price = tight_stop
                        position.stop_mechanism = "pump_probe_kill"
                        position.stop_trigger = "pump_probe_4h_dead"
            elif position.probe_stage == 1:
                strong_confirm = (
                    mfe_pct >= cfg.two_stage_strong_mfe_pct
                    or ret_4h_val >= cfg.two_stage_strong_ret_pct
                )
                if strong_confirm:
                    target_qty = position.probe_full_qty * cfg.two_stage_strong_target_pct
                    remaining_qty = target_qty - position.quantity
                    position.probe_stage = 2
                    position.probe_add_qty = max(remaining_qty, 0.0)
                    return "probe_confirm" if position.probe_add_qty > 0 else None

        if (
            not cfg.two_stage_confirm_enabled
            and not cold_squeeze
            and position.is_probe
            and not position.probe_confirmed
            and held_hours >= 3.5
        ):
            ret_4h_val = close / anchor_price - 1
            if ret_4h_val >= 0:
                target_pct = (
                    cfg.probe_confirm_target_pct_a
                    if position.probe_tier == "A"
                    else cfg.probe_confirm_target_pct_b
                )
                if cfg.staged_confirm_enabled:
                    strong_confirm = (
                        mfe_pct >= cfg.staged_confirm_strong_mfe_pct
                        or ret_4h_val >= cfg.staged_confirm_strong_ret_pct
                    )
                    if not strong_confirm:
                        target_pct = (
                            cfg.staged_confirm_weak_target_pct_a
                            if position.probe_tier == "A"
                            else cfg.staged_confirm_weak_target_pct_b
                        )
                target_qty = position.probe_full_qty * target_pct
                remaining_qty = target_qty - position.quantity
                if remaining_qty <= 0:
                    position.probe_confirmed = True
                    position.probe_add_qty = 0.0
                    return None
                if remaining_qty > 0:
                    position.probe_confirmed = True
                    position.probe_add_qty = remaining_qty
                    return 'probe_confirm'
            elif ret_4h_val <= -0.02:
                tight_stop = close - position.atr * 0.3
                if tight_stop > position.stop_price:
                    position.stop_price = tight_stop
                    position.stop_mechanism = 'pump_probe_kill'
                    position.stop_trigger = 'pump_probe_4h_dead'

        if (
            cfg.add_tranche_exit_enabled
            and position.add_qty > 0
            and position.add_entry_price > 0
            and position.add_opened_at is not None
            and not position.trailing_active
        ):
            add_history = self._position_history(current, position.add_opened_at)
            if not add_history.empty:
                add_high = float(add_history["high"].max())
                position.add_highest_price = max(position.add_highest_price or position.add_entry_price, add_high)
            add_held_hours = (ensure_utc(now) - ensure_utc(position.add_opened_at)).total_seconds() / 3600
            add_mfe = position.add_highest_price / position.add_entry_price - 1
            add_ret = close / position.add_entry_price - 1
            if add_ret <= cfg.add_tranche_stop_pct:
                position.stop_mechanism = "pump_add_tranche_exit"
                position.stop_trigger = "pump_add_confirm_price_lost"
                return "add_tranche_exit"
            if (
                add_held_hours >= cfg.add_tranche_fail_hours
                and add_mfe < cfg.add_tranche_fail_mfe_pct
                and add_ret <= cfg.add_tranche_fail_ret_pct
            ):
                position.stop_mechanism = "pump_add_tranche_exit"
                position.stop_trigger = "pump_add_no_follow_through"
                return "add_tranche_exit"

        if len(current) >= 3:
            post_close = current['close'].astype(float)
            h1 = float(post_close.iloc[-3] / anchor_price - 1) if len(post_close) >= 3 else 0
            h2 = float(post_close.iloc[-2] / anchor_price - 1) if len(post_close) >= 2 else 0
            h3 = float(post_close.iloc[-1] / anchor_price - 1)
            if h1 < 0 and h2 < h1 and h3 < h2:
                position.stop_mechanism = 'pump_3h_down'
                position.stop_trigger = 'pump_consecutive_down'
                return 'pump_3h_down'
        stagnation_hours = 3.0 if cold_squeeze else (4.0 if aggressive else (5.0 if tighten else cfg.stagnation_stop_hours))
        if cold_squeeze:
            stagnation_mfe = max(cfg.stagnation_min_mfe_pct, cfg.cold_squeeze_fail_mfe_pct)
        elif aggressive:
            stagnation_mfe = min(cfg.stagnation_min_mfe_pct, 0.06)
        else:
            stagnation_mfe = cfg.stagnation_min_mfe_pct
        if held_hours >= stagnation_hours and position.max_favorable_pct < stagnation_mfe:
            if cfg.add_tranche_stagnation_first_enabled and position.add_qty > 0 and not position.trailing_active:
                position.stop_mechanism = "pump_add_tranche_stagnation_exit"
                position.stop_trigger = "pump_stagnation_first_exit_add"
                return "add_tranche_exit"
            position.stop_mechanism = "pump_stagnation_exit"
            position.stop_trigger = "pump_no_fast_follow_through"
            return "pump_stagnation_exit"
        time_stop_hours = 6.0 if cold_squeeze else (6.0 if aggressive else (8.0 if tighten else cfg.time_stop_hours))
        if held_hours >= time_stop_hours and close < anchor_price * (1 + cfg.time_stop_min_profit_pct):
            position.stop_mechanism = "pump_time_exit"
            position.stop_trigger = "pump_time_stop"
            return "pump_time_exit"

        new_stop = position.stop_price
        mechanism = position.stop_mechanism
        trigger = position.stop_trigger
        if mfe_pct >= 0.15:
            protected = anchor_price * (1 - 0.03)
            if protected > new_stop:
                new_stop = protected
                mechanism = "pump_profit_protect"
                trigger = "low_below_pump_profit_protect"
        breakeven_trigger = 0.05 if cold_squeeze else (0.04 if aggressive else (0.06 if tighten else 0.08))
        if mfe_pct >= breakeven_trigger and anchor_price > new_stop:
            new_stop = anchor_price
            mechanism = "pump_breakeven"
            trigger = f"pump_be_{breakeven_trigger:.0%}_mfe"
        lock_trigger = 0.10 if cold_squeeze else (0.08 if aggressive else 0.10)
        lock_pct = 0.015 if cold_squeeze or aggressive else 0.02
        if mfe_pct >= lock_trigger:
            lock_stop = anchor_price * (1 + lock_pct)
            if lock_stop > new_stop:
                new_stop = lock_stop
                mechanism = "pump_lock_2pct" if abs(lock_pct - 0.02) < 1e-9 else "pump_lock_1_5pct"
                trigger = f"pump_lock_{lock_trigger:.0%}_mfe"

        trailing_multiple: float | None = None
        if mfe_pct >= cfg.trailing_3_profit_pct:
            trailing_multiple = cfg.trailing_3_atr_multiple
        elif mfe_pct >= cfg.trailing_2_profit_pct:
            trailing_multiple = cfg.trailing_2_atr_multiple
        elif mfe_pct >= cfg.trailing_1_profit_pct:
            trailing_multiple = cfg.trailing_1_atr_multiple
        if trailing_multiple is not None and position.highest_price is not None:
            trailing_stop = position.highest_price - position.atr * trailing_multiple
            if trailing_stop > new_stop:
                new_stop = min(trailing_stop, close)
                mechanism = "pump_trailing_stop"
                trigger = "low_below_pump_trailing"
                position.trailing_active = True

        if (
            cfg.scaled_avg_floor_enabled
            and position.probe_confirmed
            and trade_entry_price > 0
            and trade_mfe_pct >= cfg.scaled_avg_floor_mfe_pct
        ):
            if trade_entry_price > new_stop:
                new_stop = trade_entry_price
                mechanism = "pump_scaled_avg_floor"
                trigger = f"pump_scaled_avg_floor_{cfg.scaled_avg_floor_mfe_pct:.0%}_mfe"

        if cfg.mfe_protect_enabled and trade_entry_price > 0 and anchor_price >= trade_entry_price * 0.98:
            if trade_mfe_pct >= 0.40:
                new_stop = max(new_stop, trade_entry_price * cfg.mfe_protect_40pct_mult)
            elif trade_mfe_pct >= 0.25:
                new_stop = max(new_stop, trade_entry_price * cfg.mfe_protect_25pct_mult)
            elif trade_mfe_pct >= 0.15:
                new_stop = max(new_stop, trade_entry_price * cfg.mfe_protect_15pct_mult)

        if new_stop > position.stop_price:
            position.stop_price = new_stop
            position.stop_mechanism = mechanism
            position.stop_trigger = trigger
            position.last_stop_update = {
                "trigger": mechanism,
                "mfe_pct": mfe_pct,
                "mfe_trade_level": trade_mfe_pct,
                "stop_anchor_price": anchor_price,
                "avg_entry_price": trade_entry_price,
                "highest_price": position.highest_price,
            }
        return None

    @staticmethod
    def _effective_exit_profile(entry_profile: str, current_profile: str) -> str:
        rank = {"normal": 0, "light_tighten": 1, "tighten": 2, "aggressive_tighten": 3, "cold_squeeze": 4}
        entry_rank = rank.get(entry_profile, 0)
        current_rank = rank.get(current_profile, 0)
        effective_rank = max(entry_rank, current_rank)
        for name, value in rank.items():
            if value == effective_rank:
                return name
        return "normal"

    def _precompute_indicators(self, candles_1h: dict[str, pd.DataFrame]) -> None:
        """Pre-compute all factor columns on the full candle DataFrames.
        The hourly loop then just reads the last values instead of recomputing.
        """
        windows = self.config.momentum.windows_hours
        weights = self.config.momentum.weights
        if len(windows) != len(weights):
            raise ValueError("momentum windows and weights must have the same length")
        min_window = max(windows) if windows else 0
        for _sym, frame in candles_1h.items():
            if frame.empty:
                continue
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            volume = frame["volume"].astype(float)
            # Lightweight columns computed unconditionally (needed for snapshot cache)
            frame["pos24h"] = close / high.rolling(24, min_periods=1).max() - 1
            frame["vol_trend6"] = volume / volume.rolling(6, min_periods=1).mean()
            frame["r1"] = close.pct_change(1)
            frame["r2"] = close.pct_change(2)
            frame["r3"] = close.pct_change(3)
            if len(frame) <= min_window:
                continue
            # Momentum
            weighted_return = pd.Series(0.0, index=frame.index)
            for window, weight in zip(windows, weights, strict=True):
                ret_col = f"ret_{window}h"
                frame[ret_col] = close / close.shift(window) - 1
                weighted_return = weighted_return + frame[ret_col] * float(weight)
            frame["weighted_return"] = weighted_return
            frame["ret_6h"] = close / close.shift(6) - 1
            # Trend helpers
            frame["ma20"] = close.rolling(20).mean()
            frame["ema20"] = close.ewm(span=20, adjust=False).mean()
            ema20_dev = close / frame["ema20"].replace(0, np.nan) - 1
            frame["ema20_dev"] = ema20_dev
            frame["ema20_dev_rank_2160h"] = ema20_dev.rolling(2160, min_periods=50).rank(pct=True).fillna(0.0)
            # ATR
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            frame["atr14"] = tr.rolling(14).mean()
            if "quote_volume" in frame.columns:
                qv = frame["quote_volume"].astype(float)
                frame["qv_6h_sum"] = qv.rolling(6).sum()
                frame["qv_24h_sum"] = qv.rolling(24).sum()
                frame["qv_30_avg"] = qv.shift(6).rolling(24).mean() * 6
            frame["above_ma20"] = close > frame["ma20"]
            cr = high - low
            frame["wick_ratio"] = (high - close) / cr.replace(0, np.nan)
            frame["new_12h_high"] = high > high.rolling(12).max().shift(1)
            regime_avg_volume = volume.shift(2).rolling(48).mean()
            regime_recent_volume = volume.rolling(6).mean()
            frame["regime_vol_expansion"] = (
                regime_recent_volume / regime_avg_volume.replace(0, np.nan)
                >= self.config.pump_mode.regime_hot_volume_expansion_ratio
            )

    def _latest_atr(self, frame: pd.DataFrame | None) -> float | None:
        if frame is None or frame.empty:
            return None
        if "atr14" in frame.columns:
            atr = frame["atr14"].dropna()
            if not atr.empty:
                return float(atr.iloc[-1])
        return None

    def _build_position_cache(self, candles: dict[str, pd.DataFrame], timeline: list[datetime]) -> dict[str, dict[pd.Timestamp, int]]:
        """Pre-compute integer row positions for each (symbol, timestamp) pair.

        This converts O(log n) searchsorted to O(1) dict lookup for every slice.
        """
        cache: dict[str, dict[pd.Timestamp, int]] = {}
        timeline_ts = [pd.Timestamp(ensure_utc(t)) for t in timeline]
        for symbol, frame in candles.items():
            if frame.empty:
                continue
            if isinstance(frame.index, pd.DatetimeIndex):
                positions = frame.index.searchsorted(timeline_ts, side="right")
                cache[symbol] = {ts: int(p) for ts, p in zip(timeline_ts, positions, strict=True) if p > 0}
            elif "open_time" in frame.columns:
                times = pd.to_datetime(frame["open_time"], utc=True)
                positions = times.searchsorted(pd.DatetimeIndex(timeline_ts), side="right")
                cache[symbol] = {ts: int(p) for ts, p in zip(timeline_ts, positions, strict=True) if p > 0}
        return cache

    def _slice_frame(self, frame: pd.DataFrame, end: datetime) -> pd.DataFrame:
        """Slice a single DataFrame up to `end` — returns FULL history for position management."""
        if frame.empty:
            return frame
        end_ts = pd.Timestamp(ensure_utc(end))
        try:
            return frame.loc[:end_ts]
        except KeyError:
            if "open_time" in frame.columns:
                mask = pd.to_datetime(frame["open_time"], utc=True) <= end_ts
                return frame[mask]
            return frame

    def _position_history(self, current: pd.DataFrame, opened_at: datetime) -> pd.DataFrame:
        if current.empty:
            return current
        opened_ts = pd.Timestamp(ensure_utc(opened_at))
        if isinstance(current.index, pd.DatetimeIndex):
            return current.loc[opened_ts:]
        if "open_time" in current.columns:
            return current[pd.to_datetime(current["open_time"], utc=True) >= opened_ts]
        return current

    def _exit_details(
        self,
        position: OpenPosition,
        current: pd.DataFrame,
        market_state: MarketState,
        stop_before: float,
        exit_price: float,
        trigger_time: datetime,
    ) -> dict[str, object]:
        history = self._position_history(current, position.opened_at)
        entry_anchor = self._trade_entry_price(position)
        stop_anchor = self._pump_stop_anchor_price(position)
        if history.empty:
            high = entry_anchor
            low = entry_anchor
            close = entry_anchor
        else:
            high = float(history["high"].max())
            low = float(history["low"].min())
            close = float(history["close"].iloc[-1])
        highest_price = max(position.highest_price or entry_anchor, high)
        return {
            "entry_price": position.entry_price,
            "probe_entry_price": position.probe_entry_price or position.entry_price,
            "confirm_entry_price": position.confirm_entry_price,
            "avg_entry_price": entry_anchor,
            "stop_anchor_price": stop_anchor,
            "active_stop_price": position.stop_price,
            "exit_price": exit_price,
            "exit_reason": position.stop_mechanism,
            "final_trade_ret_pct": exit_price / entry_anchor - 1 if entry_anchor > 0 else 0,
            "atr": position.atr,
            "stop_before_update": stop_before,
            "stop_after_update": position.stop_price,
            "highest_price": highest_price,
            "mfe_trade_level": highest_price / entry_anchor - 1 if entry_anchor > 0 else 0,
            "mfe_atr": (high - entry_anchor) / position.atr if position.atr > 0 else 0,
            "mae_atr": (low - entry_anchor) / position.atr if position.atr > 0 else 0,
            "trigger_time": trigger_time.isoformat(),
            "opened_at": position.opened_at.isoformat(),
            "position_state": "closed",
            "strategy_type": "pump",
            "entry_market_phase": position.market_phase,
            "entry_exit_profile": position.exit_profile,
            "core_qty": position.core_qty,
            "add_qty": position.add_qty,
            "probe_stage": position.probe_stage,
            "add_entry_price": position.add_entry_price,
            "add_opened_at": position.add_opened_at.isoformat() if position.add_opened_at is not None else None,
            "market_state": market_state.state,
            "market_phase": market_state.phase,
            "market_transition": market_state.transition,
            "market_exit_profile": market_state.exit_profile,
            "market_reasons": market_state.reasons,
            "last_close": close,
            "last_stop_update": position.last_stop_update,
        }

    def _compute_post_entry_path(self, candles_1h, symbol, opened_at, entry_price, now):
        """Record 1h/2h/3h/4h post-entry performance for pump trades."""
        frame = candles_1h.get(symbol, pd.DataFrame())
        if frame.empty:
            return None
        opened_ts = pd.Timestamp(ensure_utc(opened_at))
        now_ts = pd.Timestamp(ensure_utc(now))
        if 'open' not in frame.columns:
            return None
        # Get candles from entry time to now
        if isinstance(frame.index, pd.DatetimeIndex):
            mask = (frame.index >= opened_ts) & (frame.index <= now_ts)
        else:
            open_time = pd.to_datetime(frame['open_time'], utc=True)
            mask = (open_time >= opened_ts) & (open_time <= now_ts)
        post = frame[mask]
        if len(post) < 1:
            return None
        result = {'entry_price': float(entry_price)}
        # Track max drawdown and max favorable excursion at 1h, 2h, 3h, 4h
        for h in [1, 2, 3, 4]:
            rows = post.head(h)
            if len(rows) == 0:
                break
            high = float(rows['high'].max()) if 'high' in rows.columns else float(rows['close'].max())
            low = float(rows['low'].min()) if 'low' in rows.columns else float(rows['close'].min())
            close_h = float(rows['close'].iloc[-1]) if len(rows) > 0 else entry_price
            result[f'mfe_{h}h'] = (high / entry_price - 1) * 100
            result[f'mae_{h}h'] = (low / entry_price - 1) * 100
            result[f'ret_{h}h'] = (close_h / entry_price - 1) * 100
            result[f'high_{h}h'] = (high > entry_price)
            result[f'low_below_entry_{h}h'] = (low < entry_price)
        return result

    def _detect_pump_regime_snapshot(self, snapshot: pd.DataFrame) -> str:
        return self._market_engine.detect_pump_regime_snapshot(snapshot)

    def _detect_market_context(
        self,
        snapshot: pd.DataFrame,
        btc_1h: pd.DataFrame,
        fast_valve: bool,
        fast_reasons: list[str],
    ) -> MarketState:
        return self._market_engine.detect_market_context(snapshot, btc_1h, fast_valve, fast_reasons, self._pump_regime)

    def _pump_candidates_from_snapshot(
        self,
        snapshot: pd.DataFrame,
        portfolio: Portfolio,
        equity: float,
        now: datetime,
    ) -> list[PumpCandidate]:
        candidates, consecutive_losses = self._candidate_engine.select_candidates(
            snapshot,
            portfolio,
            equity,
            now,
            self._pump_regime,
            self._market_context,
            self._pump_symbol_last_exit,
        )
        self._pump_consecutive_losses = consecutive_losses
        return candidates

    def _enter_pump_positions(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        next_time: datetime,
        candidates: list[PumpCandidate],
        portfolio: Portfolio,
        candles_1h: dict[str, pd.DataFrame],
        broker: BacktestBroker,
        orders: list[Order],
        prices: dict[str, float],
        market: MarketState | None = None,
    ) -> None:
        cfg = self.config.pump_mode
        market = market or self._market_context
        if not candidates:
            return
        equity = portfolio.equity(prices)
        if equity <= 0:
            return
        pump_positions = list(portfolio.positions.values())
        slots = max(cfg.max_positions - len(pump_positions), 0)
        if slots <= 0:
            return
        pump_exposure = sum(p.quantity * prices.get(p.symbol, p.entry_price) for p in pump_positions) / equity

        for candidate in candidates:
            if slots <= 0:
                break
            if candidate.symbol in portfolio.positions:
                continue
            is_cold_squeeze = "cold_squeeze" in candidate.reason
            if is_cold_squeeze:
                cold_positions = sum(1 for position in portfolio.positions.values() if position.exit_profile == "cold_squeeze")
                if cold_positions >= cfg.cold_squeeze_max_positions:
                    self._reject(session, run_id, now, candidate.symbol, "pump_cold_squeeze_slot_limit")
                    continue
            cooldown_until = self._pump_symbol_cooldowns.get(candidate.symbol)
            if cooldown_until is not None and ensure_utc(now) < ensure_utc(cooldown_until):
                self._reject(session, run_id, now, candidate.symbol, "pump_symbol_cooldown")
                continue
            next_open = self._next_open(candles_1h, candidate.symbol, next_time)
            if next_open is None or next_open <= 0:
                self._reject(session, run_id, now, candidate.symbol, "pump_missing_next_open")
                continue
            if (
                market.entry_mode == "patient"
                and next_open > candidate.price * (1 + cfg.market_context_patient_max_entry_gap_pct)
            ):
                self._reject(session, run_id, now, candidate.symbol, "pump_patient_entry_gap")
                continue

            stop_distance = max(candidate.atr * cfg.initial_stop_atr_multiple, next_open * cfg.initial_stop_pct)
            if is_cold_squeeze:
                stop_distance = min(stop_distance, next_open * cfg.cold_squeeze_initial_stop_pct)
            active_pct = self._active_capital_pct(equity)
            active_equity = equity * active_pct
            peak_ratio = equity / max(self._equity_high, 1)
            eff_risk_pct = cfg.trade_risk_pct
            if getattr(cfg, 'equity_peak_risk_enabled', False):
                floor = getattr(cfg, 'equity_peak_risk_floor', 0.50)
                eff_risk_pct *= max(floor, peak_ratio)
            risk_budget = active_equity * eff_risk_pct * candidate.risk_multiplier
            exposure_cap = cfg.max_total_exposure_pct * active_pct
            full_quantity = min(
                risk_budget / stop_distance,
                active_equity * cfg.max_symbol_position_pct / next_open,
                equity * max(exposure_cap - pump_exposure, 0) / next_open,
            )
            if full_quantity <= 0:
                self._reject(session, run_id, now, candidate.symbol, "pump_exposure_limit")
                continue

            probe_pct = cfg.cold_squeeze_probe_pct if is_cold_squeeze else (cfg.probe_pct_a if candidate.tier == "A" else cfg.probe_pct_b)
            stagnation_reentry_boosted = False
            last_exit = self._pump_symbol_last_exit.get(candidate.symbol)
            if cfg.stagnation_reentry_boost_enabled and last_exit is not None:
                last_exit_time, last_exit_reason = last_exit
                age_hours = (ensure_utc(now) - ensure_utc(last_exit_time)).total_seconds() / 3600
                if (
                    last_exit_reason == "pump_stagnation_exit"
                    and 0 <= age_hours <= cfg.stagnation_reentry_boost_hours
                ):
                    boosted_probe_pct = (
                        cfg.stagnation_reentry_probe_pct_a
                        if candidate.tier == "A"
                        else cfg.stagnation_reentry_probe_pct_b
                    )
                    probe_pct = max(probe_pct, boosted_probe_pct)
                    stagnation_reentry_boosted = True
            quantity = full_quantity * probe_pct
            if cfg.portfolio_open_risk_enabled:
                probe_cap_pct = cfg.max_probe_open_risk_pct or cfg.max_portfolio_open_risk_pct
                quantity = self._risk_capped_quantity(
                    portfolio,
                    prices,
                    equity,
                    quantity,
                    stop_distance,
                    probe_cap_pct,
                )
            if quantity <= 0:
                self._reject(session, run_id, now, candidate.symbol, "pump_probe_too_small")
                continue

            order = broker.execute_market(
                candidate.symbol, "buy", quantity, next_open,
                f"pump_{self._pump_regime}_{candidate.tier}_{candidate.reason}_"
                f"r72={candidate.ret_72h:.2f}_r24={candidate.ret_24h:.2f}_"
                f"r6={candidate.ret_6h:.2f}_vr={candidate.volume_ratio:.1f}"
            )
            notional = order.quantity * order.filled_price + order.fee
            if notional > portfolio.cash:
                self._reject(session, run_id, now, candidate.symbol, "pump_insufficient_cash")
                continue

            portfolio.cash -= notional
            stop_price = order.filled_price - stop_distance
            position = OpenPosition(
                candidate.symbol,
                order.quantity,
                order.filled_price,
                stop_price,
                candidate.atr,
                next_time,
                highest_price=order.filled_price,
                stop_mechanism="pump_initial_stop",
                stop_trigger="low_below_pump_initial_stop",
                is_probe=True,
                probe_full_qty=full_quantity,
                probe_tier=candidate.tier,
                probe_entry_price=order.filled_price,
                avg_entry_price=order.filled_price,
                entry_notional=order.quantity * order.filled_price,
                core_qty=order.quantity,
                ema20_dev_pct=candidate.ema20_dev_pct,
                signal_wick_ratio=candidate.wick_ratio,
                market_phase=market.phase,
                exit_profile="cold_squeeze" if is_cold_squeeze else market.exit_profile,
            )
            portfolio.positions[candidate.symbol] = position
            orders.append(order)
            pump_exposure += order.quantity * order.filled_price / equity
            slots -= 1
            self._write_orders(
                session,
                run_id,
                next_time,
                [order],
                [
                    OrderMetadata(
                        mechanism="pump_entry",
                        trigger=candidate.reason,
                        details={
                            "strategy_type": "pump",
                            "signal_time": now.isoformat(),
                            "entry_price": order.filled_price,
                            "probe_entry_price": order.filled_price,
                            "avg_entry_price": order.filled_price,
                            "probe_pct": probe_pct,
                            "cold_squeeze": is_cold_squeeze,
                            "stagnation_reentry_boosted": stagnation_reentry_boosted,
                            "active_capital_pct": active_pct,
                            "stop_anchor_price": order.filled_price,
                            "stop_price": stop_price,
                            "atr": candidate.atr,
                            "risk_multiplier": candidate.risk_multiplier,
                            "market_phase": market.phase,
                            "market_transition": market.transition,
                            "market_entry_mode": market.entry_mode,
                            "market_exit_profile": "cold_squeeze" if is_cold_squeeze else market.exit_profile,
                            "market_risk_multiplier": market.risk_multiplier,
                            "ema20_dev_rank_2160h": candidate.ema20_dev_rank_2160h,
                            "bad_b_ema_vr_risk_reduced": (
                                candidate.tier == "B"
                                and "early" in candidate.reason
                                and "confirmed" not in candidate.reason
                                and candidate.ema20_dev_rank_2160h >= cfg.bad_b_ema_rank_min
                                and candidate.volume_ratio > cfg.bad_b_volume_ratio_min
                            ),
                            "bad_b_ema_vr_risk_mid_reduced": (
                                cfg.bad_b_ema_vr_risk_mid_enabled
                                and candidate.tier == "B"
                                and "early" in candidate.reason
                                and "confirmed" not in candidate.reason
                                and candidate.ema20_dev_rank_2160h >= cfg.bad_b_ema_rank_min
                                and cfg.bad_b_volume_ratio_mid_min < candidate.volume_ratio <= cfg.bad_b_volume_ratio_mid_max
                            ),
                            "score": candidate.score,
                            "ret_6h": candidate.ret_6h,
                            "ret_24h": candidate.ret_24h,
                            "ret_72h": candidate.ret_72h,
                            "volume_ratio": candidate.volume_ratio,
                            "quote_volume_24h": candidate.quote_volume_24h,
                            "ema20_dev_pct": candidate.ema20_dev_pct,
                            "wick_ratio": candidate.wick_ratio,
                            "r1": candidate.r1, "r2": candidate.r2, "r3": candidate.r3,
                            "pos24h": candidate.pos24h, "vol_trend6": candidate.vol_trend6,
                        },
                    )
                ],
            )
            self._write_position(session, run_id, next_time, position, "pump_open", order.filled_price)

    def _next_open(self, candles: dict[str, pd.DataFrame], symbol: str, next_time: datetime) -> float | None:
        fill_time = self._paper_fill_time
        if fill_time is not None and ensure_utc(next_time) == ensure_utc(fill_time):
            live_price = self._paper_fill_open_prices.get(symbol)
            if live_price is not None and live_price > 0:
                return float(live_price)
        frame = candles.get(symbol, pd.DataFrame())
        if frame.empty:
            return None
        ts = pd.Timestamp(ensure_utc(next_time))
        # Try DatetimeIndex first (fast path from _prepare_candles)
        if isinstance(frame.index, pd.DatetimeIndex):
            try:
                row = frame.loc[ts]
                return float(row["open"]) if not row.empty else None
            except KeyError:
                return None
        # Fallback: column-based lookup for raw DataFrames
        if "open_time" in frame.columns:
            times = pd.to_datetime(frame["open_time"], utc=True)
            matches = frame[times == ts]
            if not matches.empty:
                return float(matches["open"].iloc[0])
        return None

    @staticmethod
    def _portfolio_open_risk(portfolio: Portfolio, prices: dict[str, float]) -> float:
        risk = 0.0
        for position in portfolio.positions.values():
            price = prices.get(position.symbol, position.avg_entry_price or position.entry_price)
            risk += position.quantity * max(float(price) - position.stop_price, 0.0)
        return risk

    def _active_capital_pct(self, equity: float) -> float:
        cfg = self.config.pump_mode
        if not cfg.profit_reserve_enabled or equity <= 0:
            return 1.0
        profit_ratio = equity / max(self.config.backtest.initial_equity, 1)
        if profit_ratio >= cfg.profit_reserve_profit_2_threshold:
            profit_cap = cfg.profit_reserve_profit_2_active_cap
        elif profit_ratio >= cfg.profit_reserve_profit_1_threshold:
            profit_cap = cfg.profit_reserve_profit_1_active_cap
        else:
            return 1.0

        drawdown = equity / max(self._equity_high, 1) - 1
        if drawdown <= -0.30:
            active_pct = cfg.profit_reserve_deep_dd_active_pct
        elif drawdown <= -0.20:
            active_pct = max(profit_cap, cfg.profit_reserve_dd_30_active_pct)
        elif drawdown <= -0.10:
            active_pct = max(profit_cap, cfg.profit_reserve_dd_20_active_pct)
        else:
            active_pct = min(profit_cap, cfg.profit_reserve_dd_10_active_pct)
        return min(max(active_pct, 0.0), 1.0)

    def _risk_capped_quantity(
        self,
        portfolio: Portfolio,
        prices: dict[str, float],
        equity: float,
        requested_qty: float,
        unit_risk: float,
        cap_pct: float,
    ) -> float:
        if requested_qty <= 0 or unit_risk <= 0 or equity <= 0:
            return 0.0
        current_risk = self._portfolio_open_risk(portfolio, prices)
        remaining_risk = equity * cap_pct - current_risk
        if remaining_risk <= 0:
            return 0.0
        return min(requested_qty, remaining_risk / unit_risk)

    def _write_orders(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        orders: list[Order],
        metadata: list[OrderMetadata] | None = None,
    ) -> None:
        self._persistence.write_orders(session, run_id, now, orders, metadata)

    def _write_position(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        position: OpenPosition,
        state: str,
        current_price: float,
        equity: float | None = None,
    ) -> None:
        entry_anchor = self._trade_entry_price(position)
        self._persistence.write_position(session, run_id, now, position, state, current_price, entry_anchor, equity)

    def _reject(self, session: Session, run_id: int, now: datetime, symbol: str, reason: str) -> None:
        self._persistence.reject(session, run_id, now, symbol, reason)
