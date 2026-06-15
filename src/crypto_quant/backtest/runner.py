from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.execution.broker import BacktestBroker, Order
from crypto_quant.factors.momentum import MomentumFactorEngine, compute_atr
from crypto_quant.risk.engine import RiskEngine
from crypto_quant.risk.market_state import (
    MarketState,
    compute_market_breadth,
    evaluate_btc_ma50_state,
    fast_risk_valve_triggered,
)
from crypto_quant.storage.candles import distinct_candle_symbols, load_candles
from crypto_quant.storage.models import (
    EquityCurveRecord,
    FactorScoreRecord,
    MarketStateRecord,
    OrderRecord,
    PositionRecord,
    RejectedSignalRecord,
    SignalRecord,
    StrategyRun,
)
from crypto_quant.strategy.engine import StrategyEngine
from crypto_quant.strategy.types import SwapRecommendation, TargetPosition
from crypto_quant.universe.service import WeeklyUniverseService
from crypto_quant.utils.time import ensure_utc, monday_utc

# Max bars needed by any factor: momentum 73 + MA trend 20 + vol avg 20 + buffer
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
    # v1: position state machine
    state: str = "strong"  # "strong" | "weakening" | "failed"
    weakened: bool = False  # True after first partial reduction
    entry_atr: float = 0.0  # ATR at entry time, for expansion tracking (§7.6-7.7)
    entry_vr: float = 0.0   # v28: volume ratio at entry, for post-entry tightening
    position_type: str = "main"  # "main" | "pump"
    max_favorable_pct: float = 0.0
    # v2.0: probe-and-confirm position sizing
    is_probe: bool = False           # True if position is a partial probe
    probe_full_qty: float = 0.0      # intended full position size
    probe_tier: str = ""             # "A" | "B" — signal quality at entry
    probe_confirmed: bool = False    # True after 4h confirmation add
    probe_add_qty: float = 0.0       # pending scale-in quantity
    entry_vr: float = 0.0            # volume ratio at entry
    probe_entry_price: float = 0.0
    confirm_entry_price: float = 0.0
    avg_entry_price: float = 0.0
    entry_notional: float = 0.0


@dataclass(frozen=True)
class PumpCandidate:
    symbol: str
    score: float
    price: float
    atr: float
    risk_multiplier: float
    reason: str
    ret_6h: float
    ret_24h: float
    ret_72h: float
    volume_ratio: float
    quote_volume_24h: float
    tier: str = "B"  # v2.0: "A" | "B" signal quality


@dataclass(frozen=True)
class OrderMetadata:
    mechanism: str | None = None
    trigger: str | None = None
    details: dict[str, object] | None = None


@dataclass
class Portfolio:
    cash: float
    positions: dict[str, OpenPosition] = field(default_factory=dict)
    # v1: risk tracking
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

    def record_trade(self, pnl: float, won: bool, position_type: str = "main") -> None:
        """Record a closed trade's PnL and outcome."""
        if pnl < 0:
            self.daily_realized_loss += pnl
        if position_type == "pump":
            self.pump_trade_results.append(won)
            if len(self.pump_trade_results) > 20:
                self.pump_trade_results = self.pump_trade_results[-20:]
        self.recent_trade_results.append(won)
        # keep only last 20 results
        if len(self.recent_trade_results) > 20:
            self.recent_trade_results = self.recent_trade_results[-20:]


@dataclass
class ResearchBacktester:
    config: AppConfig
    _last_engine: StrategyEngine | None = None
    _pos_cache_1h: dict[str, dict[pd.Timestamp, int]] = field(default_factory=dict)
    _pos_cache_4h: dict[str, dict[pd.Timestamp, int]] = field(default_factory=dict)
    _pump_cooldown_until: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))
    _pump_regime: str = "COLD"  # HOT | WARM | COLD
    _last_regime_check: datetime = field(default_factory=lambda: datetime(2000, 1, 1, tzinfo=UTC))
    _pump_consecutive_losses: int = 0
    _pump_symbol_cooldowns: dict[str, datetime] = field(default_factory=dict)  # per-symbol re-entry cooldown
    _pump_post_entry: list[dict[str, object]] = field(default_factory=list)  # post-entry price paths
    _pump_recent_exits: list[str] = field(default_factory=list)  # v20: adaptive risk tracking
    _candles_ref: dict[str, pd.DataFrame] = field(default_factory=dict)  # v2.5: full candles ref for enrichment
    _snapshot_cache_1h: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)

    def run_synthetic(self, session: Session | None = None) -> BacktestResult:
        run_id = self._create_run(session, "synthetic-mvp") if session is not None else None
        candles = self._synthetic_candles()
        factors = MomentumFactorEngine(self.config.momentum).score(candles)
        prices = {symbol: float(frame["close"].iloc[-1]) for symbol, frame in candles.items()}
        atrs = {symbol: float(compute_atr(frame, self.config.risk.atr_period).iloc[-1]) for symbol, frame in candles.items()}
        market = MarketState("risk_on", reasons=["synthetic_acceptance"])
        _, targets, _ = StrategyEngine(self.config).generate_targets(
            factors,
            market,
            prices,
            atrs,
            self.config.backtest.initial_equity,
            candles,
        )
        broker = BacktestBroker(self.config.backtest.fee_bps, self.config.backtest.slippage_bps)
        orders = [
            broker.execute_market(target.symbol, "buy", target.quantity, target.entry_price, "next_open_fill")
            for target in targets
        ]
        final_equity = self.config.backtest.initial_equity - sum(order.fee for order in orders)
        if session is not None and run_id is not None:
            self._write_orders(
                session,
                run_id,
                datetime.now(UTC),
                orders,
                [OrderMetadata(mechanism="entry", trigger="synthetic") for _ in orders],
            )
            self._finish_run(session, run_id, "completed")
        return BacktestResult(run_id, orders, final_equity)

    def run_real(
        self,
        session: Session,
        start: datetime,
        end: datetime,
        report_dir: Path | None = None,
    ) -> BacktestResult:
        start = ensure_utc(start)
        end = ensure_utc(end)
        run_id = self._create_run(session, "real-mvp")
        if run_id is None:
            raise RuntimeError("real backtest requires a database session")
        orders: list[Order] = []
        try:
            universe_map = WeeklyUniverseService(self.config).load_effective_map(session, start, end)
            all_universe = sorted(
                {symbol for symbols in universe_map.values() for symbol in symbols}
                | {self.config.market_state.btc_symbol}
            )
            if self.config.pump_mode.enabled:
                all_symbols = sorted(
                    set(distinct_candle_symbols(session, self.config.exchange_id, "1h"))
                    | {self.config.market_state.btc_symbol}
                )
            else:
                all_symbols = all_universe
            candles_1h = self._prepare_candles(load_candles(session, self.config.exchange_id, all_symbols, "1h", start, end))
            self._candles_ref = candles_1h  # for on-demand enrichment in _pump_candidates
            candles_4h = self._prepare_candles(load_candles(session, self.config.exchange_id, all_symbols, "4h", start, end))
            btc_1h = candles_1h.get(self.config.market_state.btc_symbol, pd.DataFrame())
            btc_4h_required = candles_4h.get(self.config.market_state.btc_symbol, pd.DataFrame())
            if btc_1h.empty or btc_4h_required.empty:
                missing = []
                if btc_1h.empty:
                    missing.append("1h")
                if btc_4h_required.empty:
                    missing.append("4h")
                raise RuntimeError(f"missing required BTC/USDT candles for backtest: {', '.join(missing)}")
            timeline = self._timeline(candles_1h, start, end)
            if not timeline:
                raise RuntimeError("no 1h candles available for backtest timeline")
            # Pre-build position cache: map (symbol, timestamp) → integer row position
            self._pos_cache_1h = self._build_position_cache(candles_1h, timeline)
            self._pos_cache_4h = self._build_position_cache(candles_4h, timeline)
            # Pre-compute all factor columns once on full DataFrames
            self._precompute_indicators(candles_1h, candles_4h)
            self._snapshot_cache_1h = self._build_snapshot_value_cache(candles_1h)
            portfolio = Portfolio(cash=self.config.backtest.initial_equity, initial_equity=self.config.backtest.initial_equity)
            broker = BacktestBroker(self.config.backtest.fee_bps, self._slippage_bps())
            equity_high = self.config.backtest.initial_equity
            engine = StrategyEngine(self.config)  # v1: stateful engine
            self._last_engine = engine
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

                active_symbols = [] if self.config.pump_mode.enabled else universe_map.get(monday_utc(now), [])
                if not self.config.pump_mode.enabled:
                    # Grace period: keep coins that were in the universe last week
                    prev_week = monday_utc(now) - timedelta(days=7)
                    prev_symbols = universe_map.get(prev_week, [])
                    active_symbols = sorted(set(active_symbols) | set(prev_symbols))

                # Filter mega-cap and excluded coins from trading universe.
                # BTC is still kept for market state monitoring via btc_symbol config.
                mega_caps = {c.upper() for c in self.config.universe.mega_cap_exclude}
                keywords = [kw.upper() for kw in self.config.universe.exclude_keywords]
                active_symbols = [
                    s for s in active_symbols
                    if s.split("/")[0].upper() not in mega_caps
                    and not any(kw in s.split("/")[0].upper() for kw in keywords)
                ]
                # Momentum watchlist: extreme momentum coins bypass pool volume threshold.
                # Scan ALL loaded candles, not just universe symbols.
                all_spot = [] if self.config.pump_mode.enabled else [s for s in candles_1h.keys() if s not in active_symbols
                           and s.split('/')[0].upper() not in mega_caps
                           and not any(kw in s.split('/')[0].upper() for kw in keywords)
                           and s != self.config.market_state.btc_symbol]
                if all_spot:
                    watch_1h = self._slice(candles_1h, all_spot, now, cache=self._pos_cache_1h)
                    for sym, frame in watch_1h.items():
                        if len(frame) < 73:
                            continue
                        # v20: use precomputed weighted_return
                        if "weighted_return" in frame.columns:
                            wret = float(frame["weighted_return"].iloc[-1])
                        else:
                            close = frame['close'].astype(float)
                            wret = (float(close.iloc[-1]/close.iloc[-5]-1)*0.25 +
                                    float(close.iloc[-1]/close.iloc[-25]-1)*0.35 +
                                    float(close.iloc[-1]/close.iloc[-49]-1)*0.25 +
                                    float(close.iloc[-1]/close.iloc[-73]-1)*0.15)
                        if wret >= 0.50:
                            active_symbols.append(sym)
                    active_symbols = sorted(set(active_symbols))
                # Always include held symbols for position management even if
                # they dropped out of the universe this week (white paper §2.4).
                held_symbols = list(portfolio.positions.keys())
                all_relevant = sorted(set(active_symbols) | set(held_symbols))

                if all_relevant:
                    current_1h = self._slice(candles_1h, all_relevant, now, cache=self._pos_cache_1h)
                    # v2.5: 4h slice only needed for main strategy (market state / breadth)
                    current_4h = {} if self.config.pump_mode.enabled else self._slice(candles_4h, all_relevant, now, cache=self._pos_cache_4h)
                else:
                    current_1h = {}
                    current_4h = {}

                pump_scan_1h: dict[str, pd.DataFrame] = {}
                pump_snapshot = pd.DataFrame()
                pump_prices_all: dict[str, float] = {}
                if self.config.pump_mode.enabled:
                    pump_symbols = [
                        s for s in all_symbols
                        if s != self.config.market_state.btc_symbol
                        and s.split("/")[0].upper() not in mega_caps
                        and not any(kw in s.split("/")[0].upper() for kw in keywords)
                    ]
                    pump_snapshot = self._pump_snapshot(candles_1h, pump_symbols, now)
                    pump_prices_all = self._pump_latest_prices(candles_1h, pump_symbols, now)

                # v2.5: fast path for pump-only — skip market breadth, keep fast_risk_valve
                if self.config.pump_mode.enabled:
                    current_btc_1h = self._slice(candles_1h, [self.config.market_state.btc_symbol], now, cache=self._pos_cache_1h).get(self.config.market_state.btc_symbol, pd.DataFrame())
                    fv, _ = fast_risk_valve_triggered(btc_1h=current_btc_1h)
                    market = MarketState("defensive" if fv else "risk_on", fast_risk_valve=fv, reasons=["pump_mode_fast"])
                else:
                    btc_4h = self._slice(candles_4h, [self.config.market_state.btc_symbol], now, cache=self._pos_cache_4h).get(
                        self.config.market_state.btc_symbol, pd.DataFrame())
                    current_btc_1h = self._slice(candles_1h, [self.config.market_state.btc_symbol], now, cache=self._pos_cache_1h).get(
                        self.config.market_state.btc_symbol, pd.DataFrame())
                    market = self._market_state(btc_4h, current_4h, current_btc_1h)
                prices_all = self._last_prices(current_1h)

                # ---- compute factor scores early (needed for position state rank check) ----
                factors = pd.DataFrame()
                if active_symbols:
                    # Filter from already-sliced current_* (active ⊆ all_relevant)
                    active_1h = {s: current_1h[s] for s in active_symbols if s in current_1h}
                    active_4h = {s: current_4h[s] for s in active_symbols if s in current_4h}
                    factors = self._fast_factors(active_1h)

                # ---- always run position management ----
                # Pass factor scores so position state can check ranking
                self._update_position_states(portfolio, current_1h, now, factors=factors)

                for symbol, position in list(portfolio.positions.items()):
                    if position.position_type == "main" and position.state == "weakening" and not position.weakened:
                        self._weakening_reduce(
                            session, run_id, now, next_time, symbol, position,
                            portfolio, candles_1h, broker, orders,
                        )

                # v1: check hard risk limits on held positions (§7.7)
                for symbol, position in list(portfolio.positions.items()):
                    if position.position_type != "main":
                        continue
                    current_atr = self._latest_atr(current_1h.get(symbol))
                    if self._check_hard_risk_limits(position, portfolio, prices_all, current_atr=current_atr):
                        position.state = "weakening"  # force weakening → will reduce or exit

                self._process_stops(session, run_id, now, next_time, portfolio, candles_1h, broker, orders, market, engine)

                # v2.5: detect pump regime from sliced candles, scan candidates if HOT/WARM
                pump_candidates: list = []
                pump_prices: dict[str, float] = {}
                if self.config.pump_mode.enabled and (not pump_snapshot.empty or pump_prices_all):
                    if now >= self._pump_cooldown_until:
                        if not hasattr(self, '_last_regime_check') or (now - self._last_regime_check).total_seconds() >= 14400:
                            self._pump_regime = self._detect_pump_regime_snapshot(pump_snapshot)
                            self._last_regime_check = now
                        if self._pump_regime in ("HOT", "WARM"):
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
                    )

                # ---- signal generation (only when universe is active; skip if pump-only mode) ----
                if active_symbols and not self.config.pump_mode.enabled:
                    prices_active = self._last_prices(active_1h)

                    equity = portfolio.equity(prices_all)
                    self._write_market_state(session, run_id, now, market)
                    self._write_factor_scores(session, run_id, now, factors)
                    atrs = {
                        symbol: float(compute_atr(frame, self.config.risk.atr_period).iloc[-1])
                        for symbol, frame in active_1h.items()
                        if len(frame) >= self.config.risk.atr_period + 1
                    }

                    signals, targets, rejected = engine.generate_targets(
                        factors,
                        market,
                        prices_active,
                        atrs,
                        equity,
                        active_1h,
                        candles_4h=active_4h,
                        now=now,
                        daily_realized_loss=portfolio.daily_realized_loss,
                        recent_trade_results=portfolio.recent_trade_results,
                    )
                    engine.update_false_breakout(factors, now=now)

                    for signal in signals:
                        session.add(
                            SignalRecord(
                                strategy_run_id=run_id,
                                time=now,
                                symbol=signal.symbol,
                                side=signal.side,
                                rank=signal.rank,
                                target_weight=signal.target_weight,
                                reason=signal.reason,
                            )
                        )
                    for symbol, reason in rejected:
                        self._reject(session, run_id, now, symbol, reason)
                    # --- swap mechanism (White Paper §10) ---
                    swaps = self._find_swaps(portfolio, targets, factors) if self.config.risk.swap_enabled else []
                    for swap in swaps:
                        # Sell the weak position and buy the stronger one
                        old_pos = portfolio.positions.get(swap.sell_symbol)
                        if old_pos is not None:
                            self._full_exit(
                                session, run_id, next_time, swap.sell_symbol, old_pos,
                                portfolio, candles_1h, broker, orders, market, old_pos.stop_price,
                                f"swap_out_to_{swap.buy_target.symbol}", now,
                            )

                    approved = RiskEngine(self.config.risk).approve_targets(
                        [target for target in targets if target.symbol not in portfolio.positions]
                    ).approved
                    self._enter_positions(session, run_id, now, next_time, approved, portfolio, candles_1h, broker, orders)
                else:
                    equity = portfolio.equity(prices_all)

                equity_now = portfolio.equity(prices_all)
                equity_high = max(equity_high, equity_now)
                drawdown = equity_now / equity_high - 1 if equity_high else 0
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

    def _create_run(self, session: Session | None, name: str) -> int | None:
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

    def _finish_run(self, session: Session, run_id: int, status: str) -> None:
        run = session.get(StrategyRun, run_id)
        if run is not None:
            run.status = status
            run.finished_at = datetime.now(UTC)

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
            if frame is None or frame.empty: continue
            if sym in cache:
                pos = cache[sym].get(end_ts)
                if pos is None or int(pos) < 73: continue
                idx = int(pos) - 1
            else:
                try:
                    idx = frame.index.get_loc(end_ts) - 1
                    if idx < 72: continue
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
            "new_12h_high", "regime_vol_expansion", "atr14",
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
        prices: dict[str, float] = {}
        end_ts = pd.Timestamp(ensure_utc(end))
        for symbol in symbols:
            frame = candles.get(symbol)
            if frame is None or frame.empty:
                continue
            pos = self._cached_position(symbol, end)
            if pos is not None:
                idx = int(pos) - 1
            else:
                try:
                    idx = frame.index.get_loc(end_ts)
                except KeyError:
                    continue
            if idx < 0:
                continue
            values = self._snapshot_cache_1h.get(symbol)
            if values is not None:
                prices[symbol] = float(values["close"][idx])
            else:
                prices[symbol] = float(frame["close"].iloc[idx])
        return prices

    def _pump_snapshot(self, candles: dict[str, pd.DataFrame], symbols: list[str], end: datetime) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        end_ts = pd.Timestamp(ensure_utc(end))
        for symbol in symbols:
            frame = candles.get(symbol)
            if frame is None or frame.empty:
                continue
            pos = self._cached_position(symbol, end)
            if pos is not None:
                p = int(pos)
                idx = p - 1
            else:
                try:
                    idx = frame.index.get_loc(end_ts)
                    p = idx + 1
                except KeyError:
                    continue
            if idx < 0:
                continue
            values = self._snapshot_cache_1h.get(symbol)
            if values is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "history": p,
                    "price": float(values["close"][idx]),
                    "ret_24h": float(values["ret_24h"][idx]),
                    "ret_72h": float(values["ret_72h"][idx]),
                    "ret_6h": float(values["ret_6h"][idx]),
                    "above_ma20": bool(values["above_ma20"][idx]),
                    "qv_6h": float(values["qv_6h_sum"][idx]),
                    "qv_24h": float(values["qv_24h_sum"][idx]),
                    "qv_30_avg": float(values["qv_30_avg"][idx]),
                    "wick_ratio": float(values["wick_ratio"][idx]),
                    "new_12h_high": bool(values["new_12h_high"][idx]),
                    "regime_vol_expansion": bool(values["regime_vol_expansion"][idx]),
                    "atr": float(values["atr14"][idx]),
                }
            )
        return pd.DataFrame(rows)

    def _last_prices(self, candles: dict[str, pd.DataFrame]) -> dict[str, float]:
        return {symbol: float(frame["close"].iloc[-1]) for symbol, frame in candles.items() if not frame.empty}

    @staticmethod
    def _trade_entry_price(position: OpenPosition) -> float:
        return position.avg_entry_price if position.position_type == "pump" and position.avg_entry_price > 0 else position.entry_price

    @staticmethod
    def _trade_entry_notional(position: OpenPosition) -> float:
        return position.entry_notional if position.entry_notional > 0 else position.quantity * position.entry_price

    def _market_state(
        self,
        btc_4h: pd.DataFrame,
        candles_4h: dict[str, pd.DataFrame],
        btc_1h: pd.DataFrame | None = None,
    ) -> MarketState:
        state = evaluate_btc_ma50_state(btc_4h, self.config.market_state)
        breadth = compute_market_breadth(candles_4h, ma_period=20)
        reasons = list(state.reasons)
        fast_triggered, fast_reasons = fast_risk_valve_triggered(btc_1h=btc_1h)
        reasons.extend(fast_reasons)
        if fast_triggered:
            return MarketState(
                "defensive",
                state.btc_close,
                state.btc_ma50,
                state.ma50_slope_4,
                breadth,
                fast_risk_valve=True,
                reasons=reasons,
            )
        if breadth < 0.2:
            return MarketState(
                "defensive",
                state.btc_close,
                state.btc_ma50,
                state.ma50_slope_4,
                breadth,
                reasons=reasons + ["breadth_emergency"],
            )
        if breadth < 0.25:
            return MarketState("caution", state.btc_close, state.btc_ma50, state.ma50_slope_4, breadth, reasons=reasons + ["breadth_pause"])
        if breadth < 0.35:
            return MarketState("caution", state.btc_close, state.btc_ma50, state.ma50_slope_4, breadth, reasons=reasons + ["breadth_weak"])
        return MarketState(state.state, state.btc_close, state.btc_ma50, state.ma50_slope_4, breadth, state.fast_risk_valve, reasons)

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
        engine: StrategyEngine,
    ) -> None:
        for symbol, position in list(portfolio.positions.items()):
            frame = candles_1h.get(symbol, pd.DataFrame())
            current = self._slice_frame(frame, now) if not frame.empty else pd.DataFrame()
            if current.empty:
                continue

            stop_before = position.stop_price
            if position.position_type == "pump":
                forced_reason = self._update_pump_stop(position, current, now)
                if forced_reason == "probe_confirm":
                    # v2.0: scale-in to full position
                    add_qty = position.probe_add_qty
                    next_open = self._next_open(candles_1h, symbol, next_time)
                    if add_qty > 0 and next_open is not None and next_open > 0:
                        est_cost = add_qty * next_open * 1.002  # conservative: include est fee
                        if est_cost <= portfolio.cash:
                            scale_order = broker.execute_market(symbol, "buy", add_qty, next_open, f"probe_confirm_{position.probe_tier}")
                            portfolio.cash -= scale_order.quantity * scale_order.filled_price + scale_order.fee
                            entry_notional = self._trade_entry_notional(position) + scale_order.quantity * scale_order.filled_price
                            position.quantity += scale_order.quantity
                            position.entry_notional = entry_notional
                            position.avg_entry_price = entry_notional / position.quantity if position.quantity > 0 else position.entry_price
                            position.confirm_entry_price = scale_order.filled_price
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
                                    },
                                )])
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
            else:
                self._update_position_stop(position, current, market_state)

            # v1: check for failed state — full exit
            if position.position_type == "main" and position.state == "failed" and position.weakened:
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
                        "failed_exit",
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
            if position.position_type == "pump":
                reason = mechanism if mechanism.startswith("pump_") else "pump_stop"
            else:
                reason = mechanism if mechanism in ("trailing_stop", "breakeven_stop", "defensive_exit") else "atr_stop"
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
        """Execute a full position exit and record trade result."""
        frame = candles_1h.get(symbol, pd.DataFrame())
        current = self._slice_frame(frame, trigger_time) if not frame.empty else pd.DataFrame()
        exit_price = self._next_open(candles_1h, symbol, exit_time)
        if exit_price is None:
            exit_price = float(current["close"].iloc[-1]) if not current.empty else position.entry_price
        if exit_price <= 0:
            return
        order = broker.execute_market(symbol, "sell", position.quantity, exit_price, reason)
        if order.filled_price <= 0:
            return
        orders.append(order)
        entry_anchor = self._trade_entry_price(position)
        pnl = (order.filled_price - entry_anchor) * position.quantity - order.fee
        portfolio.cash += order.quantity * order.filled_price - order.fee
        won = order.filled_price > entry_anchor
        portfolio.record_trade(pnl, won, position.position_type)
        if self._last_engine is not None:
            self._last_engine.feed_trade_result(won)
        # v20: track exit reason for adaptive risk
        if position.position_type == "pump":
            from datetime import timedelta
            cfg_ar = self.config.pump_mode
            self._pump_recent_exits.append(reason or '?')
            lookback = getattr(cfg_ar, 'adaptive_risk_lookback', 20)
            if len(self._pump_recent_exits) > lookback * 2:
                self._pump_recent_exits = self._pump_recent_exits[-lookback * 2:]
            # Record post-entry price path for analysis
            post = self._compute_post_entry_path(candles_1h, symbol, position.opened_at, entry_anchor, trigger_time)
            if post:
                post['symbol'] = symbol; post['exit_reason'] = reason; post['pnl'] = pnl
                self._pump_post_entry.append(post)
        del portfolio.positions[symbol]
        details = self._exit_details(position, current, market_state, stop_before, order.filled_price, trigger_time)
        self._write_orders(
            session,
            run_id,
            exit_time,
            [order],
            [OrderMetadata(mechanism=position.stop_mechanism, trigger=position.stop_trigger, details=details)],
        )
        self._write_position(session, run_id, exit_time, position, "closed", order.filled_price)

    def _update_position_states(
        self,
        portfolio: Portfolio,
        candles: dict[str, pd.DataFrame],
        now: datetime,
        factors: pd.DataFrame | None = None,
    ) -> None:
        """Update position states: strong → weakening → failed (White Paper §9.6).

        Args:
            factors: current factor scores DataFrame (for rank-based weakening check).
        """
        rc = self.config.risk

        # Build rank lookup from factor scores if available
        rank_map: dict[str, int] = {}
        if factors is not None and not factors.empty:
            scored = factors.sort_values("final_score", ascending=False).reset_index(drop=True)
            for idx, row in enumerate(scored.itertuples(), start=1):
                rank_map[str(row.symbol)] = idx

        for symbol, position in list(portfolio.positions.items()):
            if position.position_type != "main":
                continue
            frame = candles.get(symbol, pd.DataFrame())
            if frame.empty or position.weakened:
                continue

            close = float(frame["close"].iloc[-1])
            ma_short = float(frame["close"].rolling(20).mean().iloc[-1])
            high_since_entry = max(position.highest_price or position.entry_price, float(frame["high"].max()))

            rank = rank_map.get(symbol, 999)

            # Check weakening conditions (White Paper §9.6)
            weakening = False

            # A: rank > 5 AND close < 1H MA20
            below_short_ma = close < ma_short
            rank_dropped = rank > rc.weakening_rank_threshold

            # B: drawdown from highest > 1.5x ATR
            dd_from_high = high_since_entry - close
            deep_drawdown = dd_from_high > rc.weakening_drawdown_atr_multiple * position.atr

            # D: volume stall on held position
            from crypto_quant.factors.volume_stall import detect_volume_stall
            stall = detect_volume_stall(symbol, frame, candles, rc)

            # State transitions
            failing = False

            # Strong → Weakening
            if rank_dropped and below_short_ma:
                weakening = True
            elif rank_dropped and deep_drawdown:
                weakening = True
            elif stall.stalled:
                weakening = True

            # Weakening → Failed
            if below_short_ma and close < ma_short * 0.95:  # clear break below MA
                failing = True
            if dd_from_high > rc.weakening_drawdown_atr_multiple * 2.0 * position.atr:
                failing = True
            if rank > rc.weakening_rank_upper:  # rank > 15 per white paper
                failing = True

            if failing:
                position.state = "failed"
            elif weakening:
                position.state = "weakening"

    def _weakening_reduce(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        next_time: datetime,
        symbol: str,
        position: OpenPosition,
        portfolio: Portfolio,
        candles_1h: dict[str, pd.DataFrame],
        broker: BacktestBroker,
        orders: list[Order],
    ) -> None:
        """Reduce position by weakening_reduction_pct and tighten stops."""
        rc = self.config.risk
        reduction_pct = rc.weakening_reduction_pct
        sell_qty = position.quantity * reduction_pct

        next_open = self._next_open(candles_1h, symbol, next_time)
        if next_open is None:
            return

        order = broker.execute_market(symbol, "sell", sell_qty, next_open, "weakening_reduce")
        if order.filled_price <= 0:
            return

        orders.append(order)
        pnl = (order.filled_price - position.entry_price) * sell_qty - order.fee
        portfolio.cash += order.quantity * order.filled_price - order.fee
        won = order.filled_price > position.entry_price
        portfolio.record_trade(pnl, won, position.position_type)
        if self._last_engine is not None:
            self._last_engine.feed_trade_result(won)

        # reduce remaining position and tighten trailing
        position.quantity -= sell_qty
        position.weakened = True
        if position.quantity <= 0:
            del portfolio.positions[symbol]
            self._write_position(session, run_id, next_time, position, "closed", order.filled_price)
            return

        # tighten trailing stop for remaining position if active
        if position.trailing_active and position.highest_price is not None:
            new_trailing = position.highest_price - position.atr * rc.weakening_trailing_multiple
            if new_trailing > position.stop_price:
                position.stop_price = new_trailing
                position.stop_mechanism = "trailing_stop_tightened"
                position.stop_trigger = "low_below_tightened_trailing"

        self._write_orders(
            session,
            run_id,
            next_time,
            [order],
            [OrderMetadata(mechanism="weakening_reduce", trigger="state_transition")],
        )
        self._write_position(session, run_id, next_time, position, "weakening", order.filled_price)

    def _update_position_stop(self, position: OpenPosition, current: pd.DataFrame, market_state: MarketState) -> None:
        if position.atr <= 0:
            return
        current = self._position_history(current, position.opened_at)
        if current.empty:
            return
        high = float(current["high"].max())
        close = float(current["close"].iloc[-1])
        position.highest_price = max(position.highest_price or position.entry_price, high)

        # v1: hybrid stop — use structure level if available (White Paper §8.2 Scheme C)
        if self.config.risk.hybrid_stop_enabled and position.stop_mechanism == "initial_atr_stop":
            structure_stop = self._find_structure_stop(current, position)
            if structure_stop is not None:
                atr_stop = position.entry_price - position.atr * self.config.risk.atr_stop_multiple
                # Use structure stop if it's tighter (higher) than ATR stop
                if structure_stop > atr_stop:
                    position.stop_price = structure_stop
                    position.stop_mechanism = "structure_stop"
                    position.stop_trigger = "low_below_structure"
                    position.last_stop_update = {
                        "trigger": "structure_stop",
                        "structure_stop": structure_stop,
                        "atr_stop": atr_stop,
                    }

        if self.config.risk.enable_breakeven_stop:
            activation = position.entry_price + position.atr * self.config.risk.breakeven_activation_atr_multiple
            if high >= activation:
                breakeven = position.entry_price * (1 - self.config.risk.breakeven_buffer_bps / 10_000)
                new_stop = max(position.stop_price, min(breakeven, close))
                if new_stop > position.stop_price:
                    position.stop_price = new_stop
                    position.stop_mechanism = "breakeven_stop"
                    position.stop_trigger = "low_below_breakeven_stop"
                    position.last_stop_update = {
                        "trigger": "breakeven_activation",
                        "activation_price": activation,
                        "highest_price": position.highest_price,
                    }
        if self.config.risk.enable_trailing_stop:
            activation = position.entry_price + position.atr * self.config.risk.trailing_activation_atr_multiple
            if high >= activation:
                position.trailing_active = True
                trailing_stop = high - position.atr * self.config.risk.trailing_stop_atr_multiple
                new_stop = max(position.stop_price, min(trailing_stop, close))
                if new_stop > position.stop_price:
                    position.stop_price = new_stop
                    position.stop_mechanism = "trailing_stop"
                    position.stop_trigger = "low_below_trailing_stop"
                    position.last_stop_update = {
                        "trigger": "trailing_activation",
                        "activation_price": activation,
                        "highest_price": position.highest_price,
                        "trailing_width_atr": self.config.risk.trailing_stop_atr_multiple,
                    }
        if (
            self.config.risk.defensive_tighten_existing_positions
            and market_state.state == "defensive"
            and close <= position.entry_price + position.atr
        ):
            defensive_stop = close - position.atr * self.config.risk.defensive_tighten_atr_multiple
            new_stop = max(position.stop_price, defensive_stop)
            if new_stop > position.stop_price:
                position.stop_price = new_stop
                position.stop_mechanism = "defensive_exit"
                position.stop_trigger = "low_below_defensive_tightened_stop"
                position.last_stop_update = {
                    "trigger": "defensive_tighten",
                    "market_state": market_state.state,
                    "defensive_tighten_atr": self.config.risk.defensive_tighten_atr_multiple,
                }

    def _update_pump_stop(self, position: OpenPosition, current: pd.DataFrame, now: datetime) -> str | None:
        cfg = self.config.pump_mode
        if position.atr <= 0:
            return None
        current = self._position_history(current, position.opened_at)
        if current.empty:
            return None

        high = float(current["high"].max())
        close = float(current["close"].iloc[-1])
        anchor_price = self._trade_entry_price(position)
        position.highest_price = max(position.highest_price or anchor_price, high)
        mfe_pct = position.highest_price / anchor_price - 1
        position.max_favorable_pct = max(position.max_favorable_pct, mfe_pct)

        held_hours = (ensure_utc(now) - ensure_utc(position.opened_at)).total_seconds() / 3600
        is_unconfirmed_b = position.is_probe and position.probe_tier == "B" and not position.probe_confirmed

        # v2.0: probe confirmation at 4h
        if position.is_probe and not position.probe_confirmed and held_hours >= 3.5:
            ret_4h_val = close / anchor_price - 1
            if ret_4h_val >= 0:
                # Confirm: mark for scale-in (handled in _process_stops)
                remaining_qty = position.probe_full_qty - position.quantity
                if remaining_qty > 0:
                    position.probe_confirmed = True
                    position.probe_add_qty = remaining_qty
                    return 'probe_confirm'
            elif is_unconfirmed_b and ret_4h_val <= cfg.unconfirmed_b_ret_4h_exit_pct:
                position.stop_mechanism = "pump_b_unconfirmed_4h_down"
                position.stop_trigger = "pump_b_ret_4h_cut"
                return "pump_b_unconfirmed_4h_down"
            elif ret_4h_val <= -0.02:
                # Kill: tighten stop to near current price, force exit
                tight_stop = close - position.atr * 0.3
                if tight_stop > position.stop_price:
                    position.stop_price = tight_stop
                    position.stop_mechanism = 'pump_probe_kill'
                    position.stop_trigger = 'pump_probe_4h_dead'

        if len(current) >= 3:
            post_close = current['close'].astype(float)
            h1 = float(post_close.iloc[-3] / anchor_price - 1) if len(post_close) >= 3 else 0
            h2 = float(post_close.iloc[-2] / anchor_price - 1) if len(post_close) >= 2 else 0
            h3 = float(post_close.iloc[-1] / anchor_price - 1)
            if is_unconfirmed_b and h3 <= cfg.unconfirmed_b_ret_3h_exit_pct and close < anchor_price:
                position.stop_mechanism = "pump_b_unconfirmed_3h_down"
                position.stop_trigger = "pump_b_ret_3h_cut"
                return "pump_b_unconfirmed_3h_down"
            if h1 < 0 and h2 < h1 and h3 < h2:
                position.stop_mechanism = 'pump_3h_down'
                position.stop_trigger = 'pump_consecutive_down'
                return 'pump_3h_down'
        if held_hours >= cfg.stagnation_stop_hours and position.max_favorable_pct < cfg.stagnation_min_mfe_pct:
            position.stop_mechanism = "pump_stagnation_exit"
            position.stop_trigger = "pump_no_fast_follow_through"
            return "pump_stagnation_exit"
        if held_hours >= cfg.time_stop_hours and close < anchor_price * (1 + cfg.time_stop_min_profit_pct):
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
        # v2.2: same lock/breakeven for all positions (keep v2.0's working levels)
        if mfe_pct >= 0.08 and anchor_price > new_stop:
            new_stop = anchor_price
            mechanism = "pump_breakeven"
            trigger = "pump_be_8pct"
        if mfe_pct >= 0.10:
            lock_stop = anchor_price * 1.02
            if lock_stop > new_stop:
                new_stop = lock_stop
                mechanism = "pump_lock_2pct"
                trigger = "pump_lock_10pct_mfe"

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

        if new_stop > position.stop_price:
            position.stop_price = new_stop
            position.stop_mechanism = mechanism
            position.stop_trigger = trigger
            position.last_stop_update = {
                "trigger": mechanism,
                "mfe_pct": mfe_pct,
                "stop_anchor_price": anchor_price,
                "highest_price": position.highest_price,
            }
        return None

    def _find_structure_stop(self, current: pd.DataFrame, position: OpenPosition) -> float | None:
        """Find a structure-based stop level from recent swing lows (White Paper §8.2 Scheme C).

        Looks for the lowest low in the last 6-12 candles as a structure support.
        Uses it only if it falls within 1.0-2.5x ATR from entry.
        """
        rc = self.config.risk
        if len(current) < 6:
            return None

        low = current["low"].astype(float)
        # Find swing low in last 6-12 candles
        recent_low = float(low.iloc[-12:].min()) if len(low) >= 12 else float(low.iloc[-6:].min())
        structure_distance = position.entry_price - recent_low
        atr_distance = position.atr

        if atr_distance <= 0:
            return None

        atr_multiple = structure_distance / atr_distance
        # Only use structure stop if within reasonable ATR range
        if rc.hybrid_stop_structure_atr_min <= atr_multiple <= rc.hybrid_stop_structure_atr_max:
            return recent_low
        return None

    def _check_hard_risk_limits(
        self,
        position: OpenPosition,
        portfolio: Portfolio,
        prices: dict[str, float],
        current_atr: float | None = None,
    ) -> bool:
        """Check hard risk limits (White Paper §7.7). Returns True if risk reduction is needed."""
        rc = self.config.risk
        if position.atr <= 0:
            return False

        equity = portfolio.equity(prices)
        if equity <= 0:
            return True

        price = prices.get(position.symbol, position.entry_price)
        effective_atr = current_atr if current_atr is not None and current_atr > 0 else position.atr
        # volatility risk exposure
        vol_risk = position.quantity * effective_atr * rc.atr_stop_multiple / equity
        if vol_risk > rc.hard_risk_volatility_exposure_pct:
            return True

        # ATR expansion + drawdown check
        if hasattr(position, "entry_atr") and position.entry_atr > 0:
            atr_expansion = effective_atr / position.entry_atr
            if atr_expansion > rc.hard_risk_atr_expansion:
                drawdown_from_high = (position.highest_price or price) - price
                if position.highest_price and drawdown_from_high > rc.hard_risk_dd_atr_multiple * effective_atr:
                    return True

        return False

    def _fast_factors(self, cur_1h: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Build factor DataFrame from pre-computed columns (no per-symbol computation)."""
        rows = []
        for sym, frame in cur_1h.items():
            if frame.empty:
                continue
            last = frame.iloc[-1]
            if "weighted_return" not in frame.columns or pd.isna(last.get("weighted_return")):
                continue
            rows.append({
                "symbol": sym,
                "weighted_return": float(last["weighted_return"]),
                "momentum_score": 0.5,
                "volume_score": 0.5,
                "trend_score": 0.5,
                "final_score": 0.5,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df["momentum_score"] = df["weighted_return"].rank(pct=True)
            df["final_score"] = df["momentum_score"]
        return df.sort_values("final_score", ascending=False).reset_index(drop=True) if not df.empty else df

    def _precompute_indicators(self, candles_1h: dict[str, pd.DataFrame], candles_4h: dict[str, pd.DataFrame]) -> None:
        """Pre-compute all factor columns on the full candle DataFrames.
        The hourly loop then just reads the last values instead of recomputing.
        """
        windows = self.config.momentum.windows_hours
        weights = self.config.momentum.weights
        if len(windows) != len(weights):
            raise ValueError("momentum windows and weights must have the same length")
        min_window = max(windows) if windows else 0
        vol_cfg = self.config.volume_score
        trend_cfg = self.config.trend
        for sym, frame in candles_1h.items():
            if frame.empty or len(frame) <= min_window:
                continue
            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            volume = frame["volume"].astype(float)
            # Momentum
            weighted_return = pd.Series(0.0, index=frame.index)
            for window, weight in zip(windows, weights, strict=True):
                ret_col = f"ret_{window}h"
                frame[ret_col] = close / close.shift(window) - 1
                weighted_return = weighted_return + frame[ret_col] * float(weight)
            frame["weighted_return"] = weighted_return
            # v20: precompute ret_6h for pump candidate fast path
            frame["ret_6h"] = close / close.shift(6) - 1
            # Trend helpers
            frame["ma20"] = close.rolling(trend_cfg.ma_short_period).mean()
            # ATR
            tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
            frame["atr14"] = tr.rolling(14).mean()
            # Volume score mirrors strategy.engine._compute_volume_score.
            frame["volume_score_col"] = 0.5
            if vol_cfg.enabled:
                recent_volume = volume.rolling(vol_cfg.lookback_hours).sum()
                avg_volume = volume.shift(vol_cfg.lookback_hours).rolling(vol_cfg.average_hours).mean() * vol_cfg.lookback_hours
                vol_ratio = recent_volume / avg_volume.replace(0, np.nan)
                price_change = close / close.shift(vol_cfg.lookback_hours) - 1
                frame["vol_ratio"] = vol_ratio
                # v2.5: precompute pump-candidate fast-path columns
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

                frame.loc[(vol_ratio >= 1.2) & (price_change > 0.005), "volume_score_col"] = (
                    0.8 + ((vol_ratio - 1.2) * 0.2).clip(upper=0.2)
                )
                frame.loc[(vol_ratio >= 1.2) & (price_change <= 0.005), "volume_score_col"] = (
                    0.1 + (1.0 - vol_ratio / 3.0).clip(lower=0) * 0.2
                )
                frame.loc[(vol_ratio >= 0.7) & (vol_ratio < 1.2) & (price_change > 0), "volume_score_col"] = (
                    0.5 + price_change * 5.0
                )
                frame.loc[(vol_ratio < 0.7) & (price_change > -0.01), "volume_score_col"] = 0.4
                frame.loc[price_change < -0.01, "volume_score_col"] = 0.1
            frame["volume_score_col"] = frame["volume_score_col"].fillna(0.5)
            frame["trend_score_col"] = self._precomputed_trend_score(sym, frame, candles_4h)
        # 4H candles: just add MA50 for BTC
        btc_4h = candles_4h.get(self.config.market_state.btc_symbol)
        if btc_4h is not None and not btc_4h.empty:
            btc_4h["ma50"] = btc_4h["close"].astype(float).rolling(50).mean()

    def _latest_atr(self, frame: pd.DataFrame | None) -> float | None:
        if frame is None or frame.empty:
            return None
        if "atr14" in frame.columns:
            atr = frame["atr14"].dropna()
            if not atr.empty:
                return float(atr.iloc[-1])
        if len(frame) >= self.config.risk.atr_period + 1:
            value = compute_atr(frame, self.config.risk.atr_period).iloc[-1]
            if pd.notna(value):
                return float(value)
        return None

    def _precomputed_trend_score(
        self,
        symbol: str,
        frame_1h: pd.DataFrame,
        candles_4h: dict[str, pd.DataFrame],
    ) -> pd.Series:
        trend_cfg = self.config.trend
        if not trend_cfg.enabled:
            return pd.Series(0.5, index=frame_1h.index)

        close = frame_1h["close"].astype(float)
        high = frame_1h["high"].astype(float) if "high" in frame_1h.columns else close
        ma_short = close.rolling(trend_cfg.ma_short_period).mean()
        above_short = (close > ma_short).astype(float) * 0.33

        ma_long = close.rolling(trend_cfg.ma_long_period).mean()
        ma_long_prev = ma_long.shift(trend_cfg.ma_long_period)
        frame_4h = candles_4h.get(symbol, pd.DataFrame())
        if frame_4h is not None and not frame_4h.empty and len(frame_4h) >= trend_cfg.ma_long_period + 1:
            four_h = frame_4h.sort_index() if isinstance(frame_4h.index, pd.DatetimeIndex) else frame_4h.sort_values("open_time")
            close_4h = four_h["close"].astype(float)
            long_df = pd.DataFrame(
                {
                    "_time": pd.to_datetime(four_h["open_time"], utc=True),
                    "ma_long": close_4h.rolling(trend_cfg.ma_long_period).mean(),
                    "ma_long_prev": close_4h.rolling(trend_cfg.ma_long_period).mean().shift(trend_cfg.ma_long_period),
                }
            ).dropna(subset=["ma_long"])
            if not long_df.empty:
                base = pd.DataFrame({"_time": pd.to_datetime(frame_1h["open_time"], utc=True)}, index=frame_1h.index)
                base_sorted = base.sort_values("_time")
                aligned = pd.merge_asof(
                    base_sorted,
                    long_df.sort_values("_time"),
                    on="_time",
                    direction="backward",
                ).set_index(base_sorted.index)
                ma_long = aligned["ma_long"].reindex(frame_1h.index)
                ma_long_prev = aligned["ma_long_prev"].reindex(frame_1h.index)

        above_long = (close > ma_long).astype(float) * 0.33
        slope = ma_long / ma_long_prev - 1
        slope_score = (slope * 100).clip(lower=0.0, upper=1.0).fillna(0.0) * 0.34

        high_24h = high.rolling(24).max()
        dd_from_high = close / high_24h.replace(0, np.nan) - 1
        dd_penalty = (dd_from_high.abs() * 4).clip(upper=0.2).where(
            dd_from_high < -trend_cfg.max_drawdown_from_24h_high,
            0.0,
        )

        return (above_short + above_long + slope_score - dd_penalty.fillna(0.0)).clip(lower=0.0, upper=1.0).fillna(0.5)

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
            "stop_anchor_price": entry_anchor,
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
            "strategy_type": position.position_type,
            "market_state": market_state.state,
            "market_reasons": market_state.reasons,
            "last_close": close,
            "last_stop_update": position.last_stop_update,
        }

    def _enter_positions(
        self,
        session: Session,
        run_id: int,
        now: datetime,
        next_time: datetime,
        targets: list[TargetPosition],
        portfolio: Portfolio,
        candles_1h: dict[str, pd.DataFrame],
        broker: BacktestBroker,
        orders: list[Order],
    ) -> None:
        for target in targets:
            next_open = self._next_open(candles_1h, target.symbol, next_time)
            if next_open is None:
                self._reject(session, run_id, now, target.symbol, "missing_next_open")
                continue
            order = broker.execute_market(target.symbol, "buy", target.quantity, next_open, "next_1h_open")
            notional = order.quantity * order.filled_price + order.fee
            if notional > portfolio.cash:
                self._reject(session, run_id, now, target.symbol, "insufficient_cash")
                continue
            portfolio.cash -= notional
            stop_price = order.filled_price - target.atr * self.config.risk.atr_stop_multiple
            position = OpenPosition(
                target.symbol,
                target.quantity,
                order.filled_price,
                stop_price,
                target.atr,
                next_time,
                highest_price=order.filled_price,
                entry_atr=target.atr,
            )
            portfolio.positions[target.symbol] = position
            orders.append(order)
            self._write_orders(
                session,
                run_id,
                next_time,
                [order],
                [
                    OrderMetadata(
                        mechanism="entry",
                        trigger="next_1h_open",
                        details={
                            "signal_time": now.isoformat(),
                            "entry_price": order.filled_price,
                            "atr": target.atr,
                            "stop_price": stop_price,
                            "atr_stop_multiple": self.config.risk.atr_stop_multiple,
                            "strategy_type": "main",
                        },
                    )
                ],
            )
            self._write_position(session, run_id, next_time, position, "open", order.filled_price)

    def _compute_post_entry_path(self, candles_1h, symbol, opened_at, entry_price, now):
        """Record 1h/2h/3h/4h post-entry performance for pump trades."""
        frame = candles_1h.get(symbol, pd.DataFrame())
        if frame.empty: return None
        opened_ts = pd.Timestamp(ensure_utc(opened_at))
        now_ts = pd.Timestamp(ensure_utc(now))
        if 'open' not in frame.columns: return None
        # Get candles from entry time to now
        mask = (frame.index >= opened_ts) & (frame.index <= now_ts) if isinstance(frame.index, pd.DatetimeIndex) else (pd.to_datetime(frame['open_time'], utc=True) >= opened_ts) & (pd.to_datetime(frame['open_time'], utc=True) <= now_ts)
        post = frame[mask]
        if len(post) < 1: return None
        result = {'entry_price': float(entry_price)}
        # Track max drawdown and max favorable excursion at 1h, 2h, 3h, 4h
        for h in [1, 2, 3, 4]:
            rows = post.head(h)
            if len(rows) == 0: break
            high = float(rows['high'].max()) if 'high' in rows.columns else float(rows['close'].max())
            low = float(rows['low'].min()) if 'low' in rows.columns else float(rows['close'].min())
            close_h = float(rows['close'].iloc[-1]) if len(rows) > 0 else entry_price
            result[f'mfe_{h}h'] = (high / entry_price - 1) * 100
            result[f'mae_{h}h'] = (low / entry_price - 1) * 100
            result[f'ret_{h}h'] = (close_h / entry_price - 1) * 100
            result[f'high_{h}h'] = (high > entry_price)
            result[f'low_below_entry_{h}h'] = (low < entry_price)
        return result

    def _detect_pump_regime(self, candles_1h: dict[str, pd.DataFrame]) -> str:
        """Detect pump regime from sliced candle dict (uses precomputed columns)."""
        cfg = self.config.pump_mode
        ret_24h: list[float] = []
        new_high_count = 0
        vol_exp_count = 0
        total = 0
        for sym, frame in candles_1h.items():
            if len(frame) < 50: continue
            total += 1
            r24 = float(frame['ret_24h'].iloc[-1]) if 'ret_24h' in frame.columns else float(frame['close'].iloc[-1]/frame['close'].iloc[-25]-1)
            ret_24h.append(r24)
            if 'new_12h_high' in frame.columns:
                if bool(frame['new_12h_high'].iloc[-1]): new_high_count += 1
            volume = frame['volume']
            # v2.5: base volume for regime (quote vol tested separately)
            if len(volume) >= 50:
                avg_vol = float(volume.iloc[-50:-2].mean()); recent_vol = float(volume.iloc[-6:].mean())
                if avg_vol > 0 and recent_vol / avg_vol >= cfg.regime_hot_volume_expansion_ratio: vol_exp_count += 1
        if total < 20: return "COLD"
        median_ret = sorted(ret_24h)[len(ret_24h)//2] if ret_24h else 0.0
        nh_r = new_high_count / total; ve_r = vol_exp_count / total
        if median_ret >= cfg.regime_hot_24h_return_pct and nh_r >= cfg.regime_hot_new_high_ratio and ve_r >= 0.05: return "HOT"
        if median_ret >= cfg.regime_warm_24h_return_pct and nh_r >= cfg.regime_warm_new_high_ratio: return "WARM"
        return "COLD"

    def _detect_pump_regime_snapshot(self, snapshot: pd.DataFrame) -> str:
        cfg = self.config.pump_mode
        if snapshot.empty:
            return "COLD"
        eligible = snapshot[snapshot["history"].astype(int) >= 50]
        total = len(eligible)
        if total < 20:
            return "COLD"
        ret_24h = [float(value) for value in eligible["ret_24h"].tolist() if pd.notna(value)]
        if not ret_24h:
            return "COLD"
        median_ret = sorted(ret_24h)[len(ret_24h) // 2]
        new_high_count = int(eligible["new_12h_high"].fillna(False).astype(bool).sum())
        vol_exp_count = int(eligible["regime_vol_expansion"].fillna(False).astype(bool).sum())
        nh_r = new_high_count / total
        ve_r = vol_exp_count / total
        if median_ret >= cfg.regime_hot_24h_return_pct and nh_r >= cfg.regime_hot_new_high_ratio and ve_r >= 0.05:
            return "HOT"
        if median_ret >= cfg.regime_warm_24h_return_pct and nh_r >= cfg.regime_warm_new_high_ratio:
            return "WARM"
        return "COLD"

    def _pump_candidates(
        self,
        candles_1h: dict[str, pd.DataFrame],
        portfolio: Portfolio,
        equity: float,
        now: datetime,
    ) -> list[PumpCandidate]:
        cfg = self.config.pump_mode
        if not cfg.enabled or equity <= 0:
            return []
        if portfolio.daily_realized_loss < 0 and abs(portfolio.daily_realized_loss) > equity * cfg.max_daily_loss_pct:
            return []
        regime = self._pump_regime
        if regime == "COLD":
            return []
        consecutive_losses = 0
        for won in reversed(portfolio.pump_trade_results):
            if won: break
            consecutive_losses += 1
        self._pump_consecutive_losses = consecutive_losses

        candidates: list[PumpCandidate] = []
        for symbol, frame in candles_1h.items():
            if symbol in portfolio.positions or len(frame) < 73: continue
            if 'ret_24h' not in frame.columns: continue
            price = float(frame['close'].iloc[-1])
            if price <= 0: continue
            ret_24h = float(frame['ret_24h'].iloc[-1])
            if ret_24h < cfg.min_24h_return: continue
            ret_72h = float(frame['ret_72h'].iloc[-1])
            ret_6h = float(frame['ret_6h'].iloc[-1])
            above_ma = bool(frame['above_ma20'].iloc[-1]) if 'above_ma20' in frame.columns else (price > float(frame['ma20'].iloc[-1]))
            if not above_ma: continue
            # Volume ratio — use precomputed if available
            if 'qv_6h_sum' in frame.columns and 'qv_30_avg' in frame.columns:
                q6 = float(frame['qv_6h_sum'].iloc[-1]); q30 = float(frame['qv_30_avg'].iloc[-1])
                quote_volume_24h = float(frame['qv_24h_sum'].iloc[-1]) if 'qv_24h_sum' in frame.columns else 0
                quote_volume_6h = q6
                volume_ratio = q6 / q30 if q30 > 0 else 0
            else:
                qv = frame['quote_volume'].astype(float) if 'quote_volume' in frame.columns else frame['close'].astype(float)*frame['volume'].astype(float)
                quote_volume_24h = float(qv.iloc[-24:].sum()) if len(qv) >= 24 else 0
                quote_volume_6h = float(qv.iloc[-6:].sum()) if len(qv) >= 6 else 0
                avg_v = float(qv.iloc[-30:-6].mean())*6 if len(qv) >= 30 else 0
                volume_ratio = quote_volume_6h / avg_v if avg_v > 0 else 0
            if volume_ratio <= 0: continue
            if quote_volume_24h < cfg.min_quote_volume_24h and quote_volume_6h < cfg.min_quote_volume_6h: continue
            # Wick check — use precomputed if available
            if 'wick_ratio' in frame.columns and 'new_12h_high' in frame.columns:
                if float(frame['wick_ratio'].iloc[-1]) >= self.config.risk.blowoff_wick_ratio and bool(frame['new_12h_high'].iloc[-1]): continue
            # min_6h
            min_6h = getattr(cfg, 'min_6h_return', 0.0)
            if ret_6h < min_6h: continue
            # Signal classification
            early = ret_24h >= cfg.min_24h_return and ret_6h >= cfg.early_6h_return and volume_ratio >= cfg.early_volume_ratio_min
            confirmed_sig = ret_72h >= cfg.min_72h_return and ret_24h >= cfg.min_24h_return and volume_ratio >= cfg.volume_ratio_min
            warm_early_ok = False
            if regime == 'WARM':
                wr6 = getattr(cfg, 'warm_early_6h_return', 0.12); wvr = getattr(cfg, 'warm_early_volume_ratio_min', 2.0)
                warm_early_ok = ret_24h >= cfg.min_24h_return and ret_6h >= wr6 and volume_ratio >= wvr
            signal_ok = ((confirmed_sig or early) and regime == 'HOT') or (warm_early_ok and regime == 'WARM')
            if not signal_ok: continue
            if ret_72h > cfg.max_72h_return_chase: continue
            if ret_72h > cfg.max_72h_return_entry: continue
            if ret_72h > 1.20 and ret_6h / max(ret_24h, 0.001) < 0.30: continue
            risk_multiplier = 1.0
            if ret_72h > cfg.max_72h_return_reduced_risk: risk_multiplier = cfg.late_chase_risk_multiplier
            elif ret_72h > cfg.max_72h_return_full_risk: risk_multiplier = cfg.reduced_risk_multiplier
            if early and confirmed_sig and ret_72h <= cfg.max_72h_return_full_risk: risk_multiplier *= 1.25
            atr = self._latest_atr(frame)
            if atr is None or atr <= 0: continue
            score = ret_24h * 0.45 + ret_72h * 0.35 + ret_6h * 0.10 + min(volume_ratio / 5.0, 1.0) * 0.10
            tier = 'A' if (regime=='WARM' and (early or warm_early_ok) and 0.45<=ret_72h<=0.86 and volume_ratio<=15) else 'B'
            sig_type = "early_confirmed" if (early and confirmed_sig) else ("early" if early else "confirmed")
            reason = f"pump_{regime}_{tier}_{sig_type}"
            candidates.append(PumpCandidate(symbol=symbol, score=score, price=price, atr=atr, risk_multiplier=risk_multiplier,
                reason=reason, ret_6h=ret_6h, ret_24h=ret_24h, ret_72h=ret_72h, volume_ratio=volume_ratio,
                quote_volume_24h=quote_volume_24h, tier=tier))
        return sorted(candidates, key=lambda item: item.score, reverse=True)

    def _pump_candidates_from_snapshot(
        self,
        snapshot: pd.DataFrame,
        portfolio: Portfolio,
        equity: float,
        now: datetime,
    ) -> list[PumpCandidate]:
        cfg = self.config.pump_mode
        if not cfg.enabled or equity <= 0 or snapshot.empty:
            return []
        if portfolio.daily_realized_loss < 0 and abs(portfolio.daily_realized_loss) > equity * cfg.max_daily_loss_pct:
            return []
        regime = self._pump_regime
        if regime == "COLD":
            return []
        consecutive_losses = 0
        for won in reversed(portfolio.pump_trade_results):
            if won:
                break
            consecutive_losses += 1
        self._pump_consecutive_losses = consecutive_losses

        candidates: list[PumpCandidate] = []
        for row in snapshot.itertuples(index=False):
            symbol = str(row.symbol)
            if symbol in portfolio.positions or int(row.history) < 73:
                continue
            price = float(row.price)
            if price <= 0:
                continue
            ret_24h = float(row.ret_24h)
            if pd.isna(ret_24h) or ret_24h < cfg.min_24h_return:
                continue
            ret_72h = float(row.ret_72h)
            ret_6h = float(row.ret_6h)
            if pd.isna(ret_72h) or pd.isna(ret_6h):
                continue
            if not bool(row.above_ma20):
                continue
            q6 = float(row.qv_6h)
            q30 = float(row.qv_30_avg)
            quote_volume_24h = float(row.qv_24h)
            quote_volume_6h = q6
            volume_ratio = q6 / q30 if q30 > 0 else 0
            if volume_ratio <= 0:
                continue
            if quote_volume_24h < cfg.min_quote_volume_24h and quote_volume_6h < cfg.min_quote_volume_6h:
                continue
            if float(row.wick_ratio) >= self.config.risk.blowoff_wick_ratio and bool(row.new_12h_high):
                continue
            min_6h = getattr(cfg, "min_6h_return", 0.0)
            if ret_6h < min_6h:
                continue
            early = ret_24h >= cfg.min_24h_return and ret_6h >= cfg.early_6h_return and volume_ratio >= cfg.early_volume_ratio_min
            confirmed_sig = ret_72h >= cfg.min_72h_return and ret_24h >= cfg.min_24h_return and volume_ratio >= cfg.volume_ratio_min
            warm_early_ok = False
            if regime == "WARM":
                wr6 = getattr(cfg, "warm_early_6h_return", 0.12)
                wvr = getattr(cfg, "warm_early_volume_ratio_min", 2.0)
                warm_early_ok = ret_24h >= cfg.min_24h_return and ret_6h >= wr6 and volume_ratio >= wvr
            signal_ok = ((confirmed_sig or early) and regime == "HOT") or (warm_early_ok and regime == "WARM")
            if not signal_ok:
                continue
            if ret_72h > cfg.max_72h_return_chase:
                continue
            if ret_72h > cfg.max_72h_return_entry:
                continue
            if ret_72h > 1.20 and ret_6h / max(ret_24h, 0.001) < 0.30:
                continue
            risk_multiplier = 1.0
            if ret_72h > cfg.max_72h_return_reduced_risk:
                risk_multiplier = cfg.late_chase_risk_multiplier
            elif ret_72h > cfg.max_72h_return_full_risk:
                risk_multiplier = cfg.reduced_risk_multiplier
            if early and confirmed_sig and ret_72h <= cfg.max_72h_return_full_risk:
                risk_multiplier *= 1.25
            atr = float(row.atr)
            if pd.isna(atr) or atr <= 0:
                continue
            score = ret_24h * 0.45 + ret_72h * 0.35 + ret_6h * 0.10 + min(volume_ratio / 5.0, 1.0) * 0.10
            tier = "A" if (regime == "WARM" and (early or warm_early_ok) and 0.45 <= ret_72h <= 0.86 and volume_ratio <= 15) else "B"
            sig_type = "early_confirmed" if (early and confirmed_sig) else ("early" if early else "confirmed")
            reason = f"pump_{regime}_{tier}_{sig_type}"
            candidates.append(PumpCandidate(symbol=symbol, score=score, price=price, atr=atr, risk_multiplier=risk_multiplier,
                reason=reason, ret_6h=ret_6h, ret_24h=ret_24h, ret_72h=ret_72h, volume_ratio=volume_ratio,
                quote_volume_24h=quote_volume_24h, tier=tier))
        return sorted(candidates, key=lambda item: item.score, reverse=True)

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
    ) -> None:
        cfg = self.config.pump_mode
        if not candidates:
            return
        equity = portfolio.equity(prices)
        if equity <= 0:
            return
        pump_positions = [p for p in portfolio.positions.values() if p.position_type == "pump"]
        slots = max(cfg.max_positions - len(pump_positions), 0)
        if slots <= 0:
            return
        pump_exposure = sum(p.quantity * prices.get(p.symbol, p.entry_price) for p in pump_positions) / equity

        for candidate in candidates:
            if slots <= 0:
                break
            if candidate.symbol in portfolio.positions:
                continue
            next_open = self._next_open(candles_1h, candidate.symbol, next_time)
            if next_open is None or next_open <= 0:
                self._reject(session, run_id, now, candidate.symbol, "pump_missing_next_open")
                continue

            stop_distance = max(candidate.atr * cfg.initial_stop_atr_multiple, next_open * cfg.initial_stop_pct)
            risk_budget = equity * cfg.trade_risk_pct * candidate.risk_multiplier
            full_quantity = min(
                risk_budget / stop_distance,
                equity * cfg.max_symbol_position_pct / next_open,
                equity * max(cfg.max_total_exposure_pct - pump_exposure, 0) / next_open,
            )
            if full_quantity <= 0:
                self._reject(session, run_id, now, candidate.symbol, "pump_exposure_limit")
                continue

            # v2.5B: keep B-tier unconfirmed probes smaller than A-tier probes.
            probe_pct = cfg.probe_pct_a if candidate.tier == "A" else cfg.probe_pct_b
            quantity = full_quantity * probe_pct
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
                entry_atr=candidate.atr,
                entry_vr=candidate.volume_ratio,
                position_type="pump",
                stop_mechanism="pump_initial_stop",
                stop_trigger="low_below_pump_initial_stop",
                is_probe=True,
                probe_full_qty=full_quantity,
                probe_tier=candidate.tier,
                probe_entry_price=order.filled_price,
                avg_entry_price=order.filled_price,
                entry_notional=order.quantity * order.filled_price,
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
                            "stop_anchor_price": order.filled_price,
                            "stop_price": stop_price,
                            "atr": candidate.atr,
                            "risk_multiplier": candidate.risk_multiplier,
                            "probe_pct": probe_pct,
                            "score": candidate.score,
                            "ret_6h": candidate.ret_6h,
                            "ret_24h": candidate.ret_24h,
                            "ret_72h": candidate.ret_72h,
                            "volume_ratio": candidate.volume_ratio,
                            "quote_volume_24h": candidate.quote_volume_24h,
                        },
                    )
                ],
            )
            self._write_position(session, run_id, next_time, position, "pump_open", order.filled_price)

    def _find_swaps(
        self,
        portfolio: Portfolio,
        targets: list[TargetPosition],
        factors: pd.DataFrame,
    ) -> list[SwapRecommendation]:
        """Find swap opportunities: sell weakening position for significantly stronger candidate."""
        rc = self.config.risk
        if not portfolio.positions or not targets:
            return []

        # Build score lookup
        score_map: dict[str, float] = {}
        if not factors.empty:
            for row in factors.itertuples(index=False):
                score_map[str(row.symbol)] = float(getattr(row, "final_score", getattr(row, "momentum_score", 0)))

        swaps: list[SwapRecommendation] = []

        for target in targets:
            if target.symbol in portfolio.positions:
                continue
            buy_score = score_map.get(target.symbol, 0)
            for sym, pos in portfolio.positions.items():
                sell_score = score_map.get(sym, 0)
                if sell_score <= 0 or buy_score <= 0:
                    continue
                advantage = buy_score / sell_score if sell_score > 0 else float("inf")

                # Strong swap: new coin is Top 3 and significantly stronger
                if advantage >= rc.swap_score_advantage:
                    # Check position is weak enough to swap
                    if pos.state in ("weakening", "failed"):
                        swaps.append(SwapRecommendation(sym, target, 999, 1, sell_score, buy_score, "swap_strong"))
                elif advantage >= rc.swap_strong_score_advantage:
                    # Very strong signal — can swap even healthy positions
                    swaps.append(SwapRecommendation(sym, target, 999, 1, sell_score, buy_score, "swap_very_strong"))

                if len(swaps) >= rc.swap_max_per_day:
                    break
            if len(swaps) >= rc.swap_max_per_day:
                break

        return swaps

    def _next_open(self, candles: dict[str, pd.DataFrame], symbol: str, next_time: datetime) -> float | None:
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

    def _write_market_state(self, session: Session, run_id: int, now: datetime, state: MarketState) -> None:
        session.add(
            MarketStateRecord(
                strategy_run_id=run_id,
                time=now,
                btc_close=state.btc_close,
                btc_ma50=state.btc_ma50,
                ma50_slope_4=state.ma50_slope_4,
                breadth=state.breadth,
                state=state.state,
                fast_risk_valve=state.fast_risk_valve,
                reasons=state.reasons,
            )
        )

    def _write_factor_scores(self, session: Session, run_id: int, now: datetime, factors: pd.DataFrame) -> None:
        for row in factors.head(10).itertuples(index=False):
            raw = {key: float(getattr(row, key)) for key in factors.columns if key.startswith("return_") or key.startswith("weight_")}
            volume_score = float(getattr(row, "volume_score", 0)) if hasattr(row, "volume_score") else None
            trend_score = float(getattr(row, "trend_score", 0)) if hasattr(row, "trend_score") else None
            session.add(
                FactorScoreRecord(
                    strategy_run_id=run_id,
                    time=now,
                    symbol=str(row.symbol),
                    momentum_score=float(row.momentum_score),
                    volume_score=volume_score,
                    trend_score=trend_score,
                    final_score=float(row.final_score),
                    raw_factors=raw,
                )
            )

    def _write_orders(
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

    def _reject(self, session: Session, run_id: int, now: datetime, symbol: str, reason: str) -> None:
        session.add(RejectedSignalRecord(strategy_run_id=run_id, time=now, symbol=symbol, reason=reason, details=None))

    def _synthetic_candles(self) -> dict[str, pd.DataFrame]:
        index = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        data: dict[str, pd.DataFrame] = {}
        for offset, symbol in enumerate(["AAA/USDT", "BBB/USDT", "CCC/USDT", "DDD/USDT"], start=1):
            close = pd.Series(range(100 + offset, 220 + offset), dtype=float) * (1 + offset * 0.001)
            frame = pd.DataFrame(
                {
                    "open_time": index,
                    "open": close.shift(1).fillna(close.iloc[0]),
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": 1000,
                    "quote_volume": close * 1000,
                }
            )
            data[symbol] = frame
        return data
