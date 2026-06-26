from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.execution.broker import BacktestBroker, Order
from crypto_quant.risk.market_state import MarketState, fast_risk_valve_triggered
from crypto_quant.storage.candles import distinct_candle_symbols, load_candles
from crypto_quant.storage.models import (
    EquityCurveRecord,
    OrderRecord,
    PositionRecord,
    RejectedSignalRecord,
    StrategyRun,
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
    tier: str = "B"
    ema20_dev_rank_2160h: float = 0.0
    # Signal-bar metadata (for post-trade analysis without DB joins)
    ema20_dev_pct: float = 0.0
    wick_ratio: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    pos24h: float = 0.0
    vol_trend6: float = 0.0


@dataclass(frozen=True)
class OrderMetadata:
    mechanism: str | None = None
    trigger: str | None = None
    details: dict[str, object] | None = None


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
    _market_feature_history: list[dict[str, float]] = field(default_factory=list)
    _market_context: MarketState = field(default_factory=lambda: MarketState("risk_on"))
    _previous_market_phase: str = "normal"

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
                    "ema20_dev_rank_2160h": float(values["ema20_dev_rank_2160h"][idx]),
                    "ema20_dev": float(values["ema20_dev"][idx]),
                    "r1": float(values["r1"][idx]),
                    "r2": float(values["r2"][idx]),
                    "r3": float(values["r3"][idx]),
                    "pos24h": float(values["pos24h"][idx]),
                    "vol_trend6": float(values["vol_trend6"][idx]),
                }
            )
        return pd.DataFrame(rows)

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
        """Exit only the high-cost add tranche and keep the core position open."""
        add_qty = min(position.add_qty, position.quantity)
        if add_qty <= 0:
            return
        frame = candles_1h.get(symbol, pd.DataFrame())
        current = self._slice_frame(frame, trigger_time) if not frame.empty else pd.DataFrame()
        exit_price = self._next_open(candles_1h, symbol, exit_time)
        if exit_price is None:
            exit_price = float(current["close"].iloc[-1]) if not current.empty else position.add_entry_price
        if exit_price <= 0:
            return

        order = broker.execute_market(symbol, "sell", add_qty, exit_price, "pump_add_tranche_exit")
        if order.filled_price <= 0:
            return
        orders.append(order)
        portfolio.cash += order.quantity * order.filled_price - order.fee

        add_entry = position.add_entry_price or position.confirm_entry_price or self._trade_entry_price(position)
        add_pnl = (order.filled_price - add_entry) * order.quantity - order.fee
        position.quantity = max(position.quantity - order.quantity, 0.0)
        position.add_qty = max(position.add_qty - order.quantity, 0.0)
        if position.add_qty <= 1e-12:
            position.add_qty = 0.0
            position.add_opened_at = None
            position.add_highest_price = 0.0
            position.add_entry_price = 0.0

        remaining_notional = max(position.entry_notional - order.quantity * add_entry, 0.0)
        position.entry_notional = remaining_notional
        position.avg_entry_price = remaining_notional / position.quantity if position.quantity > 0 else position.entry_price
        position.stop_mechanism = "pump_add_tranche_exit"
        position.stop_trigger = "pump_add_failed_follow_through"

        details = self._exit_details(position, current, market_state, stop_before, order.filled_price, trigger_time)
        details.update(
            {
                "partial_exit": True,
                "add_exit_qty": order.quantity,
                "add_entry_price": add_entry,
                "add_exit_pnl": add_pnl,
                "remaining_qty": position.quantity,
                "remaining_avg_entry_price": position.avg_entry_price,
            }
        )
        partial_mechanism = position.stop_mechanism
        partial_trigger = position.stop_trigger
        self._write_orders(
            session,
            run_id,
            exit_time,
            [order],
            [OrderMetadata(mechanism=partial_mechanism, trigger=partial_trigger, details=details)],
        )
        if position.quantity <= 1e-12:
            del portfolio.positions[symbol]
        else:
            position.stop_mechanism = stop_mechanism_before
            position.stop_trigger = stop_trigger_before

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
        portfolio.record_trade(pnl, won)
        cfg_ar = self.config.pump_mode
        self._pump_recent_exits.append(reason or "?")
        self._pump_symbol_last_exit[symbol] = (exit_time, reason or "?")
        if (
            cfg_ar.failed_reentry_cooldown_enabled
            and reason in set(cfg_ar.failed_reentry_exit_reasons)
        ):
            self._pump_symbol_cooldowns[symbol] = exit_time + timedelta(hours=cfg_ar.failed_reentry_cooldown_hours)
        lookback = cfg_ar.adaptive_risk_lookback
        if len(self._pump_recent_exits) > lookback * 2:
            self._pump_recent_exits = self._pump_recent_exits[-lookback * 2:]
        post = self._compute_post_entry_path(candles_1h, symbol, position.opened_at, entry_anchor, trigger_time)
        if post:
            post["symbol"] = symbol
            post["exit_reason"] = reason
            post["pnl"] = pnl
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

    def _detect_market_context(
        self,
        snapshot: pd.DataFrame,
        btc_1h: pd.DataFrame,
        fast_valve: bool,
        fast_reasons: list[str],
    ) -> MarketState:
        cfg = self.config.pump_mode
        if fast_valve:
            return MarketState(
                "risk_off",
                fast_risk_valve=True,
                reasons=fast_reasons or ["btc_fast_valve"],
                phase="risk_off",
                transition="deteriorating",
                risk_multiplier=0.0,
                entry_mode="none",
                exit_profile="aggressive_tighten" if cfg.market_context_exit_tightening_enabled else "normal",
            )
        if not cfg.market_context_enabled:
            return MarketState("risk_on", reasons=[f"legacy_{self._pump_regime.lower()}"], phase=self._pump_regime.lower())

        metrics = self._market_context_metrics(snapshot, btc_1h)
        history = self._market_feature_history
        if len(history) < cfg.market_context_min_history:
            context = self._legacy_market_context(metrics)
        else:
            context = self._phase_market_context(metrics, history)
        self._market_feature_history.append(metrics)
        if len(self._market_feature_history) > 720:
            self._market_feature_history = self._market_feature_history[-720:]
        self._previous_market_phase = context.phase
        return context

    def _market_context_metrics(self, snapshot: pd.DataFrame, btc_1h: pd.DataFrame) -> dict[str, float]:
        metrics = {
            "eligible_count": 0.0,
            "median_ret_24h": 0.0,
            "new12h_high_ratio": 0.0,
            "ret24_gt10_ratio": 0.0,
            "ret24_gt30_ratio": 0.0,
            "ret72_gt80_ratio": 0.0,
            "new_high_high_wick_ratio": 0.0,
            "sync_down_24h_ratio": 0.0,
            "vol_expansion_ratio": 0.0,
            "candidate_count": 0.0,
            "btc_ret_24h": 0.0,
            "btc_vol_24h": 0.0,
            "heat_score": 0.0,
            "heat_delta_24h": 0.0,
        }
        if snapshot.empty:
            return metrics
        eligible = snapshot[snapshot["history"].astype(int) >= 73].copy()
        if eligible.empty:
            return metrics
        total = len(eligible)
        ret24 = eligible["ret_24h"].astype(float)
        ret72 = eligible["ret_72h"].astype(float)
        new_high = eligible["new_12h_high"].fillna(False).astype(bool)
        wick = eligible["wick_ratio"].astype(float)
        vol_exp = eligible["regime_vol_expansion"].fillna(False).astype(bool)
        q30 = eligible["qv_30_avg"].astype(float).replace(0, np.nan)
        volume_ratio = eligible["qv_6h"].astype(float) / q30

        metrics.update(
            {
                "eligible_count": float(total),
                "median_ret_24h": float(ret24.median()),
                "new12h_high_ratio": float(new_high.mean()),
                "ret24_gt10_ratio": float((ret24 > 0.10).mean()),
                "ret24_gt30_ratio": float((ret24 > 0.30).mean()),
                "ret72_gt80_ratio": float((ret72 > 0.80).mean()),
                "new_high_high_wick_ratio": float((new_high & (wick > 0.60)).mean()),
                "sync_down_24h_ratio": float((ret24 < -0.05).mean()),
                "vol_expansion_ratio": float(vol_exp.mean()),
                "candidate_count": float(self._market_context_candidate_count(eligible, volume_ratio)),
            }
        )
        if btc_1h is not None and len(btc_1h) >= 25:
            close = btc_1h["close"].astype(float)
            metrics["btc_ret_24h"] = float(close.iloc[-1] / close.iloc[-25] - 1) if close.iloc[-25] > 0 else 0.0
            metrics["btc_vol_24h"] = float(close.pct_change().tail(24).std())
        metrics["heat_score"] = (
            metrics["median_ret_24h"]
            + 0.50 * metrics["new12h_high_ratio"]
            + 0.30 * metrics["ret24_gt10_ratio"]
            + 0.20 * metrics["vol_expansion_ratio"]
            - 0.30 * metrics["ret72_gt80_ratio"]
            - 0.20 * metrics["new_high_high_wick_ratio"]
        )
        if len(self._market_feature_history) >= 6:
            metrics["heat_delta_24h"] = metrics["heat_score"] - self._market_feature_history[-6]["heat_score"]
        return metrics

    def _market_context_candidate_count(self, eligible: pd.DataFrame, volume_ratio: pd.Series) -> int:
        cfg = self.config.pump_mode
        rough = (
            (eligible["ret_24h"].astype(float) >= cfg.min_24h_return)
            & (eligible["ret_6h"].astype(float) >= cfg.min_6h_return)
            & (eligible["above_ma20"].fillna(False).astype(bool))
            & (volume_ratio >= cfg.early_volume_ratio_min)
            & (eligible["ret_72h"].astype(float) <= cfg.max_72h_return_full_risk)
        )
        return int(rough.fillna(False).sum())

    def _legacy_market_context(self, metrics: dict[str, float]) -> MarketState:
        if self._pump_regime == "COLD":
            phase, entry_mode, risk_multiplier, exit_profile = "cold", "none", 0.0, "normal"
        elif self._pump_regime == "HOT":
            phase, entry_mode, risk_multiplier, exit_profile = "expanding", "normal", 1.0, "normal"
        else:
            phase, entry_mode, risk_multiplier, exit_profile = (
                "normal",
                "patient",
                self.config.pump_mode.market_context_normal_risk_multiplier,
                "normal",
            )
        return MarketState(
            phase,
            reasons=[f"legacy_{self._pump_regime.lower()}_history_warmup"],
            phase=phase,
            transition=self._market_transition(phase),
            risk_multiplier=risk_multiplier,
            entry_mode=entry_mode,
            exit_profile=exit_profile,
            metrics=metrics,
        )

    def _phase_market_context(self, metrics: dict[str, float], history: list[dict[str, float]]) -> MarketState:
        cfg = self.config.pump_mode
        q = self._market_quantiles(history)
        risk_off = (
            metrics["heat_score"] <= q["heat_score"][0.2]
            and metrics["btc_ret_24h"] <= q["btc_ret_24h"][0.2]
            and metrics["sync_down_24h_ratio"] >= q["sync_down_24h_ratio"][0.8]
        ) or (
            metrics["heat_delta_24h"] <= q["heat_delta_24h"][0.2]
            and metrics["sync_down_24h_ratio"] >= q["sync_down_24h_ratio"][0.8]
        )
        crowded_fading = (
            metrics["ret24_gt10_ratio"] >= q["ret24_gt10_ratio"][0.8]
            and metrics["ret24_gt30_ratio"] >= q["ret24_gt30_ratio"][0.8]
            and metrics["heat_delta_24h"] < 0
        ) or (
            metrics["new_high_high_wick_ratio"] >= q["new_high_high_wick_ratio"][0.8]
            and metrics["heat_delta_24h"] < 0
        )
        expanding = (
            metrics["ret24_gt10_ratio"] >= q["ret24_gt10_ratio"][0.8]
            and metrics["heat_delta_24h"] >= q["heat_delta_24h"][0.8]
            and metrics["btc_ret_24h"] >= 0
        )
        crowded_hot = (
            metrics["ret24_gt10_ratio"] >= q["ret24_gt10_ratio"][0.8]
            and metrics["ret24_gt30_ratio"] >= q["ret24_gt30_ratio"][0.8]
            and metrics["heat_delta_24h"] >= 0
        )
        cold = (
            metrics["ret24_gt10_ratio"] <= q["ret24_gt10_ratio"][0.2]
            and metrics["new12h_high_ratio"] <= q["new12h_high_ratio"][0.2]
            and metrics["candidate_count"] <= q["candidate_count"][0.4]
        )

        if risk_off:
            phase = "risk_off"
            entry_mode = "none"
            risk_multiplier = 0.0
            exit_profile = "aggressive_tighten" if cfg.market_context_exit_tightening_enabled else "normal"
        elif expanding:
            phase, entry_mode, risk_multiplier, exit_profile = "expanding", "normal", 1.0, "normal"
        elif crowded_hot:
            phase = "crowded_hot"
            entry_mode = "normal"
            risk_multiplier = cfg.market_context_crowded_hot_risk_multiplier
            exit_profile = "light_tighten" if cfg.market_context_exit_tightening_enabled else "normal"
        elif crowded_fading:
            phase = "crowded_fading"
            entry_mode = "patient"
            risk_multiplier = cfg.market_context_crowded_fading_risk_multiplier
            exit_profile = "aggressive_tighten" if cfg.market_context_exit_tightening_enabled else "normal"
        elif cold:
            phase, entry_mode, risk_multiplier, exit_profile = "cold", "none", 0.0, "normal"
        else:
            phase = "normal"
            entry_mode = "patient"
            risk_multiplier = cfg.market_context_normal_risk_multiplier
            exit_profile = "tighten" if cfg.market_context_exit_tightening_enabled else "normal"

        transition = self._market_transition(phase)
        reasons = [
            f"phase={phase}",
            f"transition={transition}",
            f"old_regime={self._pump_regime}",
        ]
        return MarketState(
            phase,
            reasons=reasons,
            phase=phase,
            transition=transition,
            risk_multiplier=risk_multiplier,
            entry_mode=entry_mode,
            exit_profile=exit_profile,
            metrics=metrics,
        )

    def _market_quantiles(self, history: list[dict[str, float]]) -> dict[str, dict[float, float]]:
        fields = [
            "heat_score",
            "heat_delta_24h",
            "ret24_gt10_ratio",
            "ret24_gt30_ratio",
            "new12h_high_ratio",
            "new_high_high_wick_ratio",
            "btc_ret_24h",
            "sync_down_24h_ratio",
            "candidate_count",
        ]
        frame = pd.DataFrame(history)
        return {
            field: {q: float(frame[field].quantile(q)) for q in (0.2, 0.4, 0.8)}
            for field in fields
        }

    def _market_transition(self, phase: str) -> str:
        previous = self._previous_market_phase
        if phase == previous:
            return "stable"
        if phase == "risk_off" or (previous == "crowded_hot" and phase == "crowded_fading"):
            return "deteriorating"
        if phase == "expanding" and previous in {"normal", "cold", "crowded_fading", "crowded_hot"}:
            return "improving"
        if previous == "risk_off" and phase != "risk_off":
            return "recovery"
        return "stable"

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
        if regime == "COLD" and not cfg.cold_squeeze_enabled:
            return []
        market = self._market_context
        if market.entry_mode == "none" or market.risk_multiplier <= 0:
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
            if getattr(cfg, 'reject_long_wick_enabled', False):
                if float(row.wick_ratio) > 0.80 and float(row.r2) < 0:
                    continue
            min_6h = getattr(cfg, "min_6h_return", 0.0)
            if ret_6h < min_6h:
                continue
            ema20_dev_pct = float(getattr(row, 'ema20_dev', 0)) * 100
            cold_squeeze = (
                cfg.cold_squeeze_enabled
                and regime == "COLD"
                and market.entry_mode == "patient"
                and market.phase in {"normal", "crowded_fading"}
                and ret_24h >= cfg.cold_squeeze_min_24h_return
                and ret_24h <= cfg.cold_squeeze_max_24h_return
                and cfg.cold_squeeze_min_72h_return <= ret_72h <= cfg.cold_squeeze_max_72h_return
                and quote_volume_24h >= cfg.cold_squeeze_min_quote_volume_24h
                and cfg.cold_squeeze_min_volume_ratio <= volume_ratio <= cfg.cold_squeeze_max_volume_ratio
                and cfg.cold_squeeze_min_ema20_dev_pct <= ema20_dev_pct <= cfg.cold_squeeze_max_ema20_dev_pct
                and float(row.wick_ratio) < cfg.cold_squeeze_max_wick_ratio
                and ret_6h / max(ret_24h, 0.001) >= cfg.cold_squeeze_min_ret6_to_ret24
            )
            if regime == "COLD" and not cold_squeeze:
                continue
            early = ret_24h >= cfg.min_24h_return and ret_6h >= cfg.early_6h_return and volume_ratio >= cfg.early_volume_ratio_min
            confirmed_sig = ret_72h >= cfg.min_72h_return and ret_24h >= cfg.min_24h_return and volume_ratio >= cfg.volume_ratio_min
            warm_early_ok = False
            if regime == "WARM":
                wr6 = getattr(cfg, "warm_early_6h_return", 0.12)
                wvr = getattr(cfg, "warm_early_volume_ratio_min", 2.0)
                warm_early_ok = ret_24h >= cfg.min_24h_return and ret_6h >= wr6 and volume_ratio >= wvr
            signal_ok = cold_squeeze or ((confirmed_sig or early) and regime == "HOT") or (warm_early_ok and regime == "WARM")
            if not signal_ok:
                continue
            if (
                cfg.reject_hot_confirmed_high_volume_enabled
                and regime == "HOT"
                and confirmed_sig
                and volume_ratio > cfg.reject_hot_confirmed_high_volume_ratio
            ):
                continue
            # r72 overheating: hard reject > 80% for normal pump signals.
            if not cold_squeeze and ret_72h > cfg.max_72h_return_full_risk:
                continue
            if ret_72h > cfg.max_72h_return_chase:
                continue
            if ret_72h > cfg.max_72h_return_entry:
                continue
            if ret_72h > 1.20 and ret_6h / max(ret_24h, 0.001) < 0.30:
                continue
            risk_multiplier = cfg.cold_squeeze_risk_multiplier if cold_squeeze else 1.0
            if early and confirmed_sig:
                risk_multiplier *= 1.25
            risk_multiplier *= market.risk_multiplier
            atr = float(row.atr)
            if pd.isna(atr) or atr <= 0:
                continue
            score = ret_24h * 0.45 + ret_72h * 0.35 + ret_6h * 0.10 + min(volume_ratio / 5.0, 1.0) * 0.10
            if cfg.reject_high_score_enabled and score > cfg.reject_high_score_threshold:
                continue
            tier = "A" if (regime == "WARM" and (early or warm_early_ok) and 0.45 <= ret_72h <= 0.86 and volume_ratio <= 15) else "B"
            sig_type = "early_confirmed" if (early and confirmed_sig) else ("early" if early else "confirmed")
            ema20_dev_rank_2160h = float(row.ema20_dev_rank_2160h)
            r1 = float(getattr(row, 'r1', 0))
            r2 = float(getattr(row, 'r2', 0))
            r3 = float(getattr(row, 'r3', 0))
            late_accel = (
                cfg.late_accel_control_enabled
                and (
                    max(r1, r2, r3) > cfg.late_accel_max_r123
                    or (r1 + r2 + r3) > cfg.late_accel_last3_sum
                )
            )
            if late_accel and tier == "A":
                tier = "B"
            if late_accel and confirmed_sig:
                risk_multiplier *= cfg.late_accel_confirmed_risk_multiplier
            if (
                cfg.warm_a_high_ema_downgrade_enabled
                and regime == "WARM"
                and tier == "A"
                and ema20_dev_pct > cfg.warm_a_high_ema_downgrade_pct
            ):
                tier = "B"
            if (
                cfg.warm_a_late_spike_downgrade_enabled
                and regime == "WARM"
                and tier == "A"
                and (
                    max(r1, r2, r3) > cfg.warm_a_late_spike_max_r123
                    or (r1 + r2 + r3) > cfg.warm_a_late_spike_last3_sum
                )
            ):
                tier = "B"
            if getattr(cfg, 'ema_abs_min_enabled', False):
                ema_abs = ema20_dev_pct
                if ema_abs < cfg.ema_abs_min_threshold:
                    continue
                if getattr(cfg, 'ema_abs_max_enabled', False) and ema_abs > cfg.ema_abs_max_threshold:
                    continue
            if getattr(cfg, 'reject_accel_decay_enabled', False):
                if ret_6h / max(ret_24h, 0.001) < 0.5 and r1 < 0 and r2 < 0 and r3 < 0:
                    continue
            if (
                cfg.reject_high_vol_trend_enabled
                and float(getattr(row, 'vol_trend6', 0)) > cfg.reject_high_vol_trend_threshold
            ):
                continue
            if (
                cfg.bad_b_ema_vr_risk_mid_enabled
                and tier == "B"
                and sig_type == "early"
                and ema20_dev_rank_2160h >= cfg.bad_b_ema_rank_min
                and cfg.bad_b_volume_ratio_mid_min < volume_ratio <= cfg.bad_b_volume_ratio_mid_max
            ):
                risk_multiplier *= cfg.bad_b_risk_multiplier_mid
            if (
                market.phase == "crowded_fading"
                and (
                    ema20_dev_pct >= cfg.market_context_fading_ema20_dev_pct
                    or volume_ratio >= cfg.market_context_fading_volume_ratio
                    or ret_6h >= cfg.market_context_fading_ret_6h
                )
            ):
                risk_multiplier *= cfg.market_context_fading_extreme_multiplier
            if cfg.bad_b_vr30_reject_enabled and volume_ratio > cfg.bad_b_volume_ratio_min:
                continue
            if cold_squeeze:
                tier = "B"
                sig_type = "cold_squeeze"
            reason = f"pump_{regime}_{tier}_{sig_type}"
            candidates.append(PumpCandidate(symbol=symbol, score=score, price=price, atr=atr, risk_multiplier=risk_multiplier,
                reason=reason, ret_6h=ret_6h, ret_24h=ret_24h, ret_72h=ret_72h, volume_ratio=volume_ratio,
                quote_volume_24h=quote_volume_24h, tier=tier, ema20_dev_rank_2160h=ema20_dev_rank_2160h,
                ema20_dev_pct=ema20_dev_pct,
                wick_ratio=float(row.wick_ratio), r1=r1,
                r2=r2, r3=r3,
                pos24h=float(getattr(row, 'pos24h', 0)), vol_trend6=float(getattr(row, 'vol_trend6', 0))))
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
