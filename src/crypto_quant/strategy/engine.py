"""v1.0 Strategy engine with daily loss circuit-breaker, volume-stall detection,
cooldown tracking, and composite scoring (momentum + volume + trend).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from crypto_quant.config.settings import AppConfig, RiskConfig
from crypto_quant.factors.trend import compute_trend_scores
from crypto_quant.factors.volume_stall import CooldownTracker, compute_24h_returns, detect_volume_stall
from crypto_quant.risk.false_breakout import FalseBreakoutDetector, FalseBreakoutState
from crypto_quant.risk.market_state import MarketState
from crypto_quant.strategy.types import Signal, TargetPosition


def _compute_volume_score(symbol: str, candles_1h: dict[str, pd.DataFrame], lookback: int = 6, avg_hours: int = 20) -> float:
    """Return a continuous volume score (0..1).

    High score: volume expansion + price rising.
    Medium score: normal volume + stable price.
    Low score: volume expansion + price stalling.
    """
    frame = candles_1h.get(symbol)
    if frame is None or len(frame) < lookback + avg_hours + 1:
        return 0.5  # neutral for insufficient data

    volume = frame["volume"].astype(float)
    close = frame["close"].astype(float)

    recent_volume = float(volume.iloc[-lookback:].sum())
    avg_vol_window = volume.iloc[-(lookback + avg_hours) : -lookback]
    avg_volume = float(avg_vol_window.mean()) * lookback if len(avg_vol_window) > 0 else 1.0

    if avg_volume <= 0:
        return 0.5

    vol_ratio = recent_volume / avg_volume
    price_change = float(close.iloc[-1] / close.iloc[-1 - lookback] - 1)

    # volume expansion + price rising → strong confirmation
    if vol_ratio >= 1.2 and price_change > 0.005:
        return 0.8 + min(0.2, (vol_ratio - 1.2) * 0.2)
    # volume expansion + price stalling → warning
    if vol_ratio >= 1.2 and price_change <= 0.005:
        return 0.1 + max(0, (1.0 - vol_ratio / 3.0)) * 0.2
    # normal volume + price rising → moderate
    if 0.7 <= vol_ratio < 1.2 and price_change > 0:
        return 0.5 + price_change * 5.0  # scale: 1% → 0.55, 5% → 0.75
    # low volume + stable → neutral
    if vol_ratio < 0.7 and price_change > -0.01:
        return 0.4
    # falling price → low score regardless of volume
    if price_change < -0.01:
        return 0.1

    return 0.4


def _composite_score(
    momentum_df: pd.DataFrame,
    candles_1h: dict[str, pd.DataFrame],
    config: AppConfig,
    candles_4h: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute composite final_score = momentum × w_m + volume × w_v + trend × w_t."""
    if momentum_df.empty:
        return momentum_df

    trend_cfg = config.trend
    vol_cfg = config.volume_score

    # Fast path: if pre-computed columns exist on candles, skip heavy recomputation
    # Check if first symbol's candle data has pre-computed columns
    first_sym = momentum_df["symbol"].iloc[0] if "symbol" in momentum_df.columns else None
    has_precomputed = first_sym and candles_1h and first_sym in candles_1h and "trend_score_col" in candles_1h[first_sym].columns

    if has_precomputed:
        # Read pre-computed scores directly from candle data
        momentum_df = momentum_df.drop(columns=["trend_score", "volume_score"], errors="ignore")
        for i, row in momentum_df.iterrows():
            sym = row["symbol"]
            frame = candles_1h.get(sym)
            if frame is not None and not frame.empty and "trend_score_col" in frame.columns:
                momentum_df.at[i, "trend_score"] = float(frame["trend_score_col"].iloc[-1])
                momentum_df.at[i, "volume_score"] = float(frame["volume_score_col"].iloc[-1])
            else:
                momentum_df.at[i, "trend_score"] = 0.5
                momentum_df.at[i, "volume_score"] = 0.5
    else:
        # trend scores — pass 4H candles for proper long-MA calculation
        momentum_df = momentum_df.drop(columns=["trend_score", "volume_score"], errors="ignore")
        trend_df = compute_trend_scores(candles_1h, trend_cfg, candles_4h=candles_4h)
        if not trend_df.empty:
            momentum_df = momentum_df.merge(trend_df, on="symbol", how="left")
            momentum_df["trend_score"] = momentum_df["trend_score"].fillna(0.5)
        else:
            momentum_df["trend_score"] = 0.5

        # volume scores
        if vol_cfg.enabled:
            momentum_df["volume_score"] = momentum_df["symbol"].apply(
                lambda s: _compute_volume_score(s, candles_1h, vol_cfg.lookback_hours, vol_cfg.average_hours)
            )
        else:
            momentum_df["volume_score"] = 0.5

    # composite weight — normalized so weights sum to 1
    m_weight = 1.0 - trend_cfg.score_weight - vol_cfg.score_weight
    v_weight = vol_cfg.score_weight
    t_weight = trend_cfg.score_weight

    momentum_df["final_score"] = (
        momentum_df["momentum_score"].fillna(0.5) * m_weight
        + momentum_df["volume_score"].fillna(0.5) * v_weight
        + momentum_df["trend_score"].fillna(0.5) * t_weight
    )

    return momentum_df.sort_values("final_score", ascending=False).reset_index(drop=True)


@dataclass
class StrategyEngine:
    config: AppConfig

    # mutable state
    _cooldowns: CooldownTracker = field(default_factory=lambda: CooldownTracker(4.0))
    _false_breakout_detector: FalseBreakoutDetector = field(default_factory=FalseBreakoutDetector)
    _prev_state: str = "unknown"  # track defensive→risk_on transitions

    @property
    def risk_config(self) -> RiskConfig:
        return self.config.risk

    def generate_targets(
        self,
        factor_scores: pd.DataFrame,
        market_state: MarketState,
        prices: dict[str, float],
        atrs: dict[str, float],
        equity: float,
        candles_1h: dict[str, pd.DataFrame] | None = None,
        candles_4h: dict[str, pd.DataFrame] | None = None,
        now: datetime | None = None,
        daily_realized_loss: float = 0.0,
        recent_trade_results: list[bool] | None = None,
    ) -> tuple[list[Signal], list[TargetPosition], list[tuple[str, str]]]:
        """Generate entry signals and target positions.

        Args:
            factor_scores: momentum factor scores (must have symbol and momentum_score cols)
            market_state: current market regime
            prices: current close prices keyed by symbol
            atrs: current 1H ATR values keyed by symbol
            equity: current strategy equity
            candles_1h: 1H OHLCV frames keyed by symbol (for volume/trend checks)
            candles_4h: 4H OHLCV frames keyed by symbol (for long-MA trend check)
            now: current timestamp (for cooldown expiry)
            daily_realized_loss: total realized PnL today (negative = loss)
            recent_trade_results: bool list of recent closed-trade outcomes (True = win)
        """
        candles = candles_1h or {}
        recent = recent_trade_results or []

        if factor_scores.empty:
            return [], [], []

        # ---- composite scoring ----
        scored = _composite_score(factor_scores, candles, self.config, candles_4h=candles_4h)

        # retain only top candidates for further filtering
        max_candidates = max(self.risk_config.max_positions * 2, 5)
        top = scored.head(max_candidates)

        signals: list[Signal] = []
        targets: list[TargetPosition] = []
        rejected: list[tuple[str, str]] = []

        # ---- market state: fast_risk_valve hard-blocks; defensive raises the bar ----
        self._prev_state = market_state.state
        if market_state.fast_risk_valve:
            return [], [], [(str(row.symbol), "fast_risk_valve") for row in top.itertuples()]

        # defensive mode: allow but require much stronger signals
        is_defensive = market_state.state == "defensive"


        # ---- extreme momentum (>50% wret) bypasses circuit breakers ----
        max_wret = max((float(getattr(r, "weighted_return", 0)) for r in top.itertuples()), default=0)
        has_extreme = max_wret >= 0.50

        # daily loss circuit breaker (extreme bypasses)
        if not has_extreme and daily_realized_loss < 0 and abs(daily_realized_loss) > equity * self.risk_config.daily_loss_limit_pct:
            return [], [], [(str(row.symbol), "daily_loss_limit") for row in top.itertuples()]

        # consecutive loss pause (extreme bypasses; defensive→risk_on gives fresh start)
        if not has_extreme:
            consecutive_losses = 0
            for won in reversed(recent):
                if not won:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses >= self.risk_config.consecutive_loss_pause_count:
                return [], [], [(str(row.symbol), "consecutive_loss_pause") for row in top.itertuples()]

        # recent-trades-loss check (extreme bypasses)
        if not has_extreme:
            lookback_m = self.risk_config.recent_trades_lookback
            threshold_n = self.risk_config.recent_trades_loss_threshold
            if len(recent) >= lookback_m:
                last_m = recent[-lookback_m:]
                if sum(1 for w in last_m if not w) >= threshold_n:
                    return [], [], [(str(row.symbol), "recent_loss_streak") for row in top.itertuples()]

        # ---- false breakout detection ----
        fb_state = self._false_breakout_detector.current_state()
        effective_max_positions = self.risk_config.max_positions
        effective_trade_risk_pct = self.risk_config.trade_risk_pct
        effective_top_n = None  # None = no extra limit
        effective_min_score = self.risk_config.min_composite_score
        effective_min_momentum = self.risk_config.min_momentum_return

        if fb_state == FalseBreakoutState.ACTIVE:
            effective_max_positions = min(effective_max_positions, self.config.false_breakout.max_positions)
            effective_trade_risk_pct *= self.config.false_breakout.single_risk_multiplier
            effective_top_n = self.config.false_breakout.top_n_allowed

        # defensive mode: stronger signals (30%+ momentum, 0.80+ score), full risk
        if is_defensive:
            effective_max_positions = min(effective_max_positions, 1)
            effective_min_momentum = max(effective_min_momentum, 0.30)
            effective_min_score = max(effective_min_score, 0.80)

        # ---- per-symbol filtering ----
        now_dt = now or datetime.now()
        # Pre-compute 24h returns once to avoid O(n²) in volume-stall detection
        returns_24h_cache = compute_24h_returns(candles) if self.risk_config.volume_stall_enabled else {}

        for rank_idx, row in enumerate(top.itertuples(), start=1):
            symbol = str(row.symbol)

            # false breakout top-N cap
            if effective_top_n is not None and len(signals) >= effective_top_n:
                rejected.append((symbol, "false_breakout_top_n_cap"))
                continue

            # volume confirmation (binary, optional)
            if self.risk_config.volume_confirmation_enabled and not self._volume_confirmed_binary(symbol, candles):
                rejected.append((symbol, "volume_confirmation"))
                continue

            # blow-off top detection (White Paper §5.2 冲高回落)
            if self.risk_config.blowoff_enabled:
                blowoff = self._detect_blowoff(symbol, candles)
                if blowoff:
                    rejected.append((symbol, f"blowoff_top:{blowoff}"))
                    continue

            # volume-stall detection
            symbol_candles = candles.get(symbol, pd.DataFrame())
            if not symbol_candles.empty:
                stall = detect_volume_stall(symbol, symbol_candles, candles, self.risk_config, returns_24h=returns_24h_cache)
                if stall.stalled:
                    rejected.append((symbol, f"volume_stall:{stall.reason}"))
                    self._cooldowns.trigger(symbol, now_dt)
                    continue

            # Don't chase exhausted trends: if 72h return > threshold, skip
            symbol_candles_1h = candles.get(symbol, pd.DataFrame())
            if not symbol_candles_1h.empty and len(symbol_candles_1h) >= 73:
                close_72h = symbol_candles_1h["close"].astype(float)
                ret_72h = float(close_72h.iloc[-1] / close_72h.iloc[-73] - 1)
                if ret_72h > self.risk_config.max_72h_return:
                    rejected.append((symbol, f"exhausted_72h:{ret_72h:.0%}"))
                    continue

            # cooldown check
            if self._cooldowns.is_cooling_down(symbol, now_dt):
                rejected.append((symbol, "cooldown"))
                continue

            # Momentum+score gate: risk_on uses composite alone; caution/defensive require thresholds
            restricted = market_state.state in ("caution", "defensive", "unknown")
            if restricted:
                row_wret = float(getattr(row, "weighted_return", 0))
                if row_wret < effective_min_momentum:
                    rejected.append((symbol, f"momentum<{effective_min_momentum:.0%}:{row_wret:.1%}"))
                    continue
                if effective_min_score > 0:
                    row_score = float(getattr(row, "final_score", 0))
                    if row_score < effective_min_score:
                        rejected.append((symbol, f"score<{effective_min_score:.2f}:{row_score:.2f}"))
                        continue

            # price / ATR sanity
            price = float(prices.get(symbol, 0))
            atr = float(atrs.get(symbol, 0))
            stop_distance = atr * self.risk_config.atr_stop_multiple
            if price <= 0 or stop_distance <= 0:
                rejected.append((symbol, "missing_price_or_atr"))
                continue

            # position sizing
            max_trade_loss = equity * effective_trade_risk_pct
            quantity = min(
                max_trade_loss / stop_distance,
                equity * self.risk_config.max_symbol_position_pct / price,
            )
            target_weight = quantity * price / equity
            stop_risk = quantity * stop_distance / equity
            volatility_risk = quantity * atr * self.risk_config.atr_stop_multiple / equity

            signals.append(
                Signal(
                    time=now_dt,
                    symbol=symbol,
                    side="buy",
                    rank=rank_idx,
                    target_weight=target_weight,
                    reason="v1_composite_top",
                )
            )
            targets.append(
                TargetPosition(
                    symbol=symbol,
                    target_weight=target_weight,
                    quantity=quantity,
                    entry_price=price,
                    stop_price=price - stop_distance,
                    atr=atr,
                    stop_risk_exposure=stop_risk,
                    volatility_risk_exposure=volatility_risk,
                )
            )

            if len(signals) >= effective_max_positions:
                break

        # expire stale cooldowns
        self._cooldowns.expire(now_dt)

        return signals, targets, rejected

    def update_false_breakout(self, top_signals: pd.DataFrame, now: datetime | None = None) -> None:
        """Feed top-signal data to the false-breakout detector.

        Args:
            top_signals: DataFrame with columns [symbol, time, final_score, ...]
            now: current simulation time (used in backtest; defaults to real time)
        """
        self._false_breakout_detector.update(top_signals, self.config, now=now)

    def feed_trade_result(self, won: bool) -> None:
        """Inform the false-breakout detector of a closed trade outcome."""
        self._false_breakout_detector.record_trade_result(won)


    def _detect_blowoff(self, symbol: str, candles_1h: dict[str, pd.DataFrame]) -> str:
        """Detect blow-off top (冲高回落) pattern (White Paper §5.2).

        Returns a reason string if detected, empty string otherwise.
        """
        frame = candles_1h.get(symbol, pd.DataFrame())
        if len(frame) < 12:
            return ""
        if "high" not in frame.columns:
            return ""

        close = frame["close"].astype(float)
        high = frame["high"].astype(float)
        low = frame["low"].astype(float) if "low" in frame.columns else frame["close"].astype(float)
        volume = frame["volume"].astype(float)

        latest_high = float(high.iloc[-1])
        latest_low = float(low.iloc[-1])
        latest_close = float(close.iloc[-1])
        candle_range = latest_high - latest_low

        if candle_range <= 0:
            return ""

        # Check: closed in bottom portion of candle (long upper wick)
        wick_ratio = (latest_high - latest_close) / candle_range
        if wick_ratio < self.risk_config.blowoff_wick_ratio:
            return ""

        # Check: made a new short-term high
        prior_high = float(high.iloc[-12:-1].max())
        if latest_high <= prior_high:
            return ""

        # Check: elevated volume
        avg_volume = float(volume.iloc[-12:-1].mean())
        if avg_volume > 0 and float(volume.iloc[-1]) < avg_volume * self.risk_config.blowoff_volume_multiple:
            return ""

        return f"blowoff_wick:{wick_ratio:.2f}"

    def _volume_confirmed_binary(self, symbol: str, candles_1h: dict[str, pd.DataFrame]) -> bool:
        """Legacy binary volume confirmation — only used when enabled."""
        frame = candles_1h.get(symbol)
        lookback = self.risk_config.volume_confirmation_lookback_hours
        average = self.risk_config.volume_confirmation_average_hours
        if frame is None or len(frame) < lookback + average + 1:
            return False
        volume = frame["volume"].astype(float)
        recent_volume = float(volume.iloc[-lookback:].sum())
        avg_window = volume.iloc[-(lookback + average) : -lookback]
        average_volume = float(avg_window.mean()) * lookback if len(avg_window) > 0 else 0
        if average_volume <= 0:
            return False
        close = frame["close"].astype(float)
        price_rising = float(close.iloc[-1]) > float(close.iloc[-1 - lookback])
        return price_rising and recent_volume / average_volume >= self.risk_config.volume_confirmation_min_ratio
