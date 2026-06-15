from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crypto_quant.config.settings import MarketStateConfig


@dataclass(frozen=True)
class MarketState:
    state: str
    btc_close: float | None = None
    btc_ma50: float | None = None
    ma50_slope_4: float | None = None
    breadth: float | None = None
    fast_risk_valve: bool = False
    reasons: list[str] = field(default_factory=list)

    @property
    def allows_new_risk(self) -> bool:
        """Normal risk-on: full position sizing, all signals allowed."""
        return self.state == "risk_on" and not self.fast_risk_valve

    @property
    def allows_strong_only(self) -> bool:
        """Caution mode: only Top 1-2 signals with half risk."""
        return self.state == "caution" and not self.fast_risk_valve


def evaluate_btc_ma50_state(btc_4h: pd.DataFrame, config: MarketStateConfig) -> MarketState:
    if len(btc_4h) < config.ma_period + config.slope_lookback_bars:
        return MarketState(state="unknown", reasons=["insufficient_btc_history"])
    close = btc_4h["close"].astype(float)
    ma = close.rolling(config.ma_period).mean()
    current_ma = float(ma.iloc[-1])
    past_ma = float(ma.iloc[-1 - config.slope_lookback_bars])
    slope = current_ma / past_ma - 1
    btc_close = float(close.iloc[-1])
    if btc_close > current_ma and slope > config.bearish_slope_threshold:
        return MarketState("risk_on", btc_close, current_ma, slope, reasons=["btc_above_ma50"])
    if btc_close < current_ma and slope <= config.bearish_slope_threshold:
        return MarketState("defensive", btc_close, current_ma, slope, reasons=["btc_below_ma50_bearish_slope"])
    return MarketState("caution", btc_close, current_ma, slope, reasons=["mixed_btc_trend"])


def compute_market_breadth(candles_by_symbol: dict[str, pd.DataFrame], ma_period: int = 20) -> float:
    states: list[bool] = []
    for frame in candles_by_symbol.values():
        if len(frame) < ma_period:
            continue
        close = frame["close"].astype(float)
        states.append(bool(close.iloc[-1] > close.rolling(ma_period).mean().iloc[-1]))
    return sum(states) / len(states) if states else 0.0


def fast_risk_valve_triggered(
    btc_1h: pd.DataFrame | None = None,
    top_signal_returns_4h: pd.Series | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if btc_1h is not None and len(btc_1h) >= 2:
        last_return = float(btc_1h["close"].iloc[-1] / btc_1h["close"].iloc[-2] - 1)
        if last_return <= -0.07:
            reasons.append("btc_1h_drop")
    if top_signal_returns_4h is not None and len(top_signal_returns_4h) >= 10:
        if float(top_signal_returns_4h.mean()) < -0.01 and float((top_signal_returns_4h < 0).mean()) > 0.6:
            reasons.append("top_signal_deterioration")
    return bool(reasons), reasons
