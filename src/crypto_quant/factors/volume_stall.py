"""Volume-stall (放量滞涨) detector — White Paper §5.4.

Detects distribution patterns where price is elevated, volume is high,
but price fails to make new highs.  This is a hard rejection with a
cool-down period.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd

from crypto_quant.config.settings import RiskConfig


@dataclass(frozen=True)
class VolumeStallResult:
    stalled: bool
    reason: str = ""
    details: dict[str, float] | None = None


def compute_24h_returns(all_candles_1h: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Pre-compute 24h returns for all symbols (used by volume-stall condition 5)."""
    returns: dict[str, float] = {}
    for sym, frame in all_candles_1h.items():
        if len(frame) >= 25:
            returns[sym] = float(frame["close"].iloc[-1] / frame["close"].iloc[-25] - 1)
    return returns


def detect_volume_stall(
    symbol: str,
    candles_1h: pd.DataFrame,
    all_candles_1h: dict[str, pd.DataFrame],
    config: RiskConfig,
    returns_24h: dict[str, float] | None = None,
) -> VolumeStallResult:
    """Return VolumeStallResult indicating whether *symbol* is stalling.

    Requires the same 1h candle structure used elsewhere:
    columns: open_time, open, high, low, close, volume, quote_volume.

    Args:
        returns_24h: pre-computed 24h returns dict (avoids O(n²) iteration).
                     If None, computed on-the-fly from all_candles_1h.
    """
    if not config.volume_stall_enabled:
        return VolumeStallResult(False)

    lookback = config.volume_stall_lookback_bars
    vol_lookback = config.volume_stall_volume_lookback_hours

    if len(candles_1h) < vol_lookback + lookback:
        return VolumeStallResult(False, reason="insufficient_data")

    required = {"close", "high", "volume"}
    if not required.issubset(candles_1h.columns):
        return VolumeStallResult(False, reason="missing_required_columns")

    close = candles_1h["close"].astype(float)
    high = candles_1h["high"].astype(float)
    volume = candles_1h["volume"].astype(float)

    # condition 1: >= min_bars_above bars with volume > average_volume * ratio
    avg_volume = float(volume.iloc[-(vol_lookback + lookback) : -lookback].mean())
    if avg_volume <= 0:
        return VolumeStallResult(False, reason="zero_average_volume")

    recent_volume = volume.iloc[-lookback:]
    high_volume_bars = int((recent_volume > avg_volume * config.volume_stall_volume_ratio).sum())
    if high_volume_bars < config.volume_stall_min_bars_above:
        return VolumeStallResult(False, reason=f"volume_bars:{high_volume_bars}/{config.volume_stall_min_bars_above}")

    # condition 2: recent 6h high < previous 24h high * rejection_pct
    recent_high = float(high.iloc[-lookback:].max())
    prior_24h = high.iloc[-(vol_lookback + lookback) : -lookback]
    prior_24h_high = float(prior_24h.max()) if len(prior_24h) > 0 else recent_high

    if prior_24h_high <= 0:
        return VolumeStallResult(False, reason="invalid_prior_high")

    if recent_high >= prior_24h_high * config.volume_stall_high_rejection_pct:
        return VolumeStallResult(False, reason="still_making_highs")

    # condition 3: recent close return < threshold
    recent_return = float(close.iloc[-1] / close.iloc[-lookback - 1] - 1)
    if recent_return >= config.volume_stall_return_threshold:
        return VolumeStallResult(False, reason=f"return_above_threshold:{recent_return:.4f}")

    # condition 4: drawdown from 24h high > threshold
    drawdown_from_high = float(close.iloc[-1] / prior_24h_high - 1)
    if drawdown_from_high > -config.volume_stall_drawdown_threshold:
        return VolumeStallResult(False, reason=f"drawdown_nominal:{drawdown_from_high:.4f}")

    # condition 5: symbol is in top 20% of 24h returns in candidate pool
    _returns = returns_24h if returns_24h is not None else compute_24h_returns(all_candles_1h)
    if _returns:
        threshold = pd.Series(list(_returns.values())).quantile(1 - config.volume_stall_top_return_pct)
        sym_return = _returns.get(symbol, 0)
        if sym_return < threshold:
            return VolumeStallResult(False, reason="not_top_return_pct")

    return VolumeStallResult(
        True,
        reason="volume_stall",
        details={
            "high_volume_bars": high_volume_bars,
            "recent_high": recent_high,
            "prior_24h_high": prior_24h_high,
            "recent_return": recent_return,
            "drawdown_from_high": drawdown_from_high,
            "avg_volume": avg_volume,
        },
    )


class CooldownTracker:
    """Tracks per-symbol cool-down periods to prevent repeated entries
    into the same failing coin (White Paper §5.5).
    """

    def __init__(self, cooldown_hours: float = 4.0) -> None:
        self.cooldown_hours = cooldown_hours
        self._cooldowns: dict[str, datetime] = {}

    def is_cooling_down(self, symbol: str, now: datetime) -> bool:
        until = self._cooldowns.get(symbol)
        if until is None:
            return False
        if now >= until:
            del self._cooldowns[symbol]
            return False
        return True

    def trigger(self, symbol: str, now: datetime) -> None:
        from datetime import timedelta

        self._cooldowns[symbol] = now.astimezone(UTC) + timedelta(hours=self.cooldown_hours)

    def remaining(self, symbol: str, now: datetime) -> float:
        until = self._cooldowns.get(symbol)
        if until is None:
            return 0.0
        remaining = (until - now.astimezone(UTC)).total_seconds() / 3600
        return max(remaining, 0.0)

    def expire(self, now: datetime) -> None:
        expired = [sym for sym, until in self._cooldowns.items() if now >= until]
        for sym in expired:
            del self._cooldowns[sym]
