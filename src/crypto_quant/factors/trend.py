"""Trend structure factor — White Paper §4.7.

Evaluates whether a symbol's trend structure is healthy by checking:
- Price relative to 1H MA20 (short-term) and 4H MA20 (long-term)
- 4H MA20 slope direction
- Drawdown from 24h high
"""

from __future__ import annotations

import pandas as pd

from crypto_quant.config.settings import TrendConfig


def compute_trend_scores(
    candles_by_symbol: dict[str, pd.DataFrame],
    config: TrendConfig,
    candles_4h: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Return a DataFrame with symbol and trend_score (0..1) columns.

    Args:
        candles_by_symbol: 1H OHLCV frames keyed by symbol.
        config: trend factor configuration.
        candles_4h: optional 4H OHLCV frames for the long-term MA check.
                    If not provided, 1H data is used as fallback.
    """
    if not config.enabled:
        return pd.DataFrame(columns=["symbol", "trend_score"])

    four_h = candles_4h or {}

    rows: list[dict[str, float | str]] = []
    for symbol, candles in candles_by_symbol.items():
        if candles.empty or len(candles) < config.ma_short_period + 1:
            continue

        close = candles["close"].astype(float)
        high = candles["high"].astype(float) if "high" in candles.columns else close

        # short-term MA on 1H chart
        ma_short = float(close.rolling(config.ma_short_period).mean().iloc[-1])

        # long-term MA: prefer 4H chart per white paper §4.7
        frame_4h = four_h.get(symbol, pd.DataFrame())
        if not frame_4h.empty and len(frame_4h) >= config.ma_long_period + 1:
            close_4h = frame_4h["close"].astype(float)
            ma_long = float(close_4h.rolling(config.ma_long_period).mean().iloc[-1])
            ma_long_prev = float(close_4h.rolling(config.ma_long_period).mean().iloc[-1 - config.ma_long_period])
        else:
            # fallback: use 1H data for long MA
            if len(candles) < config.ma_long_period + 1:
                continue
            ma_long = float(close.rolling(config.ma_long_period).mean().iloc[-1])
            ma_long_prev = float(close.rolling(config.ma_long_period).mean().iloc[-1 - config.ma_long_period])

        if ma_short <= 0 or ma_long <= 0:
            continue

        # component 1: price above short MA (1H)? (0 or 0.33)
        above_short = 0.33 if float(close.iloc[-1]) > ma_short else 0

        # component 2: price above long MA (4H)? (0 or 0.33)
        above_long = 0.33 if float(close.iloc[-1]) > ma_long else 0

        # component 3: long MA slope positive? (0 or 0.34, scaled by steepness)
        if ma_long_prev > 0:
            slope = (ma_long / ma_long_prev) - 1
            slope_score = 0.34 * max(0.0, min(1.0, slope * 100))  # scale: 1% slope = full score
        else:
            slope_score = 0

        # component 4: drawdown from 24h high (penalty)
        if len(high) >= 24:
            high_24h = float(high.iloc[-24:].max())
            dd_from_high = float(close.iloc[-1]) / high_24h - 1 if high_24h > 0 else 0
            if dd_from_high < -config.max_drawdown_from_24h_high:
                dd_penalty = min(0.2, abs(dd_from_high) * 4)  # max 0.2 penalty
            else:
                dd_penalty = 0
        else:
            dd_penalty = 0

        trend_score = max(0.0, min(1.0, above_short + above_long + slope_score - dd_penalty))
        rows.append({"symbol": symbol, "trend_score": trend_score})

    return pd.DataFrame(rows)
