from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from crypto_quant.config.settings import MomentumConfig


def compute_atr(candles: pd.DataFrame, period: int = 14) -> pd.Series:
    high = candles["high"]
    low = candles["low"]
    close = candles["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(period, min_periods=period).mean()


@dataclass(frozen=True)
class MomentumFactorEngine:
    config: MomentumConfig

    def score(self, candles_by_symbol: dict[str, pd.DataFrame]) -> pd.DataFrame:
        rows: list[dict[str, float | str]] = []
        for symbol, candles in candles_by_symbol.items():
            if candles.empty:
                continue
            close = candles["close"].astype(float)
            raw: dict[str, float] = {}
            enough = True
            for window, weight in zip(self.config.windows_hours, self.config.weights, strict=True):
                if len(close) <= window:
                    enough = False
                    break
                raw[f"return_{window}h"] = float(close.iloc[-1] / close.iloc[-window - 1] - 1)
                raw[f"weight_{window}h"] = float(weight)
            if not enough:
                continue
            weighted_return = sum(
                raw[f"return_{window}h"] * weight
                for window, weight in zip(self.config.windows_hours, self.config.weights, strict=True)
            )
            rows.append({"symbol": symbol, "weighted_return": weighted_return, **raw})
        scores = pd.DataFrame(rows)
        if scores.empty:
            return scores
        scores["momentum_score"] = scores["weighted_return"].rank(pct=True)
        scores["volume_score"] = np.nan
        scores["trend_score"] = np.nan
        scores["final_score"] = scores["momentum_score"]
        return scores.sort_values("final_score", ascending=False).reset_index(drop=True)
