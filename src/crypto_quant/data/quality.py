from __future__ import annotations

import pandas as pd

REQUIRED_CANDLE_COLUMNS = {"open_time", "open", "high", "low", "close", "volume"}


def validate_candles(frame: pd.DataFrame) -> list[str]:
    errors: list[str] = []
    missing = REQUIRED_CANDLE_COLUMNS - set(frame.columns)
    if missing:
        errors.append(f"missing columns: {sorted(missing)}")
        return errors
    if frame["open_time"].duplicated().any():
        errors.append("duplicate open_time values")
    if not frame["open_time"].is_monotonic_increasing:
        errors.append("open_time is not monotonic increasing")
    price_cols = ["open", "high", "low", "close"]
    if frame[price_cols].isna().any().any():
        errors.append("price columns contain nulls")
    if (frame[price_cols] <= 0).any().any():
        errors.append("price columns must be positive")
    if ((frame["high"] < frame["low"]) | (frame["high"] < frame["close"]) | (frame["low"] > frame["close"])).any():
        errors.append("invalid high/low relationship")
    return errors
