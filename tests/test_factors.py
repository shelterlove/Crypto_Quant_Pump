import pandas as pd

from crypto_quant.config.settings import MomentumConfig
from crypto_quant.factors.momentum import MomentumFactorEngine, compute_atr


def _candles(start: float, step: float, periods: int = 80) -> pd.DataFrame:
    close = pd.Series([start + step * i for i in range(periods)], dtype=float)
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC"),
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1000,
        }
    )


def test_momentum_percentile_ranks_strongest_symbol_highest() -> None:
    scores = MomentumFactorEngine(MomentumConfig()).score(
        {"SLOW/USDT": _candles(100, 0.1), "FAST/USDT": _candles(100, 2.0)}
    )
    assert scores.iloc[0]["symbol"] == "FAST/USDT"
    assert scores.iloc[0]["momentum_score"] == 1.0


def test_compute_atr_uses_true_range() -> None:
    atr = compute_atr(_candles(100, 1, 20), period=14)
    assert atr.notna().sum() == 7
    assert atr.iloc[-1] > 0
