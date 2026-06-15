import pandas as pd

from crypto_quant.config.settings import UniverseConfig
from crypto_quant.universe.builder import LiquidityUniverseBuilder


def test_universe_filters_stables_leverage_and_low_volume() -> None:
    symbols = pd.DataFrame(
        [
            {"symbol": "BTC/USDT", "base": "BTC", "quote": "USDT", "status": "TRADING", "spot": True, "quote_volume_30d": 100_000_000},
            {"symbol": "USDC/USDT", "base": "USDC", "quote": "USDT", "status": "TRADING", "spot": True, "quote_volume_30d": 200_000_000},
            {"symbol": "ETHUP/USDT", "base": "ETHUP", "quote": "USDT", "status": "TRADING", "spot": True, "quote_volume_30d": 200_000_000},
            {"symbol": "ABC/USDT", "base": "ABC", "quote": "USDT", "status": "TRADING", "spot": True, "quote_volume_30d": 1_000_000},
        ]
    )
    universe = LiquidityUniverseBuilder(UniverseConfig(top_n=60)).build(symbols)
    assert universe["symbol"].tolist() == []  # BTC is now mega-cap excluded


def test_universe_respects_top_n_liquidity_rank() -> None:
    symbols = pd.DataFrame(
        [
            {
                "symbol": f"C{i}/USDT",
                "base": f"C{i}",
                "quote": "USDT",
                "status": "TRADING",
                "spot": True,
                "quote_volume_30d": 100_000_000 - i,
            }
            for i in range(5)
        ]
    )
    universe = LiquidityUniverseBuilder(UniverseConfig(top_n=3)).build(symbols)
    assert universe["symbol"].tolist() == ["C0/USDT", "C1/USDT", "C2/USDT"]
    assert universe["liquidity_rank"].tolist() == [1, 2, 3]
