import pandas as pd

from crypto_quant.config.settings import MarketStateConfig, RiskConfig
from crypto_quant.risk.engine import RiskEngine
from crypto_quant.risk.market_state import compute_market_breadth, evaluate_btc_ma50_state, fast_risk_valve_triggered
from crypto_quant.strategy.types import TargetPosition


def _btc(closes: list[float]) -> pd.DataFrame:
    values = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=len(closes), freq="4h", tz="UTC"),
            "open": values,
            "high": values + 1,
            "low": values - 1,
            "close": values,
            "volume": 1000,
        }
    )


def test_btc_ma50_state_risk_on() -> None:
    state = evaluate_btc_ma50_state(_btc(list(range(100, 160))), MarketStateConfig())
    assert state.state == "risk_on"
    assert state.ma50_slope_4 is not None


def test_market_breadth_counts_symbols_above_ma() -> None:
    rising = _btc(list(range(100, 160)))
    falling = _btc(list(range(160, 100, -1)))
    assert compute_market_breadth({"A": rising, "B": falling}, ma_period=50) == 0.5


def test_fast_risk_valve_triggers_on_btc_hourly_drop() -> None:
    btc_1h = pd.DataFrame({"close": [100.0, 92.0]})
    triggered, reasons = fast_risk_valve_triggered(btc_1h=btc_1h)
    assert triggered
    assert "btc_1h_drop" in reasons


def test_atr_position_size_respects_single_position_cap() -> None:
    quantity, stop_price, target_weight = RiskEngine(RiskConfig()).atr_position_size(100_000, 100, 1)
    assert quantity == 350
    assert stop_price == 98
    assert target_weight == 0.35


def test_hard_risk_limit_rejects_single_volatility_excess() -> None:
    decision = RiskEngine(RiskConfig()).approve_targets(
        [TargetPosition("ABC/USDT", 0.1, volatility_risk_exposure=0.04)]
    )
    assert not decision.approved
    assert decision.rejected[0][1] == "single_volatility_risk_limit"
