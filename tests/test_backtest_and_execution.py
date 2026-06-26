from datetime import UTC, datetime

import pandas as pd
import pytest

from crypto_quant.backtest.runner import OpenPosition, Portfolio, ResearchBacktester
from crypto_quant.config.settings import MomentumConfig, PumpModeConfig, load_config
from crypto_quant.execution.broker import BacktestBroker, Order
from crypto_quant.storage.models import OrderRecord, PositionRecord


def test_backtest_broker_applies_buy_slippage_and_fee() -> None:
    order = BacktestBroker(fee_bps=10, slippage_bps=5).execute_market(
        "BTC/USDT", "buy", 2, 100, "next_open_fill"
    )
    assert order.filled_price == 100.05
    assert order.fee == 0.2001
    assert order.reason == "next_open_fill"


def test_precomputed_pump_indicators_use_configured_momentum_weights() -> None:
    cfg = load_config("configs/v1.yaml").model_copy(
        update={
            "pump_mode": PumpModeConfig(enabled=True),
            "momentum": MomentumConfig(windows_hours=[4, 24], weights=[1.0, 0.0]),
        }
    )
    backtester = ResearchBacktester(cfg)
    index = pd.date_range("2024-01-01", periods=30, freq="h", tz="UTC")
    close = pd.Series(range(100, 130), dtype=float)
    candles = {
        "AAA/USDT": pd.DataFrame(
            {
                "open_time": index,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": [1000] * len(close),
                "quote_volume": close * 1000,
            }
        ).set_index("open_time", drop=False)
    }

    backtester._precompute_indicators(candles)

    expected = close.iloc[-1] / close.iloc[-5] - 1
    frame = candles["AAA/USDT"]
    assert frame["weighted_return"].iloc[-1] == expected
    assert "ema20_dev" in frame
    assert "qv_6h_sum" in frame
    assert "regime_vol_expansion" in frame


def test_pump_regime_snapshot_detects_hot_market() -> None:
    cfg = load_config("configs/v1.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    rows = []
    for i in range(25):
        rows.append(
            {
                "symbol": f"AAA{i}/USDT",
                "history": 80,
                "ret_24h": 0.06,
                "new_12h_high": i < 8,
                "regime_vol_expansion": i < 2,
            }
        )
    snapshot = pd.DataFrame(rows)

    assert backtester._detect_pump_regime_snapshot(snapshot) == "HOT"


def test_funding_cost_reduces_cash_for_open_futures_exposure() -> None:
    cfg = load_config("configs/futures_1x.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    portfolio = Portfolio(
        cash=100_000,
        positions={
            "AAA/USDT": OpenPosition(
                "AAA/USDT",
                quantity=10,
                entry_price=100,
                stop_price=90,
                atr=5,
                opened_at=datetime(2024, 1, 1, tzinfo=UTC),
            )
        },
    )

    cost = backtester._apply_funding_cost(
        portfolio,
        {"AAA/USDT": 120},
        datetime(2024, 1, 1, tzinfo=UTC),
        datetime(2024, 1, 1, 2, tzinfo=UTC),
    )

    assert cost > 0
    assert portfolio.cash == pytest.approx(100_000 - cost)


def test_pump_candidates_from_snapshot_detect_fast_volume_backed_move() -> None:
    cfg = load_config("configs/v1.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    backtester._pump_regime = "HOT"
    snapshot = pd.DataFrame(
        [
            {
                "symbol": "AAA/USDT",
                "history": 100,
                "price": 1.0,
                "ret_24h": 0.35,
                "ret_72h": 0.50,
                "ret_6h": 0.12,
                "above_ma20": True,
                "qv_6h": 5_000_000,
                "qv_24h": 20_000_000,
                "qv_30_avg": 1_000_000,
                "wick_ratio": 0.1,
                "new_12h_high": False,
                "regime_vol_expansion": True,
                "atr": 0.05,
                "ema20_dev_rank_2160h": 0.5,
                "ema20_dev": 0.15,
                "r1": 0.01,
                "r2": 0.02,
                "r3": 0.03,
                "pos24h": 0.0,
                "vol_trend6": 2.0,
            }
        ]
    )

    candidates = backtester._pump_candidates_from_snapshot(
        snapshot, Portfolio(cash=100_000), 100_000, datetime(2024, 1, 1, tzinfo=UTC)
    )

    assert len(candidates) == 1
    assert candidates[0].symbol == "AAA/USDT"
    assert candidates[0].reason == "pump_HOT_B_early_confirmed"


def test_pump_stop_trails_after_large_mfe() -> None:
    cfg = load_config("configs/v1.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    position = OpenPosition(
        "AAA/USDT",
        quantity=1,
        entry_price=100,
        stop_price=90,
        atr=5,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        stop_mechanism="pump_initial_stop",
    )
    current = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC"),
            "high": [105, 140, 165, 170],
            "low": [99, 130, 150, 160],
            "close": [104, 138, 160, 168],
        }
    ).set_index("open_time", drop=False)

    reason = backtester._update_pump_stop(position, current, datetime(2024, 1, 1, 3, tzinfo=UTC))

    assert reason is None
    assert position.stop_mechanism == "pump_trailing_stop"
    assert position.stop_price > 150


def test_pump_lock_uses_probe_anchor_when_probe_breathing_enabled() -> None:
    cfg = load_config("configs/v1.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    position = OpenPosition(
        "AAA/USDT",
        quantity=1,
        entry_price=100,
        stop_price=90,
        atr=20,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        stop_mechanism="pump_initial_stop",
        probe_entry_price=100,
        confirm_entry_price=120,
        avg_entry_price=110,
        entry_notional=110,
        probe_confirmed=True,
    )
    current = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
            "high": [111, 122],
            "low": [109, 116],
            "close": [110, 121],
        }
    ).set_index("open_time", drop=False)

    reason = backtester._update_pump_stop(position, current, datetime(2024, 1, 1, 1, tzinfo=UTC))

    assert reason is None
    assert position.stop_mechanism == "pump_lock_2pct"
    assert position.stop_price == pytest.approx(102.0)


def test_enter_pump_position_records_order_and_position(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    cfg = load_config("configs/v1.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    backtester._pump_regime = "HOT"
    next_time = datetime(2024, 1, 1, 1, tzinfo=UTC)
    candles = {
        "AAA/USDT": pd.DataFrame(
            {
                "open_time": pd.date_range("2024-01-01", periods=2, freq="h", tz="UTC"),
                "open": [100.0, 110.0],
                "high": [101.0, 111.0],
                "low": [99.0, 109.0],
                "close": [100.0, 110.0],
            }
        ).set_index("open_time", drop=False)
    }
    candidate = backtester._pump_candidates_from_snapshot(
        pd.DataFrame(
            [
                {
                    "symbol": "AAA/USDT",
                    "history": 100,
                    "price": 100.0,
                    "ret_24h": 0.35,
                    "ret_72h": 0.50,
                    "ret_6h": 0.12,
                    "above_ma20": True,
                    "qv_6h": 5_000_000,
                    "qv_24h": 20_000_000,
                    "qv_30_avg": 1_000_000,
                    "wick_ratio": 0.1,
                    "new_12h_high": False,
                    "regime_vol_expansion": True,
                    "atr": 5.0,
                    "ema20_dev_rank_2160h": 0.5,
                    "ema20_dev": 0.15,
                    "r1": 0.01,
                    "r2": 0.02,
                    "r3": 0.03,
                    "pos24h": 0.0,
                    "vol_trend6": 2.0,
                }
            ]
        ),
        Portfolio(cash=100_000),
        100_000,
        datetime(2024, 1, 1, tzinfo=UTC),
    )[0]
    portfolio = Portfolio(cash=100_000)
    orders: list[Order] = []

    backtester._enter_pump_positions(
        sqlite_session,
        1,
        datetime(2024, 1, 1, tzinfo=UTC),
        next_time,
        [candidate],
        portfolio,
        candles,
        BacktestBroker(fee_bps=10, slippage_bps=5),
        orders,
        {"AAA/USDT": 110.0},
    )

    assert "AAA/USDT" in portfolio.positions
    assert orders[0].side == "buy"
    order_record = next(item for item in sqlite_session.new if isinstance(item, OrderRecord))
    assert order_record.mechanism == "pump_entry"
    position_record = next(item for item in sqlite_session.new if isinstance(item, PositionRecord))
    assert position_record.state == "pump_open"
