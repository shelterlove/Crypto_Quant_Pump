from datetime import UTC, datetime

import pandas as pd
import pytest

from crypto_quant.backtest.runner import OpenPosition, Portfolio, ResearchBacktester
from crypto_quant.config.settings import MomentumConfig, PumpModeConfig, load_config
from crypto_quant.execution.broker import BacktestBroker, Order
from crypto_quant.factors.trend import compute_trend_scores
from crypto_quant.risk.market_state import MarketState
from crypto_quant.storage.models import OrderRecord, PositionRecord
from crypto_quant.strategy.engine import StrategyEngine


def test_backtest_broker_applies_buy_slippage_and_fee() -> None:
    order = BacktestBroker(fee_bps=10, slippage_bps=5).execute_market(
        "BTC/USDT", "buy", 2, 100, "next_open_fill"
    )
    assert order.filled_price == 100.05
    assert order.fee == 0.2001
    assert order.reason == "next_open_fill"


def test_synthetic_backtest_completes_without_database() -> None:
    result = ResearchBacktester(load_config("configs/mvp.yaml")).run_synthetic()
    assert result.strategy_run_id is None
    assert len(result.orders) <= 3
    assert result.final_equity > 0


def test_volume_confirmation_filters_low_volume_signal() -> None:
    cfg = load_config("configs/binance_mvp_plus_wave1.yaml")
    factors = pd.DataFrame([{"symbol": "AAA/USDT", "final_score": 1.0, "momentum_score": 1.0, "weighted_return": 0.20}])
    frame = pd.DataFrame(
        {
            "close": [100 + i for i in range(27)],
            "high": [101 + i for i in range(27)],
            "volume": [1000] * 21 + [100] * 6,
        }
    )
    signals, targets, rejected = StrategyEngine(cfg).generate_targets(
        factors,
        MarketState("risk_on"),
        {"AAA/USDT": 126},
        {"AAA/USDT": 2},
        100_000,
        {"AAA/USDT": frame},
    )
    assert signals == []
    assert targets == []
    assert rejected == [("AAA/USDT", "volume_confirmation")]


def test_breakeven_and_trailing_stop_update() -> None:
    cfg = load_config("configs/binance_mvp_plus_wave1.yaml")
    backtester = ResearchBacktester(cfg)
    position = OpenPosition("AAA/USDT", 1, 100, 96, 2, datetime(2024, 1, 1, tzinfo=UTC))
    current = pd.DataFrame({"high": [107], "low": [104], "close": [106]})
    backtester._update_position_stop(position, current, MarketState("risk_on"))
    assert position.trailing_active is True
    assert position.stop_price == 103
    assert position.stop_mechanism == "trailing_stop"
    assert position.stop_trigger == "low_below_trailing_stop"


def test_defensive_state_tightens_weak_position_stop() -> None:
    cfg = load_config("configs/binance_mvp_plus_wave1.yaml")
    backtester = ResearchBacktester(cfg)
    position = OpenPosition("AAA/USDT", 1, 100, 92, 3, datetime(2024, 1, 1, tzinfo=UTC))
    current = pd.DataFrame({"high": [101], "low": [99], "close": [100]})
    backtester._update_position_stop(position, current, MarketState("defensive"))
    assert position.stop_price == 97
    assert position.stop_mechanism == "defensive_exit"


def test_breakeven_stop_mechanism_is_recorded_before_trailing_activation() -> None:
    cfg = load_config("configs/binance_mvp_plus_wave1.yaml")
    backtester = ResearchBacktester(cfg)
    position = OpenPosition("AAA/USDT", 1, 100, 96, 2, datetime(2024, 1, 1, tzinfo=UTC))
    current = pd.DataFrame({"high": [103], "low": [101], "close": [102]})

    backtester._update_position_stop(position, current, MarketState("risk_on"))

    assert position.stop_price == 100
    assert position.stop_mechanism == "breakeven_stop"
    assert position.stop_trigger == "low_below_breakeven_stop"


def test_entry_stop_is_recomputed_from_filled_price(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    cfg = load_config("configs/binance_mvp_plus_wave1.yaml")
    backtester = ResearchBacktester(cfg)
    target = StrategyEngine(cfg).generate_targets(
        pd.DataFrame([{"symbol": "AAA/USDT", "final_score": 1.0, "momentum_score": 1.0, "weighted_return": 0.20}]),
        MarketState("risk_on"),
        {"AAA/USDT": 100},
        {"AAA/USDT": 2},
        100_000,
        {
            "AAA/USDT": pd.DataFrame(
                {
                    "open_time": pd.date_range("2024-01-01", periods=27, freq="h", tz="UTC"),
                    "close": list(range(100, 127)),
                    "volume": [1000] * 27,
                }
            )
        },
    )[1][0]
    now = datetime(2024, 1, 2, tzinfo=UTC)
    next_time = datetime(2024, 1, 2, 1, tzinfo=UTC)
    candles = {
        "AAA/USDT": pd.DataFrame(
            {
                "open_time": pd.date_range("2024-01-02", periods=2, freq="h", tz="UTC"),
                "_open_time_utc": pd.date_range("2024-01-02", periods=2, freq="h", tz="UTC"),
                "open": [100, 110],
                "high": [101, 111],
                "low": [99, 109],
                "close": [100, 110],
            }
        )
    }
    portfolio = Portfolio(cash=100_000)
    orders: list[Order] = []

    backtester._enter_positions(
        sqlite_session,
        1,
        now,
        next_time,
        [target],
        portfolio,
        candles,
        BacktestBroker(fee_bps=10, slippage_bps=5),
        orders,
    )
    position = portfolio.positions["AAA/USDT"]
    assert position.opened_at == next_time
    assert position.entry_price == 110.055
    assert position.stop_price == 106.055

    order = next(item for item in sqlite_session.new if isinstance(item, OrderRecord))
    assert order.time == next_time
    assert order.mechanism == "entry"
    assert order.trigger == "next_1h_open"
    assert order.details is not None
    assert order.details["stop_price"] == 106.055

    position_record = next(item for item in sqlite_session.new if isinstance(item, PositionRecord))
    assert position_record.opened_at == next_time
    assert position_record.stop_price is not None
    assert float(position_record.stop_price) == 106.055


def test_precomputed_momentum_uses_configured_weights() -> None:
    cfg = load_config("configs/mvp.yaml").model_copy(
        update={"momentum": MomentumConfig(windows_hours=[4, 24], weights=[1.0, 0.0])}
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
            }
        ).set_index("open_time", drop=False)
    }

    backtester._precompute_indicators(candles, {})

    expected = close.iloc[-1] / close.iloc[-5] - 1
    assert candles["AAA/USDT"]["weighted_return"].iloc[-1] == expected


def test_precomputed_trend_matches_standard_factor_on_latest_slice() -> None:
    cfg = load_config("configs/mvp.yaml")
    backtester = ResearchBacktester(cfg)
    index_1h = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
    close_1h = pd.Series(range(100, 220), dtype=float)
    frame_1h = pd.DataFrame(
        {
            "open_time": index_1h,
            "open": close_1h,
            "high": close_1h + 1,
            "low": close_1h - 1,
            "close": close_1h,
            "volume": [1000] * len(close_1h),
        }
    ).set_index("open_time", drop=False)
    index_4h = pd.date_range("2024-01-01", periods=40, freq="4h", tz="UTC")
    close_4h = pd.Series(range(100, 140), dtype=float)
    frame_4h = pd.DataFrame(
        {
            "open_time": index_4h,
            "open": close_4h,
            "high": close_4h + 1,
            "low": close_4h - 1,
            "close": close_4h,
            "volume": [1000] * len(close_4h),
        }
    ).set_index("open_time", drop=False)
    candles_1h = {"AAA/USDT": frame_1h}
    candles_4h = {"AAA/USDT": frame_4h}

    backtester._precompute_indicators(candles_1h, candles_4h)

    latest_time = index_1h[-1]
    expected = compute_trend_scores(
        candles_1h,
        cfg.trend,
        candles_4h={"AAA/USDT": frame_4h.loc[:latest_time]},
    )["trend_score"].iloc[0]
    actual = candles_1h["AAA/USDT"]["trend_score_col"].iloc[-1]
    assert actual == expected


def test_hard_risk_limit_uses_current_atr_expansion() -> None:
    cfg = load_config("configs/mvp.yaml")
    backtester = ResearchBacktester(cfg)
    position = OpenPosition(
        "AAA/USDT",
        quantity=10,
        entry_price=100,
        stop_price=96,
        atr=1,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        highest_price=130,
        entry_atr=1,
    )
    portfolio = Portfolio(cash=99_000, positions={"AAA/USDT": position}, initial_equity=100_000)

    assert backtester._check_hard_risk_limits(
        position,
        portfolio,
        {"AAA/USDT": 110},
        current_atr=4,
    )


def test_pump_candidates_detect_fast_volume_backed_move() -> None:
    cfg = load_config("configs/mvp.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    index = pd.date_range("2024-01-01", periods=80, freq="h", tz="UTC")
    close = pd.Series([100.0] * 50 + [150.0] * 20 + list(pd.Series(range(160, 210, 5), dtype=float)))
    frame = pd.DataFrame(
        {
            "open_time": index,
            "open": close,
            "high": close * 1.02,
            "low": close * 0.98,
            "close": close,
            "volume": [1000.0] * 70 + [5000.0] * 10,
            "quote_volume": [100_000.0] * 70 + [500_000.0] * 10,
        }
    ).set_index("open_time", drop=False)
    backtester._precompute_indicators({"AAA/USDT": frame}, {})
    backtester._pump_regime = "HOT"  # bypass regime check for single-symbol test

    candidates = backtester._pump_candidates(
        {"AAA/USDT": frame},
        Portfolio(cash=100_000),
        100_000,
        index[-1].to_pydatetime(),
    )

    assert candidates
    assert candidates[0].symbol == "AAA/USDT"
    assert candidates[0].reason in {"pump_HOT_B_early", "pump_HOT_B_confirmed", "pump_HOT_B_early_confirmed"}


def test_pump_stop_trails_after_large_mfe() -> None:
    cfg = load_config("configs/mvp.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    position = OpenPosition(
        "AAA/USDT",
        quantity=1,
        entry_price=100,
        stop_price=90,
        atr=5,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        position_type="pump",
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


def test_pump_lock_uses_weighted_average_entry_after_confirm() -> None:
    cfg = load_config("configs/mvp.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    position = OpenPosition(
        "AAA/USDT",
        quantity=1,
        entry_price=100,
        stop_price=90,
        atr=20,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        position_type="pump",
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
    assert position.stop_price == pytest.approx(112.2)


def test_b_unconfirmed_probe_exits_directly_after_4h_loss() -> None:
    cfg = load_config("configs/mvp.yaml").model_copy(update={"pump_mode": PumpModeConfig(enabled=True)})
    backtester = ResearchBacktester(cfg)
    position = OpenPosition(
        "AAA/USDT",
        quantity=1,
        entry_price=100,
        stop_price=80,
        atr=5,
        opened_at=datetime(2024, 1, 1, tzinfo=UTC),
        position_type="pump",
        stop_mechanism="pump_initial_stop",
        is_probe=True,
        probe_tier="B",
        probe_entry_price=100,
        avg_entry_price=100,
        entry_notional=100,
    )
    current = pd.DataFrame(
        {
            "open_time": pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC"),
            "high": [101, 100, 99, 99, 98],
            "low": [99, 98, 97, 96, 95],
            "close": [100, 99, 98.5, 98.2, 97.5],
        }
    ).set_index("open_time", drop=False)

    reason = backtester._update_pump_stop(position, current, datetime(2024, 1, 1, 4, tzinfo=UTC))

    assert reason == "pump_b_unconfirmed_4h_down"
    assert position.stop_mechanism == "pump_b_unconfirmed_4h_down"
