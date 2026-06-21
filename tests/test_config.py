from crypto_quant.config.settings import load_config


def test_config_loads_defaults_and_hash_is_stable() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.strategy_version == "pump-v1"
    assert cfg.risk.blowoff_wick_ratio == 0.6
    assert cfg.pump_mode.enabled is True
    assert cfg.stable_hash() == cfg.stable_hash()


def test_v1_config_extends_default() -> None:
    cfg = load_config("configs/v1.yaml")
    assert cfg.market_state.btc_symbol == "BTC/USDT"
    assert cfg.backtest.cost_mode == "basic"


def test_v1_pg_matches_final_risk_baseline() -> None:
    cfg = load_config("configs/v1_pg.yaml")
    assert cfg.pump_mode.equity_peak_risk_enabled is True
    assert cfg.pump_mode.equity_peak_risk_floor == 0.70
