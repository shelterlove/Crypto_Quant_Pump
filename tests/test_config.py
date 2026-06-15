from crypto_quant.config.settings import load_config


def test_config_loads_defaults_and_hash_is_stable() -> None:
    cfg = load_config("configs/default.yaml")
    assert cfg.strategy_version == "v1.4.2"
    assert cfg.risk.trade_risk_pct == 0.01
    assert cfg.stable_hash() == cfg.stable_hash()


def test_mvp_config_extends_default() -> None:
    cfg = load_config("configs/mvp.yaml")
    assert cfg.market_state.ma_period == 50
    assert cfg.backtest.cost_mode == "basic"


def test_v1_pg_matches_final_risk_baseline() -> None:
    cfg = load_config("configs/v1_pg.yaml")
    assert cfg.risk.trade_risk_pct == 0.01
    assert cfg.risk.max_positions == 3
    assert cfg.risk.max_symbol_position_pct == 0.60
