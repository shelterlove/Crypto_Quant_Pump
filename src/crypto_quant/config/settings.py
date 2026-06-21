from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class UniverseConfig(BaseModel):
    top_n: int = 60
    min_quote_volume_30d: float = 50_000_000
    exclude_keywords: list[str] = Field(
        default_factory=lambda: [
            "USDC",
            "USDT",
            "DAI",
            "BUSD",
            "TUSD",
            "USDP",
            "FDUSD",
            "UP",
            "DOWN",
            "BULL",
            "BEAR",
            # Quasi-stablecoins and non-crypto instruments
            "USD1",    # stablecoin-like
            "USDE",    # synthetic dollar
            "USR",     # stablecoin
            "USDY",    # yield stablecoin
            "EURI",    # euro stablecoin
            "AEUR",    # euro stablecoin
            "PAXG",    # gold token — not a crypto altcoin
            "XAUT",    # gold token
            "BFUSD",   # stablecoin
            "FRAX",    # stablecoin
        ]
    )
    # Mega-cap symbols to exclude from trading universe (kept for market state monitoring only).
    # These dominate momentum scores and prevent altcoin selection.
    mega_cap_exclude: list[str] = Field(
        default_factory=lambda: [
            "BTC",
            "ETH",
            "BNB",
            "SOL",
            "XRP",
            "ADA",
            "DOGE",
            "TRX",
            "LINK",
            "AVAX",
            "TON",
            "DOT",
            "MATIC",
            "SHIB",
            "LTC",
            "BCH",
            "UNI",
            "ATOM",
            "XLM",
            "ETC",
        ]
    )


class PumpModeConfig(BaseModel):
    enabled: bool = False
    max_positions: int = 2
    trade_risk_pct: float = 0.015
    max_symbol_position_pct: float = 0.45
    max_total_exposure_pct: float = 0.80
    min_24h_return: float = 0.32
    min_72h_return: float = 0.35
    min_6h_return: float = 0.10
    early_6h_return: float = 0.08
    min_quote_volume_24h: float = 2_000_000
    min_quote_volume_6h: float = 800_000
    volume_ratio_min: float = 1.5
    early_volume_ratio_min: float = 1.8
    max_72h_return_full_risk: float = 0.80
    max_72h_return_reduced_risk: float = 2.20
    max_72h_return_chase: float = 3.50
    max_72h_return_entry: float = 3.50
    reduced_risk_multiplier: float = 0.70
    late_chase_risk_multiplier: float = 0.40
    initial_stop_atr_multiple: float = 1.8
    initial_stop_pct: float = 0.10
    probe_pct_a: float = 0.50
    probe_pct_b: float = 0.30
    probe_confirm_target_pct_a: float = 1.00
    probe_confirm_target_pct_b: float = 1.00
    portfolio_open_risk_enabled: bool = False
    max_portfolio_open_risk_pct: float = 0.10
    max_probe_open_risk_pct: float | None = None
    profit_protect_1_pct: float = 0.15
    profit_protect_1_stop_pct: float = -0.03
    breakeven_profit_pct: float = 0.30
    probe_anchor_breathing_enabled: bool = True
    bad_b_ema_vr_risk_enabled: bool = True
    bad_b_ema_rank_min: float = 0.95
    bad_b_volume_ratio_min: float = 30.0
    bad_b_risk_multiplier: float = 0.50
    # Optional mid-tier size cut for B-tier early signals with extreme EMA extension and high volume ratio.
    bad_b_ema_vr_risk_mid_enabled: bool = False
    bad_b_volume_ratio_mid_min: float = 15.0
    bad_b_volume_ratio_mid_max: float = 30.0
    bad_b_risk_multiplier_mid: float = 0.75
    # Optional hard reject for excessive short-window volume expansion.
    bad_b_vr30_reject_enabled: bool = False
    # Optional stop floor after large favorable excursion.
    mfe_protect_enabled: bool = False
    mfe_protect_15pct_mult: float = 1.005
    mfe_protect_25pct_mult: float = 1.03
    # Optional EMA-extension filters for candidate quality.
    ema_abs_min_enabled: bool = False
    ema_abs_min_threshold: float = 10.0
    ema_abs_max_enabled: bool = False
    ema_abs_max_threshold: float = 40.0
    improved_score_enabled: bool = False
    reject_long_wick_enabled: bool = False
    reject_accel_decay_enabled: bool = False
    reject_hot_confirmed_high_volume_enabled: bool = False
    reject_hot_confirmed_high_volume_ratio: float = 5.0
    warm_a_high_ema_downgrade_enabled: bool = False
    warm_a_high_ema_downgrade_pct: float = 30.0
    reject_high_score_enabled: bool = False
    reject_high_score_threshold: float = 0.55
    reject_high_vol_trend_enabled: bool = False
    reject_high_vol_trend_threshold: float = 3.0
    warm_a_late_spike_downgrade_enabled: bool = False
    warm_a_late_spike_max_r123: float = 0.13
    warm_a_late_spike_last3_sum: float = 0.20
    late_accel_control_enabled: bool = False
    late_accel_max_r123: float = 0.24
    late_accel_last3_sum: float = 0.55
    late_accel_confirmed_risk_multiplier: float = 0.75
    exit_confidence_enabled: bool = False
    exit_confidence_wick_threshold: float = 0.80
    exit_confidence_low_ema_threshold: float = 8.0
    scaled_avg_floor_enabled: bool = False
    scaled_avg_floor_mfe_pct: float = 0.10
    early_probe_fail_enabled: bool = False
    early_probe_fail_hours: float = 2.0
    early_probe_fail_ret_pct: float = -0.06
    early_probe_fail_mfe_max_pct: float = 0.02
    staged_confirm_enabled: bool = False
    staged_confirm_strong_mfe_pct: float = 0.10
    staged_confirm_strong_ret_pct: float = 0.05
    staged_confirm_weak_target_pct_a: float = 0.70
    staged_confirm_weak_target_pct_b: float = 0.70
    two_stage_confirm_enabled: bool = False
    two_stage_weak_target_pct: float = 0.55
    two_stage_strong_target_pct: float = 1.00
    two_stage_strong_mfe_pct: float = 0.12
    two_stage_strong_ret_pct: float = 0.06
    add_tranche_exit_enabled: bool = False
    add_tranche_fail_hours: float = 2.0
    add_tranche_fail_mfe_pct: float = 0.05
    add_tranche_fail_ret_pct: float = 0.005
    add_tranche_stop_pct: float = -0.02
    add_tranche_stagnation_first_enabled: bool = False
    failed_reentry_cooldown_enabled: bool = False
    failed_reentry_cooldown_hours: float = 24.0
    failed_reentry_exit_reasons: list[str] = ["pump_probe_kill", "pump_3h_down"]
    stagnation_reentry_boost_enabled: bool = False
    stagnation_reentry_boost_hours: float = 168.0
    stagnation_reentry_probe_pct_a: float = 0.70
    stagnation_reentry_probe_pct_b: float = 0.50
    profit_reserve_enabled: bool = False
    profit_reserve_dd_10_active_pct: float = 0.70
    profit_reserve_dd_20_active_pct: float = 0.80
    profit_reserve_dd_30_active_pct: float = 0.90
    profit_reserve_deep_dd_active_pct: float = 1.00
    profit_reserve_profit_1_threshold: float = 1.50
    profit_reserve_profit_1_active_cap: float = 0.85
    profit_reserve_profit_2_threshold: float = 2.50
    profit_reserve_profit_2_active_cap: float = 0.75
    cold_squeeze_enabled: bool = False
    cold_squeeze_min_24h_return: float = 0.50
    cold_squeeze_min_72h_return: float = 0.25
    cold_squeeze_max_72h_return: float = 1.25
    cold_squeeze_min_quote_volume_24h: float = 150_000_000
    cold_squeeze_min_volume_ratio: float = 1.5
    cold_squeeze_max_volume_ratio: float = 30.0
    cold_squeeze_min_ema20_dev_pct: float = 8.0
    cold_squeeze_max_ema20_dev_pct: float = 45.0
    cold_squeeze_max_wick_ratio: float = 0.65
    cold_squeeze_max_24h_return: float = 1.20
    cold_squeeze_min_ret6_to_ret24: float = 0.10
    cold_squeeze_risk_multiplier: float = 0.70
    cold_squeeze_probe_pct: float = 0.20
    cold_squeeze_max_positions: int = 1
    cold_squeeze_initial_stop_pct: float = 0.06
    cold_squeeze_confirm_enabled: bool = True
    cold_squeeze_confirm_mfe_pct: float = 0.15
    cold_squeeze_confirm_target_pct: float = 0.50
    cold_squeeze_fail_hours: float = 3.0
    cold_squeeze_fail_mfe_pct: float = 0.10
    cold_squeeze_fail_ret_pct: float = -0.03
    # Optional risk scaling by current equity versus prior equity peak.
    equity_peak_risk_enabled: bool = False
    equity_peak_risk_floor: float = 0.50
    mfe_protect_40pct_mult: float = 1.08
    trailing_1_profit_pct: float = 0.60
    trailing_1_atr_multiple: float = 2.5
    trailing_2_profit_pct: float = 1.00
    trailing_2_atr_multiple: float = 2.0
    trailing_3_profit_pct: float = 1.80
    trailing_3_atr_multiple: float = 1.5
    time_stop_hours: float = 12.0
    time_stop_min_profit_pct: float = 0.0
    stagnation_stop_hours: float = 6.0
    stagnation_min_mfe_pct: float = 0.08
    max_daily_loss_pct: float = 0.15
    consecutive_loss_pause_count: int = 999
    cooldown_minutes: int = 0
    extended_cooldown_minutes: int = 0
    recent_trades_lookback: int = 12
    recent_trades_loss_threshold: int = 8
    regime_hot_24h_return_pct: float = 0.05
    regime_hot_new_high_ratio: float = 0.20
    regime_hot_volume_expansion_ratio: float = 1.5
    regime_warm_24h_return_pct: float = 0.02
    regime_warm_new_high_ratio: float = 0.10
    warm_early_6h_return: float = 0.12
    warm_early_volume_ratio_min: float = 2.0
    adaptive_risk_enabled: bool = True
    adaptive_risk_lookback: int = 20
    adaptive_risk_min_multiplier: float = 0.25
    market_context_enabled: bool = True
    market_context_min_history: int = 30
    market_context_normal_risk_multiplier: float = 1.00
    market_context_crowded_hot_risk_multiplier: float = 1.00
    market_context_crowded_fading_risk_multiplier: float = 1.00
    market_context_patient_max_entry_gap_pct: float = 999.0
    market_context_exit_tightening_enabled: bool = False
    market_context_fading_extreme_multiplier: float = 1.00
    market_context_fading_ema20_dev_pct: float = 18.0
    market_context_fading_volume_ratio: float = 13.0
    market_context_fading_ret_6h: float = 0.23


class RiskConfig(BaseModel):
    blowoff_wick_ratio: float = 0.6  # (high-close)/(high-low) threshold


class MarketStateConfig(BaseModel):
    btc_symbol: str = "BTC/USDT"


class MomentumConfig(BaseModel):
    windows_hours: list[int] = Field(default_factory=lambda: [4, 24, 48, 72])
    weights: list[float] = Field(default_factory=lambda: [0.25, 0.35, 0.25, 0.15])


class BacktestConfig(BaseModel):
    initial_equity: float = 100_000
    fee_bps: float = 10
    slippage_bps: float = 5
    pessimistic_slippage_bps: float = 15
    cost_mode: str = "basic"


class AppConfig(BaseModel):
    database_url: str = "postgresql+psycopg://crypto_quant:crypto_quant@localhost:5432/crypto_quant"
    exchange_id: str = "binance"
    strategy_version: str = "v1.4.2"
    base_currency: str = "USDT"
    timeframes: dict[str, str] = Field(default_factory=lambda: {"primary": "1h"})
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market_state: MarketStateConfig = Field(default_factory=MarketStateConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    pump_mode: PumpModeConfig = Field(default_factory=PumpModeConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)

    def stable_hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CRYPTO_QUANT_",
        extra="ignore",
    )

    env: str = "dev"
    database_url: str = AppConfig().database_url
    exchange_id: str = "binance"
    log_level: str = "INFO"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    extends = raw.pop("extends", None)
    if extends:
        base = load_config(extends).model_dump()
        raw = _deep_merge(base, raw)
    return AppConfig.model_validate(raw)


def get_settings() -> AppSettings:
    return AppSettings()
