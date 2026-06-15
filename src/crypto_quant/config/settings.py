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


class TrendConfig(BaseModel):
    enabled: bool = True
    ma_short_period: int = 20
    ma_long_period: int = 20
    score_weight: float = 0.25
    max_drawdown_from_24h_high: float = 0.05


class VolumeScoreConfig(BaseModel):
    enabled: bool = True
    score_weight: float = 0.25
    lookback_hours: int = 6
    average_hours: int = 20


class FalseBreakoutConfig(BaseModel):
    enabled: bool = True
    top_n: int = 10
    signal_window_hours: int = 24
    return_horizon_hours: int = 8
    negative_signal_threshold_pct: float = 0.5
    breadth_threshold: float = 0.4
    min_samples: int = 10
    single_risk_multiplier: float = 0.5
    max_positions: int = 1
    top_n_allowed: int = 2


class PumpModeConfig(BaseModel):
    enabled: bool = False
    max_positions: int = 2
    trade_risk_pct: float = 0.015
    max_symbol_position_pct: float = 0.45
    max_total_exposure_pct: float = 0.80
    min_24h_return: float = 0.18
    min_72h_return: float = 0.35
    min_6h_return: float = 0.10  # all signals must have 6h >= 10% (0% WR below this)
    early_6h_return: float = 0.08
    min_quote_volume_24h: float = 2_000_000
    min_quote_volume_6h: float = 800_000
    volume_ratio_min: float = 1.5
    early_volume_ratio_min: float = 1.8
    max_72h_return_full_risk: float = 0.80  # v19: lowered from 1.20 — r72>80% stats worse
    max_72h_return_reduced_risk: float = 2.20
    max_72h_return_chase: float = 3.50
    max_72h_return_entry: float = 3.50  # disabled (same as chase limit)
    reduced_risk_multiplier: float = 0.70
    late_chase_risk_multiplier: float = 0.40
    initial_stop_atr_multiple: float = 1.8
    initial_stop_pct: float = 0.10
    profit_protect_1_pct: float = 0.15
    profit_protect_1_stop_pct: float = -0.03
    breakeven_profit_pct: float = 0.30
    probe_anchor_breathing_enabled: bool = True
    trailing_1_profit_pct: float = 0.60
    trailing_1_atr_multiple: float = 2.5
    trailing_2_profit_pct: float = 1.00
    trailing_2_atr_multiple: float = 2.0
    trailing_3_profit_pct: float = 1.80
    trailing_3_atr_multiple: float = 1.5
    time_stop_hours: float = 12.0  # default B-tier; v23: tier-specific
    time_stop_min_profit_pct: float = 0.0
    stagnation_stop_hours: float = 6.0  # default B-tier; v23: tier-specific
    stagnation_min_mfe_pct: float = 0.08  # default B-tier; v23: tier-specific
    max_daily_loss_pct: float = 0.15
    # v19: cooldown disabled — after 3+ losses, next trade 18/18 wins
    consecutive_loss_pause_count: int = 999  # effectively disabled
    cooldown_minutes: int = 0
    extended_cooldown_minutes: int = 0
    recent_trades_lookback: int = 12
    recent_trades_loss_threshold: int = 8
    # v2: pump regime detection — only trade pumps when market is "hot"
    regime_hot_24h_return_pct: float = 0.05      # top20 24h median return
    regime_hot_new_high_ratio: float = 0.20        # coins making new 24h highs
    regime_hot_volume_expansion_ratio: float = 1.5 # volume expansion threshold
    regime_warm_24h_return_pct: float = 0.02
    regime_warm_new_high_ratio: float = 0.10
    # v19: WARM now allows early signals with slightly stricter thresholds
    warm_early_6h_return: float = 0.12
    warm_early_volume_ratio_min: float = 2.0
    # v20: adaptive risk — scale position size by recent 3h_down ratio
    adaptive_risk_enabled: bool = True
    adaptive_risk_lookback: int = 20
    adaptive_risk_min_multiplier: float = 0.25


class RiskConfig(BaseModel):
    trade_risk_pct: float = 0.01
    max_symbol_position_pct: float = 0.35
    max_positions: int = 3
    atr_period: int = 14
    atr_stop_multiple: float = 2.0
    max_single_volatility_risk_pct: float = 0.03
    max_portfolio_stop_risk_pct: float = 0.03
    max_portfolio_volatility_risk_pct: float = 0.06
    enable_breakeven_stop: bool = False
    breakeven_activation_atr_multiple: float = 2.0
    breakeven_buffer_bps: float = 0.0
    enable_trailing_stop: bool = False
    trailing_activation_atr_multiple: float = 2.0
    trailing_stop_atr_multiple: float = 2.5
    defensive_tighten_existing_positions: bool = False
    defensive_tighten_atr_multiple: float = 1.0
    volume_confirmation_enabled: bool = False
    volume_confirmation_lookback_hours: int = 6
    volume_confirmation_average_hours: int = 20
    volume_confirmation_min_ratio: float = 1.0
    # v1: daily loss circuit breaker
    daily_loss_limit_pct: float = 0.04
    daily_loss_defensive_pct: float = 0.06
    consecutive_loss_pause_count: int = 3
    consecutive_loss_pause_hours: float = 2.0
    recent_trades_lookback: int = 5
    recent_trades_loss_threshold: int = 4
    # v1: volume-stall filter (white paper §5.4)
    volume_stall_enabled: bool = True
    volume_stall_lookback_bars: int = 6
    volume_stall_volume_lookback_hours: int = 20
    volume_stall_volume_ratio: float = 1.5
    volume_stall_min_bars_above: int = 4
    volume_stall_high_rejection_pct: float = 0.995
    volume_stall_return_threshold: float = 0.015
    volume_stall_drawdown_threshold: float = 0.015
    volume_stall_top_return_pct: float = 0.2
    cooldown_hours: float = 4.0
    # v1: position state machine
    weakening_reduction_pct: float = 0.33
    weakening_trailing_multiple: float = 1.5
    weakening_rank_threshold: int = 5
    weakening_rank_upper: int = 15
    weakening_drawdown_atr_multiple: float = 1.5
    # v1: swap mechanism (White Paper §10)
    swap_enabled: bool = True
    swap_score_advantage: float = 1.15  # new coin score must be > current × this
    swap_strong_score_advantage: float = 1.25  # strong swap threshold
    swap_max_per_day: int = 2  # max swaps per day
    # v1: blow-off top detection (White Paper §5.2)
    blowoff_enabled: bool = True
    blowoff_wick_ratio: float = 0.6  # (high-close)/(high-low) threshold
    blowoff_volume_multiple: float = 1.5  # volume must be elevated
    # v1: hard risk limits during hold (White Paper §7.7)
    hard_risk_volatility_exposure_pct: float = 0.03  # single position vol risk cap
    hard_risk_atr_expansion: float = 3.0  # atr_expansion_ratio trigger
    hard_risk_dd_atr_multiple: float = 1.5  # drawdown vs ATR trigger
    # v1: don't chase exhausted trends — block if coin already ran up too much
    max_72h_return: float = 1.20  # 120%+ in 3 days → don't enter (trend exhausted)
    # v1: hybrid stop (White Paper §8.2 Scheme C)
    hybrid_stop_enabled: bool = True
    hybrid_stop_structure_atr_min: float = 1.0
    hybrid_stop_structure_atr_max: float = 2.5
    # v1: signal quality thresholds
    min_composite_score: float = 0.0  # absolute minimum final_score (0=disabled)
    min_momentum_return: float = 0.15  # must have 15%+ absolute weighted return
    # v1: caution-mode overrides — raise the bar instead of full rejection
    caution_max_positions: int = 1
    caution_risk_multiplier: float = 0.5  # half normal risk
    caution_min_score: float = 0.90  # nearly perfect composite score
    caution_min_momentum: float = 0.08  # 8%+ absolute weighted return
    # v1: low-volume overrides — coins below this volume need higher momentum
    low_volume_threshold: float = 10_000_000  # below 10M daily vol = "low volume"
    low_volume_min_momentum: float = 0.03  # must have 3%+ weighted return


class MarketStateConfig(BaseModel):
    btc_symbol: str = "BTC/USDT"
    ma_period: int = 50
    slope_lookback_bars: int = 4
    bearish_slope_threshold: float = -0.005
    breadth_threshold: float = 0.5


class MomentumConfig(BaseModel):
    windows_hours: list[int] = Field(default_factory=lambda: [4, 24, 48, 72])
    weights: list[float] = Field(default_factory=lambda: [0.25, 0.35, 0.25, 0.15])


class BacktestConfig(BaseModel):
    initial_equity: float = 100_000
    fee_bps: float = 10
    slippage_bps: float = 5
    pessimistic_slippage_bps: float = 15
    cost_mode: str = "basic"


class PipelineConfig(BaseModel):
    slippage_pressure_bps: list[float] = Field(default_factory=lambda: [5.0, 10.0, 20.0, 30.0])
    prefer_local_symbol_cache: bool = False


class AppConfig(BaseModel):
    database_url: str = "postgresql+psycopg://crypto_quant:crypto_quant@localhost:5432/crypto_quant"
    exchange_id: str = "binance"
    strategy_version: str = "v1.4.2"
    base_currency: str = "USDT"
    timeframes: dict[str, str] = Field(default_factory=lambda: {"primary": "1h", "trend": "4h"})
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    market_state: MarketStateConfig = Field(default_factory=MarketStateConfig)
    momentum: MomentumConfig = Field(default_factory=MomentumConfig)
    trend: TrendConfig = Field(default_factory=TrendConfig)
    volume_score: VolumeScoreConfig = Field(default_factory=VolumeScoreConfig)
    false_breakout: FalseBreakoutConfig = Field(default_factory=FalseBreakoutConfig)
    pump_mode: PumpModeConfig = Field(default_factory=PumpModeConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

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
