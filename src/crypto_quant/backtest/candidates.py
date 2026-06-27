from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from crypto_quant.config.settings import AppConfig
from crypto_quant.risk.market_state import MarketState
from crypto_quant.utils.time import ensure_utc


@dataclass(frozen=True)
class PumpCandidate:
    symbol: str
    score: float
    price: float
    atr: float
    risk_multiplier: float
    reason: str
    ret_6h: float
    ret_24h: float
    ret_72h: float
    volume_ratio: float
    quote_volume_24h: float
    tier: str = "B"
    ema20_dev_rank_2160h: float = 0.0
    ema20_dev_pct: float = 0.0
    wick_ratio: float = 0.0
    r1: float = 0.0
    r2: float = 0.0
    r3: float = 0.0
    pos24h: float = 0.0
    vol_trend6: float = 0.0


class CandidateEngine:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def latest_prices(
        self,
        candles: dict[str, pd.DataFrame],
        symbols: list[str],
        end: datetime,
        cached_position: Any,
        snapshot_cache: dict[str, dict[str, Any]],
    ) -> dict[str, float]:
        prices: dict[str, float] = {}
        end_ts = pd.Timestamp(ensure_utc(end))
        for symbol in symbols:
            frame = candles.get(symbol)
            if frame is None or frame.empty:
                continue
            pos = cached_position(symbol, end)
            if pos is not None:
                idx = int(pos) - 1
            else:
                try:
                    idx = frame.index.get_loc(end_ts)
                except KeyError:
                    continue
            if idx < 0:
                continue
            values = snapshot_cache.get(symbol)
            if values is not None:
                prices[symbol] = float(values["close"][idx])
            else:
                prices[symbol] = float(frame["close"].iloc[idx])
        return prices

    def snapshot(
        self,
        candles: dict[str, pd.DataFrame],
        symbols: list[str],
        end: datetime,
        cached_position: Any,
        snapshot_cache: dict[str, dict[str, Any]],
    ) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        end_ts = pd.Timestamp(ensure_utc(end))
        for symbol in symbols:
            frame = candles.get(symbol)
            if frame is None or frame.empty:
                continue
            pos = cached_position(symbol, end)
            if pos is not None:
                p = int(pos)
                idx = p - 1
            else:
                try:
                    idx = frame.index.get_loc(end_ts)
                    p = idx + 1
                except KeyError:
                    continue
            if idx < 0:
                continue
            values = snapshot_cache.get(symbol)
            if values is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "history": p,
                    "price": float(values["close"][idx]),
                    "ret_24h": float(values["ret_24h"][idx]),
                    "ret_72h": float(values["ret_72h"][idx]),
                    "ret_6h": float(values["ret_6h"][idx]),
                    "above_ma20": bool(values["above_ma20"][idx]),
                    "qv_6h": float(values["qv_6h_sum"][idx]),
                    "qv_24h": float(values["qv_24h_sum"][idx]),
                    "qv_30_avg": float(values["qv_30_avg"][idx]),
                    "wick_ratio": float(values["wick_ratio"][idx]),
                    "new_12h_high": bool(values["new_12h_high"][idx]),
                    "regime_vol_expansion": bool(values["regime_vol_expansion"][idx]),
                    "atr": float(values["atr14"][idx]),
                    "ema20_dev_rank_2160h": float(values["ema20_dev_rank_2160h"][idx]),
                    "ema20_dev": float(values["ema20_dev"][idx]),
                    "r1": float(values["r1"][idx]),
                    "r2": float(values["r2"][idx]),
                    "r3": float(values["r3"][idx]),
                    "pos24h": float(values["pos24h"][idx]),
                    "vol_trend6": float(values["vol_trend6"][idx]),
                }
            )
        return pd.DataFrame(rows)

    def select_candidates(
        self,
        snapshot: pd.DataFrame,
        portfolio: Any,
        equity: float,
        now: datetime,
        pump_regime: str,
        market: MarketState,
        symbol_last_exit: dict[str, tuple[datetime, str]],
    ) -> tuple[list[PumpCandidate], int]:
        cfg = self.config.pump_mode
        if not cfg.enabled or equity <= 0 or snapshot.empty:
            return [], 0
        if portfolio.daily_realized_loss < 0 and abs(portfolio.daily_realized_loss) > equity * cfg.max_daily_loss_pct:
            return [], 0
        if pump_regime == "COLD" and not cfg.cold_squeeze_enabled:
            return [], 0
        if market.entry_mode == "none" or market.risk_multiplier <= 0:
            return [], 0

        consecutive_losses = 0
        for won in reversed(portfolio.pump_trade_results):
            if won:
                break
            consecutive_losses += 1

        candidates: list[PumpCandidate] = []
        for row in snapshot.itertuples(index=False):
            symbol = str(row.symbol)
            if symbol in portfolio.positions or int(row.history) < 73:
                continue
            price = float(row.price)
            if price <= 0:
                continue
            ret_24h = float(row.ret_24h)
            if pd.isna(ret_24h) or ret_24h < cfg.min_24h_return:
                continue
            ret_72h = float(row.ret_72h)
            ret_6h = float(row.ret_6h)
            if pd.isna(ret_72h) or pd.isna(ret_6h):
                continue
            if not bool(row.above_ma20):
                continue
            q6 = float(row.qv_6h)
            q30 = float(row.qv_30_avg)
            quote_volume_24h = float(row.qv_24h)
            quote_volume_6h = q6
            volume_ratio = q6 / q30 if q30 > 0 else 0
            if volume_ratio <= 0:
                continue
            if quote_volume_24h < cfg.min_quote_volume_24h and quote_volume_6h < cfg.min_quote_volume_6h:
                continue
            if float(row.wick_ratio) >= self.config.risk.blowoff_wick_ratio and bool(row.new_12h_high):
                continue
            if cfg.reject_long_wick_enabled and float(row.wick_ratio) > 0.80 and float(row.r2) < 0:
                continue
            if ret_6h < cfg.min_6h_return:
                continue
            ema20_dev_pct = float(getattr(row, "ema20_dev", 0)) * 100
            cold_squeeze = (
                cfg.cold_squeeze_enabled
                and pump_regime == "COLD"
                and market.entry_mode == "patient"
                and market.phase in {"normal", "crowded_fading"}
                and ret_24h >= cfg.cold_squeeze_min_24h_return
                and ret_24h <= cfg.cold_squeeze_max_24h_return
                and cfg.cold_squeeze_min_72h_return <= ret_72h <= cfg.cold_squeeze_max_72h_return
                and quote_volume_24h >= cfg.cold_squeeze_min_quote_volume_24h
                and cfg.cold_squeeze_min_volume_ratio <= volume_ratio <= cfg.cold_squeeze_max_volume_ratio
                and cfg.cold_squeeze_min_ema20_dev_pct <= ema20_dev_pct <= cfg.cold_squeeze_max_ema20_dev_pct
                and float(row.wick_ratio) < cfg.cold_squeeze_max_wick_ratio
                and ret_6h / max(ret_24h, 0.001) >= cfg.cold_squeeze_min_ret6_to_ret24
            )
            if pump_regime == "COLD" and not cold_squeeze:
                continue
            early = ret_24h >= cfg.min_24h_return and ret_6h >= cfg.early_6h_return and volume_ratio >= cfg.early_volume_ratio_min
            confirmed_sig = ret_72h >= cfg.min_72h_return and ret_24h >= cfg.min_24h_return and volume_ratio >= cfg.volume_ratio_min
            warm_early_ok = False
            if pump_regime == "WARM":
                warm_early_ok = (
                    ret_24h >= cfg.min_24h_return
                    and ret_6h >= cfg.warm_early_6h_return
                    and volume_ratio >= cfg.warm_early_volume_ratio_min
                )
            signal_ok = cold_squeeze or ((confirmed_sig or early) and pump_regime == "HOT") or (warm_early_ok and pump_regime == "WARM")
            if not signal_ok:
                continue
            if (
                cfg.reject_hot_confirmed_high_volume_enabled
                and pump_regime == "HOT"
                and confirmed_sig
                and volume_ratio > cfg.reject_hot_confirmed_high_volume_ratio
            ):
                continue
            if not cold_squeeze and ret_72h > cfg.max_72h_return_full_risk:
                continue
            if ret_72h > cfg.max_72h_return_chase or ret_72h > cfg.max_72h_return_entry:
                continue
            if ret_72h > 1.20 and ret_6h / max(ret_24h, 0.001) < 0.30:
                continue

            risk_multiplier = (cfg.cold_squeeze_risk_multiplier if cold_squeeze else 1.0) * market.risk_multiplier
            if early and confirmed_sig:
                risk_multiplier *= 1.25
            atr = float(row.atr)
            if pd.isna(atr) or atr <= 0:
                continue
            score = ret_24h * 0.45 + ret_72h * 0.35 + ret_6h * 0.10 + min(volume_ratio / 5.0, 1.0) * 0.10
            if cfg.reject_high_score_enabled and score > cfg.reject_high_score_threshold:
                continue
            tier = "A" if (pump_regime == "WARM" and (early or warm_early_ok) and 0.45 <= ret_72h <= 0.86 and volume_ratio <= 15) else "B"
            sig_type = "early_confirmed" if (early and confirmed_sig) else ("early" if early else "confirmed")
            ema20_dev_rank_2160h = float(row.ema20_dev_rank_2160h)
            r1 = float(getattr(row, "r1", 0))
            r2 = float(getattr(row, "r2", 0))
            r3 = float(getattr(row, "r3", 0))
            late_accel = cfg.late_accel_control_enabled and (
                max(r1, r2, r3) > cfg.late_accel_max_r123 or (r1 + r2 + r3) > cfg.late_accel_last3_sum
            )
            if late_accel and tier == "A":
                tier = "B"
            if late_accel and confirmed_sig:
                risk_multiplier *= cfg.late_accel_confirmed_risk_multiplier
            if (
                cfg.warm_a_high_ema_downgrade_enabled
                and pump_regime == "WARM"
                and tier == "A"
                and ema20_dev_pct > cfg.warm_a_high_ema_downgrade_pct
            ):
                tier = "B"
            if (
                cfg.warm_a_late_spike_downgrade_enabled
                and pump_regime == "WARM"
                and tier == "A"
                and (max(r1, r2, r3) > cfg.warm_a_late_spike_max_r123 or (r1 + r2 + r3) > cfg.warm_a_late_spike_last3_sum)
            ):
                tier = "B"
            if cfg.ema_abs_min_enabled:
                if ema20_dev_pct < cfg.ema_abs_min_threshold:
                    continue
                if cfg.ema_abs_max_enabled and ema20_dev_pct > cfg.ema_abs_max_threshold:
                    continue
            if cfg.reject_accel_decay_enabled and ret_6h / max(ret_24h, 0.001) < 0.5 and r1 < 0 and r2 < 0 and r3 < 0:
                continue
            if cfg.reject_high_vol_trend_enabled and float(getattr(row, "vol_trend6", 0)) > cfg.reject_high_vol_trend_threshold:
                continue
            if (
                cfg.bad_b_ema_vr_risk_mid_enabled
                and tier == "B"
                and sig_type == "early"
                and ema20_dev_rank_2160h >= cfg.bad_b_ema_rank_min
                and cfg.bad_b_volume_ratio_mid_min < volume_ratio <= cfg.bad_b_volume_ratio_mid_max
            ):
                risk_multiplier *= cfg.bad_b_risk_multiplier_mid
            if (
                market.phase == "crowded_fading"
                and (
                    ema20_dev_pct >= cfg.market_context_fading_ema20_dev_pct
                    or volume_ratio >= cfg.market_context_fading_volume_ratio
                    or ret_6h >= cfg.market_context_fading_ret_6h
                )
            ):
                risk_multiplier *= cfg.market_context_fading_extreme_multiplier
            if cfg.bad_b_vr30_reject_enabled and volume_ratio > cfg.bad_b_volume_ratio_min:
                continue
            if cold_squeeze:
                tier = "B"
                sig_type = "cold_squeeze"
            reason = f"pump_{pump_regime}_{tier}_{sig_type}"
            candidates.append(
                PumpCandidate(
                    symbol=symbol,
                    score=score,
                    price=price,
                    atr=atr,
                    risk_multiplier=risk_multiplier,
                    reason=reason,
                    ret_6h=ret_6h,
                    ret_24h=ret_24h,
                    ret_72h=ret_72h,
                    volume_ratio=volume_ratio,
                    quote_volume_24h=quote_volume_24h,
                    tier=tier,
                    ema20_dev_rank_2160h=ema20_dev_rank_2160h,
                    ema20_dev_pct=ema20_dev_pct,
                    wick_ratio=float(row.wick_ratio),
                    r1=r1,
                    r2=r2,
                    r3=r3,
                    pos24h=float(getattr(row, "pos24h", 0)),
                    vol_trend6=float(getattr(row, "vol_trend6", 0)),
                )
            )
        return sorted(candidates, key=lambda item: item.score, reverse=True), consecutive_losses
