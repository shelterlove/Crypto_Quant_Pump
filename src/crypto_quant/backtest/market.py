from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from crypto_quant.config.settings import AppConfig
from crypto_quant.risk.market_state import MarketState


@dataclass
class MarketContextEngine:
    config: AppConfig
    feature_history: list[dict[str, float]] = field(default_factory=list)
    previous_phase: str = "normal"

    def detect_pump_regime_snapshot(self, snapshot: pd.DataFrame) -> str:
        cfg = self.config.pump_mode
        if snapshot.empty:
            return "COLD"
        eligible = snapshot[snapshot["history"].astype(int) >= 50]
        total = len(eligible)
        if total < 20:
            return "COLD"
        ret_24h = [float(value) for value in eligible["ret_24h"].tolist() if pd.notna(value)]
        if not ret_24h:
            return "COLD"
        median_ret = sorted(ret_24h)[len(ret_24h) // 2]
        new_high_count = int(eligible["new_12h_high"].fillna(False).astype(bool).sum())
        vol_exp_count = int(eligible["regime_vol_expansion"].fillna(False).astype(bool).sum())
        nh_r = new_high_count / total
        ve_r = vol_exp_count / total
        if median_ret >= cfg.regime_hot_24h_return_pct and nh_r >= cfg.regime_hot_new_high_ratio and ve_r >= 0.05:
            return "HOT"
        if median_ret >= cfg.regime_warm_24h_return_pct and nh_r >= cfg.regime_warm_new_high_ratio:
            return "WARM"
        return "COLD"

    def detect_market_context(
        self,
        snapshot: pd.DataFrame,
        btc_1h: pd.DataFrame,
        fast_valve: bool,
        fast_reasons: list[str],
        pump_regime: str,
    ) -> MarketState:
        cfg = self.config.pump_mode
        if fast_valve:
            return MarketState(
                "risk_off",
                fast_risk_valve=True,
                reasons=fast_reasons or ["btc_fast_valve"],
                phase="risk_off",
                transition="deteriorating",
                risk_multiplier=0.0,
                entry_mode="none",
                exit_profile="aggressive_tighten" if cfg.market_context_exit_tightening_enabled else "normal",
            )
        if not cfg.market_context_enabled:
            return MarketState("risk_on", reasons=[f"legacy_{pump_regime.lower()}"], phase=pump_regime.lower())

        metrics = self._context_metrics(snapshot, btc_1h)
        history = self.feature_history
        if len(history) < cfg.market_context_min_history:
            context = self._legacy_context(metrics, pump_regime)
        else:
            context = self._phase_context(metrics, history, pump_regime)
        self.feature_history.append(metrics)
        if len(self.feature_history) > 720:
            self.feature_history = self.feature_history[-720:]
        self.previous_phase = context.phase
        return context

    def _context_metrics(self, snapshot: pd.DataFrame, btc_1h: pd.DataFrame) -> dict[str, float]:
        metrics = {
            "eligible_count": 0.0,
            "median_ret_24h": 0.0,
            "new12h_high_ratio": 0.0,
            "ret24_gt10_ratio": 0.0,
            "ret24_gt30_ratio": 0.0,
            "ret72_gt80_ratio": 0.0,
            "new_high_high_wick_ratio": 0.0,
            "sync_down_24h_ratio": 0.0,
            "vol_expansion_ratio": 0.0,
            "candidate_count": 0.0,
            "btc_ret_24h": 0.0,
            "btc_vol_24h": 0.0,
            "heat_score": 0.0,
            "heat_delta_24h": 0.0,
        }
        if snapshot.empty:
            return metrics
        eligible = snapshot[snapshot["history"].astype(int) >= 73].copy()
        if eligible.empty:
            return metrics
        total = len(eligible)
        ret24 = eligible["ret_24h"].astype(float)
        ret72 = eligible["ret_72h"].astype(float)
        new_high = eligible["new_12h_high"].fillna(False).astype(bool)
        wick = eligible["wick_ratio"].astype(float)
        vol_exp = eligible["regime_vol_expansion"].fillna(False).astype(bool)
        q30 = eligible["qv_30_avg"].astype(float).replace(0, np.nan)
        volume_ratio = eligible["qv_6h"].astype(float) / q30

        metrics.update(
            {
                "eligible_count": float(total),
                "median_ret_24h": float(ret24.median()),
                "new12h_high_ratio": float(new_high.mean()),
                "ret24_gt10_ratio": float((ret24 > 0.10).mean()),
                "ret24_gt30_ratio": float((ret24 > 0.30).mean()),
                "ret72_gt80_ratio": float((ret72 > 0.80).mean()),
                "new_high_high_wick_ratio": float((new_high & (wick > 0.60)).mean()),
                "sync_down_24h_ratio": float((ret24 < -0.05).mean()),
                "vol_expansion_ratio": float(vol_exp.mean()),
                "candidate_count": float(self._candidate_count(eligible, volume_ratio)),
            }
        )
        if btc_1h is not None and len(btc_1h) >= 25:
            close = btc_1h["close"].astype(float)
            metrics["btc_ret_24h"] = float(close.iloc[-1] / close.iloc[-25] - 1) if close.iloc[-25] > 0 else 0.0
            metrics["btc_vol_24h"] = float(close.pct_change().tail(24).std())
        metrics["heat_score"] = (
            metrics["median_ret_24h"]
            + 0.50 * metrics["new12h_high_ratio"]
            + 0.30 * metrics["ret24_gt10_ratio"]
            + 0.20 * metrics["vol_expansion_ratio"]
            - 0.30 * metrics["ret72_gt80_ratio"]
            - 0.20 * metrics["new_high_high_wick_ratio"]
        )
        if len(self.feature_history) >= 6:
            metrics["heat_delta_24h"] = metrics["heat_score"] - self.feature_history[-6]["heat_score"]
        return metrics

    def _candidate_count(self, eligible: pd.DataFrame, volume_ratio: pd.Series) -> int:
        cfg = self.config.pump_mode
        rough = (
            (eligible["ret_24h"].astype(float) >= cfg.min_24h_return)
            & (eligible["ret_6h"].astype(float) >= cfg.min_6h_return)
            & (eligible["above_ma20"].fillna(False).astype(bool))
            & (volume_ratio >= cfg.early_volume_ratio_min)
            & (eligible["ret_72h"].astype(float) <= cfg.max_72h_return_full_risk)
        )
        return int(rough.fillna(False).sum())

    def _legacy_context(self, metrics: dict[str, float], pump_regime: str) -> MarketState:
        if pump_regime == "COLD":
            phase, entry_mode, risk_multiplier, exit_profile = "cold", "none", 0.0, "normal"
        elif pump_regime == "HOT":
            phase, entry_mode, risk_multiplier, exit_profile = "expanding", "normal", 1.0, "normal"
        else:
            phase, entry_mode, risk_multiplier, exit_profile = (
                "normal",
                "patient",
                self.config.pump_mode.market_context_normal_risk_multiplier,
                "normal",
            )
        return MarketState(
            phase,
            reasons=[f"legacy_{pump_regime.lower()}_history_warmup"],
            phase=phase,
            transition=self._transition(phase),
            risk_multiplier=risk_multiplier,
            entry_mode=entry_mode,
            exit_profile=exit_profile,
            metrics=metrics,
        )

    def _phase_context(self, metrics: dict[str, float], history: list[dict[str, float]], pump_regime: str) -> MarketState:
        cfg = self.config.pump_mode
        q = self._quantiles(history)
        risk_off = (
            metrics["heat_score"] <= q["heat_score"][0.2]
            and metrics["btc_ret_24h"] <= q["btc_ret_24h"][0.2]
            and metrics["sync_down_24h_ratio"] >= q["sync_down_24h_ratio"][0.8]
        ) or (
            metrics["heat_delta_24h"] <= q["heat_delta_24h"][0.2]
            and metrics["sync_down_24h_ratio"] >= q["sync_down_24h_ratio"][0.8]
        )
        crowded_fading = (
            metrics["ret24_gt10_ratio"] >= q["ret24_gt10_ratio"][0.8]
            and metrics["ret24_gt30_ratio"] >= q["ret24_gt30_ratio"][0.8]
            and metrics["heat_delta_24h"] < 0
        ) or (
            metrics["new_high_high_wick_ratio"] >= q["new_high_high_wick_ratio"][0.8]
            and metrics["heat_delta_24h"] < 0
        )
        expanding = (
            metrics["ret24_gt10_ratio"] >= q["ret24_gt10_ratio"][0.8]
            and metrics["heat_delta_24h"] >= q["heat_delta_24h"][0.8]
            and metrics["btc_ret_24h"] >= 0
        )
        crowded_hot = (
            metrics["ret24_gt10_ratio"] >= q["ret24_gt10_ratio"][0.8]
            and metrics["ret24_gt30_ratio"] >= q["ret24_gt30_ratio"][0.8]
            and metrics["heat_delta_24h"] >= 0
        )
        cold = (
            metrics["ret24_gt10_ratio"] <= q["ret24_gt10_ratio"][0.2]
            and metrics["new12h_high_ratio"] <= q["new12h_high_ratio"][0.2]
            and metrics["candidate_count"] <= q["candidate_count"][0.4]
        )

        if risk_off:
            phase = "risk_off"
            entry_mode = "none"
            risk_multiplier = 0.0
            exit_profile = "aggressive_tighten" if cfg.market_context_exit_tightening_enabled else "normal"
        elif expanding:
            phase, entry_mode, risk_multiplier, exit_profile = "expanding", "normal", 1.0, "normal"
        elif crowded_hot:
            phase = "crowded_hot"
            entry_mode = "normal"
            risk_multiplier = cfg.market_context_crowded_hot_risk_multiplier
            exit_profile = "light_tighten" if cfg.market_context_exit_tightening_enabled else "normal"
        elif crowded_fading:
            phase = "crowded_fading"
            entry_mode = "patient"
            risk_multiplier = cfg.market_context_crowded_fading_risk_multiplier
            exit_profile = "aggressive_tighten" if cfg.market_context_exit_tightening_enabled else "normal"
        elif cold:
            phase, entry_mode, risk_multiplier, exit_profile = "cold", "none", 0.0, "normal"
        else:
            phase = "normal"
            entry_mode = "patient"
            risk_multiplier = cfg.market_context_normal_risk_multiplier
            exit_profile = "tighten" if cfg.market_context_exit_tightening_enabled else "normal"

        transition = self._transition(phase)
        reasons = [
            f"phase={phase}",
            f"transition={transition}",
            f"old_regime={pump_regime}",
        ]
        return MarketState(
            phase,
            reasons=reasons,
            phase=phase,
            transition=transition,
            risk_multiplier=risk_multiplier,
            entry_mode=entry_mode,
            exit_profile=exit_profile,
            metrics=metrics,
        )

    def _quantiles(self, history: list[dict[str, float]]) -> dict[str, dict[float, float]]:
        fields = [
            "heat_score",
            "heat_delta_24h",
            "ret24_gt10_ratio",
            "ret24_gt30_ratio",
            "new12h_high_ratio",
            "new_high_high_wick_ratio",
            "btc_ret_24h",
            "sync_down_24h_ratio",
            "candidate_count",
        ]
        frame = pd.DataFrame(history)
        return {
            field: {q: float(frame[field].quantile(q)) for q in (0.2, 0.4, 0.8)}
            for field in fields
        }

    def _transition(self, phase: str) -> str:
        previous = self.previous_phase
        if phase == previous:
            return "stable"
        if phase == "risk_off" or (previous == "crowded_hot" and phase == "crowded_fading"):
            return "deteriorating"
        if phase == "expanding" and previous in {"normal", "cold", "crowded_fading", "crowded_hot"}:
            return "improving"
        if previous == "risk_off" and phase != "risk_off":
            return "recovery"
        return "stable"
