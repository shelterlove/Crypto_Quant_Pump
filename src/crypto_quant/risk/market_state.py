from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class MarketState:
    state: str
    fast_risk_valve: bool = False
    reasons: list[str] = field(default_factory=list)
    phase: str = "normal"
    transition: str = "stable"
    risk_multiplier: float = 1.0
    entry_mode: str = "normal"
    exit_profile: str = "normal"
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def allows_new_risk(self) -> bool:
        """Normal risk-on: full position sizing, all signals allowed."""
        return self.state == "risk_on" and not self.fast_risk_valve

    @property
    def allows_strong_only(self) -> bool:
        """Caution mode: only Top 1-2 signals with half risk."""
        return self.state == "caution" and not self.fast_risk_valve


def fast_risk_valve_triggered(
    btc_1h: pd.DataFrame | None = None,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if btc_1h is not None and len(btc_1h) >= 2:
        last_return = float(btc_1h["close"].iloc[-1] / btc_1h["close"].iloc[-2] - 1)
        if last_return <= -0.07:
            reasons.append("btc_1h_drop")
    return bool(reasons), reasons
