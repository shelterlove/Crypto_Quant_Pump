"""False-breakout environment detector — White Paper §11.

Tracks top-signal forward performance to detect when "strong signals"
are systematically failing.  When active, reduces risk exposure.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import pandas as pd

from crypto_quant.config.settings import AppConfig


class FalseBreakoutState(Enum):
    NORMAL = "normal"
    ACTIVE = "active"


@dataclass
class SignalSnapshot:
    time: datetime
    symbol: str
    score: float
    entry_price: float | None = None
    return_8h: float | None = None


@dataclass
class FalseBreakoutDetector:
    """Tracks recent top signals and detects false-breakout regimes."""

    _snapshots: deque[SignalSnapshot] = field(default_factory=deque)
    _recent_trade_results: deque[bool] = field(default_factory=lambda: deque(maxlen=10))
    _state: FalseBreakoutState = FalseBreakoutState.NORMAL
    _last_evaluation: datetime | None = None
    _state_since: datetime | None = None
    _breadth_at_trigger: float | None = None

    def update(self, top_signals: pd.DataFrame, config: AppConfig, now: datetime | None = None) -> None:
        """Record the current set of top-ranked signals.

        Called each hour with the current factor score table.
        """
        if top_signals.empty:
            return

        fb_cfg = config.false_breakout
        if not fb_cfg.enabled:
            return

        now = now or datetime.now()

        # snapshot the current top-N signals for future return evaluation
        top = top_signals.head(fb_cfg.top_n).copy()
        for row in top.itertuples(index=False):
            self._snapshots.append(
                SignalSnapshot(
                    time=now,
                    symbol=str(row.symbol),
                    score=float(getattr(row, "final_score", getattr(row, "momentum_score", 0))),
                )
            )

        # cull old snapshots (beyond window)
        window = pd.Timedelta(hours=fb_cfg.signal_window_hours)
        cutoff = now - window
        while self._snapshots and self._snapshots[0].time < cutoff:
            self._snapshots.popleft()

        # evaluate
        self._evaluate(config)
        self._last_evaluation = now

    def record_trade_result(self, won: bool) -> None:
        self._recent_trade_results.append(won)

    def current_state(self) -> FalseBreakoutState:
        return self._state

    def _evaluate(self, config: AppConfig) -> None:
        fb_cfg = config.false_breakout

        if len(self._snapshots) < fb_cfg.min_samples:
            self._state = FalseBreakoutState.NORMAL
            return

        # Check 1: >50% of recent top signals have negative forward returns
        # (We can't know real forward returns in real-time, but in backtest
        # we key off recent closed-trade outcomes as a proxy)
        recent_loss_rate = 0.0
        if len(self._recent_trade_results) >= 3:
            recent_loss_rate = 1.0 - sum(self._recent_trade_results) / len(self._recent_trade_results)

        # Check 2: using the number of consecutive losses as proxy
        consecutive = 0
        for won in reversed(list(self._recent_trade_results)):
            if not won:
                consecutive += 1
            else:
                break

        triggers = 0
        if recent_loss_rate >= fb_cfg.negative_signal_threshold_pct:
            triggers += 1
        if consecutive >= 3:
            triggers += 1

        if triggers >= (1 if consecutive >= 4 else 2):
            if self._state != FalseBreakoutState.ACTIVE:
                self._state_since = datetime.now()
            self._state = FalseBreakoutState.ACTIVE
        else:
            self._state = FalseBreakoutState.NORMAL
            self._state_since = None
