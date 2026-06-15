from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Signal:
    time: datetime | None
    symbol: str
    side: str
    rank: int
    target_weight: float
    reason: str


@dataclass(frozen=True)
class TargetPosition:
    symbol: str
    target_weight: float
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_price: float = 0.0
    atr: float = 0.0
    stop_risk_exposure: float = 0.0
    volatility_risk_exposure: float = 0.0


@dataclass(frozen=True)
class SwapRecommendation:
    """Recommend swapping out `sell_symbol` for `buy_target`."""
    sell_symbol: str
    buy_target: TargetPosition
    sell_rank: int
    buy_rank: int
    sell_score: float
    buy_score: float
    reason: str = "swap"
