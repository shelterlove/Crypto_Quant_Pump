from __future__ import annotations

from dataclasses import dataclass

from crypto_quant.strategy.types import Signal, TargetPosition


@dataclass(frozen=True)
class FreqtradeSignalMapper:
    """Future mapping layer from core strategy decisions to Freqtrade callbacks."""

    def entry_signal(self, signal: Signal) -> bool:
        return signal.side == "buy"

    def custom_stake_amount(self, target: TargetPosition, equity: float) -> float:
        return max(target.target_weight * equity, 0.0)

    def custom_stoploss(self, target: TargetPosition) -> float:
        if target.entry_price <= 0:
            return 0.0
        return (target.stop_price / target.entry_price) - 1


class FreqtradeBrokerAdapter:
    """Skeleton only. Real Freqtrade order execution is outside phase 1."""

    def confirm_trade_entry(self, risk_vetoed: bool) -> bool:
        return not risk_vetoed
