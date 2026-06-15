from __future__ import annotations

from dataclasses import dataclass, field

from crypto_quant.config.settings import RiskConfig
from crypto_quant.strategy.types import TargetPosition


@dataclass(frozen=True)
class RiskDecision:
    approved: list[TargetPosition] = field(default_factory=list)
    rejected: list[tuple[TargetPosition, str]] = field(default_factory=list)


@dataclass(frozen=True)
class RiskEngine:
    config: RiskConfig

    def atr_position_size(self, equity: float, entry_price: float, atr: float) -> tuple[float, float, float]:
        stop_distance = atr * self.config.atr_stop_multiple
        if entry_price <= 0 or stop_distance <= 0:
            return 0.0, 0.0, 0.0
        max_trade_loss = equity * self.config.trade_risk_pct
        quantity_by_risk = max_trade_loss / stop_distance
        quantity_by_cap = equity * self.config.max_symbol_position_pct / entry_price
        quantity = min(quantity_by_risk, quantity_by_cap)
        stop_price = entry_price - stop_distance
        target_weight = quantity * entry_price / equity
        return quantity, stop_price, target_weight

    def approve_targets(
        self,
        targets: list[TargetPosition],
        portfolio_stop_risk: float = 0.0,
        portfolio_volatility_risk: float = 0.0,
    ) -> RiskDecision:
        approved: list[TargetPosition] = []
        rejected: list[tuple[TargetPosition, str]] = []
        if portfolio_stop_risk > self.config.max_portfolio_stop_risk_pct:
            return RiskDecision(rejected=[(target, "portfolio_stop_risk_limit") for target in targets])
        if portfolio_volatility_risk > self.config.max_portfolio_volatility_risk_pct:
            return RiskDecision(rejected=[(target, "portfolio_volatility_risk_limit") for target in targets])
        for target in targets:
            if target.volatility_risk_exposure > self.config.max_single_volatility_risk_pct:
                rejected.append((target, "single_volatility_risk_limit"))
            elif len(approved) >= self.config.max_positions:
                rejected.append((target, "max_positions"))
            else:
                approved.append(target)
        return RiskDecision(approved=approved, rejected=rejected)
