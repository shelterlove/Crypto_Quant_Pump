from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    symbol: str
    side: str
    quantity: float
    expected_price: float
    filled_price: float
    fee: float
    slippage: float
    status: str
    reason: str


class Broker(ABC):
    @abstractmethod
    def execute_market(self, symbol: str, side: str, quantity: float, next_open: float, reason: str) -> Order:
        ...


@dataclass(frozen=True)
class BacktestBroker(Broker):
    fee_bps: float = 10
    slippage_bps: float = 5

    def execute_market(self, symbol: str, side: str, quantity: float, next_open: float, reason: str) -> Order:
        direction = 1 if side.lower() == "buy" else -1
        slippage = next_open * self.slippage_bps / 10_000 * direction
        filled = next_open + slippage
        fee = abs(quantity * filled) * self.fee_bps / 10_000
        return Order(symbol, side, quantity, next_open, filled, fee, abs(slippage), "filled", reason)
