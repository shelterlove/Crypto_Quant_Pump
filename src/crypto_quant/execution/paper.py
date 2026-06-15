from __future__ import annotations

from crypto_quant.execution.broker import Broker, Order


class PaperBroker(Broker):
    def execute_market(self, symbol: str, side: str, quantity: float, next_open: float, reason: str) -> Order:
        return Order(symbol, side, quantity, next_open, next_open, 0.0, 0.0, "accepted", reason)
