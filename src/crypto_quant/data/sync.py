from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from sqlalchemy.orm import Session

from crypto_quant.data.binance import BinanceSpotDataProvider
from crypto_quant.data.quality import validate_candles
from crypto_quant.storage.candles import CandleCoverage, coverage, has_complete_candle_coverage, upsert_candles


@dataclass(frozen=True)
class CandleSyncPlan:
    exchange: str
    timeframe: str
    symbols: list[str]
    start: datetime
    end: datetime


@dataclass
class CandleSyncResult:
    inserted_or_updated: int = 0
    coverage: list[CandleCoverage] = field(default_factory=list)
    errors: dict[str, list[str]] = field(default_factory=dict)


class CandleSyncService:
    def __init__(self, provider: BinanceSpotDataProvider | None = None, exchange: str = "binance") -> None:
        self.provider = provider or BinanceSpotDataProvider(exchange)
        self.exchange = exchange

    def resolve_symbols(self, explicit_symbols: list[str] | None, all_usdt_spot: bool) -> list[str]:
        if explicit_symbols:
            return sorted(set(explicit_symbols))
        if all_usdt_spot:
            return self.provider.fetch_usdt_spot_symbols()
        return ["BTC/USDT"]

    def build_plan(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> CandleSyncPlan:
        return CandleSyncPlan(self.exchange, timeframe, sorted(set(symbols)), start, end)

    def run(self, session: Session, plan: CandleSyncPlan) -> CandleSyncResult:
        result = CandleSyncResult()
        for symbol in plan.symbols:
            if has_complete_candle_coverage(session, plan.exchange, symbol, plan.timeframe, plan.start, plan.end):
                logger.info("skipping {} {}: complete local coverage", symbol, plan.timeframe)
                continue
            logger.info("syncing {} {} {} -> {}", symbol, plan.timeframe, plan.start, plan.end)
            frame = self.provider.fetch_ohlcv_paginated(symbol, plan.timeframe, plan.start, plan.end)
            errors = validate_candles(frame) if not frame.empty else ["no_candles_returned"]
            if errors:
                result.errors[symbol] = errors
                if frame.empty:
                    continue
            result.inserted_or_updated += upsert_candles(session, plan.exchange, symbol, plan.timeframe, frame)
            result.coverage.append(coverage(frame, symbol, plan.timeframe))
        return result
