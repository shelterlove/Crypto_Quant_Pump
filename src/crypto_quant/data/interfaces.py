from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd


class MarketDataProvider(ABC):
    @abstractmethod
    def get_candles(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Return closed OHLCV candles keyed by symbol."""
