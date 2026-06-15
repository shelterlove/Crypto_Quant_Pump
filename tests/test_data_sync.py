from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import requests

from crypto_quant.data.binance import BinanceSpotDataProvider
from crypto_quant.data.sync import CandleSyncService


class FakeProvider:
    def fetch_usdt_spot_symbols(self) -> list[str]:
        return ["BTC/USDT", "ETH/USDT"]

    def fetch_ohlcv_paginated(self, symbol: str, timeframe: str, start: datetime, end: datetime) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open_time": pd.date_range(start, periods=3, freq="h", tz="UTC"),
                "open": [1, 2, 3],
                "high": [2, 3, 4],
                "low": [0.5, 1.5, 2.5],
                "close": [1.5, 2.5, 3.5],
                "volume": [10, 10, 10],
                "quote_volume": [15, 25, 35],
            }
        )


def test_candle_sync_plan_resolves_all_usdt_symbols() -> None:
    service = CandleSyncService(provider=FakeProvider(), exchange="binance")  # type: ignore[arg-type]
    symbols = service.resolve_symbols(None, all_usdt_spot=True)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    plan = service.build_plan(symbols, "1h", start, start + timedelta(hours=2))
    assert plan.symbols == ["BTC/USDT", "ETH/USDT"]
    assert plan.timeframe == "1h"


def test_binance_spot_data_provider_selects_configured_exchange() -> None:
    provider = BinanceSpotDataProvider("binanceus")
    assert provider.exchange.id == "binanceus"


def test_binance_spot_data_provider_passes_proxy_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:10808")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:10808")
    provider = BinanceSpotDataProvider("binanceus")
    assert provider.exchange.proxies == {
        "http": "http://127.0.0.1:10808",
        "https": "http://127.0.0.1:10808",
    }


def test_binance_spot_data_provider_rejects_unknown_exchange() -> None:
    with pytest.raises(ValueError, match="unsupported ccxt exchange_id"):
        BinanceSpotDataProvider("not_an_exchange")


def test_binance_spot_data_provider_retries_transient_request_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = BinanceSpotDataProvider("binanceus")
    monkeypatch.setattr(provider, "RETRY_BACKOFF_SECONDS", 0)
    calls = 0

    class FakeResponse:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, list[dict[str, object]]]:
            return {
                "symbols": [
                    {
                        "baseAsset": "BTC",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "isSpotTradingAllowed": True,
                    }
                ]
            }

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise requests.exceptions.ReadTimeout("temporary read timeout")
        return FakeResponse()

    monkeypatch.setattr(provider.session, "get", fake_get)

    assert provider.fetch_usdt_spot_symbols() == ["BTC/USDT"]
    assert calls == 2


def test_binance_spot_data_provider_retries_retryable_http_status(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = BinanceSpotDataProvider("binanceus")
    monkeypatch.setattr(provider, "RETRY_BACKOFF_SECONDS", 0)
    calls = 0

    class FakeResponse:
        def __init__(self, status_code: int) -> None:
            self.status_code = status_code

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise requests.exceptions.HTTPError(response=self)

        def json(self) -> dict[str, list[dict[str, object]]]:
            return {
                "symbols": [
                    {
                        "baseAsset": "ETH",
                        "quoteAsset": "USDT",
                        "status": "TRADING",
                        "isSpotTradingAllowed": True,
                    }
                ]
            }

    def fake_get(*args: object, **kwargs: object) -> FakeResponse:
        nonlocal calls
        calls += 1
        return FakeResponse(451 if calls == 1 else 200)

    monkeypatch.setattr(provider.session, "get", fake_get)

    assert provider.fetch_usdt_spot_symbols() == ["ETH/USDT"]
    assert calls == 2
