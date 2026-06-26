from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
import requests

from crypto_quant.data.binance import BinanceSpotDataProvider, BinanceUsdmDataProvider, binance_provider_for_exchange
from crypto_quant.data.sync import CandleSyncService
from crypto_quant.storage.candles import distinct_candle_symbols, upsert_candles


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


class FakeFuturesProvider(FakeProvider):
    def fetch_usdt_perp_symbols(self) -> list[str]:
        return ["BTC/USDT", "1000PEPE/USDT"]


def test_candle_sync_plan_resolves_all_usdt_symbols() -> None:
    service = CandleSyncService(provider=FakeProvider(), exchange="binance")  # type: ignore[arg-type]
    symbols = service.resolve_symbols(None, all_usdt_spot=True)
    start = datetime(2024, 1, 1, tzinfo=UTC)
    plan = service.build_plan(symbols, "1h", start, start + timedelta(hours=2))
    assert plan.symbols == ["BTC/USDT", "ETH/USDT"]
    assert plan.timeframe == "1h"


def test_candle_sync_plan_resolves_all_usdt_perps_for_usdm() -> None:
    service = CandleSyncService(provider=FakeFuturesProvider(), exchange="binance_usdm")  # type: ignore[arg-type]
    assert service.resolve_symbols(None, all_usdt_spot=True) == ["1000PEPE/USDT", "BTC/USDT"]


def test_binance_provider_factory_selects_futures_provider() -> None:
    assert isinstance(binance_provider_for_exchange("binance_usdm"), BinanceUsdmDataProvider)
    with pytest.raises(ValueError, match="unsupported Binance-compatible exchange_id"):
        binance_provider_for_exchange("not_supported")


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


def test_binance_usdm_provider_parses_only_trading_usdt_perpetuals(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = BinanceUsdmDataProvider("binance_usdm")

    def fake_json(path: str, params: dict[str, object] | None = None) -> dict[str, list[dict[str, object]]]:
        assert path == "/exchangeInfo"
        assert params is None
        return {
            "symbols": [
                {"baseAsset": "BTC", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "TRADING"},
                {"baseAsset": "ETH", "quoteAsset": "USDT", "contractType": "CURRENT_QUARTER", "status": "TRADING"},
                {"baseAsset": "BNB", "quoteAsset": "USDC", "contractType": "PERPETUAL", "status": "TRADING"},
                {"baseAsset": "XRP", "quoteAsset": "USDT", "contractType": "PERPETUAL", "status": "BREAK"},
            ]
        }

    monkeypatch.setattr(provider, "_get_json", fake_json)

    assert provider.fetch_usdt_perp_symbols() == ["BTC/USDT"]


def test_binance_usdm_kline_fields_match_candle_schema() -> None:
    provider = BinanceUsdmDataProvider("binance_usdm")
    frame = provider._klines_to_frame(
        [
            [
                1_704_067_200_000,
                "1.0",
                "2.0",
                "0.5",
                "1.5",
                "10.0",
                1_704_070_799_999,
                "15.0",
            ]
        ]
    )

    assert list(frame.columns) == ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
    assert frame["quote_volume"].iloc[0] == 15.0


def test_candles_for_spot_and_usdm_same_symbol_can_coexist(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    start = datetime(2024, 1, 1, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "open_time": pd.date_range(start, periods=1, freq="h", tz="UTC"),
            "open": [1],
            "high": [2],
            "low": [1],
            "close": [1.5],
            "volume": [10],
            "quote_volume": [15],
        }
    )

    upsert_candles(sqlite_session, "binance", "BTC/USDT", "1h", frame)
    upsert_candles(sqlite_session, "binance_usdm", "BTC/USDT", "1h", frame)
    sqlite_session.commit()

    assert distinct_candle_symbols(sqlite_session, "binance", "1h") == ["BTC/USDT"]
    assert distinct_candle_symbols(sqlite_session, "binance_usdm", "1h") == ["BTC/USDT"]


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
