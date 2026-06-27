from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, cast

import ccxt
import pandas as pd
import requests
from loguru import logger

from crypto_quant.data.interfaces import MarketDataProvider
from crypto_quant.utils.time import TIMEFRAME_MS, ensure_utc, previous_closed_open_time


class BinanceBaseDataProvider(MarketDataProvider):
    REQUEST_RETRIES = 20
    RETRY_BACKOFF_SECONDS = 1.0
    RETRYABLE_HTTP_STATUSES = {418, 429, 451, 500, 502, 503, 504}
    RETRYABLE_ERRORS = (
        requests.exceptions.ConnectionError,
        requests.exceptions.ProxyError,
        requests.exceptions.ReadTimeout,
        requests.exceptions.Timeout,
    )

    BASE_URLS: dict[str, str] = {
        "binance": "https://api.binance.com/api/v3",
        "binanceus": "https://api.binance.us/api/v3",
    }

    def __init__(self, exchange_id: str = "binance") -> None:
        exchange_class = getattr(ccxt, exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"unsupported ccxt exchange_id: {exchange_id}")
        if exchange_id not in self.BASE_URLS:
            raise ValueError(f"unsupported Binance-compatible exchange_id: {exchange_id}")
        self.base_url = self.BASE_URLS[exchange_id]
        self.session = requests.Session()
        self.session.proxies.update(self._proxy_options())
        options: dict[str, Any] = {"enableRateLimit": True}
        proxies = self._proxy_options()
        if proxies:
            options["proxies"] = proxies
        self.exchange = exchange_class(options)

    def _proxy_options(self) -> dict[str, str]:
        proxies: dict[str, str] = {}
        http_proxy = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
        https_proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
        if http_proxy:
            proxies["http"] = http_proxy
        if https_proxy:
            proxies["https"] = https_proxy
        return proxies

    def get_candles(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame]:
        since = int(start.timestamp() * 1000) if start else None
        data: dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            frame = self._fetch_klines(symbol, timeframe, since, int(end.timestamp() * 1000) if end else None)
            if end:
                frame = frame[frame["open_time"] <= pd.Timestamp(end)]
            data[symbol] = frame
        return data

    def fetch_ohlcv_paginated(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        limit: int = 1000,
    ) -> pd.DataFrame:
        start = ensure_utc(start)
        end = ensure_utc(end)
        since = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        step_ms = TIMEFRAME_MS[timeframe]
        rows: list[list[Any]] = []
        while since <= end_ms:
            batch = self._fetch_klines_raw(symbol, timeframe, since, end_ms, limit)
            if not batch:
                break
            for candle in batch:
                open_ms = int(candle[0])
                if open_ms > end_ms:
                    break
                rows.append(candle)
            next_since = int(batch[-1][0]) + step_ms
            if next_since <= since:
                break
            since = next_since
            if int(batch[-1][0]) >= end_ms:
                break
        frame = self._klines_to_frame(rows)
        if frame.empty:
            return frame
        closed_cutoff = previous_closed_open_time(datetime.now(tz=end.tzinfo), timeframe)
        frame = frame[(frame["open_time"] <= pd.Timestamp(end)) & (frame["open_time"] <= pd.Timestamp(closed_cutoff))]
        frame = frame.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
        return frame

    def _fetch_klines(self, symbol: str, timeframe: str, since: int | None, end_ms: int | None, limit: int = 1000) -> pd.DataFrame:
        return self._klines_to_frame(self._fetch_klines_raw(symbol, timeframe, since, end_ms, limit))

    def fetch_last_prices(self, symbols: list[str]) -> dict[str, float]:
        requested = {symbol.replace("/", ""): symbol for symbol in symbols}
        if not requested:
            return {}
        payload = self._get_json("/ticker/price")
        if not isinstance(payload, list):
            return {}
        prices: dict[str, float] = {}
        for row in payload:
            market_symbol = str(row.get("symbol", ""))
            mapped = requested.get(market_symbol)
            if mapped is None:
                continue
            try:
                prices[mapped] = float(row["price"])
            except (KeyError, TypeError, ValueError):
                continue
        return prices

    def fetch_open_prices_at(self, symbols: list[str], timeframe: str, open_time: datetime) -> dict[str, float]:
        requested = sorted(set(symbols))
        if not requested:
            return {}
        open_time = ensure_utc(open_time)
        open_ms = int(open_time.timestamp() * 1000)
        end_ms = open_ms + TIMEFRAME_MS[timeframe] - 1
        prices: dict[str, float] = {}
        for symbol in requested:
            batch = self._fetch_klines_raw(symbol, timeframe, open_ms, end_ms, limit=2)
            if not batch:
                continue
            candle = next((row for row in batch if int(row[0]) == open_ms), None)
            if candle is None:
                continue
            try:
                prices[symbol] = float(candle[1])
            except (TypeError, ValueError, IndexError):
                continue
        return prices

    def _fetch_klines_raw(
        self,
        symbol: str,
        timeframe: str,
        since: int | None,
        end_ms: int | None,
        limit: int = 1000,
    ) -> list[list[Any]]:
        params: dict[str, Any] = {
            "symbol": symbol.replace("/", ""),
            "interval": timeframe,
            "limit": limit,
        }
        if since is not None:
            params["startTime"] = since
        if end_ms is not None:
            params["endTime"] = end_ms
        return cast(list[list[Any]], self._get_json("/klines", params=params))

    def _get_json(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_error: requests.exceptions.RequestException | None = None
        for attempt in range(1, self.REQUEST_RETRIES + 1):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code not in self.RETRYABLE_HTTP_STATUSES or attempt >= self.REQUEST_RETRIES:
                    raise
                logger.warning("retrying Binance HTTP {} for {} attempt {}/{}", status_code, path, attempt, self.REQUEST_RETRIES)
                time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
            except self.RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt >= self.REQUEST_RETRIES:
                    raise
                logger.warning("retrying Binance request for {} attempt {}/{}: {}", path, attempt, self.REQUEST_RETRIES, exc)
                time.sleep(self.RETRY_BACKOFF_SECONDS * attempt)
        if last_error is not None:
            raise last_error
        raise RuntimeError("unreachable Binance request retry state")

    def _klines_to_frame(self, rows: list[list[Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["open_time", "open", "high", "low", "close", "volume", "quote_volume"])
        frame = frame.iloc[:, [0, 1, 2, 3, 4, 5, 7]]
        frame.columns = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]
        frame["open_time"] = pd.to_datetime(frame["open_time"], unit="ms", utc=True)
        for column in ["open", "high", "low", "close", "volume", "quote_volume"]:
            frame[column] = frame[column].astype(float)
        return frame


class BinanceSpotDataProvider(BinanceBaseDataProvider):
    BASE_URLS = {
        "binance": "https://api.binance.com/api/v3",
        "binanceus": "https://api.binance.us/api/v3",
    }

    def fetch_spot_symbols(self) -> list[dict[str, Any]]:
        markets = self._get_json("/exchangeInfo").get("symbols", [])
        rows: list[dict[str, Any]] = []
        for market in markets:
            base = market.get("baseAsset")
            quote = market.get("quoteAsset")
            symbol = f"{base}/{quote}"
            rows.append(
                {
                    "symbol": symbol,
                    "base": base,
                    "quote": quote,
                    "active": market.get("status") == "TRADING",
                    "spot": bool(market.get("isSpotTradingAllowed", False)),
                    "info": market,
                }
            )
        return rows

    def fetch_usdt_spot_symbols(self) -> list[str]:
        symbols: list[str] = []
        for row in self.fetch_spot_symbols():
            info = row.get("info") or {}
            status = str(info.get("status") or ("TRADING" if row.get("active") else ""))
            spot_allowed = bool(info.get("isSpotTradingAllowed", row.get("spot", False)))
            if row.get("quote") == "USDT" and status == "TRADING" and spot_allowed:
                symbols.append(str(row["symbol"]))
        return sorted(symbols)


class BinanceUsdmDataProvider(BinanceBaseDataProvider):
    BASE_URLS = {
        "binance_usdm": "https://fapi.binance.com/fapi/v1",
    }

    def __init__(self, exchange_id: str = "binance_usdm") -> None:
        if exchange_id not in self.BASE_URLS:
            raise ValueError(f"unsupported Binance-compatible exchange_id: {exchange_id}")
        self.base_url = self.BASE_URLS[exchange_id]
        self.session = requests.Session()
        self.session.proxies.update(self._proxy_options())
        options: dict[str, Any] = {"enableRateLimit": True}
        proxies = self._proxy_options()
        if proxies:
            options["proxies"] = proxies
        exchange_class = getattr(ccxt, "binanceusdm", None)
        self.exchange = exchange_class(options) if exchange_class is not None else None

    def fetch_usdt_perp_symbols(self) -> list[str]:
        markets = self._get_json("/exchangeInfo").get("symbols", [])
        symbols: list[str] = []
        for market in markets:
            if (
                market.get("quoteAsset") == "USDT"
                and market.get("contractType") == "PERPETUAL"
                and market.get("status") == "TRADING"
            ):
                symbols.append(f"{market.get('baseAsset')}/USDT")
        return sorted(symbols)


def binance_provider_for_exchange(exchange_id: str) -> BinanceBaseDataProvider:
    if exchange_id == "binance_usdm":
        return BinanceUsdmDataProvider(exchange_id)
    if exchange_id in BinanceSpotDataProvider.BASE_URLS:
        return BinanceSpotDataProvider(exchange_id)
    raise ValueError(f"unsupported Binance-compatible exchange_id: {exchange_id}")
