from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from crypto_quant.config.settings import UniverseConfig


class UniverseBuilder(Protocol):
    def build(self, symbols: pd.DataFrame) -> pd.DataFrame:
        ...


@dataclass(frozen=True)
class LiquidityUniverseBuilder:
    config: UniverseConfig
    quote_asset: str = "USDT"

    def build(self, symbols: pd.DataFrame) -> pd.DataFrame:
        frame = symbols.copy()
        frame["eligible"] = True
        frame["reason"] = ""
        quote_col = "quote_asset" if "quote_asset" in frame.columns else "quote"
        base_col = "base_asset" if "base_asset" in frame.columns else "base"
        volume_col = "quote_volume_30d"
        frame.loc[frame[quote_col] != self.quote_asset, ["eligible", "reason"]] = [False, "quote_asset"]
        if "status" in frame.columns:
            frame.loc[frame["status"] != "TRADING", ["eligible", "reason"]] = [False, "status"]
        if "spot" in frame.columns:
            frame.loc[~frame["spot"].astype(bool), ["eligible", "reason"]] = [False, "not_spot"]
        if "is_spot_trading_allowed" in frame.columns:
            frame.loc[
                ~frame["is_spot_trading_allowed"].astype(bool),
                ["eligible", "reason"],
            ] = [False, "not_spot"]
        for keyword in self.config.exclude_keywords:
            mask = frame[base_col].astype(str).str.upper().str.contains(keyword)
            frame.loc[mask, ["eligible", "reason"]] = [False, f"excluded:{keyword}"]
        # Exclude mega-cap coins from trading universe so altcoins get selected
        for cap in self.config.mega_cap_exclude:
            mask = frame[base_col].astype(str).str.upper() == cap.upper()
            frame.loc[mask, ["eligible", "reason"]] = [False, f"mega_cap:{cap}"]
        frame.loc[
            frame[volume_col].fillna(0) < self.config.min_quote_volume_30d,
            ["eligible", "reason"],
        ] = [False, "low_volume"]
        eligible = frame[frame["eligible"]].copy()
        eligible["liquidity_rank"] = eligible[volume_col].rank(ascending=False, method="first").astype(int)
        eligible = eligible.sort_values("liquidity_rank").head(self.config.top_n)
        return eligible
