from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.storage.candles import load_candles
from crypto_quant.storage.models import UniverseMember, UniverseSnapshot
from crypto_quant.universe.builder import LiquidityUniverseBuilder
from crypto_quant.utils.time import ensure_utc, monday_utc


@dataclass(frozen=True)
class WeeklyUniverse:
    effective_from: datetime
    symbols: list[str]


@dataclass
class UniverseBuildResult:
    snapshots: list[WeeklyUniverse] = field(default_factory=list)
    candidate_union: set[str] = field(default_factory=set)


class WeeklyUniverseService:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.builder = LiquidityUniverseBuilder(config.universe, config.base_currency)

    def build(
        self,
        session: Session,
        symbols: list[str],
        start: datetime,
        end: datetime,
        persist: bool = True,
    ) -> UniverseBuildResult:
        start = monday_utc(start)
        end = ensure_utc(end)
        result = UniverseBuildResult()
        week = start
        while week <= end:
            universe = self._build_for_week(session, symbols, week)
            selected = universe["symbol"].astype(str).tolist() if not universe.empty else []
            result.snapshots.append(WeeklyUniverse(week, selected))
            result.candidate_union.update(selected)
            if persist:
                self._persist_week(session, week, universe)
            week += timedelta(days=7)
        return result

    def load_effective_map(self, session: Session, start: datetime, end: datetime) -> dict[datetime, list[str]]:
        rows = (
            session.execute(
                select(UniverseSnapshot)
                .where(UniverseSnapshot.effective_from >= monday_utc(start))
                .where(UniverseSnapshot.effective_from <= ensure_utc(end))
                .order_by(UniverseSnapshot.effective_from)
            )
            .scalars()
            .all()
        )
        return {
            snapshot.effective_from: [member.symbol for member in sorted(snapshot.members, key=lambda item: item.liquidity_rank)]
            for snapshot in rows
        }

    def _build_for_week(self, session: Session, symbols: list[str], effective_from: datetime) -> pd.DataFrame:
        lookback_start = effective_from - timedelta(days=30)
        lookback_end = effective_from - timedelta(days=1)
        daily = load_candles(session, self.config.exchange_id, symbols, "1d", lookback_start, lookback_end)
        rows: list[dict[str, object]] = []
        for symbol, frame in daily.items():
            if frame.empty or len(frame) < 5:
                continue
            base, quote = symbol.split("/", 1)
            rows.append(
                {
                    "symbol": symbol,
                    "base": base,
                    "quote": quote,
                    "status": "TRADING",
                    "spot": True,
                    "quote_volume_30d": float(frame["quote_volume"].tail(30).mean()),
                }
            )
        if not rows:
            return pd.DataFrame(columns=["symbol", "quote_volume_30d", "liquidity_rank"])
        return self.builder.build(pd.DataFrame(rows))

    def _persist_week(self, session: Session, effective_from: datetime, universe: pd.DataFrame) -> None:
        existing_ids = session.execute(
            select(UniverseSnapshot.id)
            .where(UniverseSnapshot.exchange == self.config.exchange_id)
            .where(UniverseSnapshot.effective_from == effective_from)
        ).scalars()
        for snapshot_id in existing_ids:
            session.execute(delete(UniverseMember).where(UniverseMember.snapshot_id == snapshot_id))
            session.execute(delete(UniverseSnapshot).where(UniverseSnapshot.id == snapshot_id))
        snapshot = UniverseSnapshot(
            exchange=self.config.exchange_id,
            snapshot_time=effective_from,
            effective_from=effective_from,
            config_hash=self.config.stable_hash(),
        )
        if session.get_bind().dialect.name == "sqlite":
            snapshot.id = int(session.execute(select(func.max(UniverseSnapshot.id))).scalar() or 0) + 1
        session.add(snapshot)
        session.flush()
        next_member_id = int(session.execute(select(func.max(UniverseMember.id))).scalar() or 0) + 1
        for row in universe.itertuples(index=False):
            member = UniverseMember(
                snapshot_id=snapshot.id,
                symbol=str(row.symbol),
                liquidity_rank=int(row.liquidity_rank),
                quote_volume_30d=float(row.quote_volume_30d),
                eligible=True,
                reason=None,
            )
            if session.get_bind().dialect.name == "sqlite":
                member.id = next_member_id
                next_member_id += 1
            session.add(member)
