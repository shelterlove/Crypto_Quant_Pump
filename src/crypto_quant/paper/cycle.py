from __future__ import annotations

import fcntl
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import TracebackType

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from crypto_quant.config.settings import AppConfig
from crypto_quant.data.sync import CandleSyncResult, CandleSyncService
from crypto_quant.paper.runner import PaperRunner, PaperRunResult
from crypto_quant.reporting import BacktestReportWriter
from crypto_quant.storage.candles import distinct_candle_symbols
from crypto_quant.storage.models import Candle
from crypto_quant.utils.time import ensure_utc, previous_closed_open_time, timeframe_delta


class PaperCycleLocked(RuntimeError):
    pass


class PaperCycleDataStale(RuntimeError):
    pass


class FileLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._handle: object | None = None

    def __enter__(self) -> FileLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            raise PaperCycleLocked(f"another paper cycle is running: {self.path}") from exc
        handle.write(f"{datetime.now(UTC).isoformat()}\n")
        handle.flush()
        self._handle = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._handle is None:
            return
        handle = self._handle
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
        self._handle = None


@dataclass(frozen=True)
class FreshnessStatus:
    latest_candle_time: datetime | None
    expected_candle_time: datetime
    stale: bool
    lag_seconds: int | None


@dataclass(frozen=True)
class PaperCycleResult:
    status: str
    run_id: int | None
    orders: int
    equity: float
    signal_time: datetime | None
    fill_time: datetime | None
    report_path: str | None
    state_path: str
    synced_rows: int
    sync_errors: int
    latest_candle_time: datetime | None
    expected_candle_time: datetime
    lag_seconds: int | None
    reason: str | None = None

    def to_json(self) -> str:
        def default(value: object) -> str:
            if isinstance(value, datetime):
                return ensure_utc(value).isoformat()
            return str(value)

        return json.dumps(asdict(self), default=default, sort_keys=True)


class PaperCycleRunner:
    def __init__(
        self,
        config: AppConfig,
        state_path: Path = Path("paper_state/main.json"),
        lock_path: Path = Path("paper_state/paper.lock"),
        report_dir: Path = Path("reports"),
        lookback_days: int = 120,
        sync_lookback_hours: int = 6,
        max_staleness_hours: int = 2,
        settle_minutes: int = 3,
        all_usdt_spot: bool = False,
        skip_sync: bool = False,
        allow_stale: bool = False,
    ) -> None:
        self.config = config
        self.state_path = state_path
        self.lock_path = lock_path
        self.report_dir = report_dir
        self.lookback_days = lookback_days
        self.sync_lookback_hours = sync_lookback_hours
        self.max_staleness_hours = max_staleness_hours
        self.settle_minutes = settle_minutes
        self.all_usdt_spot = all_usdt_spot
        self.skip_sync = skip_sync
        self.allow_stale = allow_stale

    def run(self, session: Session, now: datetime | None = None) -> PaperCycleResult:
        with FileLock(self.lock_path):
            cycle_now = ensure_utc(now or datetime.now(UTC))
            sync_result = CandleSyncResult()
            if not self.skip_sync:
                sync_result = self._sync_latest(session, cycle_now)
                session.commit()

            freshness = self._freshness(session, cycle_now)
            if freshness.stale and not self.allow_stale:
                raise PaperCycleDataStale(
                    "latest BTC candle is stale: "
                    f"latest={freshness.latest_candle_time} expected={freshness.expected_candle_time} "
                    f"lag_seconds={freshness.lag_seconds}"
                )

            paper = PaperRunner(self.config, state_path=self.state_path, lookback_days=self.lookback_days).run_once(session)
            session.commit()
            report_path = None
            if paper.strategy_run_id is not None:
                report_path = str(BacktestReportWriter(self.report_dir).write(session, paper.strategy_run_id).html)
                session.commit()
            return self._result(sync_result, freshness, paper, report_path)

    def _sync_latest(self, session: Session, now: datetime) -> CandleSyncResult:
        end = self._expected_candle_time(now)
        latest = self._latest_candle_time(session)
        if latest is None:
            start = end - timedelta(days=self.lookback_days)
        else:
            start = max(ensure_utc(latest) - timedelta(hours=self.sync_lookback_hours), end - timedelta(days=self.lookback_days))
        if start > end:
            return CandleSyncResult()
        symbols = self._sync_symbols(session)
        service = CandleSyncService(exchange=self.config.exchange_id)
        plan = service.build_plan(symbols, "1h", start, end)
        return service.run(session, plan)

    def _sync_symbols(self, session: Session) -> list[str]:
        if self.all_usdt_spot:
            symbols = CandleSyncService(exchange=self.config.exchange_id).resolve_symbols(None, all_usdt_spot=True)
        else:
            symbols = distinct_candle_symbols(session, self.config.exchange_id, "1h")
        return sorted(set(symbols) | {self.config.market_state.btc_symbol})

    def _freshness(self, session: Session, now: datetime) -> FreshnessStatus:
        latest = self._latest_candle_time(session)
        expected = self._expected_candle_time(now)
        if latest is None:
            return FreshnessStatus(None, expected, True, None)
        latest = ensure_utc(latest)
        lag_seconds = int((expected - latest).total_seconds())
        stale = lag_seconds > int(timedelta(hours=self.max_staleness_hours).total_seconds())
        return FreshnessStatus(latest, expected, stale, lag_seconds)

    def _latest_candle_time(self, session: Session) -> datetime | None:
        return session.execute(
            select(func.max(Candle.open_time))
            .where(Candle.exchange == self.config.exchange_id)
            .where(Candle.timeframe == "1h")
            .where(Candle.symbol == self.config.market_state.btc_symbol)
        ).scalar_one_or_none()

    def _expected_candle_time(self, now: datetime) -> datetime:
        settled_now = ensure_utc(now) - timedelta(minutes=self.settle_minutes)
        return previous_closed_open_time(settled_now + timeframe_delta("1h"), "1h")

    def _result(
        self,
        sync_result: CandleSyncResult,
        freshness: FreshnessStatus,
        paper: PaperRunResult,
        report_path: str | None,
    ) -> PaperCycleResult:
        status = "skipped" if paper.skipped else "completed"
        return PaperCycleResult(
            status=status,
            run_id=paper.strategy_run_id,
            orders=len(paper.orders),
            equity=paper.equity,
            signal_time=paper.processed_signal_time,
            fill_time=paper.fill_time,
            report_path=report_path,
            state_path=str(paper.state_path),
            synced_rows=sync_result.inserted_or_updated,
            sync_errors=len(sync_result.errors),
            latest_candle_time=freshness.latest_candle_time,
            expected_candle_time=freshness.expected_candle_time,
            lag_seconds=freshness.lag_seconds,
            reason=paper.reason,
        )
