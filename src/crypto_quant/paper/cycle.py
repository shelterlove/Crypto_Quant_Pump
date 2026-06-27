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
from crypto_quant.paper.monitor import PaperMonitorWriter
from crypto_quant.paper.runner import PaperRunner, PaperRunResult
from crypto_quant.reporting import BacktestReportWriter, RunSummaryBuilder
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
    buys: int
    sells: int
    candidates: int
    open_positions: int
    equity: float
    pnl: float
    return_pct: float
    signal_time: datetime | None
    fill_time: datetime | None
    report_path: str | None
    report_exists: bool
    state_path: str
    working_dir: str
    synced_rows: int
    sync_errors: int
    sync_error_symbols: list[str]
    latest_candle_time: datetime | None
    expected_candle_time: datetime
    lag_seconds: int | None
    latest_db_run_id: int | None = None
    latest_db_run_status: str | None = None
    last_completed_run_id: int | None = None
    last_completed_report_path: str | None = None
    last_completed_report_exists: bool = False
    market_phase: str | None = None
    market_entry_mode: str | None = None
    pump_regime: str | None = None
    reason: str | None = None

    def to_json(self) -> str:
        def default(value: object) -> str:
            if isinstance(value, datetime):
                return ensure_utc(value).isoformat()
            return str(value)

        return json.dumps(asdict(self), default=default, sort_keys=True)

    def to_text(self) -> str:
        report = self.report_path or "-"
        reason = self.reason or "-"
        lag = "-" if self.lag_seconds is None else f"{self.lag_seconds}s"
        return "\n".join(
            [
                f"status={self.status} run_id={self.run_id} reason={reason}",
                f"equity={self.equity:.2f} pnl={self.pnl:.2f} return={self.return_pct:.2%}",
                f"orders={self.orders} buys={self.buys} sells={self.sells} "
                f"candidates={self.candidates} open_positions={self.open_positions}",
                f"market_phase={self.market_phase or '-'} entry_mode={self.market_entry_mode or '-'} pump_regime={self.pump_regime or '-'}",
                f"signal_time={self.signal_time} fill_time={self.fill_time}",
                f"latest_candle={self.latest_candle_time} expected_candle={self.expected_candle_time} lag={lag}",
                f"synced_rows={self.synced_rows} sync_errors={self.sync_errors}",
                f"sync_error_symbols={','.join(self.sync_error_symbols) or '-'}",
                f"report={report} report_exists={self.report_exists}",
                f"latest_db_run_id={self.latest_db_run_id} latest_db_run_status={self.latest_db_run_status or '-'}",
                f"last_completed_run_id={self.last_completed_run_id} "
                f"last_completed_report={self.last_completed_report_path or '-'} "
                f"last_completed_report_exists={self.last_completed_report_exists}",
                f"state={self.state_path}",
                f"working_dir={self.working_dir}",
            ]
        )


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
        self.summary_builder = RunSummaryBuilder(report_dir)

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
            report_exists = False
            if paper.strategy_run_id is not None and paper.orders:
                report_file = BacktestReportWriter(self.report_dir).write(session, paper.strategy_run_id).html.resolve()
                report_path = str(report_file)
                report_exists = report_file.exists()
                if not report_exists:
                    raise RuntimeError(f"report file missing after write: {report_file}")
                session.commit()
            result = self._result(session, sync_result, freshness, paper, report_path, report_exists)
            self._write_monitor_files(session, result)
            return result

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
        service = CandleSyncService(exchange=self.config.exchange_id)
        if self.all_usdt_spot:
            symbols = service.resolve_symbols(None, all_usdt_spot=True)
        else:
            local_symbols = set(distinct_candle_symbols(session, self.config.exchange_id, "1h"))
            active_symbols = set(service.resolve_symbols(None, all_usdt_spot=True))
            symbols = sorted(local_symbols & active_symbols)
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
        session: Session,
        sync_result: CandleSyncResult,
        freshness: FreshnessStatus,
        paper: PaperRunResult,
        report_path: str | None,
        report_exists: bool,
    ) -> PaperCycleResult:
        status = "skipped" if paper.skipped else "completed"
        buys = sum(1 for order in paper.orders if order.side == "buy")
        sells = sum(1 for order in paper.orders if order.side == "sell")
        initial_equity = self.config.backtest.initial_equity
        pnl = paper.equity - initial_equity
        return_pct = pnl / initial_equity if initial_equity else 0.0
        latest_db_run = self.summary_builder.latest_run(session, "paper")
        last_completed_run = self.summary_builder.latest_completed_run(session, "paper")
        return PaperCycleResult(
            status=status,
            run_id=paper.strategy_run_id,
            orders=len(paper.orders),
            buys=buys,
            sells=sells,
            candidates=paper.candidate_count,
            open_positions=paper.open_positions,
            equity=paper.equity,
            pnl=pnl,
            return_pct=return_pct,
            signal_time=paper.processed_signal_time,
            fill_time=paper.fill_time,
            report_path=report_path,
            report_exists=report_exists,
            state_path=str(paper.state_path.resolve()),
            working_dir=str(Path.cwd()),
            synced_rows=sync_result.inserted_or_updated,
            sync_errors=len(sync_result.errors),
            sync_error_symbols=sorted(sync_result.errors),
            latest_candle_time=freshness.latest_candle_time,
            expected_candle_time=freshness.expected_candle_time,
            lag_seconds=freshness.lag_seconds,
            latest_db_run_id=latest_db_run.run_id if latest_db_run is not None else None,
            latest_db_run_status=latest_db_run.status if latest_db_run is not None else None,
            last_completed_run_id=last_completed_run.run_id if last_completed_run is not None else None,
            last_completed_report_path=last_completed_run.report_path if last_completed_run is not None else None,
            last_completed_report_exists=last_completed_run.report_exists if last_completed_run is not None else False,
            market_phase=paper.market_phase,
            market_entry_mode=paper.market_entry_mode,
            pump_regime=paper.pump_regime,
            reason=paper.reason,
        )

    def _write_monitor_files(self, session: Session, result: PaperCycleResult) -> None:
        status_dir = self.state_path.parent
        status_dir.mkdir(parents=True, exist_ok=True)
        (status_dir / "latest_status.json").write_text(result.to_json() + "\n", encoding="utf-8")
        (status_dir / "latest_status.txt").write_text(result.to_text() + "\n", encoding="utf-8")
        PaperMonitorWriter(status_dir, self.report_dir).write(session, asdict(result))
