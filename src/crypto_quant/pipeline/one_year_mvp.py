from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from crypto_quant.backtest.runner import BacktestResult, ResearchBacktester
from crypto_quant.config.settings import AppConfig
from crypto_quant.data.sync import CandleSyncResult, CandleSyncService
from crypto_quant.reporting import BacktestReportWriter, ReportPaths
from crypto_quant.storage.candles import distinct_candle_symbols
from crypto_quant.storage.models import Candle, EquityCurveRecord, OrderRecord, StrategyRun
from crypto_quant.universe.service import UniverseBuildResult, WeeklyUniverseService
from crypto_quant.utils.time import default_one_year_window, ensure_utc

REQUIRED_TABLES = {
    "candles",
    "universe_snapshots",
    "universe_members",
    "strategy_runs",
    "orders",
    "positions",
    "equity_curve",
    "signals",
    "rejected_signals",
    "factor_scores",
    "market_state",
}


class PipelineError(RuntimeError):
    pass


class BacktesterProtocol(Protocol):
    def run_real(self, session: Session, start: datetime, end: datetime, report_dir: Path | None = None) -> BacktestResult:
        pass


class ReportWriterProtocol(Protocol):
    def write(self, session: Session, strategy_run_id: int) -> ReportPaths:
        pass

    def write_overview(self, run_ids: list[int], rows: list[dict[str, object]], warnings: list[str] | None = None) -> ReportPaths:
        pass


@dataclass(frozen=True)
class PipelineStep:
    name: str
    detail: str


@dataclass(frozen=True)
class DataQualitySummary:
    timeframe: str
    symbol_count: int
    row_count: int
    missing_intervals: int


@dataclass(frozen=True)
class PipelineRunSummary:
    label: str
    run_id: int
    report_html: Path
    final_equity: float


@dataclass
class RunOneYearMvpResult:
    start: datetime
    end: datetime
    daily_start: datetime
    dry_run: bool = False
    steps: list[PipelineStep] = field(default_factory=list)
    run_summaries: list[PipelineRunSummary] = field(default_factory=list)
    data_quality: list[DataQualitySummary] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    overview: ReportPaths | None = None

    @property
    def run_ids(self) -> list[int]:
        return [summary.run_id for summary in self.run_summaries]


class RunOneYearMvpPipeline:
    def __init__(
        self,
        config: AppConfig,
        sync_service: CandleSyncService | None = None,
        universe_service: WeeklyUniverseService | None = None,
        backtester_factory: Callable[[AppConfig], BacktesterProtocol] | None = None,
        report_writer: ReportWriterProtocol | None = None,
        now: datetime | None = None,
    ) -> None:
        self.config = config
        self.sync_service = sync_service or CandleSyncService(exchange=config.exchange_id)
        self.universe_service = universe_service or WeeklyUniverseService(config)
        self.backtester_factory = backtester_factory or (lambda cfg: ResearchBacktester(cfg))
        self.report_writer = report_writer or BacktestReportWriter()
        self.now = now

    def build_dry_run(self, start: datetime | None = None, end: datetime | None = None) -> RunOneYearMvpResult:
        start_dt, end_dt = self._window(start, end)
        daily_start = start_dt - timedelta(days=30)
        return RunOneYearMvpResult(
            start=start_dt,
            end=end_dt,
            daily_start=daily_start,
            dry_run=True,
            steps=self._planned_steps(start_dt, end_dt, daily_start),
        )

    def run(self, session: Session, start: datetime | None = None, end: datetime | None = None) -> RunOneYearMvpResult:
        start_dt, end_dt = self._window(start, end)
        daily_start = start_dt - timedelta(days=30)
        result = RunOneYearMvpResult(
            start=start_dt,
            end=end_dt,
            daily_start=daily_start,
            steps=self._planned_steps(start_dt, end_dt, daily_start),
        )

        spot_symbols = self._preflight(session)
        daily_sync = self._sync(session, spot_symbols, "1d", daily_start, end_dt)
        if daily_sync.inserted_or_updated == 0 and not distinct_candle_symbols(session, self.config.exchange_id, "1d"):
            raise PipelineError("universe stage requires 1d candles; sync returned no rows")

        daily_symbols = distinct_candle_symbols(session, self.config.exchange_id, "1d")
        if not daily_symbols:
            raise PipelineError("universe stage requires 1d candles; run the 1d sync first")
        universe = self.universe_service.build(session, daily_symbols, start_dt, end_dt, persist=True)
        session.commit()
        if not universe.candidate_union:
            raise PipelineError("weekly universe is empty; check 1d quote_volume coverage and liquidity filters")

        candidate_symbols = self.candidate_symbols(universe, self.config.market_state.btc_symbol)
        for timeframe in ["1h", "4h"]:
            self._sync(session, candidate_symbols, timeframe, start_dt, end_dt)

        quality, warnings = self._data_quality(session, candidate_symbols, start_dt, end_dt)
        result.data_quality = quality
        result.warnings.extend(warnings)
        self._assert_btc_ready(session, start_dt, end_dt)

        run_rows: list[dict[str, object]] = []
        for label, run_cfg in self._run_configs():
            backtest = self.backtester_factory(run_cfg)
            backtest_result = backtest.run_real(session, start_dt, end_dt)
            session.commit()
            run_id = backtest_result.strategy_run_id
            if run_id is None:
                raise PipelineError(f"{label} backtest did not produce a strategy_run_id")
            paths = self.report_writer.write(session, run_id)
            session.commit()
            final_equity = self._final_equity(session, run_id, backtest_result.final_equity)
            result.run_summaries.append(PipelineRunSummary(label, run_id, paths.html, final_equity))
            run_rows.append(self._overview_row(session, run_id, label, run_cfg.backtest.slippage_bps, final_equity))

        result.overview = self.report_writer.write_overview(result.run_ids, run_rows, result.warnings)
        return result

    def candidate_symbols(self, universe: UniverseBuildResult, btc_symbol: str) -> list[str]:
        return sorted(set(universe.candidate_union) | {btc_symbol})

    def _window(self, start: datetime | None, end: datetime | None) -> tuple[datetime, datetime]:
        default_start, default_end = default_one_year_window(self.now or datetime.now(UTC))
        return ensure_utc(start or default_start), ensure_utc(end or default_end)

    def _planned_steps(self, start: datetime, end: datetime, daily_start: datetime) -> list[PipelineStep]:
        return [
            PipelineStep("preflight", f"check PostgreSQL connection, schema, {self.config.exchange_id} API, and UTC one-year window"),
            PipelineStep(
                "sync_1d",
                f"sync all {self.config.exchange_id} USDT spot 1d candles from {daily_start.isoformat()} to {end.isoformat()}",
            ),
            PipelineStep(
                "build_universe",
                f"build weekly Top{self.config.universe.top_n} universe snapshots from {start.isoformat()} to {end.isoformat()}",
            ),
            PipelineStep("sync_candidates", "sync 1h and 4h candles for weekly Top60 union plus BTC/USDT"),
            PipelineStep("data_quality", "summarize symbol counts, row counts, and missing candle intervals"),
            PipelineStep("backtest", self._backtest_step_detail()),
            PipelineStep("reports", "write per-run CSV/HTML reports, signal-quality diagnostics, and one slippage overview CSV/HTML"),
        ]

    def _preflight(self, session: Session) -> list[str]:
        session.execute(text("select 1"))
        tables = set(inspect(session.get_bind()).get_table_names())
        missing = sorted(REQUIRED_TABLES - tables)
        if missing:
            raise PipelineError(f"database schema is not upgraded; missing tables: {', '.join(missing)}")
        local_symbols = distinct_candle_symbols(session, self.config.exchange_id, "1d")
        if self.config.pipeline.prefer_local_symbol_cache and local_symbols:
            return sorted(local_symbols)
        try:
            symbols = self.sync_service.resolve_symbols(None, all_usdt_spot=True)
        except Exception as exc:
            if local_symbols:
                return sorted(local_symbols)
            raise PipelineError(f"Binance USDT spot symbol lookup failed and no local symbol cache is available: {exc}") from exc
        if not symbols:
            raise PipelineError("Binance USDT spot symbol lookup returned no symbols")
        return symbols

    def _sync(self, session: Session, symbols: list[str], timeframe: str, start: datetime, end: datetime) -> CandleSyncResult:
        plan = self.sync_service.build_plan(symbols, timeframe, start, end)
        sync_result = self.sync_service.run(session, plan)
        session.commit()
        return sync_result

    def _data_quality(
        self,
        session: Session,
        symbols: list[str],
        start: datetime,
        end: datetime,
    ) -> tuple[list[DataQualitySummary], list[str]]:
        summaries: list[DataQualitySummary] = []
        warnings: list[str] = []
        for timeframe in ["1h", "4h"]:
            rows = session.execute(
                select(
                    Candle.symbol,
                    func.count(Candle.id),
                    func.min(Candle.open_time),
                    func.max(Candle.open_time),
                )
                .where(Candle.exchange == self.config.exchange_id)
                .where(Candle.symbol.in_(symbols))
                .where(Candle.timeframe == timeframe)
                .where(Candle.open_time >= ensure_utc(start))
                .where(Candle.open_time <= ensure_utc(end))
                .group_by(Candle.symbol)
            ).all()
            symbol_count = len(rows)
            row_count = sum(int(row[1]) for row in rows)
            missing = sum(
                self._missing_interval_count(timeframe, int(row[1]), row[2], row[3])
                for row in rows
                if row[2] is not None and row[3] is not None
            )
            summaries.append(DataQualitySummary(timeframe, symbol_count, row_count, missing))
            if missing:
                warnings.append(f"{timeframe}: detected {missing} missing intervals across candidate symbols")
        return summaries, warnings

    def _assert_btc_ready(self, session: Session, start: datetime, end: datetime) -> None:
        btc = self.config.market_state.btc_symbol
        missing = []
        for timeframe in ["1h", "4h"]:
            count = session.execute(
                select(func.count(Candle.id))
                .where(Candle.exchange == self.config.exchange_id)
                .where(Candle.symbol == btc)
                .where(Candle.timeframe == timeframe)
                .where(Candle.open_time >= ensure_utc(start))
                .where(Candle.open_time <= ensure_utc(end))
            ).scalar_one()
            if int(count) == 0:
                missing.append(timeframe)
        if missing:
            raise PipelineError(f"missing required BTC/USDT candles for backtest: {', '.join(missing)}")

    def _missing_interval_count(self, timeframe: str, row_count: int, start: datetime, end: datetime) -> int:
        step_seconds = {"1d": 86_400, "4h": 14_400, "1h": 3_600, "15m": 900}[timeframe]
        start_dt = ensure_utc(start)
        end_dt = ensure_utc(end)
        expected = int((end_dt - start_dt).total_seconds() // step_seconds) + 1
        return max(expected - row_count, 0)

    def _run_configs(self) -> list[tuple[str, AppConfig]]:
        base = self.config.model_copy(update={"backtest": self.config.backtest.model_copy(update={"cost_mode": "basic"})})
        configs = [("basic", base)]
        for bps in self.config.pipeline.slippage_pressure_bps:
            configs.append(
                (
                    f"slippage_{bps}bps",
                    self.config.model_copy(
                        update={
                            "backtest": self.config.backtest.model_copy(
                                update={"cost_mode": "basic", "slippage_bps": bps}
                            )
                        }
                    ),
                )
            )
        return configs

    def _backtest_step_detail(self) -> str:
        pressure = self.config.pipeline.slippage_pressure_bps
        if not pressure:
            return "run basic cost MVP only"
        labels = ", ".join(f"{bps / 100:.2f}%" for bps in pressure)
        return f"run basic cost MVP plus slippage pressure tests at {labels}"

    def _final_equity(self, session: Session, run_id: int, fallback: float) -> float:
        row = (
            session.execute(
                select(EquityCurveRecord)
                .where(EquityCurveRecord.strategy_run_id == run_id)
                .order_by(EquityCurveRecord.time.desc())
                .limit(1)
            )
            .scalars()
            .first()
        )
        return float(row.equity) if row is not None else fallback

    def _overview_row(self, session: Session, run_id: int, label: str, slippage_bps: float, final_equity: float) -> dict[str, object]:
        run = session.get(StrategyRun, run_id)
        initial_equity = float(run.config["backtest"]["initial_equity"]) if run is not None else self.config.backtest.initial_equity
        orders = session.execute(select(OrderRecord).where(OrderRecord.strategy_run_id == run_id)).scalars().all()
        drawdowns = (
            session.execute(select(EquityCurveRecord.drawdown).where(EquityCurveRecord.strategy_run_id == run_id))
            .scalars()
            .all()
        )
        return {
            "label": label,
            "run_id": run_id,
            "exchange": run.config.get("exchange_id", self.config.exchange_id) if run is not None else self.config.exchange_id,
            "slippage_pct": slippage_bps / 100,
            "final_equity": final_equity,
            "total_return_pct": (final_equity / initial_equity - 1) * 100 if initial_equity else 0,
            "max_drawdown_pct": min([float(item) for item in drawdowns], default=0) * 100,
            "orders": len(orders),
            "fees": sum(float(order.fee) for order in orders),
            **self._trade_stats(orders),
        }

    def _trade_stats(self, orders: Sequence[OrderRecord]) -> dict[str, object]:
        open_entries: dict[str, list[float]] = {}
        trade_returns: list[float] = []
        for order in sorted(orders, key=lambda item: item.time):
            if order.side == "buy":
                open_entries.setdefault(order.symbol, []).append(float(order.filled_price or 0))
            elif order.side == "sell" and open_entries.get(order.symbol):
                entry = open_entries[order.symbol].pop(0)
                exit_price = float(order.filled_price or 0)
                if entry > 0:
                    trade_returns.append(exit_price / entry - 1)
        sell_orders = [order for order in orders if order.side == "sell"]
        net_wins = [
            float(order.filled_price or 0) > float(order.expected_price or 0)
            for order in sell_orders
        ]
        return {
            "closed_trade_win_rate": sum(1 for item in trade_returns if item > 0) / len(trade_returns) if trade_returns else 0,
            "net_win_rate": sum(1 for item in net_wins if item) / len(net_wins) if net_wins else 0,
            "atr_stop_count": sum(1 for order in orders if order.reason == "atr_stop"),
            "trailing_stop_count": sum(1 for order in orders if order.reason == "trailing_stop"),
            "breakeven_stop_count": sum(1 for order in orders if order.mechanism == "breakeven_stop"),
            "defensive_exit_count": sum(1 for order in orders if order.mechanism == "defensive_exit"),
        }
