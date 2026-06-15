from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from crypto_quant.backtest.runner import BacktestResult
from crypto_quant.cli import app
from crypto_quant.config.settings import AppConfig, PipelineConfig, UniverseConfig, load_config
from crypto_quant.data.sync import CandleSyncPlan, CandleSyncResult
from crypto_quant.pipeline import PipelineError, RunOneYearMvpPipeline
from crypto_quant.reporting import BacktestReportWriter
from crypto_quant.storage.candles import upsert_candles
from crypto_quant.storage.models import EquityCurveRecord, OrderRecord, StrategyRun
from crypto_quant.universe.service import UniverseBuildResult


class NoRowsSyncService:
    calls: list[str]

    def __init__(self) -> None:
        self.calls = []

    def resolve_symbols(self, explicit_symbols: list[str] | None, all_usdt_spot: bool) -> list[str]:
        return ["AAA/USDT"]

    def build_plan(self, symbols: list[str], timeframe: str, start: datetime, end: datetime) -> CandleSyncPlan:
        return CandleSyncPlan("binance", timeframe, symbols, start, end)

    def run(self, session, plan: CandleSyncPlan) -> CandleSyncResult:  # type: ignore[no-untyped-def]
        self.calls.append(plan.timeframe)
        return CandleSyncResult()


class FixtureSyncService:
    calls: list[str]

    def __init__(self, include_btc_intraday: bool = True) -> None:
        self.include_btc_intraday = include_btc_intraday
        self.calls = []

    def resolve_symbols(self, explicit_symbols: list[str] | None, all_usdt_spot: bool) -> list[str]:
        return ["AAA/USDT"]

    def build_plan(self, symbols: list[str], timeframe: str, start: datetime, end: datetime) -> CandleSyncPlan:
        return CandleSyncPlan("binance", timeframe, symbols, start, end)

    def run(self, session, plan: CandleSyncPlan) -> CandleSyncResult:  # type: ignore[no-untyped-def]
        self.calls.append(plan.timeframe)
        symbols = plan.symbols
        if plan.timeframe in {"1h", "4h"} and not self.include_btc_intraday:
            symbols = [symbol for symbol in symbols if symbol != "BTC/USDT"]
        inserted = 0
        for symbol in symbols:
            frame = _candles(plan.start, plan.end, plan.timeframe)
            inserted += upsert_candles(session, plan.exchange, symbol, plan.timeframe, frame)
        return CandleSyncResult(inserted_or_updated=inserted)


class FailingSymbolSyncService(FixtureSyncService):
    resolve_called: bool = False

    def resolve_symbols(self, explicit_symbols: list[str] | None, all_usdt_spot: bool) -> list[str]:
        self.resolve_called = True
        raise RuntimeError("remote symbol lookup failed")


@dataclass
class FakeBacktesterFactory:
    next_id: int = 1

    def __call__(self, config: AppConfig) -> FakeBacktester:
        run_id = self.next_id
        self.next_id += 1
        return FakeBacktester(config, run_id)


@dataclass
class FakeBacktester:
    config: AppConfig
    run_id: int

    def run_real(self, session, start: datetime, end: datetime, report_dir: Path | None = None) -> BacktestResult:  # type: ignore[no-untyped-def]
        session.add(
            StrategyRun(
                id=self.run_id,
                run_name=f"fake-{self.run_id}",
                strategy_version=self.config.strategy_version,
                config=self.config.model_dump(),
                config_hash=self.config.stable_hash(),
                started_at=start,
                finished_at=end,
                status="completed",
            )
        )
        session.flush()
        final_equity = self.config.backtest.initial_equity + self.run_id * 100
        session.add(
            EquityCurveRecord(
                id=self.run_id,
                strategy_run_id=self.run_id,
                time=end,
                equity=final_equity,
                cash=final_equity,
                gross_exposure=0,
                drawdown=-0.01 * self.run_id,
            )
        )
        session.add(
            OrderRecord(
                id=self.run_id,
                strategy_run_id=self.run_id,
                time=end,
                symbol="AAA/USDT",
                side="buy",
                quantity=1,
                expected_price=10,
                limit_price=10,
                filled_price=10,
                fee=self.run_id,
                slippage=0,
                status="filled",
                reason="fixture",
            )
        )
        return BacktestResult(self.run_id, final_equity=final_equity)


def test_pipeline_dry_run_lists_full_ordered_plan() -> None:
    cfg = _test_config()
    pipeline = RunOneYearMvpPipeline(cfg, now=datetime(2025, 1, 2, tzinfo=UTC))
    result = pipeline.build_dry_run()
    assert [step.name for step in result.steps] == [
        "preflight",
        "sync_1d",
        "build_universe",
        "sync_candidates",
        "data_quality",
        "backtest",
        "reports",
    ]
    assert result.daily_start == result.start - pd.Timedelta(days=30)


def test_pipeline_candidate_union_includes_btc() -> None:
    cfg = _test_config()
    universe = UniverseBuildResult(candidate_union={"AAA/USDT", "BBB/USDT"})
    symbols = RunOneYearMvpPipeline(cfg).candidate_symbols(universe, "BTC/USDT")
    assert symbols == ["AAA/USDT", "BBB/USDT", "BTC/USDT"]


def test_pipeline_fails_when_daily_sync_produces_no_universe_data(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    cfg = _test_config()
    pipeline = RunOneYearMvpPipeline(cfg, sync_service=NoRowsSyncService())  # type: ignore[arg-type]
    start = datetime(2024, 2, 5, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)
    try:
        pipeline.run(sqlite_session, start, end)
    except PipelineError as exc:
        assert "requires 1d candles" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_pipeline_fails_when_btc_intraday_data_is_missing(sqlite_session) -> None:  # type: ignore[no-untyped-def]
    cfg = _test_config()
    pipeline = RunOneYearMvpPipeline(cfg, sync_service=FixtureSyncService(include_btc_intraday=False))  # type: ignore[arg-type]
    start = datetime(2024, 2, 5, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)
    try:
        pipeline.run(sqlite_session, start, end)
    except PipelineError as exc:
        assert "missing required BTC/USDT candles" in str(exc)
    else:
        raise AssertionError("expected PipelineError")


def test_pipeline_uses_local_symbol_cache_when_remote_lookup_fails(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    start = datetime(2024, 2, 5, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)
    upsert_candles(sqlite_session, "binance", "AAA/USDT", "1d", _candles(start - pd.Timedelta(days=30), end, "1d"))
    sqlite_session.commit()
    sync = FailingSymbolSyncService(include_btc_intraday=True)
    pipeline = RunOneYearMvpPipeline(
        _test_config(),
        sync_service=sync,  # type: ignore[arg-type]
        backtester_factory=FakeBacktesterFactory(),
        report_writer=BacktestReportWriter(tmp_path),
    )

    result = pipeline.run(sqlite_session, start, end)

    assert sync.calls == ["1d", "1h", "4h"]
    assert result.run_ids == [1, 2, 3, 4, 5]


def test_pipeline_can_prefer_local_symbol_cache(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    start = datetime(2024, 2, 5, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)
    upsert_candles(sqlite_session, "binance", "AAA/USDT", "1d", _candles(start - pd.Timedelta(days=30), end, "1d"))
    sqlite_session.commit()
    cfg = _test_config().model_copy(update={"pipeline": PipelineConfig(prefer_local_symbol_cache=True)})
    sync = FailingSymbolSyncService(include_btc_intraday=True)
    pipeline = RunOneYearMvpPipeline(
        cfg,
        sync_service=sync,  # type: ignore[arg-type]
        backtester_factory=FakeBacktesterFactory(),
        report_writer=BacktestReportWriter(tmp_path),
    )

    result = pipeline.run(sqlite_session, start, end)

    assert not sync.resolve_called
    assert result.run_ids == [1, 2, 3, 4, 5]


def test_pipeline_runs_all_backtests_and_writes_overview(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    sync = FixtureSyncService(include_btc_intraday=True)
    pipeline = RunOneYearMvpPipeline(
        _test_config(),
        sync_service=sync,  # type: ignore[arg-type]
        backtester_factory=FakeBacktesterFactory(),
        report_writer=BacktestReportWriter(tmp_path),
    )
    start = datetime(2024, 2, 5, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)
    result = pipeline.run(sqlite_session, start, end)
    assert sync.calls == ["1d", "1h", "4h"]
    assert result.run_ids == [1, 2, 3, 4, 5]
    assert result.overview is not None
    assert result.overview.html.exists()
    overview_html = result.overview.html.read_text(encoding="utf-8")
    assert "Survivorship bias risk" in overview_html
    assert "Signal Quality" not in overview_html
    assert (tmp_path / "overview" / "slippage_pressure_overview.csv").exists()


def test_pipeline_can_run_basic_backtest_only(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    cfg = _test_config().model_copy(update={"pipeline": PipelineConfig(slippage_pressure_bps=[])})
    sync = FixtureSyncService(include_btc_intraday=True)
    pipeline = RunOneYearMvpPipeline(
        cfg,
        sync_service=sync,  # type: ignore[arg-type]
        backtester_factory=FakeBacktesterFactory(),
        report_writer=BacktestReportWriter(tmp_path),
    )
    start = datetime(2024, 2, 5, tzinfo=UTC)
    end = datetime(2024, 2, 12, tzinfo=UTC)

    result = pipeline.run(sqlite_session, start, end)

    assert [summary.label for summary in result.run_summaries] == ["basic"]
    assert result.run_ids == [1]


def test_cli_pipeline_dry_run_does_not_require_database_or_network() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["pipeline", "run-one-year-mvp", "--dry-run"])
    assert result.exit_code == 0
    assert "sync_1d" in result.output
    assert "dry-run: no database, Binance, or backtest work executed" in result.output


def _test_config() -> AppConfig:
    cfg = load_config("configs/mvp.yaml")
    return cfg.model_copy(
        update={
            "database_url": "sqlite://",
            "universe": UniverseConfig(top_n=60, min_quote_volume_30d=0, exclude_keywords=[]),
        }
    )


def _candles(start: datetime, end: datetime, timeframe: str) -> pd.DataFrame:
    freq = {"1d": "D", "1h": "h", "4h": "4h"}[timeframe]
    index = pd.date_range(start, end, freq=freq, tz="UTC")
    close = pd.Series(range(10, 10 + len(index)), dtype=float)
    return pd.DataFrame(
        {
            "open_time": index,
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 1_000_000,
            "quote_volume": 100_000_000,
        }
    )
