from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from alembic.config import Config
from loguru import logger

from alembic import command
from crypto_quant.backtest.runner import ResearchBacktester
from crypto_quant.config.settings import load_config
from crypto_quant.data.sync import CandleSyncService
from crypto_quant.pipeline import PipelineError, RunOneYearMvpPipeline
from crypto_quant.reporting import BacktestReportWriter
from crypto_quant.storage.candles import distinct_candle_symbols
from crypto_quant.storage.database import get_session_factory
from crypto_quant.universe.service import WeeklyUniverseService
from crypto_quant.utils.time import default_one_year_window, parse_utc_datetime

app = typer.Typer(help="Crypto quant research framework.")
db_app = typer.Typer(help="Database commands.")
data_app = typer.Typer(help="Market data commands.")
universe_app = typer.Typer(help="Universe commands.")
backtest_app = typer.Typer(help="Backtest commands.")
pipeline_app = typer.Typer(help="End-to-end research pipelines.")
app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(universe_app, name="universe")
app.add_typer(backtest_app, name="backtest")
app.add_typer(pipeline_app, name="pipeline")


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    command.upgrade(Config("alembic.ini"), revision)


@data_app.command("sync-candles")
def sync_candles(
    config: Path = Path("configs/mvp.yaml"),
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    timeframe: str = "1h",
    start: str | None = None,
    end: str | None = None,
    all_usdt_spot: bool = False,
    dry_run: bool = False,
) -> None:
    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start)
    end_dt = parse_utc_datetime(end, default_end)
    service = CandleSyncService(exchange=cfg.exchange_id)
    resolved = service.resolve_symbols(symbols, all_usdt_spot)
    plan = service.build_plan(resolved, timeframe, start_dt, end_dt)
    if dry_run:
        typer.echo(
            f"dry-run: would sync {len(plan.symbols)} symbols "
            f"exchange={plan.exchange} timeframe={plan.timeframe} start={plan.start} end={plan.end}"
        )
        typer.echo(", ".join(plan.symbols[:20]) + (" ..." if len(plan.symbols) > 20 else ""))
        return
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        result = service.run(session, plan)
        session.commit()
    typer.echo(f"synced rows={result.inserted_or_updated} errors={len(result.errors)}")
    for item in result.coverage[:20]:
        typer.echo(
            f"{item.symbol} {item.timeframe}: rows={item.rows} "
            f"start={item.start} end={item.end} missing={item.missing_intervals}"
        )


@universe_app.command("build")
def build_universe(
    config: Path = Path("configs/mvp.yaml"),
    symbols: Annotated[list[str] | None, typer.Option("--symbol")] = None,
    start: str | None = None,
    end: str | None = None,
    dry_run: bool = False,
) -> None:
    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start)
    end_dt = parse_utc_datetime(end, default_end)
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        selected = symbols or distinct_candle_symbols(session, cfg.exchange_id, "1d")
        if not selected:
            raise typer.BadParameter("no 1d candle symbols found; run data sync-candles first")
        result = WeeklyUniverseService(cfg).build(session, selected, start_dt, end_dt, persist=not dry_run)
        if not dry_run:
            session.commit()
    typer.echo(f"weekly_snapshots={len(result.snapshots)} candidate_union={len(result.candidate_union)}")
    for snapshot in result.snapshots[:8]:
        typer.echo(f"{snapshot.effective_from.date()}: {len(snapshot.symbols)} symbols")


@backtest_app.command("run")
def run_backtest(
    config: Path = Path("configs/mvp.yaml"),
    start: str | None = None,
    end: str | None = None,
    write_db: bool = True,
    report_dir: Path = Path("reports"),
) -> None:
    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start)
    end_dt = parse_utc_datetime(end, default_end)
    if not write_db:
        result = ResearchBacktester(cfg).run_synthetic()
        typer.echo(
            f"completed synthetic: run_id={result.strategy_run_id} "
            f"orders={len(result.orders)} final_equity={result.final_equity:.2f}"
        )
        return
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        result = ResearchBacktester(cfg).run_real(session, start_dt, end_dt)
        session.commit()
        paths = BacktestReportWriter(report_dir).write(session, result.strategy_run_id or 0)
        session.commit()
    logger.info("backtest completed run_id={} orders={} final_equity={}", result.strategy_run_id, len(result.orders), result.final_equity)
    typer.echo(
        f"completed: run_id={result.strategy_run_id} orders={len(result.orders)} "
        f"final_equity={result.final_equity:.2f} report={paths.html}"
    )


@backtest_app.command("pressure-test")
def pressure_test(
    config: Path = Path("configs/mvp.yaml"),
    start: str | None = None,
    end: str | None = None,
    report_dir: Path = Path("reports"),
) -> None:
    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start)
    end_dt = parse_utc_datetime(end, default_end)
    session_factory = get_session_factory(cfg.database_url)
    slippage_levels = [5, 10, 20, 30]
    with session_factory() as session:
        for bps in slippage_levels:
            test_cfg = cfg.model_copy(update={"backtest": cfg.backtest.model_copy(update={"slippage_bps": bps, "cost_mode": "basic"})})
            result = ResearchBacktester(test_cfg).run_real(session, start_dt, end_dt)
            session.commit()
            paths = BacktestReportWriter(report_dir).write(session, result.strategy_run_id or 0)
            typer.echo(
                f"slippage={bps / 100:.2f}% run_id={result.strategy_run_id} "
                f"final_equity={result.final_equity:.2f} report={paths.html}"
            )


@data_app.command("sync-one-year-mvp")
def sync_one_year_mvp(config: Path = Path("configs/mvp.yaml"), dry_run: bool = False) -> None:
    cfg = load_config(config)
    start, end = default_one_year_window(datetime.now(UTC))
    daily_start = start - timedelta(days=30)
    typer.echo("Step 1: sync all USDT spot 1d for liquidity screening")
    typer.echo(f"start={daily_start} end={end} dry_run={dry_run}")
    typer.echo("Step 2: build weekly universe from 1d data")
    typer.echo("Step 3: sync 1h/4h for weekly Top60 union + BTC")
    typer.echo(f"database={cfg.database_url}")


@pipeline_app.command("run-one-year-mvp")
def run_one_year_mvp(
    config: Path = Path("configs/mvp.yaml"),
    start: str | None = None,
    end: str | None = None,
    report_dir: Path = Path("reports"),
    dry_run: bool = False,
) -> None:
    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start) if start else None
    end_dt = parse_utc_datetime(end, default_end) if end else None
    pipeline = RunOneYearMvpPipeline(cfg, report_writer=BacktestReportWriter(report_dir))
    if dry_run:
        result = pipeline.build_dry_run(start_dt, end_dt)
    else:
        session_factory = get_session_factory(cfg.database_url)
        try:
            with session_factory() as session:
                result = pipeline.run(session, start_dt, end_dt)
        except PipelineError as exc:
            raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"window: start={result.start.isoformat()} end={result.end.isoformat()} daily_start={result.daily_start.isoformat()}")
    for index, step in enumerate(result.steps, start=1):
        typer.echo(f"{index}. {step.name}: {step.detail}")
    for quality in result.data_quality:
        typer.echo(
            f"coverage {quality.timeframe}: symbols={quality.symbol_count} "
            f"rows={quality.row_count} missing={quality.missing_intervals}"
        )
    for warning in result.warnings:
        typer.echo(f"warning: {warning}")
    if result.run_summaries:
        typer.echo("runs:")
        for summary in result.run_summaries:
            typer.echo(
                f"  {summary.label}: run_id={summary.run_id} "
                f"final_equity={summary.final_equity:.2f} report={summary.report_html}"
            )
    if result.overview is not None:
        typer.echo(f"overview_html={result.overview.html}")
        if result.overview.csv is not None:
            typer.echo(f"overview_csv={result.overview.csv}")
    if dry_run:
        typer.echo("dry-run: no database, Binance, or backtest work executed")
    typer.echo(
        "whitepaper_not_covered: 15m fast-risk valve, order book depth, full volume/trend factors, "
        "volume-stall filter, delisted-symbol backfill"
    )
