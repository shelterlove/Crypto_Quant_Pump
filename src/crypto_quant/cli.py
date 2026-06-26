from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from alembic.config import Config
from loguru import logger

from alembic import command
from crypto_quant.backtest.runner import ResearchBacktester
from crypto_quant.config.settings import load_config
from crypto_quant.data.sync import CandleSyncService
from crypto_quant.paper.cycle import PaperCycleDataStale, PaperCycleLocked, PaperCycleRunner
from crypto_quant.paper.runner import PaperRunner
from crypto_quant.reporting import BacktestReportWriter
from crypto_quant.storage.database import get_session_factory
from crypto_quant.utils.time import default_one_year_window, parse_utc_datetime

app = typer.Typer(help="Crypto quant research framework.")
db_app = typer.Typer(help="Database commands.")
data_app = typer.Typer(help="Market data commands.")
backtest_app = typer.Typer(help="Backtest commands.")
paper_app = typer.Typer(help="Local paper trading commands.")
analysis_app = typer.Typer(help="Research analysis commands.")
futures_analysis_app = typer.Typer(help="Futures research diagnostics.")
app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(paper_app, name="paper")
app.add_typer(analysis_app, name="analysis")
analysis_app.add_typer(futures_analysis_app, name="futures")


@db_app.command("upgrade")
def db_upgrade(revision: str = "head") -> None:
    command.upgrade(Config("alembic.ini"), revision)


@data_app.command("sync-candles")
def sync_candles(
    config: Path = Path("configs/v1.yaml"),
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


@backtest_app.command("run")
def run_backtest(
    config: Path = Path("configs/v1.yaml"),
    start: str | None = None,
    end: str | None = None,
    report_dir: Path = Path("reports"),
) -> None:
    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start)
    end_dt = parse_utc_datetime(end, default_end)
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
    config: Path = Path("configs/v1.yaml"),
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


@analysis_app.command("futures-coverage")
def futures_coverage(
    config: Path = Path("configs/futures_1x.yaml"),
    start: str | None = "2023-01-01",
    end: str | None = "2025-05-31",
    report_dir: Path = Path("reports/futures_diagnostics"),
) -> None:
    _run_futures_coverage(config, start, end, report_dir)


@futures_analysis_app.command("coverage")
def futures_coverage_nested(
    config: Path = Path("configs/futures_1x.yaml"),
    start: str | None = "2023-01-01",
    end: str | None = "2025-05-31",
    report_dir: Path = Path("reports/futures_diagnostics"),
) -> None:
    _run_futures_coverage(config, start, end, report_dir)


def _run_futures_coverage(
    config: Path,
    start: str | None,
    end: str | None,
    report_dir: Path,
) -> None:
    from crypto_quant.analysis.futures import write_futures_coverage_report

    cfg = load_config(config)
    default_start, default_end = default_one_year_window()
    start_dt = parse_utc_datetime(start, default_start)
    end_dt = parse_utc_datetime(end, default_end)
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        paths = write_futures_coverage_report(session, cfg, start_dt, end_dt, report_dir)
    typer.echo(f"futures coverage report={paths.html} summary={paths.summary_csv} candidates={paths.candidates_csv}")


@paper_app.command("run")
def run_paper(
    config: Path = Path("configs/main.yaml"),
    state_path: Path = Path("paper_state/main.json"),
    report_dir: Path = Path("reports"),
    lookback_days: int = 120,
) -> None:
    cfg = load_config(config)
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        result = PaperRunner(cfg, state_path=state_path, lookback_days=lookback_days).run_once(session)
        session.commit()
        report_path = None
        if result.strategy_run_id is not None:
            report_path = BacktestReportWriter(report_dir).write(session, result.strategy_run_id).html
            session.commit()
    if result.skipped:
        typer.echo(f"paper skipped: {result.reason} state={result.state_path}")
        return
    typer.echo(
        f"paper completed: run_id={result.strategy_run_id} orders={len(result.orders)} "
        f"signal_time={result.processed_signal_time} fill_time={result.fill_time} "
        f"equity={result.equity:.2f} state={result.state_path} report={report_path}"
    )


@paper_app.command("cycle")
def run_paper_cycle(
    config: Path = Path("configs/main.yaml"),
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
    json_output: bool = False,
) -> None:
    cfg = load_config(config)
    session_factory = get_session_factory(cfg.database_url)
    runner = PaperCycleRunner(
        cfg,
        state_path=state_path,
        lock_path=lock_path,
        report_dir=report_dir,
        lookback_days=lookback_days,
        sync_lookback_hours=sync_lookback_hours,
        max_staleness_hours=max_staleness_hours,
        settle_minutes=settle_minutes,
        all_usdt_spot=all_usdt_spot,
        skip_sync=skip_sync,
        allow_stale=allow_stale,
    )
    try:
        with session_factory() as session:
            result = runner.run(session)
    except PaperCycleLocked as exc:
        typer.echo(f"paper cycle skipped: {exc}", err=True)
        raise typer.Exit(code=75) from exc
    except PaperCycleDataStale as exc:
        typer.echo(f"paper cycle failed: {exc}", err=True)
        raise typer.Exit(code=70) from exc
    if json_output:
        typer.echo(result.to_json())
        return
    typer.echo(
        f"paper cycle {result.status}: run_id={result.run_id} orders={result.orders} "
        f"buys={result.buys} sells={result.sells} candidates={result.candidates} open_positions={result.open_positions} "
        f"equity={result.equity:.2f} pnl={result.pnl:.2f} return={result.return_pct:.2%} "
        f"synced_rows={result.synced_rows} sync_errors={result.sync_errors} "
        f"latest={result.latest_candle_time} expected={result.expected_candle_time} lag_seconds={result.lag_seconds} "
        f"state={result.state_path} report={result.report_path} reason={result.reason}"
    )
