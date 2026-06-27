from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
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
from crypto_quant.paper.monitor import PaperMonitorWriter
from crypto_quant.paper.runner import PaperRunner
from crypto_quant.reporting import BacktestReportWriter, RunSummaryBuilder
from crypto_quant.storage.database import get_session_factory
from crypto_quant.utils.time import default_one_year_window, parse_utc_datetime

app = typer.Typer(help="Crypto quant research framework.")
db_app = typer.Typer(help="Database commands.")
data_app = typer.Typer(help="Market data commands.")
backtest_app = typer.Typer(help="Backtest commands.")
paper_app = typer.Typer(help="Local paper trading commands.")
app.add_typer(db_app, name="db")
app.add_typer(data_app, name="data")
app.add_typer(backtest_app, name="backtest")
app.add_typer(paper_app, name="paper")


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
        f"final_equity={result.final_equity:.2f} report={paths.html.resolve()}"
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
                f"final_equity={result.final_equity:.2f} report={paths.html.resolve()}"
            )

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
        if result.strategy_run_id is not None and result.orders:
            report_path = BacktestReportWriter(report_dir).write(session, result.strategy_run_id).html.resolve()
            session.commit()
    if result.skipped:
        typer.echo(f"paper skipped: {result.reason} state={result.state_path}")
        return
    typer.echo(
        f"paper completed: run_id={result.strategy_run_id} orders={len(result.orders)} "
        f"signal_time={result.processed_signal_time} fill_time={result.fill_time} "
        f"equity={result.equity:.2f} state={result.state_path} report={report_path}"
    )


@paper_app.command("report")
def write_paper_report(
    config: Path = Path("configs/main.yaml"),
    run_id: int | None = None,
    report_dir: Path = Path("reports"),
) -> None:
    cfg = load_config(config)
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        target_run_id = run_id
        if target_run_id is None:
            latest = RunSummaryBuilder(report_dir).latest_completed_run(session, "paper")
            if latest is None:
                raise typer.BadParameter("no completed paper run found")
            target_run_id = latest.run_id
        path = BacktestReportWriter(report_dir).write(session, target_run_id).html.resolve()
        session.commit()
    typer.echo(f"paper report generated: run_id={target_run_id} report={path}")


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


@paper_app.command("serve-monitor")
def serve_monitor(
    state_dir: Path = Path("paper_state"),
    report_dir: Path = Path("reports"),
    config: Path = Path("configs/main.yaml"),
    host: str = "0.0.0.0",
    port: int = 8000,
) -> None:
    cfg = load_config(config)
    session_factory = get_session_factory(cfg.database_url)
    with session_factory() as session:
        latest = PaperMonitorWriter(state_dir, report_dir).load_latest_status()
        PaperMonitorWriter(state_dir, report_dir).write(session, latest)
    root = state_dir.resolve().parent
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    typer.echo(f"paper monitor serving http://{host}:{port}/paper_state/dashboard.html from {root}")
    ThreadingHTTPServer((host, port), handler).serve_forever()
