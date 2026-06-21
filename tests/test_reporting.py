from datetime import UTC, datetime

from crypto_quant.reporting import BacktestReportWriter
from crypto_quant.storage.models import EquityCurveRecord, RejectedSignalRecord, StrategyRun


def test_report_writer_creates_csv_and_html(sqlite_session, tmp_path) -> None:  # type: ignore[no-untyped-def]
    run = StrategyRun(
        id=1,
        run_name="test",
        strategy_version="v1.4.2",
        config={"exchange_id": "binance", "backtest": {"initial_equity": 1000}},
        config_hash="abc",
        started_at=datetime.now(UTC),
        status="completed",
    )
    sqlite_session.add(run)
    sqlite_session.flush()
    sqlite_session.add(
        EquityCurveRecord(
            id=1,
            strategy_run_id=run.id,
            time=datetime.now(UTC),
            equity=1000,
            cash=1000,
            gross_exposure=0,
            drawdown=0,
        )
    )
    sqlite_session.add(
        RejectedSignalRecord(
            id=1,
            strategy_run_id=run.id,
            time=datetime(2024, 1, 1, tzinfo=UTC),
            symbol="AAA/USDT",
            reason="market_state:defensive",
            details=None,
        )
    )
    sqlite_session.commit()
    paths = BacktestReportWriter(tmp_path).write(sqlite_session, run.id)
    assert paths.html.exists()
    assert (paths.directory / "equity_curve.csv").exists()
    assert "Survivorship bias risk" in paths.html.read_text(encoding="utf-8")
    assert "Exit Mechanism Breakdown" in paths.html.read_text(encoding="utf-8")
