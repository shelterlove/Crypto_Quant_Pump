from __future__ import annotations

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from crypto_quant.reporting import ClosedTradeSummary, PositionSummary, RunSummary, RunSummaryBuilder
from crypto_quant.utils.time import ensure_utc


class PaperMonitorWriter:
    def __init__(self, state_dir: Path, report_dir: Path) -> None:
        self.state_dir = state_dir
        self.report_dir = report_dir
        self.summary_builder = RunSummaryBuilder(report_dir)

    def write(self, session: Session, current: dict[str, Any], limit: int = 20) -> Path:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        runs = self._recent_runs(session, limit)
        positions = self._current_positions(session, current)
        trades = self.summary_builder.recent_closed_trades(session, "paper", 12)
        dashboard_html = self.state_dir / "dashboard.html"
        dashboard_html.write_text(self._render_html(current, runs, positions, trades), encoding="utf-8")
        return dashboard_html

    def load_latest_status(self) -> dict[str, Any]:
        latest = self.state_dir / "latest_status.json"
        if not latest.exists():
            return {}
        return json.loads(latest.read_text(encoding="utf-8"))

    def _recent_runs(self, session: Session, limit: int) -> list[RunSummary]:
        return self.summary_builder.latest_runs(session, "paper", limit)

    def _current_positions(self, session: Session, current: dict[str, Any]) -> list[PositionSummary]:
        run_id = current.get("last_completed_run_id")
        if run_id in (None, "", 0):
            return []
        try:
            return self.summary_builder.open_positions_for_run(session, int(run_id))
        except (TypeError, ValueError):
            return []

    def _render_html(
        self,
        current: dict[str, Any],
        runs: list[RunSummary],
        positions: list[PositionSummary],
        trades: list[ClosedTradeSummary],
    ) -> str:
        generated_at = ensure_utc(datetime.now(UTC)).isoformat()
        latest_run = runs[0] if runs else None
        last_completed = next((run for run in runs if run.status == "completed"), None)
        rows = "\n".join(self._table_row(run) for run in runs) or "<tr><td colspan='7'>暂无运行记录。</td></tr>"
        position_rows = "\n".join(self._position_row(item) for item in positions) or "<tr><td colspan='8'>当前没有持仓。</td></tr>"
        trade_rows = "\n".join(self._trade_row(item) for item in trades) or "<tr><td colspan='7'>暂时没有已平仓交易。</td></tr>"
        return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="60">
  <title>Crypto Quant 策略监控</title>
  <style>
    :root {{
      --bg: #f3f5f7;
      --panel: #ffffff;
      --text: #14202b;
      --muted: #667788;
      --line: #d7dde5;
      --good: #0f766e;
      --warn: #b45309;
      --bad: #b42318;
      --accent: #0f4c81;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 Arial, sans-serif; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0; font-size: 28px; }}
    h2 {{ margin: 0 0 12px; font-size: 16px; }}
    p {{ margin: 6px 0 0; color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin-top: 20px;
    }}
    .metric, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .metric-label {{ font-size: 12px; color: var(--muted); }}
    .metric-value {{ margin-top: 6px; font-size: 22px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 16px;
      margin-top: 16px;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 148px 1fr;
      gap: 8px 12px;
    }}
    .kv div:nth-child(odd) {{ color: var(--muted); }}
    .ok {{ color: var(--good); }}
    .warn {{ color: var(--warn); }}
    .bad {{ color: var(--bad); }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    th {{ background: #edf2f7; color: var(--muted); font-size: 12px; }}
    code {{ font-family: "SFMono-Regular", Consolas, monospace; font-size: 12px; }}
    a {{ color: var(--accent); text-decoration: none; }}
  </style>
</head>
<body>
  <main>
    <h1>策略监控面板</h1>
    <p>生成时间 {html.escape(generated_at)}，页面每 60 秒自动刷新一次。</p>
    <div class="metrics">
      {self._metric("本轮状态", self._status_span(str(current.get("status") or "missing")))}
      {self._metric("数据延迟", self._lag_value(current.get("lag_seconds")))}
      {self._metric("候选数量", str(current.get("candidates") or 0))}
      {self._metric("本轮订单", str(current.get("orders") or 0))}
      {self._metric("当前持仓", str(current.get("open_positions") or 0))}
      {self._metric("当前净值", self._fmt_float(current.get("equity")))}
    </div>
    <div class="grid">
      {self._current_cycle_panel(current, latest_run)}
      {self._last_completed_panel(last_completed, current)}
    </div>
    <h2 style="margin-top:20px;">当前持仓</h2>
    <table>
      <thead>
        <tr>
          <th>币种</th>
          <th>开仓时间</th>
          <th>数量</th>
          <th>成本价</th>
          <th>现价</th>
          <th>浮动盈亏</th>
          <th>止损价</th>
          <th>风险标签</th>
        </tr>
      </thead>
      <tbody>
        {position_rows}
      </tbody>
    </table>
    <h2 style="margin-top:20px;">最近已平仓交易</h2>
    <table>
      <thead>
        <tr>
          <th>时间</th>
          <th>币种</th>
          <th>卖出价</th>
          <th>数量</th>
          <th>收益金额</th>
          <th>收益率</th>
          <th>退出原因</th>
        </tr>
      </thead>
      <tbody>
        {trade_rows}
      </tbody>
    </table>
    <h2 style="margin-top:20px;">最近运行记录</h2>
    <table>
      <thead>
        <tr>
          <th>Run ID</th>
          <th>开始时间</th>
          <th>状态</th>
          <th>订单</th>
          <th>候选</th>
          <th>净值</th>
          <th>报告</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </main>
</body>
</html>
"""

    def _current_cycle_panel(self, current: dict[str, Any], latest_run: RunSummary | None) -> str:
        latest_run_id = latest_run.run_id if latest_run is not None else None
        latest_status = latest_run.status if latest_run is not None else None
        rows = [
            ("状态", self._status_span(str(current.get("status") or "missing"))),
            ("原因", html.escape(str(current.get("reason") or "-"))),
            ("当前 run_id", html.escape(str(current.get("run_id") or "-"))),
            ("数据库最新 run", html.escape(str(latest_run_id or "-"))),
            ("数据库最新状态", self._status_span(str(latest_status or "missing"))),
            ("候选数量", html.escape(str(current.get("candidates") or 0))),
            ("本轮订单", html.escape(str(current.get("orders") or 0))),
            ("持仓数量", html.escape(str(current.get("open_positions") or 0))),
            ("市场状态", self._market_summary(current)),
            ("信号时间", html.escape(str(current.get("signal_time") or "-"))),
            ("成交时间", html.escape(str(current.get("fill_time") or "-"))),
            ("数据新鲜度", self._freshness_summary(current)),
        ]
        return self._panel("当前轮次", rows)

    def _last_completed_panel(self, run: RunSummary | None, current: dict[str, Any]) -> str:
        if run is None:
            return self._panel("最近一次完整运行", [("状态", "缺失")])
        rows = [
            ("run_id", str(run.run_id)),
            ("状态", self._status_span(run.status)),
            ("开始时间", html.escape(run.started_at.isoformat())),
            ("订单", f"{run.orders} ({run.buys} 买 / {run.sells} 卖)"),
            ("净值", "-" if run.equity is None else f"{run.equity:,.2f}"),
            ("报告是否存在", self._yes_no(run.report_exists)),
            ("报告", self._report_link(run.run_id)),
            ("状态文件", f"<code>{html.escape(str(current.get('state_path') or '-'))}</code>"),
        ]
        return self._panel("最近一次完整运行", rows)

    def _panel(self, title: str, rows: list[tuple[str, str]]) -> str:
        body = "".join(f"<div>{html.escape(label)}</div><div>{value}</div>" for label, value in rows)
        return f"<section class='panel'><h2>{html.escape(title)}</h2><div class='kv'>{body}</div></section>"

    def _metric(self, label: str, value: str) -> str:
        return (
            "<section class='metric'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{value}</div>"
            "</section>"
        )

    def _table_row(self, run: RunSummary) -> str:
        return (
            "<tr>"
            f"<td>{run.run_id}</td>"
            f"<td>{html.escape(run.started_at.isoformat())}</td>"
            f"<td>{self._status_span(run.status)}</td>"
            f"<td>{run.orders} ({run.buys}/{run.sells})</td>"
            f"<td>{run.signals}</td>"
            f"<td>{'-' if run.equity is None else f'{run.equity:,.2f}'}</td>"
            f"<td>{self._report_link(run.run_id)}</td>"
            "</tr>"
        )

    def _position_row(self, position: PositionSummary) -> str:
        pnl = self._fmt_signed_float(position.unrealized_pnl)
        ret = self._fmt_pct(position.unrealized_return_pct)
        return (
            "<tr>"
            f"<td>{html.escape(position.symbol)}</td>"
            f"<td>{html.escape(position.opened_at.isoformat()) if position.opened_at is not None else '-'}</td>"
            f"<td>{position.quantity:,.4f}</td>"
            f"<td>{self._fmt_float(position.entry_price)}</td>"
            f"<td>{self._fmt_float(position.current_price)}</td>"
            f"<td>{pnl} {ret}</td>"
            f"<td>{self._fmt_float(position.stop_price)}</td>"
            f"<td>{html.escape(position.risk_tag or '-')}</td>"
            "</tr>"
        )

    def _trade_row(self, trade: ClosedTradeSummary) -> str:
        return (
            "<tr>"
            f"<td>{html.escape(trade.time.isoformat())}</td>"
            f"<td>{html.escape(trade.symbol)}</td>"
            f"<td>{self._fmt_float(trade.filled_price)}</td>"
            f"<td>{trade.quantity:,.4f}</td>"
            f"<td>{self._fmt_signed_float(trade.pnl)}</td>"
            f"<td>{self._fmt_pct(trade.return_pct)}</td>"
            f"<td>{html.escape(trade.reason)}</td>"
            "</tr>"
        )

    def _report_link(self, run_id: int) -> str:
        if not self.summary_builder.report_file(run_id).exists():
            return "<span class='bad'>缺失</span>"
        return f"<a href='../reports/{run_id}/report.html'>查看</a>"

    def _fmt_float(self, value: Any) -> str:
        if value is None:
            return "-"
        return f"{float(value):,.2f}"

    def _fmt_signed_float(self, value: Any) -> str:
        if value is None:
            return "-"
        number = float(value)
        klass = "ok" if number > 0 else "bad" if number < 0 else ""
        text = f"{number:+,.2f}"
        return f"<span class='{klass}'>{text}</span>" if klass else text

    def _fmt_pct(self, value: Any) -> str:
        if value is None:
            return "-"
        number = float(value)
        klass = "ok" if number > 0 else "bad" if number < 0 else ""
        text = f"{number:+.2%}"
        return f"<span class='{klass}'>{text}</span>" if klass else text

    def _lag_value(self, value: Any) -> str:
        if value is None:
            return "-"
        return f"{int(value)}s"

    def _freshness_summary(self, current: dict[str, Any]) -> str:
        latest = str(current.get("latest_candle_time") or "-")
        expected = str(current.get("expected_candle_time") or "-")
        lag = self._lag_value(current.get("lag_seconds"))
        errors = str(current.get("sync_errors") or 0)
        return f"<code>{html.escape(latest)}</code> -> <code>{html.escape(expected)}</code> | 延迟 {lag} | 同步错误 {errors}"

    def _market_summary(self, current: dict[str, Any]) -> str:
        phase = str(current.get("market_phase") or "-")
        entry_mode = str(current.get("market_entry_mode") or "-")
        regime = str(current.get("pump_regime") or "-")
        return f"{html.escape(phase)} / {html.escape(entry_mode)} / {html.escape(regime)}"

    def _yes_no(self, value: bool) -> str:
        label = "是" if value else "否"
        klass = "ok" if value else "bad"
        return f"<span class='{klass}'>{label}</span>"

    def _status_span(self, status: str) -> str:
        labels = {
            "completed": "已完成",
            "running": "运行中",
            "skipped": "已跳过",
            "failed": "失败",
            "missing": "缺失",
        }
        text = labels.get(status, status)
        return f"<span class='{self._status_class(status)}'>{html.escape(text)}</span>"

    def _status_class(self, status: str) -> str:
        if status in {"completed", "running"}:
            return "ok"
        if status == "skipped":
            return "warn"
        return "bad"
