# 下一轮优化交接提示词

你将接手一个加密货币动量交易策略代码仓库。请先完整阅读本提示词，再审查代码和诊断结果。不要一上来改策略；先理解、复核、提出可证伪的优化假设，再按 ablation 小步实验。

## 0. 工作方式要求

- 使用中文沟通。
- 注意使用 git 做版本管理。每个独立实验单独 commit，失败实验也要保留记录或用 revert 留痕。
- 不要一次性混改多个逻辑。每次只验证一个假设。
- 优先做诊断和审查，再做策略修改。
- 避免过拟合。任何新规则必须有市场解释、样本诊断和跨窗口验证。
- 不要破坏当前核心机制：`probe-anchor breathing`。
- 所有回测尽量带进度输出：`CQ_BACKTEST_PROGRESS_EVERY=2000`。
- 回测前检查是否已有后台进程：

```bash
ps -eo pid,ppid,etime,cmd | rg "uv run python|ResearchBacktester|run_real"
```

## 1. 项目与代码架构

仓库路径：

```text
/home/jerry/my_first_crypto_quant
```

这是一个研究优先的 Python 加密货币小时级回测框架，当前核心是捕捉短期暴涨的山寨币/pump/"妖币"。当前实盘回测主要走 `pump_mode.enabled=true`，主策略基本被跳过。

主要目录：

```text
src/crypto_quant/
├── backtest/runner.py          # 回测主循环，核心文件
├── config/settings.py          # Pydantic 配置模型，含 PumpModeConfig
├── strategy/engine.py          # 主策略信号生成，pump-only 时基本不跑
├── factors/                    # 动量、趋势、成交量、volume stall 等因子
├── risk/                       # 市场状态、假突破、风险引擎
├── storage/                    # SQLAlchemy 模型、candles 加载、数据库连接
├── universe/                   # 周度币池构建，pump-only 下不是主要交易池
├── data/                       # Binance 数据拉取
├── reporting/                  # 报告输出
└── cli.py                      # CLI 入口

configs/
├── default.yaml
├── v1.yaml                     # 当前策略主要参数
└── v1_pg.yaml                  # 继承 v1.yaml，连接 PostgreSQL

scripts/
└── diagnose_run.py             # 诊断脚本，可查看任意 run 的退出统计
```

关键类和函数：

```text
src/crypto_quant/backtest/runner.py
- ResearchBacktester.run_real()
  回测主入口。加载全市场 candles、预计算指标、逐小时循环、写入 DB。

- ResearchBacktester._precompute_indicators()
  一次性预计算所有因子列（ret_6h/24h/72h、MA20、EMA20、EMA20偏离、ATR、wick_ratio、r1/r2/r3、
  pos24h、vol_trend6 等）。轻量列（r1/r2/r3/pos24h/vol_trend6）在 min_window check 之前计算，
  确保所有帧都有。

- ResearchBacktester._build_snapshot_value_cache()
  为 pump-only 快路径构建 numpy cache。必须列包括 ema20_dev, r1, r2, r3, pos24h, vol_trend6。

- ResearchBacktester._pump_snapshot()
  每小时从 cache 生成全市场 pump 扫描 snapshot。

- ResearchBacktester._detect_pump_regime_snapshot()
  用全市场山寨币 median 24h return、new high ratio、volume expansion 判定 HOT/WARM/COLD。
  注意当前 regime 有 4h cache，避免逐小时 COLD/WARM 抖动。4h 是黄金平衡点，不要改。

- ResearchBacktester._pump_candidates_from_snapshot()
  生成 pump 候选。核心条件、tier A/B、signal_type、risk_multiplier、硬拒绝都在这里。
  r72 > max_72h_return_full_risk 是第一个检查（硬拒绝 >80%）。
  rm 系统已简化：只有 early_confirmed ×1.25 和 bad-b mid vr15-30 ×0.75。

- ResearchBacktester._enter_pump_positions()
  执行 pump 入场。当前是 probe-and-confirm：A 探仓 50%，B 探仓 30%。
  仓位公式：risk_budget = equity × eff_risk_pct × risk_multiplier
  eff_risk_pct = trade_risk_pct × max(peak_ratio_floor, equity/peak_equity)

- ResearchBacktester._update_pump_stop()
  pump 持仓退出。退出顺序：probe confirm/kill → 3h_down → stagnation → time_exit →
  breakeven → lock_2pct → trailing → MFE 地板（仅未确认）。

- ResearchBacktester._pump_stop_anchor_price()
  确认加仓后使用 probe_entry_price 作为呼吸锚点；真实 PnL/统计使用 avg_entry_price。

src/crypto_quant/config/settings.py
- PumpModeConfig (Pydantic BaseModel)
  新增参数必须加到此模型，否则 getattr 读不到。YAML 未知字段被 Pydantic 忽略。
```

## 2. 环境与运行方法

依赖和测试：

```bash
cd /home/jerry/my_first_crypto_quant
uv sync --extra dev
uv run pytest
```

数据库：

```bash
docker compose up -d postgres
```

默认 PostgreSQL：

```text
postgresql+psycopg://crypto_quant:crypto_quant@localhost:5439/crypto_quant
```

OOS-1 回测命令（主窗口 2024-01 → 2025-06，~7 分钟）：

```bash
CQ_BACKTEST_PROGRESS_EVERY=2000 PYTHONPATH=src uv run python - <<'PY'
from crypto_quant.config.settings import load_config
from crypto_quant.backtest.runner import ResearchBacktester
from crypto_quant.storage.database import get_session_factory
from datetime import datetime, UTC
from time import perf_counter

cfg = load_config('configs/v1_pg.yaml')
sf = get_session_factory(cfg.database_url)
start = datetime(2024, 1, 1, tzinfo=UTC)
end = datetime(2025, 6, 1, tzinfo=UTC)
t0 = perf_counter()
with sf() as session:
    result = ResearchBacktester(cfg).run_real(session, start, end)
    session.commit()
elapsed = perf_counter() - t0
orders = result.orders
print(f'Equity: {result.final_equity:,.2f}')
print(f'Return: {(result.final_equity/cfg.backtest.initial_equity-1)*100:.2f}%')
print(f'Orders: {len(orders)} buys={sum(o.side == "buy" for o in orders)} sells={sum(o.side == "sell" for o in orders)}')
print(f'Elapsed: {elapsed/60:.2f} min')
PY
```

OOS-2（2023-01 → 2024-01）：

```bash
CQ_BACKTEST_PROGRESS_EVERY=2000 PYTHONPATH=src uv run python - <<'PY'
from crypto_quant.config.settings import load_config
from crypto_quant.backtest.runner import ResearchBacktester
from crypto_quant.storage.database import get_session_factory
from datetime import datetime, UTC
from time import perf_counter

cfg = load_config('configs/v1_pg.yaml')
sf = get_session_factory(cfg.database_url)
start = datetime(2023, 1, 1, tzinfo=UTC)
end = datetime(2024, 1, 1, tzinfo=UTC)
t0 = perf_counter()
with sf() as session:
    result = ResearchBacktester(cfg).run_real(session, start, end)
    session.commit()
elapsed = perf_counter() - t0
orders = result.orders
print(f'Equity: {result.final_equity:,.2f}')
print(f'Return: {(result.final_equity/cfg.backtest.initial_equity-1)*100:.2f}%')
print(f'Orders: {len(orders)} buys={sum(o.side == "buy" for o in orders)} sells={sum(o.side == "sell" for o in orders)}')
print(f'Elapsed: {elapsed/60:.2f} min')
PY
```

全窗口（2023-01 → 2026-06，~15 分钟）：

```bash
CQ_BACKTEST_PROGRESS_EVERY=2000 PYTHONPATH=src uv run python - <<'PY'
from crypto_quant.config.settings import load_config
from crypto_quant.backtest.runner import ResearchBacktester
from crypto_quant.storage.database import get_session_factory
from datetime import datetime, UTC
from time import perf_counter

cfg = load_config('configs/v1_pg.yaml')
sf = get_session_factory(cfg.database_url)
start = datetime(2023, 1, 1, tzinfo=UTC)
end = datetime(2026, 6, 1, tzinfo=UTC)
t0 = perf_counter()
with sf() as session:
    result = ResearchBacktester(cfg).run_real(session, start, end)
    session.commit()
elapsed = perf_counter() - t0
orders = result.orders
print(f'Equity: {result.final_equity:,.2f}')
print(f'Return: {(result.final_equity/cfg.backtest.initial_equity-1)*100:.2f}%')
print(f'Orders: {len(orders)} buys={sum(o.side == "buy" for o in orders)} sells={sum(o.side == "sell" for o in orders)}')
print(f'Elapsed: {elapsed/60:.2f} min')
PY
```

诊断命令：

```bash
PYTHONPATH=src uv run python scripts/diagnose_run.py \
  --database-url postgresql+psycopg://crypto_quant:crypto_quant@localhost:5439/crypto_quant \
  --run-id <RUN_ID> \
  --output-dir reports/diagnostics/<RUN_ID>_<label>
```

## 3. 当前策略逻辑框架

### 3.1 Regime 判定（每 4h，有缓存）

```text
对全市场山寨币 snapshot：
  HOT:  median_24h_return >= 5% AND new_high_ratio >= 20% AND vol_expansion >= 5%
  WARM: median_24h_return >= 2% AND new_high_ratio >= 10%
  COLD: 其他 → return [] 不开新仓

4h 缓存是血泪教训——无缓存时 WARM/COLD 逐小时抖动错过爆发窗口。不要动这个参数。
```

### 3.2 候选过滤链（信号 bar = 当前 K 线收盘数据，无 look-ahead）

```text
基础过滤（逐币）:
  ret_24h >= 18%
  ret_6h  >= 10%
  price > MA20
  成交量达标 (24h>=200 万 or 6h>=80 万)
  非吹顶 (wick_ratio<0.6 or NOT new_12h_high)
  r72 <= 80%（硬拒绝）
  r72 > 120% && r6/r24 < 0.3 → skip
  vr > 30 → skip
  ema20_dev_pct < 10% → skip
  ema20_dev_pct > 40% → skip

信号分类:
  HOT:  early (r6>=8%, vr>=1.8) 或 confirmed (r72>=35%, vr>=1.5)
  WARM: warm_early (r6>=12%, vr>=2.0)

Tier:
  A: WARM + warm_early + r72 45~80% + vr<=15
  B: 其他

RM（剩余有效乘数）:
  early_confirmed: ×1.25
  B + unconfirmed + ema_rank>=95% + vr 15-30: ×0.75

Score:
  r24×0.45 + r72×0.35 + r6×0.10 + min(vr/5,1)×0.10
  按 score 降序，取前 max_positions=2 个
```

### 3.3 仓位计算（入场时）

```text
stop_distance = max(atr × 1.8, price × 0.10)
eff_risk_pct  = trade_risk_pct(0.05) × max(0.70, equity/peak_equity)
risk_budget   = equity × eff_risk_pct × risk_multiplier
full_quantity = risk_budget / stop_distance
（受 max_symbol_pct=60% 和 max_exposure=100% 约束）

probe: A-tier=50%, B-tier=30% of full_quantity
```

### 3.4 退出机制（t+1 起每小触发）

```text
t+3h:   3h_down: h1<0 && h2<h1 && h3<h2 → 立即退出
        三个条件：①3h前已在 anchor 下方 ②继续跌 ③还在跌。无幅度要求。
        305笔/933，平均亏-5.5%，MFE从未超8%。主要是止血。

t+3.5h: probe confirm / probe_kill
        ret_4h >= 0  → 确认加仓到满仓
        ret_4h <= -2% → 收紧止损到 close-0.3×ATR

t+6h:   stagnation: MFE 从未达 8% → 退出
        184笔/933。筹码不死不活，退出后 max_mfe 平均 21.6%。

t+12h:  time_exit: close < anchor → 退出（很少触发）

breakeven: MFE≥8% → stop = anchor × 1.00
lock_2pct: MFE≥10% → stop = anchor × 1.02
trailing:  MFE≥60% → stop = highest - ATR × 1.5~2.5

MFE 地板 (v2.5G, 仅未确认仓位):
  trade_mfe≥15% → stop ≥ avg × 1.005
  trade_mfe≥25% → stop ≥ avg × 1.03
  trade_mfe≥40% → stop ≥ avg × 1.08
  anchor≈avg 时才触发（未确认仓位）。确认仓位有 breathing 后不触发。
```

### 3.5 probe-anchor breathing（核心设计，不可触碰）

```text
确认加仓后:
  真实成本 PnL: 使用 avg_entry_price
  Stop 呼吸锚点: 使用 probe_entry_price（更低，给妖币洗盘空间）
  lock_2pct/breakeven/trailing 全部锚在 probe_entry_price

这是策略右尾的核心。任何改动（MFE 地板加 confirmed、lock 参考系改 avg）都会
收紧呼吸空间、损失妖币。已验证 3 次。
```

## 4. 当前结果

v2.5G 最终状态（所有参数在 configs/v1.yaml 中）：

```text
OOS-1 2024-01 → 2025-06: Equity ~401k, Return ~+301% (peak_ratio floor 0.70 不含时)
Full 2023-01 → 2026-06: Equity ~199k, +99%, Max DD -72.6%, Calmar 1.36
  （注意：含 peak_ratio floor 0.70 时约 +117%, DD -59.7%）

退出分布 (baseline, 933 笔):
  trailing:      22笔, +1,065k
  lock_2pct:    204笔, +288k
  breakeven:     81笔, +12k
  3h_down:      305笔, -604k
  stagnation:   184笔, -118k
  probe_kill:    79笔, -125k
  init_stop:     55笔, -204k
  time_exit:      3笔
```

## 5. 已验证的结论

### 成功实验（已保留）

- vr>30 硬拒绝, r72>80% 硬拒绝, abs<10% 硬拒绝, abs>40% 硬拒绝
- MFE 地板（仅未确认仓位）
- peak_ratio 仓位缩放 (floor 0.70)
- 代码清理（RM 死代码删除）

### 失败实验（代表性）

- 3h_down 重入 (3 versions): 槽位竞争主导
- 退出分层 (多种): 延迟=占坑→错过新 trailing
- 成交 bar 早期 kill: 系统级联效应吃掉增量
- lock/be 参考系改 avg: 废掉 breathing
- 候选层所有新过滤维度: 信号时刻无区分力，或丢 trailing
- 候选池大小控制/WARM 强度/BTC 环境: 无法预测策略表现

### 核心经验

1. **静态诊断 ≠ 动态回测**：信号 bar(t) vs 成交 bar(t+1) 的 1h 差距导致大量误判
2. **槽位竞争主导一切**：max_positions=2 下，任何改动通过坑位释放/占据产生级联效应
3. **breathing 架构不可触碰**：动一部分就崩溃
4. **做减法比做加法好**：硬拒绝成功，准入/退出型全部失败

## 6. 已知未探索方向

1. **置信度降仓（仓位控制层）**：ema<10% 109笔0 trail，只缩小仓位不拒绝
2. **跨层信息传递**：入场质量→退出耐心（之前实现有数据对齐 bug）
3. **A-tier 重新审视**：只在 WARM 存在，不一定合理

## 7. 防护原则

```text
1. 每次只改一个假设
2. 同时看 OOS-1 和 OOS-2，不只看单窗口
3. INS (2025-06 → 2026-06) 是测试集，不要用于优化
4. trailing 不下降是第一校验指标
5. 失败实验要 revert 或保留记录
```

## 8. OOS 窗口定义

```text
OOS-1: 2024-01-01 → 2025-06-01 (主开发窗口)
OOS-2: 2023-01-01 → 2024-01-01 (熊市验证)
Full: 2023-01-01 → 2026-06-01 (全窗口)
INS:  2025-06-01 → 2026-06-01 (测试集，不可用于优化)
```

## 9. 新 session 检查清单

```text
1. git status --short && git log --oneline -8
2. uv run pytest
3. docker compose up -d postgres
4. pg_isready -h localhost -p 5439
5. 阅读 configs/v1.yaml 了解当前启用参数
6. reports/diagnostics/ 中有 596_v25g_final（OOS-1）和 603_v25g_oos2_fail（OOS-2）的诊断数据
```
