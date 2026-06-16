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

这是一个研究优先的 Python 加密货币小时级回测框架，当前核心是捕捉短期暴涨的山寨币/pump/“妖币”。当前实盘回测主要走 `pump_mode.enabled=true`，主策略基本被跳过。

主要目录：

```text
src/crypto_quant/
├── backtest/runner.py          # 回测主循环，核心文件，约 2000 行
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
└── diagnose_run.py             # 当前重点诊断脚本，已扩展 pump/EMA 交叉诊断

reports/diagnostics/
├── 584_v25d/                   # v2.5D 诊断
└── 586_v25e/                   # v2.5E 诊断
```

关键类和函数：

```text
src/crypto_quant/backtest/runner.py
- ResearchBacktester.run_real()
  回测主入口。加载 universe、全市场 candles、预计算指标、逐小时循环、写入 DB。

- ResearchBacktester._precompute_indicators()
  一次性预计算 ret_6h/24h/72h、MA20、EMA20、EMA20 dev rank、ATR、quote-volume rolling、wick、regime volume expansion 等。

- ResearchBacktester._build_snapshot_value_cache()
  为 pump-only 快路径构建 numpy cache，避免每小时 DataFrame 切片。

- ResearchBacktester._pump_snapshot()
  每小时从 cache 生成全市场 pump 扫描 snapshot。

- ResearchBacktester._detect_pump_regime_snapshot()
  用全市场 median 24h return、new high ratio、volume expansion 判定 HOT/WARM/COLD。
  注意当前 regime 有 4h cache，避免逐小时 COLD/WARM 抖动。

- ResearchBacktester._pump_candidates_from_snapshot()
  生成 pump 候选。核心条件、tier A/B、early/confirmed、risk_multiplier 和 v2.5E bad-B 降风险都在这里。

- ResearchBacktester._enter_pump_positions()
  执行 pump 入场。当前是 probe-and-confirm：A 探仓 50%，B 探仓 30%。

- ResearchBacktester._update_pump_stop()
  pump 持仓退出/止损升级。核心包括 probe confirm、3h_down、stagnation、time_stop、breakeven/lock/trailing。

- ResearchBacktester._pump_stop_anchor_price()
  当前核心设计：确认加仓后，pump stop 使用 probe_entry_price 作为呼吸锚点；真实 PnL/统计使用 avg_entry_price。

src/crypto_quant/config/settings.py
- PumpModeConfig
  pump 策略参数。当前新增了：
  probe_anchor_breathing_enabled
  bad_b_ema_vr_risk_enabled
  bad_b_ema_rank_min
  bad_b_volume_ratio_min
  bad_b_risk_multiplier

scripts/diagnose_run.py
- 读取指定 strategy_run_id，生成 CSV/markdown 诊断。
- 已支持 pump_trade_diagnostics、exit summary、EMA/r6/vr/tier/confirmed 交叉分组、trailing 明细。
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

OOS 回测命令，当前主要基准窗口：

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

INS 建议窗口：

```text
2025-06-01 UTC → 2026-06-01 UTC
```

诊断命令：

```bash
PYTHONPATH=src uv run python scripts/diagnose_run.py \
  --database-url postgresql+psycopg://crypto_quant:crypto_quant@localhost:5439/crypto_quant \
  --run-id <RUN_ID> \
  --output-dir reports/diagnostics/<RUN_ID>_<label>
```

当前已生成：

```text
reports/diagnostics/584_v25d
reports/diagnostics/586_v25e
```

## 3. 当前策略逻辑框架

### 3.1 Pump-only 主流程

当前 `configs/v1_pg.yaml` 继承 `configs/v1.yaml`，`pump_mode.enabled=true`。回测启动时加载数据库中全部 1h spot symbols，而不是只加载 Top60 universe。这是为了不漏 universe 外的妖币。

每小时流程简化为：

```text
1. 加载当前 hour
2. 更新已有持仓 stop/退出
3. pump regime 每 4h 更新一次
4. HOT/WARM 时扫描全市场 pump candidates
5. 按 score 排序，最多开 cfg.pump_mode.max_positions 个 pump 持仓
6. 写 equity curve
```

### 3.2 Pump regime

`_detect_pump_regime_snapshot()`：

```text
HOT:
  median_24h_return >= cfg.regime_hot_24h_return_pct
  new_high_ratio >= cfg.regime_hot_new_high_ratio
  volume_expansion_ratio >= 0.05

WARM:
  median_24h_return >= cfg.regime_warm_24h_return_pct
  new_high_ratio >= cfg.regime_warm_new_high_ratio

COLD:
  其他情况，不开 pump 新仓
```

重要历史经验：regime 必须 4h cache。无 cache 时 WARM/COLD 逐小时抖动，会错过妖币爆发窗口，收益大幅下降。

### 3.3 Candidate 条件

核心在 `_pump_candidates_from_snapshot()`。

基础条件：

```text
history >= 73
price > 0
ret_24h >= min_24h_return
ret_6h >= min_6h_return
ret_72h 有效
price above MA20
quote_volume_24h >= min_quote_volume_24h 或 quote_volume_6h >= min_quote_volume_6h
wick blowoff 不触发
ret_72h 不超过 chase/entry 上限
```

成交量口径：

```text
pump 入场使用 quote_volume，也就是成交额
volume_ratio = 6h quote volume / previous avg 6h quote volume
```

注意：pump regime 的 volume expansion 当前仍用 base volume，而不是 quote_volume。诊断显示差异很小：12,359 小时里只有 21 小时不同，优先级低。

信号分类：

```text
early:
  ret_24h >= 18%
  ret_6h >= early_6h_return
  volume_ratio >= early_volume_ratio_min

confirmed:
  ret_72h >= 35%
  ret_24h >= 18%
  volume_ratio >= volume_ratio_min

early_confirmed:
  同时满足 early 和 confirmed
```

HOT/WARM 权限：

```text
HOT:
  confirmed 或 early 可交易

WARM:
  warm_early_ok 可交易
```

Tier：

```text
A:
  regime == WARM
  early or warm_early_ok
  0.45 <= ret_72h <= 0.86
  volume_ratio <= 15

B:
  其他可交易 pump 候选
```

Score：

```text
score = ret_24h * 0.45
      + ret_72h * 0.35
      + ret_6h * 0.10
      + min(volume_ratio / 5.0, 1.0) * 0.10
```

### 3.4 Probe-and-confirm

入场不是一次满仓，而是探仓后确认加仓：

```text
probe_entry_price:
  第一次探仓成交价

confirm_entry_price:
  约 4h 后走势未坏时，确认加仓成交价

avg_entry_price:
  整笔交易真实加权成本
```

当前探仓比例：

```text
A: 50% full_quantity
B: 30% full_quantity
```

确认规则：

```text
持仓 >= 3.5h 且 close / stop_anchor - 1 >= 0:
  加仓到 full_quantity
```

### 3.5 当前最核心设计：probe-anchor breathing

这来自 v2.0 的“bug”复盘，但已经被显式工程化。

v2.0 bug 行为：

```text
probe = 100
confirm add = 120
avg_entry = 110
但 stop 仍使用 entry_price=100
```

这导致所谓 breakeven/lock_2pct 对整笔交易其实不是保本/锁盈，而是给确认后的妖币 6%-10% 以上洗盘空间。

当前正确设计：

```text
真实成本、PnL、final_trade_ret_pct、PositionRecord:
  使用 avg_entry_price

pump stop 呼吸锚点:
  如果 probe_anchor_breathing_enabled 且 position.probe_confirmed:
    使用 probe_entry_price
  否则:
    使用 avg_entry_price
```

对应函数：

```text
_trade_entry_price(position)       # 真实成本锚点
_pump_stop_anchor_price(position)  # pump stop 呼吸锚点
```

不要轻易改动这个机制。它是当前右尾收益核心。

### 3.6 Pump exits

`_update_pump_stop()` 顺序：

```text
1. probe confirm / probe kill
2. 3h_down 强制退出
3. stagnation_exit
4. time_exit
5. profit_protect
6. pump_breakeven
7. pump_lock_2pct
8. pump_trailing_stop
```

当前名字仍有语义歧义：

```text
pump_breakeven / pump_lock_2pct
```

在 probe-anchor breathing 下，相对整笔 avg_entry 不一定是真保本/锁 2%。诊断时必须看：

```text
avg_entry_price
probe_entry_price
stop_anchor_price
final_trade_ret_pct
stop_vs_avg_pct
```

### 3.7 v2.5E 当前新增规则

当前最新 commit：

```text
136756e v2.5E reduce bad B EMA-volume chase risk
```

只对最窄坏 B 子集降风险：

```text
bad_b_ema_vr_risk_enabled = True
bad_b_ema_rank_min = 0.95
bad_b_volume_ratio_min = 30.0
bad_b_risk_multiplier = 0.50
```

触发条件：

```text
tier == B
sig_type == early
ema20_dev_rank_2160h >= 0.95
volume_ratio > 30
```

动作：

```text
risk_multiplier *= 0.5
```

不拒绝，不改 stop，不动 probe-anchor breathing。

## 4. 当前结果与诊断

关键版本：

```text
cb03363 v2.5D make probe anchor breathing explicit
136756e v2.5E reduce bad B EMA-volume chase risk
```

OOS 窗口：

```text
2024-01-01 UTC → 2025-06-01 UTC
```

结果：

```text
v2.5D:
  Equity: 303,148.48
  Return: +203.15%
  Orders: 1371

v2.5E:
  Equity: 316,561.37
  Return: +216.56%
  Orders: 1372
```

v2.5E 相比 v2.5D：

```text
bad-B 降风险触发 28 次
trailing_stop 数量仍为 16
没有砍掉右尾
```

退出 PnL 对比：

```text
pump_3h_down:
  -452,107 -> -436,738

pump_lock_2pct:
  +210,832 -> +221,003

pump_trailing_stop:
  +810,007 -> +827,978
```

v2.5D 诊断重点：

```text
pump_trailing_stop:
  16 笔，+810,007
  这是策略生命线

pump_3h_down:
  203 笔，-452,107

pump_initial_stop:
  45 笔，-173,977

early_confirmed:
  259 笔，+263,707

early:
  354 笔，-62,736
  但也藏着 8 笔 trailing
```

EMA 诊断重点：

```text
ema20_dev_rank_2160h 95-100%:
  595 笔
  +209,244
  trailing 15 / 16
```

结论：

```text
EMA20 高偏离不是退出信号，也不是硬过滤信号。
妖币天然处在 EMA20 高偏离区。
EMA20 有价值的方式是和 vr/r6/tier/confirmed 组合识别坏追高。
```

坏组合：

```text
ema2160 95-100% + vr > 30:
  -54,415
  trailing 0

B + unconfirmed + ema2160 95-100% + vr > 30:
  -25,126
  trailing 0
```

这就是 v2.5E 的依据。

## 5. 过去优化失败经验

### 5.1 v2.5A：只修 avg_entry stop anchor

目标：修复 v2.0 的 lock/breakeven 锚点 bug。

结果：

```text
OOS Equity: 114,825
```

失败原因：

```text
语义正确，但砍掉了确认加仓后妖币的呼吸空间。
很多妖币会先冲、再洗、再拉。
用 avg_entry 锚定 breakeven/lock 太紧，会在二次拉升前被洗出去。
```

教训：

```text
真实成本和 stop 呼吸锚点必须分离。
```

### 5.2 v2.5B：简单压缩 B 未确认

动作：B 未确认降仓、快速退出。

结果：

```text
OOS Equity: 97,540
```

失败原因：

```text
B 不是纯垃圾，B unconfirmed 里也藏着右尾。
一刀切 B 会误杀后续 trailing。
```

教训：

```text
不能按 B 标签粗暴处理；必须找坏 B 子集。
```

### 5.3 v2.5C：粗糙 core confirmed breathing

动作：只给按 r72/r6/vr 定义的 core_confirmed 固定 -8% breathing。

结果：

```text
OOS Equity: 86,864
```

失败原因：

```text
core_confirmed 定义太粗糙。
真正右尾不只来自健康 core，也来自 burst/early/unconfirmed 的强势延续。
```

教训：

```text
不要用未经诊断的标签决定 breathing。
```

### 5.4 其他重要经验

```text
regime 不能逐小时重算无缓存；必须 4h cache，否则 WARM/COLD 抖动会错过爆发。

不要把 EMA20 偏离作为硬退出。

不要硬拒绝高 EMA 偏离，因为 15/16 trailing 都在 2160h EMA 偏离 95-100% 分箱。

不要一次混改 EMA、B、stop、仓位、regime。
```

## 6. 过拟合防护原则

后续优化必须遵守：

```text
1. 每次只改一个假设。
2. 优先 risk_multiplier 轻微调权，不优先 hard filter。
3. 阈值用宽区间和圆数字，不寻找精确最优点。
4. 新规则必须有市场解释，不只是分箱收益好看。
5. 必须确认 trailing_stop 数量不下降，或下降原因可解释。
6. 同时看 OOS 和 INS，不只看单窗口。
7. 失败实验要 revert 或保留记录，不要悄悄覆盖。
```

每个实验必须记录：

```text
final equity
return
orders
max drawdown
trailing_stop 数量和 PnL
3h_down 数量和 PnL
initial_stop 数量和 PnL
新规则触发次数
新规则是否改变交易路径
新规则是否减少亏损而不是偶然多吃一笔大赢家
```

## 7. 下一版优化方案，请先审查再执行

### 7.1 当前推荐方向

主线：

```text
保留 v2.5D/v2.5E 的右尾机制。
继续用 EMA + vr + r6 识别坏追高 B。
只做风险降权，不做硬过滤。
每次只扩大一个边界。
```

### 7.2 候选实验 v2.5F

谨慎扩大 v2.5E 的坏 B 子集。

当前 v2.5E 已处理：

```text
B + unconfirmed + ema2160 >= 95% + vr > 30
risk *= 0.5
```

候选 v2.5F：

```text
B + unconfirmed
ema20_dev_rank_2160h >= 95%
15 < volume_ratio <= 30
risk *= 0.75
```

不要直接 0.5，因为 `vr 15-30` 区间还有少量 trailing。建议先用 0.75。

执行前请先审查诊断：

```text
reports/diagnostics/584_v25d/pump_tier_confirmed_ema_vr.csv
reports/diagnostics/586_v25e/pump_tier_confirmed_ema_vr.csv
```

确认：

```text
B + unconfirmed + ema2160 95-100% + vr 15-30
是否仍为明显负贡献
是否存在 trailing
样本是否足够
```

### 7.3 候选实验 v2.5G

如果 v2.5F 有效，再考虑 r6 维度。

候选：

```text
B + unconfirmed
ema2160 >= 95%
r6 25%-50%
risk *= 0.75
```

执行前看：

```text
reports/diagnostics/584_v25d/pump_tier_confirmed_ema_r6.csv
reports/diagnostics/586_v25e/pump_tier_confirmed_ema_r6.csv
```

注意：不能对 `early_confirmed` 动手。当前 `early_confirmed` 是正贡献主力。

### 7.4 暂不建议做的事

暂不建议：

```text
1. 硬拒绝 B。
2. 硬拒绝 EMA 高偏离。
3. 改 probe-anchor breathing。
4. 改 pump_breakeven / pump_lock_2pct 为 avg_entry 锚点。
5. 做复杂利润锁定阶梯。
6. 一次扩大到多个 vr/r6/r72 组合。
```

利润保护可以继续诊断，但不要贸然改，因为右尾高度集中，任何锁盈阶梯都可能砍掉 `pump_trailing_stop`。

## 8. 建议新 agent 的第一步

请按这个顺序开始：

```text
1. git status --short && git log --oneline -8
2. uv run pytest
3. 阅读：
   - src/crypto_quant/backtest/runner.py
   - src/crypto_quant/config/settings.py
   - scripts/diagnose_run.py
   - reports/diagnostics/584_v25d/*.csv 中的关键表
   - reports/diagnostics/586_v25e/*.csv 中的关键表
4. 用诊断结果复核 v2.5F 是否合理。
5. 如果合理，只做 v2.5F 一个实验。
6. 跑短窗口 2024-11。
7. 跑 OOS 2024-01 → 2025-06。
8. 跑 INS 2025-06 → 2026-06。
9. 基于三组结果决定保留、回退或继续。
```

短窗口命令可用：

```bash
CQ_BACKTEST_PROGRESS_EVERY=250 PYTHONPATH=src uv run python - <<'PY'
from crypto_quant.config.settings import load_config
from crypto_quant.backtest.runner import ResearchBacktester
from crypto_quant.storage.database import get_session_factory
from datetime import datetime, UTC
from time import perf_counter

cfg = load_config('configs/v1_pg.yaml')
sf = get_session_factory(cfg.database_url)
start = datetime(2024, 11, 1, tzinfo=UTC)
end = datetime(2024, 12, 1, tzinfo=UTC)
t0 = perf_counter()
with sf() as session:
    result = ResearchBacktester(cfg).run_real(session, start, end)
    session.commit()
elapsed = perf_counter() - t0
orders = result.orders
print(f'Equity: {result.final_equity:,.2f}')
print(f'Return: {(result.final_equity/cfg.backtest.initial_equity-1)*100:.2f}%')
print(f'Orders: {len(orders)} buys={sum(o.side == "buy" for o in orders)} sells={sum(o.side == "sell" for o in orders)}')
print(f'Elapsed: {elapsed:.2f}s')
PY
```

## 9. 当前最终状态

当前最新 commit：

```text
136756e v2.5E reduce bad B EMA-volume chase risk
```

当前策略基线：

```text
v2.5E
OOS 2024-01 → 2025-06:
  Equity: 316,561.37
  Return: +216.56%
  Orders: 1372
```

下一步不是追求单次 OOS 更高，而是验证规则是否稳健，尤其是 INS 和 trailing 右尾是否保持。
