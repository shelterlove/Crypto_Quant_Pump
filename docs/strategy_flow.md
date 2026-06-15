# 🐲 妖币捕手 v1.4.2 — 策略全流程技术文档

> 最后更新：2026-06-12
> 基于 tag v1.4.2 / configs/v1.yaml

---

## 一、数据层

### 1.1 Universe 构建（周频）

`WeeklyUniverseService` 每周一 UTC 00:00 运行一次：

1. 取过去 **30 天**的 1d K 线数据
2. 过滤：日均 quote_volume < `min_quote_volume_30d`（默认 $50M）的币剔除
3. 按 quote_volume 排名，取 **Top N**（默认 60）
4. 排除 mega-cap（BTC、ETH、BNB、SOL 等 20 个）和稳定币/黄金币
5. 结果持久化到 `universe_snapshots` 表，key 为 `effective_from`（周一 00:00 UTC）

**Grace Period**：回测时当前周的 universe = 本周 snapshot ∪ 上周 snapshot（防止周一刚进榜的妖币周二就被踢出）。

### 1.2 Candle 预加载与预处理

回测启动时一次性加载全部所需 candles：
- **Main 模式**：只加载 universe 中出现的所有币种 + BTC
- **Pump 模式**：加载数据库中**全部** 1h spot 币种（430+），确保不会漏掉 universe 外的妖币

然后对所有 candle DataFrame 做 **预计算**（`_precompute_indicators`）：

| 列名 | 来源 | 内容 |
|------|------|------|
| `ret_4h`, `ret_24h`, `ret_48h`, `ret_72h` | `close / close.shift(window) - 1` | 各窗口收益率 |
| `weighted_return` | `Σ(ret × weight)` | 加权动量分 |
| `ma20` | `close.rolling(20).mean()` | 1H MA20 |
| `atr14` | `tr.rolling(14).mean()` | 14-period ATR |
| `volume_score_col` | 量价关系评分（0-1） | 放量涨=高分，放量滞涨=低分 |
| `trend_score_col` | 1H MA20 + 4H MA20 + 斜率 + 24h回撤 | 趋势结构分（0-1） |
| 4H 专用 | `ma50`（仅 BTC） | BTC 4H MA50 |

### 1.3 Position Cache（性能优化）

为每个 `(symbol, timestamp)` 预建 O(1) 查找表。回测每小时切片时不用 `searchsorted`（O(log n)），直接字典查找 → 大幅加速。

---

## 二、主循环（逐小时）

回测区间 `[start, end]`，以 BTC 1h K 线的时间轴为基准，逐小时迭代。

### 每小时的执行顺序：

```
 1. reset_daily_loss        ← 过 UTC 0 点则清零
 2. 确定 active_symbols      ← universe_map[monday(now)]
 3. Slice candles            ← 取最近 200 根 bar
 4. 极端动量扫描              ← 扫描 universe 外的币，wret≥50% 加入 active
 5. Market State 判定        ← BTC 4H MA50 + breadth + fast valve
 6. Pump Routine（如启用）
    ├── Regime 检测          ← 全体扫描：24h median ret, new high%, vol exp%
    ├── Candidate 扫描       ← 逐一检查所有币的 pump 条件
    ├── Enter Pump           ← 入场执行
    └── Update Pump Stop     ← 3h down / stagnation / time / trailing
 7. 更新 Position State      ← strong → weakening → failed
 8. 执行 Weakening Reduce    ← 减仓 1/3 + 收紧止损
 9. 检查 Hard Risk Limits    ← 超限则标记 weakening
10. Process Stops（全部持仓）
    ├── Pump: _update_pump_stop → 可能强制退出（3h_down 等）
    ├── Main: _update_position_stop → 更新止损价
    └── Stop 触发检查: low[-1] ≤ stop_price → _full_exit
11. 如 pump_mode.enabled=false：
    ├── 计算 Factor Scores
    ├── generate_targets()   ← 信号生成 + 过滤
    ├── Swap 机制            ← 换仓逻辑
    ├── RiskEngine 批准      ← 二次风控
    └── _enter_positions     ← 入场执行
12. 写入 Equity Curve        ← equity, cash, exposure, drawdown
```

---

## 三、Market State（市场状态判定）

每根 K 线判定一次，四个组件：

### 3.1 BTC 4H MA50 趋势

```
close > MA50 且 MA50 斜率 > -0.5% → risk_on
close < MA50 且 MA50 斜率 ≤ -0.5% → defensive
其他 → caution
```

### 3.2 Market Breadth（市场广度）

扫描所有 4H candle 的币种：`close > MA20` 的比例。

```
breadth < 0.20 → defensive（极度恶化）
breadth < 0.25 → caution（广度暂停）
breadth < 0.35 → caution（广度偏弱）
```

### 3.3 Fast Risk Valve（快风险阀）

BTC 1h 跌幅 ≥ **7%** → 立即触发 defensive，禁止一切开仓。

### 3.4 最终状态与交易权限

| 状态 | max_positions | risk_multiplier | momentum gate | score gate |
|------|:---:|:---:|:---:|:---:|
| **risk_on** | 3 | 1.0 | 0% | 0.00 |
| **caution** | 1 | 0.5 | 8% | 0.90 |
| **defensive** | 1 | 1.0 | 30% | 0.80 |
| **fast_risk_valve** | 0 | — | — | — |

**极端动量绕过**（wret ≥ 50%）：无视所有断路器，直接开仓。这是为妖币特设的逃生门。

---

## 四、Pump 模块（妖币捕手）🐲

### 4.1 Pump Regime 检测

每小时扫描**全量** 1h candles（非仅 universe），计算三个指标：

| 指标 | 计算方式 | HOT 阈值 | WARM 阈值 |
|------|---------|:---:|:---:|
| median_24h_return | 全市场 24h 收益中位数 | ≥ 5% | ≥ 2% |
| new_high_ratio | 创新 24h 高点的比例 | ≥ 20% | ≥ 10% |
| vol_exp_ratio | 近 6h 量超 50h 均值 1.5x 的比例 | ≥ 5% | — |

- **HOT**：median_ret ≥ 5% 且 new_high ≥ 20% 且 vol_exp_rate ≥ 5%
- **WARM**：median_ret ≥ 2% 且 new_high ≥ 10%
- **COLD**：不满足以上 → **不扫描 candidate，不交易 pump**

### 4.2 Pump Candidate 扫描

每个币逐项检查（全部满足才入选）：

| 检查项 | 条件 | 配置 |
|--------|------|------|
| 最低成交量 | `quote_vol_24h ≥ $2M` 或 `quote_vol_6h ≥ $800k` | `min_quote_volume_24h`, `min_quote_volume_6h` |
| 成交量放大 | `6h vol / avg_6h_vol_prev_24h` | `volume_ratio_min: 1.5` / `early_volume_ratio_min: 1.8` |
| 价格在 MA20 之上 | `close > MA20` | 硬性条件 |
| **confirmed** 信号 | `ret_72h ≥ 35%` 且 `ret_24h ≥ 18%` 且 `vol_ratio ≥ 1.5` | WARM 和 HOT 都允许 |
| **early** 信号 | `ret_24h ≥ 18%` 且 `ret_6h ≥ 8%` 且 `vol_ratio ≥ 1.8` | **仅 HOT 允许** |
| 冲高回落排除 | 上影线 wick ≥ 60% 且创新高 → 排除 | `blowoff_wick_ratio` |
| 涨太多排除 | `ret_72h > 3.50` → 排除（当前 disabled） | `max_72h_return_chase` |
| 动量衰减排除 | `ret_72h > 120%` 且 `ret_6h / ret_24h < 0.30` → 排除 | 防止追顶部 |
| 仓位冷却 | 同一 symbol 上次 trailing 止盈后 **24h** 内禁止重入 | per-symbol cooldown |
| 全局冷却 | 连亏 3 次 → **12h** 冷却；连亏 5 次 → **24h** 冷却 | `cooldown_minutes`, `extended_cooldown_minutes` |

### 4.3 Pump 候选评分

```
score = ret_24h × 0.45 + ret_72h × 0.35 + ret_6h × 0.10
      + min(vol_ratio / 5.0, 1.0) × 0.10
```

按 score 降序排列，取前 `max_positions`（默认 2）个。

### 4.4 Risk Multiplier（风险系数）

| ret_72h 区间 | risk_multiplier | 说明 |
|:---|:---:|:---|
| ≤ 120% | 1.00 | 全风险 |
| 120% ~ 220% | 0.70 | 降风险 |
| 220% ~ 350% | 0.40 | 追涨模式 |
| **early + confirmed 同时满足**（ret_72h ≤ 120%） | × 1.25 | 加码 |

### 4.5 Pump 仓位计算

```
stop_distance  = max(ATR × 1.8,  price × 10%)
risk_budget    = equity × 5% × risk_multiplier
quantity_risk  = risk_budget / stop_distance
quantity_cap   = equity × 60% / price        ← 单币最高 60%
quantity_total = equity × (100% - 已有pump敞口) / price
quantity       = min(quantity_risk, quantity_cap, quantity_total)
```

典型结果：price=$1, ATR=0.06, risk=5% → `stop_dist = $0.108`, `qty_risk = $5000/$0.108 ≈ 46,298`, qty ≤ $60k / $1 = 60,000 → 实际取约 46% 仓位。

### 4.6 Pump 退出机制（多层）

执行顺序（最先触发的胜出）：

#### ① 3h 连续下跌退出（v16 新增，零假阳性）

```python
# 入场后，找最近的 3 根小时 K 线
h1 = close[-3] / entry_price - 1    # 第 3 根收益
h2 = close[-2] / entry_price - 1    # 第 2 根收益
h3 = close[-1] / entry_price - 1    # 第 1 根收益
if h1 < 0 and h2 < h1 and h3 < h2:  # 三根加速下跌
    → "pump_3h_down" 强制退出
```

数据支撑：在 v16 中此规则贡献了 -$2,477 PnL（24 笔退出），但在拖尾赢家上零假阳性——从未误杀大赢家。

#### ② Stagnation 退出

```
持仓 ≥ 6h 且 MFE < 8% → "pump_stagnation_exit"
```

#### ③ Time 退出

```
持仓 ≥ 12h 且当前价 < entry_price → "pump_time_exit"
```

#### ④ Pump 止损升级（多层 Trailing Stop）

| MFE 达到 | 止损位 | 机制 |
|:---|:---|:---|
| ≥ 15% | `entry × (1 - 3%)` | `profit_protect` |
| ≥ 30% | `entry_price` | `breakeven` |
| ≥ 60% | `highest - ATR × 2.5` | `trailing_1` |
| ≥ 100% | `highest - ATR × 2.0` | `trailing_2` |
| ≥ 180% | `highest - ATR × 1.5` | `trailing_3` |

#### ⑤ 初始止损

```
stop = entry - max(ATR × 1.8, entry × 10%)
```

### 4.7 Pump 交易后 Cooldown

- **Trailing 止盈**：该 symbol **24h** 禁止重入（防止盈利回吐）
- **亏损退出**：累计连亏次数
  - 3 连亏 → 全局 **12h** 冷却
  - 5 连亏 → 全局 **24h** 冷却

---

## 五、Main 策略（传统动量轮动）

> **当前默认关闭**（`pump_mode.enabled: true` 时跳过 main 信号生成）
> 760 笔交易仅贡献 +$4 PnL，已被 pump 模块取代。

### 5.1 Factor 计算

只在 universe 内的 active_symbols 上计算。

#### Momentum Score（PctRank）

```
weighted_return = ret_4h × 0.25 + ret_24h × 0.35
                + ret_48h × 0.25 + ret_72h × 0.15
```

→ `momentum_score = weighted_return` 的 cross-sectional rank（0-1）

#### Volume Score

来自预计算列 `volume_score_col`：

| 量价关系 | 评分 |
|:---|:---|
| `vol_ratio ≥ 1.2` 且涨 | `0.8 + bonus` |
| `vol_ratio ≥ 1.2` 且横盘/跌 | `0.1 ~ 0.3` |
| `0.7 ≤ vol_ratio < 1.2` 且涨 | `0.5 + return × 5` |
| `vol_ratio < 0.7` 且横盘 | `0.4` |
| 跌幅 > 1% | `0.1` |

#### Trend Score

来自预计算列 `trend_score_col`：

```
= above_1h_ma20 × 0.33     （close > 1H MA20 ?）
+ above_4h_ma20 × 0.33     （close > 4H MA20 ?）
+ slope_4h_ma20 × 0.34     （4H MA20 斜率，clip 0-1）
- dd_penalty                （距 24h 高点回撤 > 5% 惩罚，最多扣 0.2）
```

clip 到 `[0, 1]`。

#### Composite Final Score

```
final_score = momentum_score × 0.50
            + volume_score  × 0.25
            + trend_score   × 0.25
```

### 5.2 信号过滤（逐层递减）

每个候选要过以下关卡：

| # | 过滤 | 条件 | 动作 |
|:--:|:---|:---|:---|
| 1 | False Breakout Top-N Cap | 假突破环境 | 只取 Top 2 |
| 2 | Blow-off Top | 上影线 ≥ 60% + 创新高 + 放量 | 排除 |
| 3 | Volume Stall（放量滞涨） | 6 根中 ≥ 4 根放量 1.5x，价格不涨反跌 | 排除 + 4h 冷却 |
| 4 | Exhausted 72h | `ret_72h > 120%` | 排除 |
| 5 | Cooldown | 放量滞涨冷却期内 | 跳过 |
| 6 | Score/Momentum Gate | 取决于市场状态（见 §3.4） | 排除 |

### 5.3 仓位计算

```
stop_distance   = ATR × 2.0
max_trade_loss  = equity × 1%
quantity_risk   = max_trade_loss / stop_distance
quantity_cap    = equity × 35% / price
quantity        = min(quantity_risk, quantity_cap)
```

### 5.4 Main 退出机制

#### Hybrid Stop（入场后优先）

```
structure_stop = 最近 6-12 根 K 线的 swing low
如果 structure_stop ∈ [1.0, 2.5] × ATR 且 > ATR_stop：
    → 使用 structure stop
否则：
    → stop = entry - ATR × 2.0
```

#### Breakeven Stop

```
high ≥ entry + ATR × 2.0 → stop 提升至 entry_price
```

#### Trailing Stop

```
high ≥ entry + ATR × 2.0 → stop = max(high - ATR × 2.5, close) 逐根上移
```

#### Defensive Tighten

```
market = defensive 且 close ≤ entry + ATR → stop 收紧至 close - ATR × 1.0
```

---

## 六、Position State Machine（主策略持仓管理）

每个 main 持仓有三种状态：

### strong → weakening（触发任一条件）

| 条件 | 含义 |
|:---|:---|
| `rank > 5` 且 `close < 1H MA20` | 动量排名下滑 + 跌破短均 |
| `rank > 5` 且 距最高点回撤 > `1.5 × ATR` | 排名下滑 + 深回撤 |
| 放量滞涨触发 | 滞涨警告 |

### weakening → failed（触发任一条件）

| 条件 | 含义 |
|:---|:---|
| `close < 1H MA20 × 0.95` | 明显跌破 MA20 |
| 距最高点回撤 > `3.0 × ATR` | 严重回撤 |
| `rank > 15` | 排名彻底出局 |

### Weakening Reduce（减仓）

```
卖出 1/3 仓位（weakening_reduction_pct = 0.33）
剩余仓位 trailing stop 收紧至 highest - ATR × 1.5
```

### Failed → 全退出

---

## 七、Swap 换仓机制

每小时检查：如果有新 signal 的 score 比已有持仓高 **15%** 以上（`swap_score_advantage: 1.15`）：

- 持仓状态为 weakening/failed → 可换出
- 新 signal 优势 ≥ 25%（`swap_strong_score_advantage`）→ 即使 strong 也可换

每天最多换 2 次（`swap_max_per_day`）。

---

## 八、全局风控

### 8.1 日亏损限制

```
daily_realized_loss > equity × 8%  → 当日禁止开仓
daily_realized_loss > equity × 12% → 进入 defensive
```

极端动量（wret ≥ 50%）可绕过。

### 8.2 连续亏损暂停

```
连续 3 笔亏损             → 禁止开仓（cooldown 自恢复）
最近 15 笔中 10 笔亏损     → 禁止开仓
```

### 8.3 Hard Risk Limits（main 持仓中）

| 限制 | 触发条件 | 动作 |
|:---|:---|:---|
| 单仓波动率风险 | `qty × ATR × 2.0 / equity > 3%` | → weakening |
| ATR 扩张 + 回撤 | `atr_expansion > 3x` 且 `DD > 1.5 × ATR` | → weakening |

### 8.4 Pump 特有风控

- 单日总 pump 亏损 > `equity × 15%` → 当日禁止 pump
- 总 pump 敞口 ≤ `100%` equity
- 连亏 3/5 次全局 cooldown

---

## 九、数据流全景速查

```
启动
 │
 ├── 加载 config/v1.yaml（继承 default.yaml）
 ├── 从 DB 加载 universe snapshots（按周）
 ├── 加载全量 1h/4h candles 到内存
 ├── 预计算所有因子列（weighted_return, ma20, atr14, volume_score, trend_score）
 ├── 构建 position cache（O(1) 切片）
 │
 └── 逐小时循环 ──────────────────────────────────────┐
     │                                                  │
     ├── 确定 active_symbols（universe + grace）        │
     ├── BTC 1h 暴跌 ≥ 7%？→ fast_risk_valve，全天禁止  │
     ├── 极端动量扫描（wret≥50% 币加入 active）         │
     ├── Market State 判定                              │
     │                                                   │
     ├── Pump Routine ───────────────────────────────┐  │
     │   ├── regime = _detect_pump_regime(全量扫描)  │  │
     │   ├── regime ∈ {HOT, WARM}？                  │  │
     │   │   ├── _pump_candidates() 逐币筛选        │  │
     │   │   ├── 按 score 排序取 top 2              │  │
     │   │   └── _enter_pump_positions() 执行入场   │  │
     │   └── _update_pump_stop() 每个 pump 持仓     │  │
     │       ├── 3h 连跌？→ 强制退出                │  │
     │       ├── 停滞 ≥ 6h？→ 退出                  │  │
     │       ├── 超时 ≥ 12h 且亏损？→ 退出          │  │
     │       └── 更新 trailing/breakeven/profit 停止 │  │
     │                                                   │
     ├── Main Routine（pump_mode.enabled → 跳过）      │
     │   ├── 更新 position state（strong→weak→failed）  │
     │   ├── weakening reduce（减仓 1/3）               │
     │   ├── hard risk limits（ATR 扩张触发 weakening） │
     │   ├── process stops（ATR/breakeven/trailing）    │
     │   ├── generate_targets()                        │
     │   ├── swap 换仓                                 │
     │   └── enter_positions()                          │
     │                                                   │
     ├── 检查所有持仓 stop 是否被击穿 → full_exit       │
     └── 写入 equity curve ────────────────────────────┘
```

---

## 十、Exit Reason 分布（v16, 3% risk, +315.8%）

| Exit Reason | 笔数 | PnL |
|:---|---:|---:|
| trailing（pump 拖尾止盈） | 11 | **+$9,489** |
| other | 6 | +$7 |
| init（初始止损） | 2 | -$337 |
| stagnation（停滞） | 20 | -$380 |
| time（超时） | 16 | -$419 |
| **3h_down（3h 连跌）** | 24 | **-$2,477** |

---

## 十一、关键结果记录

| 版本 | 配置 | Return | MaxDD | 笔数 |
|:---|:---|---:|---:|---:|
| v7 (pump-only, 24h cool) | 3% risk, pump only | +287.0% | -34.0% | 200 |
| v16 (+3h_down exit) | 3% risk, pump only | **+315.8%** | -33.5% | 196 |
| 5% risk test | 5% risk, pump only | **+574.9%** | -42.8% | 158 |
| Main strategy alone | 1% risk, 3pos | ~+$4 | — | 759 |

---

## 十二、未完成的待办

1. **Walk-forward 验证**：数据下载中（2024-2025 全量），下载后用 2024 年训练、2025 年测试
2. **Survivorship bias**：需要从 data.binance.vision 下载退市币历史数据
3. **3h_down 优化**：24 笔退出亏 $2,477 —— 需分析是否误杀、能否减少
4. **Early 信号分析**：early 信号砍了 53 笔交易但利润只降 $398 —— 信号本身是否有价值需进一步验证
