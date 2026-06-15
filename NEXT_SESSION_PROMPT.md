## 项目概述

两个加密货币量化交易项目：

1. **`/home/jerry/my_first_crypto_quant/`** — 自研回测框架（当前使用中），白皮书驱动的山寨币小时级轮动策略
2. **`/home/jerry/my_btc_eth_bnb_spot/my_first_crypto_spot/`** — Freqtrade 版本（策略代码已写，网络问题未跑通）

## 当前策略配置（自研框架，最终优化版）

```
策略版本: v1.0.0 (基于白皮书 v1.4.2)
配置文件: configs/v1_pg.yaml (继承 v1.yaml → default.yaml)

核心参数:
  trade_risk_pct: 1%          (单笔风险)
  max_positions: 3            (最大持仓)
  max_symbol_position_pct: 60% (单币上限)
  min_quote_volume_30d: 10M   (交易池成交量门槛)
  mega_cap_exclude: 20个大市值币

市场环境:
  risk_on:   复合评分选币，不限动量
  caution:   15% 动量门槛
  defensive: 30% 动量 + 0.80 综合分 + 不限仓
  fast_risk_valve: BTC -7%/1h → 全拒

关键过滤器:
  - 120% 耗尽过滤 (72h涨幅>120%不再追)
  - 50%+ 极端动量绕过熔断
  - 1周宽限期 (防止泵前被踢出池)
  - 放量滞涨检测 + 4h冷却
  - 冲高回落检测
  - 混合止损 (ATR + 结构位)

熔断:
  daily_loss_limit: 8%
  consecutive_loss_pause: 8次
  recent_trades_loss: 15次中10次
```

## 回测结果

| 版本 | 年化收益 | 最大回撤 | 说明 |
|------|:---:|:---:|------|
| 分月独立 | +47.7% | -13% | 每月重置本金，参考值 |
| **连续复利** | **+16.5%** | **-16.9%** | **真实结果** |

最佳月 +19.4%（11月，GIGGLE/FIL/DASH 爆发），最差月 -13.1%（10月，全市场下跌）

## 关键命令

```bash
# 进入项目
cd /home/jerry/my_first_crypto_quant

# 跑测试
uv run pytest tests/ -q

# 单月回测
uv run python -c "
from crypto_quant.config.settings import load_config
from crypto_quant.storage.database import get_session_factory
from crypto_quant.backtest.runner import ResearchBacktester
from datetime import datetime, UTC

cfg = load_config('configs/v1_pg.yaml')
session_factory = get_session_factory(cfg.database_url)
with session_factory() as session:
    r = ResearchBacktester(cfg).run_real(session, 
        datetime(2025,11,1,tzinfo=UTC), datetime(2025,12,1,tzinfo=UTC))
    print(f'{r.final_equity:.2f} {len(r.orders)} orders')
"

# 连续12个月回测 ($1000起始，约7分钟)
uv run python -c "
from crypto_quant.config.settings import load_config
from crypto_quant.storage.database import get_session_factory
from crypto_quant.backtest.runner import ResearchBacktester
from datetime import datetime, UTC
from sqlalchemy import text

cfg = load_config('configs/v1_pg.yaml')
cfg = cfg.model_copy(update={
    'backtest': cfg.backtest.model_copy(update={'initial_equity': 1000}),
    'risk': cfg.risk.model_copy(update={'trade_risk_pct': 0.01, 'max_positions': 3})
})
session_factory = get_session_factory(cfg.database_url)
with session_factory() as session:
    r = ResearchBacktester(cfg).run_real(session,
        datetime(2025,6,1,tzinfo=UTC), datetime(2026,6,1,tzinfo=UTC))
    ret = (r.final_equity/1000-1)*100
    dd = session.execute(text(\"SELECT MIN(CAST(drawdown AS DOUBLE PRECISION))*100 FROM equity_curve WHERE strategy_run_id=:rid\"),{'rid':r.strategy_run_id}).scalar()
    print(f'\${r.final_equity:.0f} ({ret:+.1f}%) MaxDD: {dd:.1f}% {len(r.orders)}ord')
"

# 重建交易池 (改配置后需要)
uv run python -c "
from crypto_quant.config.settings import load_config
from crypto_quant.storage.database import get_session_factory
from crypto_quant.storage.candles import distinct_candle_symbols
from crypto_quant.universe.service import WeeklyUniverseService
from datetime import datetime, UTC
from sqlalchemy import text

cfg = load_config('configs/v1_pg.yaml')
session_factory = get_session_factory(cfg.database_url)
with session_factory() as session:
    session.execute(text('DELETE FROM universe_members'))
    session.execute(text('DELETE FROM universe_snapshots'))
    session.commit()
    symbols = distinct_candle_symbols(session, cfg.exchange_id, '1d')
    result = WeeklyUniverseService(cfg).build(session, symbols,
        datetime(2025,6,1,tzinfo=UTC), datetime(2026,6,12,tzinfo=UTC), persist=True)
    session.commit()
    print(f'{len(result.candidate_union)} coins, {len(result.snapshots)} weeks')
"

# 数据库连接 (Docker PostgreSQL)
# 用户: crypto_quant 密码: crypto_quant 端口: 5439 (或5433) 数据库: crypto_quant
# docker compose -f docker-compose.yml up -d postgres
```

## 核心代码结构

```
src/crypto_quant/
├── backtest/runner.py      ← 回测主循环
├── strategy/engine.py      ← 信号生成(熔断/过滤/动量门)
├── factors/momentum.py     ← 裸动量因子+ATR
├── factors/trend.py        ← 趋势结构因子(1H+4H MA)
├── factors/volume_stall.py ← 放量滞涨+冷却期
├── risk/market_state.py    ← BTC MA50/市场广度/快风险阀
├── risk/engine.py          ← 仓位计算
├── risk/false_breakout.py  ← 假突破检测
├── universe/service.py     ← 周度交易池构建
├── universe/builder.py     ← 流动性筛选
├── config/settings.py      ← 所有配置模型
├── execution/broker.py     ← 回测订单模拟
└── data/binance.py         ← Binance 数据获取
configs/
├── default.yaml            ← 默认配置
├── v1.yaml                 ← v1策略参数
└── v1_pg.yaml              ← v1+PostgreSQL连接
```

## 已知限制

1. **幸存者偏差**: 回测用当前在线币种，缺历史下架币数据
2. **回测速度**: 7-8分钟/年，已接近框架极限
3. **Freqtrade迁移**: 策略代码已写在 `my_btc_eth_bnb_spot/.../MonsterMomentumV1.py`，但WSL Python SSL握手超时无法连接Binance API
4. **低成交量妖币**: 10M门槛排除了成交量<10M的微型妖币（ENSO +164%、SOMI +71%等）

## 已验证无效的尝试

- 去掉熔断 → 坏月全面崩溃
- 去掉弱化减仓 → 全面恶化
- 去掉换仓 → 全面恶化
- defensive全放开 → 噪声盖过信号
- 止损价成交模拟 → 完全崩溃
- 降池子到5M/1M → 噪声太多
- 加动量watchlist扫全市场 → 收益下降

## 待探索方向

1. 提高单笔风险（1%→1.5%或2%）但松绑熔断
2. 不同市场环境用不同参数（risk_on激进/defensive保守）
3. 动态仓位（盈利后加仓/亏损后减仓）
4. 修复Freqtrade网络问题 → 享受100倍回测速度
5. 获取下架币历史数据消除幸存者偏差
