# Pump Strategy Flow

Last updated: 2026-06-20

The active codebase is pump-only. Legacy weekly universe rotation, ordinary factor scoring, BTC 4h MA50 market state, swap logic, and non-pump position management have been removed.

## Data

The backtester loads all local Binance USDT spot symbols with `1h` candles plus `BTC/USDT`.

For each symbol, `_precompute_indicators` adds the columns required by pump regime detection and candidate selection:

| Column | Meaning |
|---|---|
| `ret_4h`, `ret_24h`, `ret_48h`, `ret_72h` | close-to-close returns |
| `weighted_return` | weighted momentum diagnostic |
| `ret_6h` | short-term pump confirmation |
| `ma20`, `above_ma20` | basic trend gate |
| `ema20_dev`, `ema20_dev_rank_2160h` | signal strength metadata |
| `atr14` | position stop distance input |
| `qv_6h_sum`, `qv_24h_sum`, `qv_30_avg` | quote-volume gates and volume ratio |
| `wick_ratio`, `new_12h_high` | blow-off / long-wick filters |
| `regime_vol_expansion` | market heat input |

## Main Loop

Each hour:

1. Reset daily realized loss if UTC date changed.
2. Slice currently held symbols for stop management.
3. Build a full pump snapshot from all eligible spot symbols.
4. Check BTC 1h fast risk valve. If BTC drops at least 7% in one hour, skip new entries.
5. Every 4 hours, classify pump regime as `HOT`, `WARM`, or `COLD`.
6. Every 4 hours, update market context from rolling cross-sectional features.
7. If regime is `HOT` or `WARM` and market context allows entries, select pump candidates from the snapshot.
8. Open probe positions for top candidates while respecting max positions and exposure.
9. Update pump stops, process exits, and write equity curve.

## Market State

There are three remaining market-state concepts:

- `fast_risk_valve`: BTC 1h return <= -7%, blocks new entries.
- `pump_regime`: cross-sectional altcoin heat state used by candidate admission.
- `market_context`: 4h rolling market risk state used as a hard entry gate.

Pump regime rules:

| Regime | Conditions |
|---|---|
| `HOT` | median 24h return >= 5%, 12h-new-high ratio >= 20%, volume-expansion ratio >= 5% |
| `WARM` | median 24h return >= 2%, 12h-new-high ratio >= 10% |
| `COLD` | otherwise, or fewer than 20 eligible symbols |

Market context uses only prior 4h observations for rolling quantiles. The current production setting is deliberately conservative: it blocks new entries only in robustly bad states and does not reduce position size, force patient entries, or tighten exits.

| Context | Main intent |
|---|---|
| `risk_off` | Block new entries when breadth, BTC trend, and cross-sectional synchronization are unfavorable |
| `cold` | Block new entries when pump participation is too weak |
| `normal`, `expanding`, `crowded_hot`, `crowded_fading` | Allow entries with unchanged risk |

## Candidate Rules

A symbol must pass:

- no existing position
- at least 73 hours of history
- 24h return >= `min_24h_return`; current default is 32%
- 6h return >= `min_6h_return`
- price above MA20
- quote volume: 24h >= `min_quote_volume_24h` or 6h >= `min_quote_volume_6h`
- no blow-off candle: `wick_ratio >= blowoff_wick_ratio` and `new_12h_high`
- no excessive 72h return beyond configured caps
- optional experimental filters if enabled: EMA absolute bounds, long-wick rejection, acceleration decay rejection

Signal admission:

- In `HOT`: confirmed or early signals are allowed.
- In `WARM`: only stricter warm-early signals are allowed.
- In `COLD`: no new pump entries.

Candidate score:

```text
score = ret_24h * 0.45
      + ret_72h * 0.35
      + ret_6h  * 0.10
      + min(volume_ratio / 5, 1) * 0.10
```

## Position Sizing

The current production config is `configs/main.yaml`.

The strategy uses small probe entries and two-stage confirmation:

- A-tier probe: 25% of full quantity.
- B-tier probe: 25% of full quantity.
- Weak confirmation: after 3.5h, if close is at or above the probe anchor, scale to 50% of full quantity.
- Strong confirmation: if MFE reaches 12% or close return reaches 6%, scale to 100% of full quantity.

Full quantity is capped by:

- risk budget divided by stop distance
- max single-symbol exposure
- max total pump exposure
- available cash

The two-stage structure is the active risk-control layer. It keeps the initial test cost small and only allocates full size after post-entry follow-through.

If `equity_peak_risk_enabled` is true, risk is scaled by `max(equity / peak_equity, equity_peak_risk_floor)`.

The market context risk multiplier defaults to `1.0` for tradable states and `0.0` for blocked states.

## Exits

Pump exits include:

- 4h probe kill if confirmation return <= -2%
- 3h consecutive down forced exit
- stagnation exit
- time stop
- profit protection
- breakeven / lock stops
- multi-level ATR trailing
- optional signal-confidence exit metadata experiment
- initial stop

All orders and position state snapshots are persisted to PostgreSQL for diagnostics.
