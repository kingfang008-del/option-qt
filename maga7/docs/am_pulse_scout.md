# AM Pulse Scout（侦察-only · 09:30–10:30）

Alert-only Mag7 opening pulse detector. **Does not place orders.** Parallel to CORE Rule-A (10:30–14:00) and AM research sleeves (`launch_slope` / `impulse_scout`).

## Why

CORE cannot catch open by design. On 2026-07-24 TSLA first hit fo≤−1% at **09:44**; Rule-A only saw the chase at ~10:39 and lost on a late put. Ops need an early `AM_SCOUT_ALERT` without moving CORE earlier or wiring a new OMS book.

## Arms

| arm | trigger (1m bars) | default |
|-----|-------------------|--------:|
| **FO** | \|fav_from_open\| ≥ thr | 1% |
| **LB** | \|ret over lookback\| ≥ thr | 2 bars / 0.8% |

Window: **09:30 ≤ t < 10:30** (exclusive end → CORE ownership). Each arm ≤1 alert per symbol×day; same bar prefers FO.

## Offline

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pulse_scout \
  --start-date 2026-07-24 --end-date 2026-07-24 \
  --tag am_pulse_scout_20260724
```

Writes `alerts.csv` / `alerts.jsonl` / `summary.json` under `results/<tag>`.

## Live / stream

Feed completed 1m OHLCV via `AmPulseScout.begin_day` + `on_bar`. Event name: `AM_SCOUT_ALERT`.

## Status

| id | lane | status | spine |
|----|------|--------|-------|
| `am_pulse_scout` | watchdog.scout | **ALERT_ONLY** | off |
| `am_pulse_sleeve` | research_sleeve | **ACCEPT_RESEARCH** | peer3 **shadow** wire |

CORE chase still blocked by `fo_lod_chase_gate` (DN30). Sleeve does **not** move Rule-A earlier.

## Live / shadow wire (2026-07-26)

三点落地：

1. **独立卫星**：`profile.am_pulse` + `Mag7Scanner.drain_am_pulse`（`event_source=am_pulse_sleeve`），不改 Rule-A TopK。
2. **互斥**：信号窗 `09:30–10:25`；`exit_flatten_before=10:30` + hold 截断到 CORE 前。
3. **先 shadow**：peer3 默认 `execute_mode=shadow` → OMS **仅**在 `mode=shadow` 成交；paper/live 拒单（`am_pulse_shadow_only`）。确认后改 `execute_mode=live`。

```json
"am_pulse": {
  "enabled": true,
  "execute_mode": "shadow",
  "arm": "FO", "dirs": ["DN"],
  "window_start": "09:30", "window_end": "10:25",
  "flatten_before": "10:30",
  "min_fav_from_open": 0.008, "tp": 0.15, "sl": 0.20
}
```

## Tradable sleeve dual-window (2026-07-26)

Windows: `may_jul09` / `jul10_23`. DN-only. Signal **09:30–10:25**.

| book | tool | verdict |
|------|------|---------|
| trades last±1% | `scan_am_pulse_trades_dual` | **PASS** (96/112 cells) |
| quote FillSpec | `scan_am_pulse_quote_dual` (lag≤3) | **REJECT** (Jul n≈4–5) |
| quote FillSpec | `scan_am_pulse_quote_dual_v2` (lag≤5) | **PASS** (19 cells) |

**Champion (research):** `pulse_FO_t0.008_tp0.15_sl0.2_sp0.15_lag5`

| window | n | mean | day_win |
|--------|--:|-----:|--------:|
| may_jul09 | 38 | +14.5% | 0.90 |
| jul10_23 | 6 | +23.3% | 1.00 |

Profile: `strategy_profiles/am_pulse_fo08_tp15_sl20_v1.json`.  
Caveats: Jul quote **n=6 floor**; strict **lag≤3** still fails (same Mag7 AM quote scarcity as impulse/launch). **Do not wire live** until denser Jul NBBO / live drain.

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pulse_trades_dual --dirs DN --tag research_am_pulse_trades_dual
PYTHONPATH=. python -m maga7.tools.scan_am_pulse_quote_dual \
  --dirs DN --tag research_am_pulse_quote_dual_v2 \
  --champions-json /mnt/s990/data/maga7/results/research_am_pulse_trades_dual/champions_broad.json \
  --max-spreads 0.10,0.15,0.20 --max-lags 2,3,5
```

## 07-24 smoke (alert)

Default Mag7 scan: **TSLA DN LB @ 09:36** + **FO @ 09:44** (fo≈−1.1%).
