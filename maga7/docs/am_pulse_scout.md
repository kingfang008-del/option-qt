# AM Pulse Scout / A+B 卫星（A 09:30–10:30 · B 10:30–11:30）

Alert-only Mag7 opening pulse detector. **Does not place orders.** Parallel to CORE Rule-A (10:30–14:00) and AM research sleeves (`launch_slope` / `impulse_scout`).

## Why

CORE cannot catch open by design. On 2026-07-24 TSLA first hit fo≤−1% at **09:44**; Rule-A only saw the chase at ~10:39 and lost on a late put. Ops need an early `AM_SCOUT_ALERT` without moving CORE earlier or wiring a new OMS book.

## Arms

| arm | trigger (1m bars) | default |
|-----|-------------------|--------:|
| **FO** | \|fav_from_open\| ≥ thr | 1% |
| **LB** | \|ret over lookback\| ≥ thr | 2 bars / 0.8% |

A 窗：**09:30 ≤ t < 10:30**；B 窗：**10:30 ≤ t < 11:30**。两段使用独立 symbol×day 触发预算；same bar prefers FO。

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

**2026-07-27 stack freeze:** research stack = **peer3 baseline CORE + this AM sleeve only**. Other TOD sleeves (lunch FO / LOD-reclaim, `qqq_open_cont` on peer3) are off / abandoned.

CORE chase still blocked by `fo_lod_chase_gate` (DN30) — **live Scanner** now runs the same feature-clock morph gates as offline replay (before TopK seat reserve). Sleeve does **not** move Rule-A earlier.

## Lunch TOD (ABANDONED · 2026-07-27)

Tried copying AM FO/LB and a dedicated **LOD-reclaim UP** into **12:30–13:30** (V-reversal hour). Trades dual either failed when matched to AM DN-only, or passed with ~+1–3% mean (vs AM +14%/+23%) and overlaps Rule-A. Tight NVDA/TSLA + short hold did not help.

| tag | verdict |
|-----|---------|
| `results/research_lunch_pulse_dual_v1` | weak FO mixed-dir only; DN-only FAIL |
| `results/research_lunch_lod_reclaim_v1` | weak PASS (~+3% May); not promote |
| `results/research_lunch_lod_reclaim_tight_v1` | worse (~+1%); **stop** |

Tools kept for archive: `scan_tod_pulse_trades_dual.py`, `scan_lunch_lod_reclaim_dual.py`. **Do not wire.**

## Live / shadow wire (2026-07-26)

三点落地：

1. **独立卫星**：`profile.am_pulse` + `Mag7Scanner.drain_am_pulse`（`event_source=am_pulse_sleeve`），不改 Rule-A TopK。
2. **分段**：A 信号窗 `09:30–10:30`、flatten `10:45`；B 信号窗 `10:30–11:30`、flatten `11:45`。
3. **先 shadow**：peer3 默认 `execute_mode=shadow` → OMS **仅**在 `mode=shadow` 成交；paper/live 拒单（`am_pulse_shadow_only`）。确认后改 `execute_mode=live`。

```json
"am_pulse": {
  "enabled": true,
  "execute_mode": "shadow",
  "arm": "FO", "dirs": ["DN", "UP"],
  "window_start": "09:30", "window_end": "10:30",
  "flatten_before": "10:45",
  "min_fav_from_open": 0.008,
  "max_fav_from_open": 0.015,
  "tp": 0.15, "sl": 0.20,
  "profit_protect": {"enabled": true, "arm_ret": 0.08, "floor_ret": 0.03}
}
```

## Tradable sleeve dual-window (2026-07-26)

历史发现窗：`may_jul09` / `jul10_23`，早期 DN-only，信号 **09:30–10:25**。该结果只作参数发现；当前 LOCK 必须用 profile 驱动的 **09:30–10:30、双向、FO≤1.5%** 重新验收。

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

## B 窗 10:30–11:30（由 AM extension 升格）

当前 LOCK 为双向 FO@0.8%、ATM、TP15/SL20、单笔最多 15 分钟；UP 使用 60 秒 confirm_abort（确认 +2%，早切 −10%），B 强制 0DTE。延伸段独立扫描，不复用 A 的每标的触发计数，也不占 CORE TopK。

| layer | window | n | mean | day_win | verdict |
|-------|--------|--:|-----:|--------:|---------|
| trades | may_jul09 | 76 | +10.39% | 85.29% | PASS |
| trades | jul10_23 | 13 | +15.85% | 100.00% | PASS |
| quote lag5/sp15 | May | 36 | +14.78933782% | — | PASS |
| quote lag5/sp15 | Jul | 8 | +14.29419958% | — | PASS |

Quote verdict: **QUOTE_PASS**（3 cells）；strict lag≤2/3 因 Jul `n=5` 未过最小样本门槛。研究 profile：`strategy_profiles/am_pulse_extend_1025_1130_fo08_tp15_sl20_v1.json`。

```bash
PYTHONPATH=. python -m maga7.tools.scan_tod_pulse_trades_dual \
  --window-start 10:30 --window-end 11:30 --session-tag AM_EXT_1030_1130 \
  --dirs DN --arms FO --fo-thr 0.008 --tp 0.15 --sl 0.20 \
  --max-hold-sec 900 --champions-only \
  --tag research_am_ext_1025_1130_fo_dn_champion

PYTHONPATH=. python -m maga7.tools.scan_am_pulse_quote_dual \
  --lane am_pulse_extension \
  --window-start 10:30 --window-end 11:30 --session-tag AM_EXT_1030_1130 \
  --dirs DN --max-spreads 0.10,0.15,0.20 --max-lags 2,3,5 \
  --champions-json /mnt/s990/data/maga7/results/research_am_ext_1025_1130_fo_dn_champion/dual_pass.json \
  --tag research_am_ext_1025_1130_quote_dual
```

接线约束：使用独立 `AM_EXT` detector/counter 并保持 shadow；不能直接把原 `am_pulse.window_end` 改到 11:30，否则 A 已触发的 symbol 不会得到 B 的第二次机会。B 最后一笔最多持有 15 分钟，统一在 11:45 前平仓。

## 回测 / Shadow 一致性（2026-07-28 修复）

- 当前唯一权威口径来自 research spine profile：A `09:30–10:30` / `flatten=10:45`；B `10:30–11:30` / `flatten=11:45`。
- Scanner 将 `max_lag_sec`、`max_spread_pct`、`min_mid` 写入信号；OMS 对 signal→quote lag、spread、mid 三门硬校验。
- B 的 `allowed_dte=[0]` 同时用于 Scanner 与 `scan_am_pulse_quote_dual --lane am_pulse_extension`。
- Redis consumer 与 LiveEngine 均显式 drain A/B 两段。
- 旧 `10:25` 产物保留作发现证据，不可直接作为当前 LOCK 的最终升线账本。

## 亏损复盘后的 Shadow 优化（2026-07-28）

- Profile 顶层 `regime.event_calendar_block` 只约束 CORE 基线；A/B 默认
  `event_calendar_block=false`，宏观事件日不触发全局 OMS halt。若后续研究证明某个
  sleeve 需要事件闸门，必须在对应 A/B 配置中显式 opt-in。
- A/B 的分钟特征时间与可交易时间分离：`feature_ts` 保留分钟标签，
  `decision_ts` 是完整分钟可用后的实际决策时间。期权报价 5 秒 lag gate 以
  `decision_ts` 为锚，同时仍要求报价相对墙钟新鲜。入场还启用股票漂移保护：
  决策后同向追价超过 0.3% 或反向回撤超过 0.15% 时拒绝，避免延迟盲追。
- A：`ladder_08_03`，可成交 MTM 曾达到 +8% 后，把退出地板抬到 +3%。在冻结 126 笔账本上，Feb–Mar 与 May–Jul09 的资金收益均提高，研究最大回撤约从 −9.0% 降至 −5.1%；但配对 bootstrap 95% CI 仍跨零，因此只进入 shadow。
- B：保留 UP-only 60 秒确认，将 `abort_thr` 从 8% 放宽至 10%。冻结 225 笔账本与当前结果逐笔对拍一致；候选在 Feb–Mar 无变化，May–Jul09 与 Jul10–23 均改善。B 不启用 A 的利润地板，因为该候选明显损害 B 总收益。
- 已完成分钟资金流 gate 在安全的 `entry_ts−60s` 口径下不稳定；旧 post-hoc `MF1_SAME` 使用左标分钟时可能包含尚未完成的当前分钟，不作为升格证据。
- 联合结果按每笔 10% 归一：Feb–Mar、May–Jul09、Jul10–23 均改善；April 小幅变差。两项优化继续保持 `execute_mode=shadow`，不自动升 live。

## 07-24 smoke (alert)

Default Mag7 scan: **TSLA DN LB @ 09:36** + **FO @ 09:44** (fo≈−1.1%).
