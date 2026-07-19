# Research baseline: full_day peer3 + streak3 + mf_idio + L2 Hunt

**Profile:** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
**Frozen:** 2026-07-18 (L0) · **L1 on:** 2026-07-18 · **L2 on:** 2026-07-18（旧 Hunt 退出 T30+extend）  
**Ladder:** 2026-07-19 收到 **OTM3**（`ladder_otm_rungs=3`；OTM4/5 消融无边际）  
**TT1 05:** 2026-07-19 — `max_entry_abs_otm_pct=0.005` + 仅二四 `entry_confirm_bars=2` / `mode=mf`  
（细则：[`tt1_uplift_05.md`](tt1_uplift_05.md)）  
**Hard SL 55:** 2026-07-19 — `sl_mult=0.45`（≈−55%；见 [`sl55_hard_stop_research.md`](sl55_hard_stop_research.md)）  
**Trade toxic:** 2026-07-19 — cut25/mfe05 + max_cut=600s + **div_mfe6%/stock&lt;0.5%**（见 [`trade_mark_toxic_path_research.md`](trade_mark_toxic_path_research.md)）  
**Role:** `research_baseline` (research / shadow default; not yet production freeze)

## Stack

| Layer | Setting |
|-------|---------|
| Causal core | Mag7+GOOGL open_ladder **OTM3**, peer_align_min=3, QQQ align, rails delay 60s |
| Exit | `hold_extend` T30→T45, MTM≥0, `hold_extend_require_mf=false`（Hunt 同此） |
| Events | `event_calendar_block` + `event_calendar_live.json` |
| Day halt | `day_loss_streak_halt=3` |
| Size gate | `mf_idio_mode=pos`, `action=scale`, `scale=0.5`, `after_loss_streak=1`, ret β 5d |
| Watchdog L1 | Degrade `reclaim_disp55` + Halt `washout_and_reclaim` |
| Watchdog L2 | `hunter.enabled=true` — `washout_reclaim` wd=1.5% · opp on · mutex `symbol_dir` |
| Risk | **`sl_mult=0.45`（≈−55%）**, `position_frac=0.2`, concurrent max 2 |
| Fill | `entry_frac`/`exit_frac`=**0.75**（2026-07-19：对齐实盘中位数，原 0.8） |
| TT1 05 | `|otm|>0.5%` 不做；**仅 Tue/Thu** 入场后再确认 2 根 1m mf |
| Trade toxic | 成交 last：MFE&lt;5%（股价不利&lt;0.5% 时放宽到 **6%**）且 MTM≤−25%，min_hold 60s，**仅 fill 后 600s 内**；exit 仍 quote |

## Snapshot metrics (offline replay)

### 勿混淆：May–Jul 档位

「约 +1856%」是 **当前研究基线（L2 + TT1 05 + sl55 + trade_toxic max600+div06）**；无 tox 的 sl55 约 +1555%。

| 档 | 开什么 | May–Jul total_ret | vs L0 | 是否在 research_baseline |
|----|--------|------------------:|------:|--------------------------|
| L0 | 无 Watchdog | **+810%** | 100% | 临时关 `watchdog.enabled` 可复现 |
| L1 | Degrade + Halt（Hunter off） | **+875%** | ~108% | 对照（关 `hunter.enabled`） |
| L2 | L1 + Hunt v2（T30+extend） | ~+1282% | ~158% | 对照（关 TT1 05） |
| L2+05 | L2 + max_otm0.5% + 二四 confirm2 · sl60 | **+1528%** | ~188% | 对照（`sl_mult=0.40`） |
| L2+05+sl55 | 上 + `sl_mult=0.45` | **+1555%** | ~192% | 对照（关 `trade_toxic`） |
| L2+05+sl55+tt | 上 + cut25（无 max_cut） | +1721% | ~212% | 对照 |
| L2+05+sl55+tt600 | 上 + cut25 + max_cut=600s | +1829% | ~226% | 对照 |
| **L2+05+sl55+tt600d** | 上 + **div_mfe6%/stock&lt;0.5%** | **+1856%** | **~229%** | **是（默认）** |

### 验收窗重划（2026-07-19）

旧「弱窗 Feb–Apr」把 **4 月牛市** 和 1–3 月波动市混在一起，会高估弱窗、低估强窗 MaxDD 来源。

| 窗 | 区间 | 市场叙事 | tt600d total_ret | MaxDD | n |
|----|------|----------|-----------------:|------:|--:|
| **强窗（新）** | **Apr–Jul（→07-16）** | QQQ 总量拉升 / 趋势市 | **+3987%** | **−24.4%** | 86 |
| 其中 May–Jul | 05-01→07-16 | 同上 | +1856% | **−5.4%** | 57 |
| 其中 Apr only | 04-01→04-30 | 牛市起步 | +109% | −24.4% | 29 |
| **弱窗（新）** | **Jan–Mar** | 波动市、无 QQQ 总量拉升 | **+37%** | −21.7% | 66 |
| 其中 Feb–Mar | 02-01→03-31 | 同上 | +35% | −21.7% | 50 |
| ~~旧弱窗~~ | Feb–Apr（废弃作弱窗） | 混入 4 月牛 | +164% | −24.4% | 77 |

要点：
- **4 月不应再算弱窗**（单月 +109%、胜率 59%）。
- Apr–Jul 复利很高，但 MaxDD **−24.4% 主要来自 4 月硬 SL**（如 04-08/04-22 AAPL）；May–Jul 单独 MaxDD 仅 −5.4%。
- 弱窗 Jan–Mar 仍为正（+37%），比「旧 Feb–Apr +164%」更诚实。

产物：`results/research_windows_apr_jul_vs_jan_mar_tt600d/`  
升线门槛（Hunt 历史）：曾用 Feb–Apr vs L0；**此后双窗验收默认用 Apr–Jul + Jan–Mar**。  
trade_toxic 细则：[`trade_mark_toxic_path_research.md`](trade_mark_toxic_path_research.md)。

- L1 产物：`results/research_extend_mtm_full_day_peer3_l1_may_jul/`  
- L2 产物：`results/watchdog/hunter_washout_reclaim_v2_opp/`  
- L2+05（sl60）产物：`results/research_extend_mtm_full_day_peer3_l2_tt1_05_may_jul/`  
- sl55 对照：`results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_may_jul/`  
- May–Jul tt600d：`results/trade_toxic_div_ablation_dual_window/strong_may_jul__04_div06_adv05/`  
- **新窗产物**：`results/research_windows_apr_jul_vs_jan_mar_tt600d/`  
- P2.1 `hold20_noext`（~+840%）仅为备选退出研究，**未**进基线。

| Window | total_ret | MaxDD | notes |
|--------|----------:|------:|-------|
| **Apr–Jul (→07-16) tt600d** | **+3987%** | **−24.4%** | **新强窗**（DD 含 4 月） |
| May–Jul (→07-16) **tt600d** | **+1856%** | **−5.4%** | 趋势市干净段 |
| Apr only tt600d | +109% | −24.4% | 牛市；勿算弱窗 |
| **Jan–Mar tt600d** | **+37%** | −21.7% | **新弱窗** |
| Feb–Mar tt600d | +35% | −21.7% | 弱窗子集 |
| Feb–Apr tt600d（旧标签） | +164% | −24.4% | 已废弃作弱窗 |
| May–Jul L2+05+sl55（关 tox） | +1555% | −11.1% | 对照 |
| May–Jul L0 | +810% | −13.2% | 无 Watchdog |

### 工程闸门（防忘）

| 项 | 状态 |
|----|------|
| L1 流式对拍 | **通过** |
| L2 流式对拍（旧退出） | **通过**（May–Jul 60 笔） |
| Paper | 上实盘前工程门 |
| VIX/VIXY 急拉直闸 | **无**；扫描见 `results/watchdog/vixy_spike_scan_2026h1/` |
| 持仓 Watchdog（QQQ 冲击平仓） | 旋钮已接、**默认 off**；见 [`hold_watchdog_research.md`](hold_watchdog_research.md) |
| 三次大亏日 / 全局 `mae_cut` | **REJECT**；分段建仓 `scale_in` 亦 **REJECT** |
| 硬 SL 55 | **已升**；见 [`sl55_hard_stop_research.md`](sl55_hard_stop_research.md) |
| trade-mark toxic cut | **已升**（cut25/mfe05/max_cut600/**div06**）；见 [`trade_mark_toxic_path_research.md`](trade_mark_toxic_path_research.md) |

## Ops

- **操作手册**：[`maga7_operations_guide.md`](maga7_operations_guide.md)
- Live/shadow: `maga7/SHELL/start_maga7_live_session.sh`
- Day stream check: `maga7/SHELL/run_day_stream_check.sh`
- Dashboard: `python dash/run.py`
- Replay results: `maga7/results/`（gitignored）

## Next validation

1. **入场质量（假突破）** — 首轮 `|fp|` 帽 / 扩 confirm **未过双窗**；见 [`entry_quality_false_break_research.md`](entry_quality_false_break_research.md)  

2. 见 [`l2_next_acceptance_checklist.md`](l2_next_acceptance_checklist.md)：P3 Live/shadow → P2.3 Hunt 日熔断 → P0.3/P0.4 敏感度

## Non-goals in this freeze

- Do not bare-enable `mf_flip` / streak early exits (hurts May–Jul)
- Do not further tighten `sl_mult` below 0.45 without dual-window re-acceptance
- `tcn_gate.enabled` stays **false**；见 [`tcn_gate_research.md`](tcn_gate_research.md)
- `trade.topk_backfill_on_block` stays **false**
- TopK 默认 `rank_by=earliest`
- 勿把 `wash_drop_min` 降到 1.2%（弱窗断崖）
- Hunt 专用 `hold20_noext` **不**进基线（除非另开决策）
- `scale_in` / 全局 `mae_cut` **不**进基线（`trade_toxic` 已升，勿与 quote `mae_cut` 混淆）
