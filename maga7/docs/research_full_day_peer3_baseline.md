# Research baseline: full_day peer3 + streak3 + mf_idio + L2 Hunt

**Profile:** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
**Frozen:** 2026-07-18 (L0) · **L1 on:** 2026-07-18 · **L2 on:** 2026-07-18（旧 Hunt 退出 T30+extend）  
**Ladder:** 2026-07-19 收到 **OTM3**（`ladder_otm_rungs=3`；OTM4/5 消融无边际）  
**TT1 05:** 2026-07-19 — `max_entry_abs_otm_pct=0.005` + 仅二四 `entry_confirm_bars=2` / `mode=mf`  
（细则：[`tt1_uplift_05.md`](tt1_uplift_05.md)）  
**Hard SL 55:** 2026-07-19 — `sl_mult=0.45`（≈−55%；见 [`sl55_hard_stop_research.md`](sl55_hard_stop_research.md)）  
**Trade toxic:** 2026-07-19 — cut25/mfe05 + max_cut=600s + **div_mfe6%/stock&lt;0.5%**；**2026-07-26** 加 `quote_fallback` + `quote_fallback_cut_ret=0.20`（**仅缺 prints**）；2–4 月 trades 回填后 02-17 改由 **`dn_gap_stall_gate` DGS18** 进场拦（见 [`trade_mark_toxic_path_research.md`](trade_mark_toxic_path_research.md) / `results/dn_gap_stall_dual_v1`）  
**Event calendar feb_jul:** 2026-07-20 — 在 May–Jul 原集上并入 loss-scan：02-05 capex / 02-11 NFP / 02-13 CPI / 03-03 geopol / 04-29 FOMC（见 [`loss_day_event_scan.md`](../results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul/loss_day_event_scan.md)；消融 `research_event_calendar_extend_ablation_*`）  
**Event calendar +AAPL CEO:** 2026-07-20 — `feb_jul_aapl_ceo` = feb_jul + 04-21/22 Cook→Ternus（见 [`remaining9_stock_news.md`](../results/research_extend_mtm_full_day_peer3_l2_tt1_05_sl55_tt600d_feb_jul/remaining9_stock_news.md)；消融 `research_event_calendar_aapl_ceo_ablation_*`）  
**Company-news policy:** 2026-07-20 — **不定方向**；live `company_news_mode=hard_risk`（仅 CEO 可自动禁票）；大单/合作只审计+LLM；宏观/财报分层禁入（见 [`event_news_policy.py`](../common/event_news_policy.py)）  
**Jul Mag7 AH 补丁：** 2026-07-26 — `2026-07-22` **TSLA/GOOGL `earnings_ah`** 写入 live+manual，peer3 `event_symbol_blackout`（AH 当日禁该标的；**不**禁 07-23；长波动/straddle sleeve 未接线）。见 [`event_calendar_finnhub_jul2026_validation.md`](event_calendar_finnhub_jul2026_validation.md)。  
**AH straddle scoreboard：** 2026-07-26 跑分 → **2026-07-27 ABANDONED**（隔夜双边不晋级；AH 禁入保留）。见 [`earnings_ah_straddle_scoreboard.md`](earnings_ah_straddle_scoreboard.md)。  
**Hunt×RS peer（06-24 AMD）：** 2026-07-27 — `HUNT_RS_PEER_T10` 清 AMD Hunt，但砍 07-01 META Hunt TP，strong keep≈0.92 → **DUAL_FAIL 不接线**。见 [`hunt_range_stall_peer_research.md`](hunt_range_stall_peer_research.md)。  
**crowd6（06-24 GOOGL）：** 2026-07-27 — `crowd_min_peer=6` 清 GOOGL、留 AMZN，但弱窗 keep≈0.74（误杀 Apr TP）→ **DUAL_FAIL**；GOOGL 仍靠 TRADE_TOX。见 [`crowd6_googl_0624_research.md`](crowd6_googl_0624_research.md)。  
**L3 causal exit (STOCK_REV):** 2026-07-22 — 细网格已过晋级条，**shadow 候选已落盘**，**未**并入本基线；见 [`peer3_l3_causal_exit_research.md`](peer3_l3_causal_exit_research.md)  
**Tail-first shadow:** 2026-07-22 — `tox_cut20 + wash_m3`（压单笔 ≤−25%）；见 [`peer3_tail_loss_research.md`](peer3_tail_loss_research.md) · profile `..._peer3_tail_tox20_wash_m3_v1`  
**S1 soft path_confirm:** 2026-07-23 — `trade.stock_path_confirm` enabled（thr_pos=+15bp / thr_neg=−30bp / max_wait=300s / **on_timeout=allow** / delay_on_pos=false / tod 10:30–14:00）。**已并入本 research_baseline**。  
**软加仓 `dvol_size_scale`:** 2026-07-23 — 截面 $vol rk1×1.25 / rk2×1.15（只抬不砍，不改 TopK 座位）。双窗 `PROMOTE_LIQ_RESEARCH` 后写入本 profile（`research_revision=2026-07-23_s1_dvol_soft`）。见 [`dvol_liq_soft_research.md`](dvol_liq_soft_research.md)。  
**窄形态专家注册表:** 2026-07-24 — `narrow_experts.catalog_path` → `CONFIG/narrow_experts/catalog_v1.json`（脊骨仍 L0+L1+L2；`core_dn_sync` / AM HF 为 **QUOTE_REJECT** 研究队列，不接线）。见 [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)。  
**Overnight gap+adv BLOCK:** 2026-07-26 接线 → **2026-08-14 C5 DEPRECATED**（strip 强窗 keep=1.06；合关确认 keep=1.025）。`enabled=false`。见 [`core_architecture_adaptability.md`](core_architecture_adaptability.md) §8.4。  
**Peer+gap stall BLOCK:** 2026-07-26 — `trade.peer_gap_gate` **已接线**（peer≤3 · fav_gap≥1.5% · `mode=block`）。`DUAL_PASS_WIRE` `peer_gap_dual_v1` / `P3_G15`（清 04-08 AAPL SL、砍 02-18 NVDA TOX）；见 [`peer_gap_stall_research.md`](peer_gap_stall_research.md)。**C5 KEEP**（strip 强窗 keep=0.87）。  
**Range-chase+pre5 stall BLOCK:** 2026-07-26 — `trade.range_stall_gate` **已接线**（entry 时钟 · chase≥0.9 · pre5≤**2.5bp** · peer≤5 · fav_ffo≥**1.2%** + `peer_pre5_max_peer=3` + **`crowd_min_peer=7`/`crowd_max_pre5=25bp`/`crowd_min_fav_from_open=1%`**）。`C7_FFO012_P25`+`CROWD25` 清 02-06/02-25/03-18/**02-18**/**03-12**；见 [`range_stall_research.md`](range_stall_research.md) / `results/priority3_dual_v2`。**C5 KEEP**（strip 弱+17 / 强+7 笔，keep 0.90/0.73）。  
**UP gap early stall BLOCK:** 2026-07-26 接线 → **2026-08-14 C5 DEPRECATED**（strip 强窗 keep=0.965）。`enabled=false`。  
**FO+LOD chase BLOCK:** 2026-07-26 接线 → **2026-08-14 C5 DEPRECATED**（strip nΔ=0，block 未变成成交）。`enabled=false`。  
**C5 morph freeze:** 2026-08-14 — 硬 BLOCK **6→3**；`morph_debt.policy=no_net_new_hard_block`。禁止再净增替换门。生产 freeze 不变。  
**C6 session risk budget:** 2026-08-14 — `trade.session_risk_budget` **已接线**（`dd_step` · 已实现权益 DD≤−5% → size×0.5）。弱 keep **0.975** MaxDD −8.8%→−7.1%；强 keep **0.998**。不是气候标签（C2 仍 FAIL），不是 morph。见 [`core_architecture_adaptability.md`](core_architecture_adaptability.md) §8.5。  
**Live Scanner 对齐（entry morph）：** 2026-07-26 — `Mag7Scanner` 已挂同一组 price/vol 门：feature 时钟 `overnight_gap` / `peer_gap` / `dn_gap_stall` / `up_gap_stall` / `fo_lod_chase`（**TopK 占座前**）；entry 时钟 `range_stall`；size scale 并入 `meta.regime_size_scale`。计数器见 snapshot `entry_morph.*`。

**AM Pulse sleeve (shadow):** 2026-07-26 — Mag7 DN FO≥0.8% · 09:30–10:25 · flatten≤10:30；`drain_am_pulse` 已挂 peer3，默认 `execute_mode=shadow`（OMS shadow-only）。见 [`am_pulse_scout.md`](am_pulse_scout.md) / `results/research_am_pulse_quote_dual_v2`。  
**Stack freeze (2026-07-27):** 研究栈只保留 **本基线 CORE + AM pulse**。`qqq_open_cont` 已关；午盘 FO / LOD-reclaim（12:30–13:30）**ABANDONED**（弱边、与 CORE 抢钟）。见 [`am_pulse_scout.md`](am_pulse_scout.md) §Lunch。  
生产 / freeze / `default_profile` 仍是 `googl_peer3_v1`。  
**S1 离线验收（标准双窗，2026-07-23）：** PRE（关 S1）vs S1 — **`KEEP_S1_RESEARCH_BASELINE`**  
| 窗 | PRE total_ret | S1 total_ret | keep | MaxDD PRE→S1 |
|----|-------------:|-------------:|-----:|--------------|
| 强 Apr–Jul（→07-21） | +4205% | **+4504%** | **1.07** | −16.1%→**−14.1%** |
| 弱 Jan–Mar | +90.6% | **+97.7%** | 1.08 | −15.5%→**−15.3%** |
| 七月切片 | +95.7% | +95.7% | **1.00** | 同 |
产物：`results/s1_research_baseline_accept_apr_jul_jan_mar_v1/` · 脚本 `tools/run_s1_research_baseline_accept.py`  
完整记录：[`s1_research_baseline_offline_pack.md`](s1_research_baseline_offline_pack.md) · 对拍节见 [`replay_stream_parity.md`](replay_stream_parity.md)  
工程对拍（stock_1s，offline↔stream）：  
- 弱窗 Jan–Mar：`parity_s1_research_baseline_jan_mar_stock1s` — **ok**（54 笔）  
- 强窗 Apr–Jul：`parity_s1_research_baseline_apr_jul_stock1s` — **ok**（79 笔）  
- 七月三路：`parity_s1_fix_tox_hunt_20260701_21` — **ok**（15 笔，含 scanner）  
**WAVE_ABORT UP-only:** 七月更好，但强窗 keep≈0.82 **未过** 0.85 — 仅 overlay，**未**并入本基线。  
**Role:** `research_baseline` (research / shadow default; not yet production freeze)

## Stack

| Layer | Setting |
|-------|---------|
| Causal core | Mag7+GOOGL open_ladder **OTM3**, peer_align_min=3, QQQ align, rails delay 60s |
| Exit | `hold_extend` T30→T45, MTM≥0, `hold_extend_require_mf=false`；**giveback gb12_p15**（peak≥15% 且 peak−mtm≥12% → 拒延期；2026-07-28 双窗 PASS） |
| Events | `event_calendar_block` + **`feb_jul_aapl_ceo`** + live **`company_news_mode=hard_risk`**（新闻不定方向；CEO/财报 symbol、宏观 full-day） |
| Day halt | `day_loss_streak_halt=3` |
| Size gate | `mf_idio_mode=pos`, `action=scale`, `scale=0.5`, `after_loss_streak=1`, ret β 5d |
| Watchdog L1 | Degrade `reclaim_disp55` + Halt `washout_and_reclaim` |
| Watchdog L2 | `hunter.enabled=true` — `washout_reclaim` wd=1.5% · opp on · mutex `symbol_dir` |
| Risk | **`sl_mult=0.45`（≈−55%）**, `position_frac=0.2`, concurrent max 2 |
| Fill | `entry_frac`/`exit_frac`=**0.75**（2026-07-19：对齐实盘中位数，原 0.8） |
| TT1 05 | `|otm|>0.5%` 不做；**仅 Tue/Thu** 入场后再确认 2 根 1m mf |
| Trade toxic | 成交 last：MFE&lt;5%（股价不利&lt;0.5% 时放宽到 **6%**）且 MTM≤−25%，min_hold 60s，**仅 fill 后 600s 内**；缺 prints → quote-sell + **qf_cut20**；exit 仍 quote |

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
| **弱窗（新）** | **Jan–Mar** | 波动市、无 QQQ 总量拉升 | **+42%** | −23.9% | 69 |
| 其中 Feb–Mar | 02-01→03-31 | 同上 | +40% | −23.9% | 53 |
| ~~旧弱窗~~ | Feb–Apr（废弃作弱窗） | 混入 4 月牛 | +164% | −24.4% | 77 |

要点：
- **4 月不应再算弱窗**（单月 +109%、胜率 59%）。
- Apr–Jul 复利很高，但 MaxDD **−24.4% 主要来自 4 月硬 SL**（如 04-08/04-22 AAPL）；May–Jul 单独 MaxDD 仅 −5.4%。
- 弱窗 Jan–Mar 仍为正（+42%），比「旧 Feb–Apr +164%」更诚实。  
- **2026-07-19：** 补齐 GOOGL Feb `open_lock`（quote 本已有）后，弱窗由 +37%/+35% → **+42%/+40%**；见 [`missed_movers_feb_mar.md`](missed_movers_feb_mar.md) §5.1。AMD Mon/Tue 仍为短 DTE 结构性不可交易。

产物：`results/research_windows_apr_jul_vs_jan_mar_tt600d/` · 弱窗刷新：`…_tt600d_jan_mar_googl_feb_lock/`  
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
| **Jan–Mar tt600d** | **+42%** | −23.9% | **新弱窗**（含 GOOGL Feb lock） |
| Feb–Mar tt600d | +40% | −23.9% | 弱窗子集 |
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

1. **入场质量（假突破 / 开盘延伸 / 震荡日）** — `|fp|` 帽未过双窗；`from_open_gate` / `chop_gate` 最佳仅 `OVERLAY_ONLY` → **不升基线**；见 [`entry_quality_false_break_research.md`](entry_quality_false_break_research.md) · [`from_open_gate_research.md`](from_open_gate_research.md) · [`chop_gate_research.md`](chop_gate_research.md)  



2. 见 [`l2_next_acceptance_checklist.md`](l2_next_acceptance_checklist.md)：P3 Live/shadow → P2.3 Hunt 日熔断 → P0.3/P0.4 敏感度

3. **L3 STOCK_REV shadow** — 主推 `peer3_l3_wash_m3_v1`（备选 `peer3_l3_uw_m3_h15_v1`）；多日毒性 + OMS 确认后再议 peer3_v2。状态见 [`peer3_l3_causal_exit_research.md`](peer3_l3_causal_exit_research.md)

4. **尾损优先 shadow** — `peer3_tail_tox20_wash_m3_v1`（tox cut20 + wash_m3）；验收见 [`peer3_tail_loss_research.md`](peer3_tail_loss_research.md)

5. **波段确认规格（未实现）** — 生产级前提：fill 后路径确认失败则 `WAVE_ABORT`，禁止未确认拖到 T30；见 [`wave_confirm_spec.md`](wave_confirm_spec.md)

## Non-goals in this freeze

- Do not bare-enable `mf_flip` / streak early exits (hurts May–Jul)
- Do not further tighten `sl_mult` below 0.45 without dual-window re-acceptance
- `tcn_gate.enabled` stays **false**；见 [`tcn_gate_research.md`](tcn_gate_research.md)
- `trade.topk_backfill_on_block` stays **false**（S1 基线双窗复验仍 `REJECT_FOR_BASELINE`；见 [`topk_backfill_research.md`](topk_backfill_research.md) §2026-07-23）
- TopK 默认 `rank_by=earliest`（`dollar_vol` 座位重排仍拒升线）
- `trade.dvol_size_scale` **enabled** on research_baseline（软加仓；见上）
- `trade.seat_score_gate` stays **false**（窄触发仍 `REJECT_FOR_BASELINE`；见 [`seat_score_gate_research.md`](seat_score_gate_research.md)）
- `trade.from_open_gate` stays **off**（硬拒/软缩仓无弱窗 lift；见 [`from_open_gate_research.md`](from_open_gate_research.md)）
- `chop_gate` stays **off**（`SOFT_NOISE` 为 OVERLAY_ONLY：July/chop 窗改善、弱窗持平；见 [`chop_gate_research.md`](chop_gate_research.md)）
- **三袖组合**（AM launch_slope / CORE 本基线 / PM fade）见 [`sleeve_portfolio_research.md`](sleeve_portfolio_research.md)；AM/PM **不**写进本 profile
- 勿把 `wash_drop_min` 降到 1.2%（弱窗断崖）
- Hunt 专用 `hold20_noext` **不**进基线（除非另开决策）
- `scale_in` / 全局 `mae_cut` **不**进基线（`trade_toxic` 已升，勿与 quote `mae_cut` 混淆）
- **L3 `STOCK_REV` 不写进本 `peer3_v1`**（shadow only；升线见 L3 文档门槛）
- **尾损栈 `tox20+wash_m3` 不写进本 `peer3_v1`**（`research_tail` shadow；见尾损文档）
