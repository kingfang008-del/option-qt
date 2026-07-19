# Research baseline: full_day peer3 + streak3 + mf_idio + L2 Hunt

**Profile:** `CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json`  
**Frozen:** 2026-07-18 (L0) · **L1 on:** 2026-07-18 · **L2 on:** 2026-07-18（旧 Hunt 退出 T30+extend）  
**Ladder:** 2026-07-19 收到 **OTM3**（`ladder_otm_rungs=3`；OTM4/5 消融无边际）  
**TT1 05:** 2026-07-19 — `max_entry_abs_otm_pct=0.005` + 仅二四 `entry_confirm_bars=2` / `mode=mf`  
（细则：[`tt1_uplift_05.md`](tt1_uplift_05.md)）  
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
| Risk | `sl_mult=0.40`, `position_frac=0.2`, concurrent max 2 |
| Fill | `entry_frac`/`exit_frac`=**0.75**（2026-07-19：对齐实盘中位数，原 0.8） |
| TT1 05 | `|otm|>0.5%` 不做；**仅 Tue/Thu** 入场后再确认 2 根 1m mf |

## Snapshot metrics (offline replay)

### 勿混淆：May–Jul 三档收益（2026-07-18 锁参）

「约 1500% / +1528%」是 **当前研究基线（L2 + TT1 05）**；旧 L2 锁参约 +1255%/+1282%。

| 档 | 开什么 | May–Jul total_ret | vs L0 | 是否在 research_baseline |
|----|--------|------------------:|------:|--------------------------|
| L0 | 无 Watchdog | **+810%** | 100% | 临时关 `watchdog.enabled` 可复现 |
| L1 | Degrade + Halt（Hunter off） | **+875%** | ~108% | 对照（关 `hunter.enabled`） |
| L2 | L1 + Hunt v2（T30+extend） | ~+1282% | ~158% | 对照（关 TT1 05） |
| **L2+05** | L2 + max_otm0.5% + 二四 confirm2 | **+1528%** | **~188%** | **是（默认）** |

升线门槛（本次）：**Feb–Apr vs L0 有提升**（L2 ~**+152%** ≈ **108% of L0**；L1 ~104%）。  
2025H2 不作硬否决（波动市 + 周五期权 vs 2026 短 DTE 不可比）；见 [`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md)。

- L1 产物：`results/research_extend_mtm_full_day_peer3_l1_may_jul/`  
- L2 产物：`results/watchdog/hunter_washout_reclaim_v2_opp/`  
- **L2+05 产物**：`results/research_extend_mtm_full_day_peer3_l2_tt1_05_may_jul/`（含 `daily_may/jun/jul.csv`）  
- P2.1 `hold20_noext`（~+840%）仅为备选退出研究，**未**进基线。

| Window | total_ret | MaxDD | notes |
|--------|----------:|------:|-------|
| May–Jul (→07-17) **L2+05** | **+1528%** | −12.2% | **当前基线**；日明细见上 |
| May–Jul (→07-17) L2 对照 | ~+1282% | −12.2% | 无 TT1 05 |
| May–Jul (→07-17) L1 对照 | +874.7% | −13.2% | Hunter off |
| May–Jul (→07-17) L0 对照 | +810.4% | −13.2% | 无 Watchdog |
| Feb–Apr L0 | ~+140% | −28.9% | weaker tape |
| Feb–Apr **L2** | ~+152% | — | **~108% of L0**（升线依据；05 待补跑） |

### 工程闸门（防忘）

| 项 | 状态 |
|----|------|
| L1 流式对拍 | **通过** |
| L2 流式对拍（旧退出） | **通过**（May–Jul 60 笔） |
| Paper | 上实盘前工程门 |
| VIX/VIXY 急拉直闸 | **无**；扫描见 `results/watchdog/vixy_spike_scan_2026h1/` |
| 三次大亏日 | 基线毒性路径；全局 `mae_cut` **REJECT** |

## Ops

- **操作手册**：[`maga7_operations_guide.md`](maga7_operations_guide.md)
- Live/shadow: `maga7/SHELL/start_maga7_live_session.sh`
- Day stream check: `maga7/SHELL/run_day_stream_check.sh`
- Dashboard: `python dash/run.py`
- Replay results: `maga7/results/`（gitignored）

## Next validation（不动本 profile 旋钮）

见 [`l2_next_acceptance_checklist.md`](l2_next_acceptance_checklist.md)：P3 Live/shadow → P2.3 Hunt 日熔断 → P0.3/P0.4 敏感度。

## Non-goals in this freeze

- Do not bare-enable `mf_flip` / streak early exits (hurts May–Jul)
- Do not global-tighten `sl_mult` for premium stops
- `tcn_gate.enabled` stays **false**；见 [`tcn_gate_research.md`](tcn_gate_research.md)
- `trade.topk_backfill_on_block` stays **false**
- TopK 默认 `rank_by=earliest`
- 勿把 `wash_drop_min` 降到 1.2%（弱窗断崖）
- Hunt 专用 `hold20_noext` **不**进基线（除非另开决策）
