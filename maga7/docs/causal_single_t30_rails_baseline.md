# Mag7 因果基线：Mag7+GOOGL + peer_min3 + T+30

> 冻结日期：**2026-07-17**（由 2026-07-16 Mag7-only 基线升级）  
> 时钟：`bar_availability_delay_seconds=60`（预聚合 1m 股票表 → 分钟完成后才可交易）  
> 成交：open_ladder OTM5，`fill_frac=0.8`，QQQ align regime，concurrent `position_frac=0.20`  
> 入场宽度：`peer_align_min=3`，`peer_align_mode=mf10`，peers=Mag7 七只（不含 GOOGL 自身计数池外仍可交易 GOOGL）

## 1. 当前基线（以此为准，后续优化都相对它）

| 项 | 值 |
|---|---|
| **Profile** | [`single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json) |
| Universe | Mag7 + **GOOGL** |
| Scheme | **`single`**（每标的每日至多首笔 Rule-A） |
| 入场过滤 | **`peer_align_min=3`**：信号时刻 Mag7 中至少 3 只 `mf10` 与开仓方向同向（含自身） |
| 出场 | **`exit_mode=none`**：TP×1.6 / SL×0.4 / **T+30**（无 `mf_flip`） |
| 选约 | `open_ladder`，`ladder_otm_rungs=5` |
| 仓位 | concurrent p20（独处 20%；并发第二腿 10%；最多 2） |

```bash
python -m maga7.tools.run_replay_offline \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1.json \
  --scheme single \
  --tag replay_single_t30_rails_p20_mag7_googl_peer3_may_jul_delay60
```

冻结产物：`maga7/results/replay_single_t30_rails_p20_mag7_googl_peer3_may_jul_delay60/`  
消融来源：`maga7/results/peer_align_ablation_mag7_googl_may_jul/`、`si_pe_ablation_mag7_googl_may_jul/`

### 1.1 相对上一版（无 peer / Mag7-only）的提升（May–Jul）

| 配置 | Ret / MaxDD | n | 备注 |
|---|---|---|---|
| Mag7-only T+30（旧基线） | +252.8% / -25.1% | 64 | 2026-07-16 冻结 |
| Mag7+GOOGL 无 peer | +303.9% / -27.0% | 64 | 扩池对照 |
| **Mag7+GOOGL peer_min3（新基线）** | **+374.8% / -16.5%** | **53** | 拒 11 笔弱宽度单 |

连亏窗（05-06~11、05-20~22、07-07~09）日复利约 **-41% → -15%**。

---

## 2. 因果基线成绩（冻结窗）

### 2.1 May–Jul（2026-05-01 ~ 2026-07-13）— 主证据窗

| 指标 | 值 |
|---|---|
| 账户总收益 / MaxDD | **+374.8% / -16.5%** |
| 笔数 / 胜率 | 53 / **64.2%** |
| 单笔期望 | **+18.2%** |
| `n_peer_block` | 11 |

### 2.2 对照：Mag7-only + peer_min3（非基线）

| 窗 | Mag7-only 无 peer | Mag7-only peer_min3 |
|---|---|---|
| May–Jul | +252.8% / -25.1% | +280.8% / -16.7%（改善） |
| Jan–Jul | **+863.6% / -28.6%** | +797.8% / -28.6%（全年略差） |

**说明**：新基线主证据是 **Mag7+GOOGL May–Jul**。Mag7-only Jan–Jul 加 peer 尚未打赢旧 Mag7-only 全年数字；后续若做全年 Mag7+GOOGL+peer3，需补齐数据后再冻结一版全窗成绩。

---

## 3. 为何升格 peer_min3（而不是 SI/PE/动态退出）

同 delay=60 消融结论：

1. **动态退出**（`mtm_floor` / `flow_die` / `flow_mtm` / `mtm_trail`）：连亏窗可止血，但误杀赢家，全期远逊硬 T+30。  
2. **文中 SI≥0.57（≈6/7 同向）**：连亏更好，但笔数砍到 ~26，May–Jul 仅 ~+213%。  
3. **PE 吸收过滤**：单独略损或略帮；`peer3+PE` DD 略好但收益低于纯 peer3。  
4. **`peer_align_min=3` + mf10 + Mag7 peers**：May–Jul 收益与回撤同时改善，拒单均质偏弱（11 笔均约 -0.5%）。

实现：`signal.peer_align_min` / `peer_align_mode` / `peer_symbols`（`maga7/common/signals.py` → replay / stream_engine）。

---

## 4. 上一版 Mag7-only 基线（保留对照）

| 项 | 值 |
|---|---|
| Profile | [`single_qqq_open_ladder_atm5otm_t30_rails_p20_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_v1.json) |
| Role | `prior_causal_baseline_mag7_only` |
| Jan–Jul | **+863.6% / -28.6%**（134 笔） |
| May–Jul | +252.8% / -25.1%（64 笔） |
| 产物 | `results/replay_single_t30_rails_p20_jan_jul_delay60/` |

无 peer 的 Mag7+GOOGL 对照：[`…_googl_v1`](../CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_v1.json)。

---

## 5. 为什么旧「约 33 倍 / -22%」是虚幻的

旧 headline（`+3375% / MaxDD -22%`）来自 **未加分钟可用性延迟** 的研究口径，不能当作可交易业绩。根因见历史说明：预聚合 1m 在 `delay=0` 下前视约 60 秒。  
**+3375% / -22%**：已作废为上线证据。

---

## 6. OTM5 vs ATM-only（不要「简化」成 ATM）

| 配置 | May–Jul（Mag7-only 旧窗） | 角色 |
|---|---|---|
| **OTM5（`ladder_otm_rungs=5`）** | **+252.8% / -25.1%** | 选约不变，仍用 OTM5 |
| ATM-only（`rungs=0`） | +52.1% / -24.6% | 研究对照，勿升格 |

---

## 7. 后续优化起点

相对 **本基线**（Mag7+GOOGL + peer_min3 + delay=60 + single + T+30 rails + OTM5）：

- 补 Mag7+GOOGL+peer3 的 **Jan–Jul 全窗**成绩后再决定是否改冻结窗  
- 出场：保持硬 T+30 rails 为默认；勿启用 flow/MTM **软退出**  
- **条件化延长**（`hold_extend` T30→T45）：研究中，见 [`hold_extend_t30_t45_research.md`](hold_extend_t30_t45_research.md)；候选 `extend_mtm_mf`（更保守）/ `extend_mtm_only`（收益更高），尚未升格  
- **mf10 + 快窗提前开火**（`early_on_mf_fast` / `mf_fast_window` 3\|5）：研究中，见 [`mf_fast_early_research.md`](mf_fast_early_research.md)；候选 `early_mf5_s6`（MaxDD 更好），勿默认写入本基线  
- **Regime 翻面 / Put–VIXY / QQQ mf10 对齐**：见 [`regime_flip_research.md`](regime_flip_research.md)；候选 `qqq_mf10_align`（MaxDD -16.5%→-12.2%，ret 有代价）；裸 `qqq_day_flip_mode=block` / `put_vixy_z_min=0` 过猛，勿默认  
- **事件日禁入**（`event_calendar_block`）：见 [`event_calendar_full_day.md`](event_calendar_full_day.md)、[`event_calendar_block_research.md`](event_calendar_block_research.md)；叠 `extend_mtm_only` 时 `full_day` May–Jul **+673% / -12.2%**；研究候选，日历需维护/API 接入  
- **07-07~09 选标失误**（事件日历未覆盖）：见 [`jul7_9_mover_vs_pick_analysis.md`](jul7_9_mover_vs_pick_analysis.md) — TopK 抢最早信号，错过 TSLA/META 主趋势  


- SI≥0.57 / PE：仅研究；勿替换 peer_min3  
- MU/AVGO 扩池仍弱于 Mag7+GOOGL，不默认并入  
- stream / live 对齐本 profile + `scheme=single`

旧临时生产 [`m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1`](../CONFIG/strategy_profiles/m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json) 仅作对照。
