# DN 结构门 ablation（反抽日假空）

针对 07-17 类「低开反抽后再回落 → Rule-A DN → put 亏」行情。

## 开关

| 开关 | 位置 | 语义 |
|------|------|------|
| `signal.block_dn_if_above_open` | replay | DN 且标的现价 **> 今开** → 挡 |
| `regime.block_dn_if_qqq_above_open` | Mag7RegimeGate | DN 且 QQQ 现价 **> 今开** → 挡 |

默认均 **false**（freeze 不打开）。

## Scoreboard（May–Jul → 2026-07-17，freeze 底仓）

| variant | total_ret | MaxDD | n | 07-17 |
|---------|-----------|-------|---|-------|
| baseline | **+810%** | −13.2% | 50 | −6.6%（NVDA+TSLA DN） |
| A `above_open` | +807% | −13.2% | 49 | −7.0%（仅挡 NVDA，TSLA 仍在） |
| B `qqq_above_open` | +584% | −13.2% | 41 | **0**（两笔都挡） |
| A+B | +584% | −13.2% | 41 | 0 |

明细：`maga7/results/dn_structure_gate_ablation_may_jul_to_0717/`。

## 结论

- **A**：几乎只动 07-17 NVDA，日线仍被 TSLA 拖累；整段收益几乎不变 → 可作轻量可选，**不解决当天**。  
- **B**：能清空 07-17，但顺带砍掉多笔盈利空单，May–Jul 从 +810% → +584% → **REJECT for freeze**。  
- `QQQ>开 ∧ LOD bounce≥2%`：样本极少，且 May–Jul toxic 命中 **0** → 不作硬门。

## 续：QQQ>今开 时 DN **缩仓**（非硬挡）

开关：`regime.scale_dn_if_qqq_above_open`（float，默认 null/off）。  
脚本：`maga7/tools/run_dn_qqq_scale_ablation.py` → `results/dn_qqq_scale_ablation_dual_window/`。

| window | variant | total_ret | vs base | 07-17 day_ret |
|--------|---------|-----------|---------|---------------|
| May–Jul | baseline | +810% | — | −6.6% |
| May–Jul | scale 0.5 | +701% | **86%** | −3.3% |
| May–Jul | scale 0.25 | +648% | 80% | −1.7% |
| May–Jul | hard block | +584% | 72% | 0 |
| Feb–Apr | baseline | +140% | — | — |
| Feb–Apr | scale 0.5 | +125% | **90%** | — |
| Feb–Apr | hard block | +97% | 69% | — |

缩仓明显优于硬挡（07-17 半损、强窗少砍），但双窗仍 <95% 且弱窗略损 → **暂不升 freeze**；若只想压单日尾部风险，可作可选研究开关。
