# AM pocket：持仓特征退出（trade-mark）

目标：同一 `vd_soft` / `no_b_up` 入场上，用**因果持仓特征**替代固定 TP，把 `mean_capture` 从 ~9–11% 抬到 **≥20%**，且双窗 compound > 0。

协议：`maga7/tools/scan_am_pocket_hold_feature_exit.py`  
结果：`/mnt/s990/data/maga7/results/research_am_pocket_hold_feature_exit/`

## 设定

| 项 | 值 |
|---|---|
| entry | `vd_soft` on `no_b_up`（n=65） |
| mark | option trade-last，slip 1% |
| size | 20% / max5 / cd10 |
| 特征 | 正股 1s micro-mom、期权 stall（无新高）、mf100 flip、soft floor |
| 对照 | `TP8/SL15/h240`、`widen+stock_up` |

## 主网格（36 cfg）

| 策略 | capture | disc | blind | 备注 |
|---|---:|---:|---:|---|
| baseline TP8 | 0.093 | +33% | +21% | 胜率高，吃不满 |
| widen+stock_up | 0.107 | +43% | +21% | 已有 research 候选 |
| **stall25_tp0.25** | **0.119** | **+51%** | **+22%** | 主网格最佳 capture |
| mom_L10_c-0.0005_tp0.15 | 0.119 | +59% | +16% | disc 最好；blind 偏弱但仍正 |

**n_hit_capture20 = 0。**

## 捕获上限探针（高硬 TP + 特征骑行）

| 策略 | capture | mean_ret | disc | blind |
|---|---:|---:|---:|---:|
| ride_stall20_tp40 | 0.133 | 6.9% | +43% | +34% |
| ride_momL10_tp40 | 0.138 | 7.1% | +47% | +31% |
| **ride_combo_st15_f03_tp50** | **0.148** | **7.7%** | **+58%** | **+24%** |
| path_mfe_oracle（诊断） | 1.000 | 51.9% | — | — |

同入场路径 MFE 均值 ≈ oracle ≈ **+52%**；要 capture≥20% 需要规则成交 mean ≈ **+10.4%**。当前最好因果规则 ≈ **+7.7%**（~15% capture）。

## 结论

1. **特征退出有效**：相对 TP8，capture +2～5pp，disc 常明显更好；econ 双窗可 PASS。
2. **未达 20%**：在同一稀疏口袋上，因果 stall/mom/floor 抬升有上限；再把硬 TP 拉到 40–50% 也只到 ~15%。
3. **promote（research）**  
   - 经济优先：`stall25_tp0.25` 或 `ride_combo_st15_f03_tp50`  
   - **不** promote `CAPTURE20_*`  
4. **若仍要 ≥20%**，单靠 exit 不够，需要换杠杆之一：  
   - 更高 MFE 浓度的入场（或 scale-out 多档）  
   - 期权侧更细的 peak/trail（不只正股 mom）  
   - 接受更密入口 + 更强失败剪枝（densify 另一条线）

## promote 快照

```text
entry: vd_soft / no_b_up
mark: trade-last slip 1%
exit_research: combo stall15 + floor+3% + tp2=50%
  (or stall25 / tp25 from main grid)
capture: ~0.15 (target 0.20 NOT met on vd_soft alone)
Scanner: not production-wired
```

后续杠杆见 [`am_pocket_capture_levers.md`](am_pocket_capture_levers.md)：  
`vd_acc0bp` 曾达 cap 0.21 但太稀（~20/3月），**已退回**本档 `vd_soft + ride`。
