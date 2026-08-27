# AM：期权活跃触发 × 正股 MF 定边（短持）

> 工具：`tools/scan_am_activity_mf_scalp.py`  
> 产物：`results/research_am_activity_mf_scalp/`  
> 口径修正：不是期权 mark 无脑 call↔put 换边；而是  
> **① 期权成交活跃异常 → ② 正股 MF/ret/vol 滑动窗定 UP/DN → ③ 短持 TP/SL**。

## 协议（因果）

| 层 | 定义 |
|----|------|
| 活跃触发 | 标的全合约 put+call 量窗速率 / 基线 ≥ `opt_vol_z`，且窗内总量 ≥ `min_v` |
| 定边 | `mf100` / `ret_60` / `mf100∧ret_60` / `+volr≥1.2`（既有 `session_1s_features`） |
| 车辆 | open-ladder ATM±（同 profile） |
| 出场 | TP8–12 / SL10–15 / h30–60 |
| 窗 | 09:30–11:30；双历 may_jul09 / jul10_23；@20%/5 |

资金流滑动窗**仍在用**；变的是触发源（期权活跃）与持仓时长（短 scalp）。

## 结果（144 细胞）

| 指标 | 情况 |
|------|------|
| 发现复利中位 | **−36%**（多数亏损） |
| 发现复利>0 | 仅 **4** 格 |
| 同时 blind>0 | **0**（软闸全灭） |
| 胜率≥55% | 仅 1 格，且 May/Jul 为负 |

近优（发现略正，**盲测仍负**）：

```text
act_w120_z2_v150_mf100+ret60+volr12_tp8_sl15_h60
  disc: win≈55%  cmp≈+16%  dd≈−24%
  blind: cmp≈−7%   jul≈−17%  may≈−4%
```

`dir_mode` 中位全部为负；`mf100` 单独略好于 `ret60`，但不够形成口袋。

## 结论

1. **问题提法对了**：活跃度 + MF 定边，不是 mark 路径换边。  
2. **这一轮参数下策略不成立**：活跃触发后跟 MF 短持，账本整体为负，过不了双窗。  
3. 既有 MF 窗在**多指标口袋入场**里仍有用（见 `am_pocket_multi_gate`）；单独扛「活跃→短 scalp」不够。  
4. 与已 REJECT 的 `option_flow_scout` / `stock_flow_opt` 同族风险：局部窗能刷出正数，**双窗不稳**。

## 下一步（若继续此假说）

- 活跃度改用**正股** `vol_z` / `volume_ratio`（开盘期权 quote/成交覆盖差）  
- 或：活跃只作**已有 AM 口袋的加严门**，不另开高频 scalp  
- 持仓不要锁死 30–60s；对照 h240 看是否只是出场太短  
- 未过双窗前 **不进 profile / shadow**
