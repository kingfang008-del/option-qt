# AM Pulse：活跃度 / MF 叠加（能否提升 LOCK）

> 工具：`tools/scan_am_pulse_activity_overlay.py`  
> 产物：`results/research_am_pulse_activity_overlay/`  
> 对象：**现有 AM Pulse LOCK**（A+B FO@0.8%，`decision_ts = feature+60s` 因果）  
> 思路：保留 FO 触发，在决策时刻用正股 1s 活跃度 / MF 窗加严

## 协议

| 项 | 值 |
|----|-----|
| 基线入场 | FO≥0.8%，A 09:30–10:30 + B 10:30–11:30，双向 |
| 决策 | `bar_delay=60`（与泄露修复后一致） |
| 叠加门 | none / mf100 / ret60 / mf+ret60 / volz15 / volr12 / 组合 |
| 出场 | LOCK tp15/sl20/h900；PP arm8/floor3；短持对照 |
| 组合 | @10% / max2（pulse 口径） |

## 相对基线（同 LOCK 出场）

| overlay | n | 发现胜率 | maxDD | 发现复利 | blind 复利 | dual |
|---------|--:|--------:|------:|--------:|----------:|:----:|
| **none（LOCK）** | 422 | 50% | **−51%** | **−47%** | −8% | ✗ |
| **volz15** | 125 | 52% | −20% | **−7%** | **+3%** | ✗ |
| volr12 | 165 | 53% | −26% | −13% | +3% | ✗ |
| volz15+mf+ret60 | 98 | 50% | −26% | −15% | +5% | ✗ |
| mf+ret60 | 307 | 47% | −58% | −53% | −11% | ✗ |
| mf+ret60+volr12 | 136 | 51% | −26% | −20% | −2% | ✗ |

短持 / profit-protect 叠加：**没有**把任何细胞拧成 dual PASS；多数更差。

`IMPROVED` 软闸（相对基线抬复利且 blind 不崩）：**0**。

## 结论

1. **因果 FO@0.8% LOCK 本身在 May–Jul 双窗已是大亏**（延迟 60s 后），不是「再叠一层就能翻正」。  
2. **活跃度门（尤其 `volz15`）有相对改善**：回撤 −51%→−20%，复利 −47%→−7%，blind 略正 —— 属于**减损过滤器**，不是新 alpha。  
3. **纯 MF/ret 对齐 FO 方向无效甚至更差**（和「FO 已含方向、再要同向动量」冗余/过拟合有关）。  
4. **不能据此改 live LOCK 为可交易**；dual_pass=0。若 shadow 继续，最多把 `vol_z≥1.5` 当研究减损开关，勿升 live。

## 与口袋研究的关系

因果 AM **口袋**（vd+MF+volr + TP8）仍是另一条线；本文件只回答：**旧 FO Pulse 上叠活跃/MF 能否提升 → 仅减损，不成立为提升版 LOCK。**
