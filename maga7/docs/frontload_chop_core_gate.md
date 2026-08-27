# FRONTLOAD_CHOP：识别早盘定价日 × CORE 降权/停手

> 模块：`common/frontload_chop.py`  
> 扫描：`tools/scan_frontload_chop_days.py` → `results/research_frontload_chop_days_v2/`  
> 验收：`tools/run_frontload_core_ab_accept.py`  
> 前置：`docs/core_peer3_week_20260720_24.md` §5

## 标签 v2（因果 @10:30）

```text
FRONTLOAD_CHOP iff
  median(|Mag7 open→10:30|) ≥ 0.8%
  AND (#names with |h1|≥0.8%) ≥ 4
  AND median(|1m ret| in 10:15–10:30) ≤ 8.5bp
  AND median(|9:30–10:00|) ≥ 0.6%
  AND median(|9:30–10:00| / |10:00→10:30|) ≥ 1.85
```

May–Jul 打标 **10/61 ≈16%**；本周 21/22/24 命中（含 MSFT 07-22）。

## 子状态 overlay（研究主推）

全年 AND（v2）仍伤强窗。改为：**日标签命中后，仅当入场时 regime 偏弱才 scale/block**。

```text
weak_substate (entry-time, causal via Mag7RegimeGate):
  vixy_z ≥ 0.75
  OR  |qqq_from_prev| ≤ 0.8%     # optional chop leg
```

| 配置 | 强 keep | 弱 keep | 本周 A | promote |
|------|--------:|--------:|-------:|--------|
| v2 always（无 overlay） | 0.77 | 0.76 | −1.3% | **NONE** |
| **weak: vixy≥0.75** | **1.01** | **0.87** | −1.3% | **A_scale_research** |
| weak: vixy≥0.75 **OR** \|fp\|≤0.8% | 0.96 | 0.87 | −1.3% | **A_scale_research** |
| B_block + 同上 overlay | ≥0.93 | **0.80** | 0% | FAIL（弱窗） |

产物：
- `results/research_frontload_core_ab_overlay_vixy/`
- `results/research_frontload_core_ab_overlay_or/`

闸门：`strong≥0.90` ∧ `weak≥0.85` ∧ 本周≥PRE → **A_scale 过闸**；B 弱窗 keep≈0.80 不过。

## 裁决

1. **研究候选：`A_scale` + `overlay=weak` + `vixy_z≥0.75`（优先）。** 强窗几乎满保留甚至略好（砍掉 07-22 半仓/空手的拖累）；弱窗 keep≈0.87；本周 −2.6%→−1.3%。  
2. **OR 腿（再加 \|fp\|≤0.8%）** 也可过闸，但强窗 keep 略低（多砍几笔偏平 QQQ 日）；作备选。  
3. **B_block 仍不过闸**（弱窗 keep≈0.80）。  
4. **尚未写入 peer3 基线 profile**——先 shadow / 显式 research 键；注意部分强窗日 `vixy_z=NaN` 会自然跳过 overlay（数据覆盖副作用，不是特征）。  
5. 开盘短持袖若做，与 v2 日标签互斥；CORE 侧只用 **weak overlay** 降权。

## 复现

```bash
# 日标签
PYTHONPATH=. python -m maga7.tools.scan_frontload_chop_days \
  --tag research_frontload_chop_days_v2

# 主推 overlay（vixy-only）
PYTHONPATH=. python -m maga7.tools.run_frontload_core_ab_accept \
  --tag research_frontload_core_ab_overlay_vixy \
  --overlay weak --overlay-vixy-z-min 0.75 --overlay-max-abs-qqq-fp none

# 备选 OR
PYTHONPATH=. python -m maga7.tools.run_frontload_core_ab_accept \
  --tag research_frontload_core_ab_overlay_or \
  --overlay weak --overlay-vixy-z-min 0.75 --overlay-max-abs-qqq-fp 0.008
```
