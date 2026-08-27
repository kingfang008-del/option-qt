# Session Flow Gate（QQQ+VIXY chop × 日累积资金流领袖）

> 主动入场门控（非亏损后降档）。配套模块：`common/session_flow_gate.py` · 消融：`tools/run_session_flow_gate_ablation.py`

## 1. 动机

Jul10–23：指数磨底/震荡，个股单边。滑动 `mf10` / peer 对齐抓不到「当天资金领袖」；亏后 size cut 太晚。

## 2. 因果定义（asof 默认 10:30）

| 片 | 规则 |
|----|------|
| A 指数 chop | `|QQQ from_open| ≤ q_am_max`（默认 0.5%）；可选 `|VIXY from_open| ≤ vixy_am_max`（缺 VIXY 缓存 fail-soft，只绑 QQQ） |
| B 领袖 | 当日 RTH `cum=cumsum(net$)`，在 Mag7 内按 `|cum|` Top-K；且 `sign(cum)` 与方向对齐 |

`when`：

- `chop_only`：仅 chop 日强制 B；趋势日放行
- `always`：始终强制 B
- `chop_block`：chop 日直接禁基线（无领袖豁免）

失败 B：`mode=block|scale`。

## 3. 消融臂

相对同一 spine `peer3_v1`（只拧 `session_flow_gate`）：

| arm | 含义 |
|-----|------|
| PRE | 关 |
| CHOP_BLOCK | chop → 全日禁基线 |
| LEADER_ALWAYS | 始终 Top-K+符号 |
| CHOP_AND_LEADER | chop → 只放领袖 |
| CHOP_LEADER_SOFT | chop → 非领袖 ×0.5 |

窗：`weak_jan_mar` / `mid_may_jul9` / `jul10_23`。

**PASS 提示**：jul `vs_PRE > 1` 且 weak/mid `vs_PRE ≥ 0.85` → `SESSION_FLOW_LIFT`。否则不接线。

## 4. 结果（2026-07-25，`session_flow_gate_ablation_v1`）

```bash
PYTHONPATH=. python -m maga7.tools.run_session_flow_gate_ablation \
  --out maga7/results/session_flow_gate_ablation_v1
```

**裁决：`NO_LIFT` — 不接线。**

| arm | weak vsPRE | mid vsPRE | jul ret (PRE=0.207) |
|-----|------------|-----------|---------------------|
| CHOP_BLOCK | 0.55 | 0.21 | 0.00（n=0） |
| LEADER_ALWAYS | 0.79 | 0.22 | −0.143 |
| CHOP_AND_LEADER | **1.13** | 0.49 | −0.143 |
| CHOP_LEADER_SOFT | 0.99 | 0.67 | 0.031 |

要点：

- Jul10–23 有 **8/10** 日被标成 chop（`q_am_max=0.5%`），门控过宽。
- 硬拦领袖在 Jul **砍掉赢家、留下毒单**（见 trades 对照）；soft×0.5 也伤 Jul。
- `CHOP_AND_LEADER` 仅在 Jan–Mar 有 lift，May–Jul9 腰斩 → 单窗假象。

## 5. v2：收窄 chop + leader boost（2026-07-25）

动机：v1 硬拦误杀 Jul 赢家；改 `mode=boost`（领袖×boost，非领袖默认×1）+ `q_am_max=0.3%`。

```bash
PYTHONPATH=. python -m maga7.tools.run_session_flow_gate_ablation \
  --suite v2 --out maga7/results/session_flow_gate_ablation_v2
```

| arm | weak | mid | jul vsPRE |
|-----|------|-----|-----------|
| NARROW_BLOCK | 1.04 | 0.61 | −0.30 |
| NARROW_SOFT | 1.02 | 0.79 | 0.37 |
| BOOST_CHOP | **1.32** | **1.42** | 0.59 |
| BOOST_NARROW | 1.08 | **1.26** | **0.95** |
| BOOST_TILT | 1.09 | 1.12 | 0.64 |
| BOOST_ALWAYS | 1.29 | 1.44 | 0.79 |

**裁决：仍 `NO_LIFT`（Jul 未抬升）。**  
`BOOST_NARROW` 是唯一接近「不伤 Jul + 抬 mid/weak」的臂，但 **解决不了 Jul10–23 口袋**（毒单 META/AMD 本身常是 cumflow 近领袖；加仓只会放大 mid 赢家，也会放大 Jul 毒单）。

诊断备忘：`q_am≤0.5%` 把 Jul 交易日几乎全标 chop；`≤0.3%` 降到约一半。硬拦方向已证伪。

## 6. 接线纪律

- **Do not wire / freeze off**（含 boost）。
- Session cumflow **不宜再当 Jul 救援旋钮**；若另开课题，可把 `BOOST_NARROW` 当「先验窗 size tilt」单独验收（目标≠修 Jul）。
- 下一主动方向建议离开 cumflow 领袖：事件/新闻门、入场 toxic/adverse、或卫星臂（`qqq_open_cont`）参与度。
- 不替代 Watchdog。
