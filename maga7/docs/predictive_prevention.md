# Predictive Prevention：早盘坏日预测（非连亏熔断）

## 目标

在 Rule-A 开火**之前**，用因果早盘特征判断「今天是否 UP-toxic 混合洗盘日」，并自动切到 expert overlay（降仓 / 禁 UP）。

这与「连亏 N 笔 → 熔断」无关：后者是事后止血；本模块是**事前预防**。

## 为什么需要第三轨（Prevention）

L1 Halt / Degrade 在 2026-07-20 类日子上可能仍为 NORMAL：

| 条件 | 当日大致值 | L1 门槛 | 结果 |
|------|-----------|---------|------|
| washout breadth | 4 | halt ≥5 | 未 halt |
| QQQ bounce from LOD | ~0.48% | reclaim ≥0.8% | 未 degrade |
| Mag7 `frac_above_open` | ~0.625 | reclaim 要求 ≤0.55 | 未 degrade |

特征是：**若干名字被洗、但 Mag7 并非全面偏红** → 经典 `washout_and_reclaim` / `reclaim_disp55` 都漏掉，随后 UP 火并亏钱。

## 规则表

| 旋钮 | 默认（research baseline） | 含义 |
|------|---------------------------|------|
| `watchdog.prevention.enabled` | **`false`（双窗 FAIL 后默认关）** | 见 [`predictive_prevention_scoreboard.md`](predictive_prevention_scoreboard.md) |
| `rule` | `mixed_wash_up` | washout breadth ≥3（`wash_drop_min=0.8%`）且 `frac_above ∈ [0.35, 0.70]` — **过宽，待收窄** |
| `prefer_risk_off` | `true`（研究旋钮） | 硬防：`up_toxic_block`；当前勿当 freeze |
| soft arm | `up_toxic` | `prefer_risk_off=false` 时 UP×0.5；强窗仍仅 54% of off |

别名：`dispersion_wash_up` / `up_dispersion_risk` / `predict_up_toxic`。

专家定义：`maga7/CONFIG/regime_router/experts_v1.json`（含 `up_toxic_block`）。

## 接线

| 层 | 位置 |
|----|------|
| 规则 | `maga7/common/watchdog.py` → `eval_router_rule` + prevention lane（halt/degrade 之后） |
| 门控 | `LiveRegimeGate.check` 已兑现 `block_directions` / `direction_size_scale`（此前 live 会忽略 overlay） |
| 门面 | `maga7/common/predictive_prevention.py` |
| Profile | research baseline `…peer3_v1.json` → `watchdog.prevention` |
| 日志 | scanner：`PREVENTION …` / reason=`prevention:mixed_wash_up` |
| 产物 | session `prevention.json` + `scanner_state.prevention`；Dash「早盘 Prevention」卡 |

优先级仍是：**HALT > DEGRADE > PREVENTION > NORMAL**（prevention 命中且仅禁一侧时记为 DEGRADE + overlay）。

## 消融注意

- `prefer_risk_off=true` 会在混合洗盘日**禁全部 UP**，需双窗 scoreboard 后再当 freeze。
- 软臂 `up_toxic`（半仓 UP）适合作为对照。
- DL / TCN 只适合日后做「选 expert」；低置信度应回退 baseline / 保守，而不是更激进。勿用连亏熔断代替本轨。

## 相关

- [`watchdog_architecture.md`](watchdog_architecture.md)
- [`regime_router_research.md`](regime_router_research.md)
- 单测：`maga7/tests/test_predictive_prevention.py`
