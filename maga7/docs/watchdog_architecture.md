# Watchdog 架构：保住基线 · 自动降级 · 短窗猎手槽

> **总览（2026-07-18 落盘）**：[`watchdog_stack_architecture.md`](watchdog_stack_architecture.md)  
> 本文保留状态机与接线细节；验收数字以总览与各研究文档为准。

## 目标

信号层微调已近天花板。系统改成三层，而不是继续拧 Rule-A：

1. **Baseline**（freeze）— 只读默认路径  
2. **Degrade / Halt** — 特殊行情自动降级或停开  
3. **Hunter** — 短时机 Watchdog 臂/撤（研究可开，freeze 默认空）

## 状态机

```text
NORMAL ──毒→ DEGRADE ──更毒→ HALT
   │                      │
   └──(臂)→ HUNT ─────────┘   ※ 默认不可从 HALT 进入 HUNT
            TTL 强制回 NORMAL
```

优先级：**HALT > DEGRADE > HUNT > NORMAL**。

| 状态 | 含义 | 当前研究动作 |
|------|------|----------------|
| `normal` | 跑 freeze | 无 overlay |
| `degrade` | 软降级 | `reclaim_disp55` → `rebound_trap_dn`（DN×0.5 when QQQ>open） |
| `halt` | 硬停 | Mag7 广度 `washout_and_reclaim` → 禁 UP/DN |
| `hunt` | 短窗机会 | 研究默认 `washout_reclaim` v2；**freeze 上 enabled=false** |

每个非 NORMAL 状态可带 `ttl_minutes`；到期回到 NORMAL。

## 代码

| 模块 | 作用 |
|------|------|
| `maga7/common/watchdog.py` | 状态机、overlay、规则评估、TTL、Hunt 候选 |
| `maga7/common/orb_open.py` | washout / ORB / washout_reclaim |
| `maga7/common/replay.py` | 按日 `begin_day`；Hunt 注入；mutex / opposite |
| `CONFIG/watchdog/degrade_halt_v1.json` | Degrade+Halt |
| `CONFIG/watchdog/degrade_halt_hunter_washout_reclaim_v1.json` | + Hunter v2 |
| `…_peer3_watchdog_v1.json` | L1 研究 profile |
| `…_watchdog_hunter_washout_reclaim_v1.json` | L2 研究 profile |

观测字段：`watchdog_state` / `watchdog_reason` / `route` / `event_source`。  
Summary：`watchdog_enabled`、`n_watchdog_days`、`watchdog_state_counts`、`n_hunt_*`。

### Live / OMS

`maga7/live/scanner.py`：

- 日切时若 `watchdog.enabled` 且 `stock_by` 可用 → `begin_day` + overlay 进 `Mag7RegimeGate`  
- 信号 `meta` / OMS exec payload 带 `watchdog_state`、`watchdog_reason`、`route`  
- Hunter 实盘开火弱于 replay（默认不注入）；Degrade 缩仓已接 OMS  

**仓位缩放（已接）**：`regime_gate.check` → `meta.regime_size_scale` →  
`oms_dry` / `oms_stub` / `broker_oms` 在 `resolve_size_frac` 之后乘上该系数。

## 与 `regime_router` 关系

- **新代码走 `profile.watchdog`**。  
- 若仅 `regime_router.enabled=true`，自动桥接为 **degrade-only** Watchdog。  
- Freeze 上两者均 **enabled=false**。

## Hunter 槽

| 检测器 | May–Jul vs freeze | Feb–Apr | 判定 |
|--------|-------------------|---------|------|
| `orb_fractal` | ~45% | ~46% | **REJECT** |
| `early_mf` | ~22% | ~107% | **REJECT** |
| `washout_reclaim` v2（wd≥1.5% + opposite baseline） | **~155%** | **~108%** | **ACCEPT_RESEARCH**（默认仍 off） |

v2 mutex：`mutex_scope=symbol_dir` + `allow_baseline_opposite=true`（详见  
[`hunter_washout_reclaim_research.md`](hunter_washout_reclaim_research.md)）。

## 验收

| 项 | 门槛 |
|----|------|
| 强/弱窗 vs freeze | ≥95% |
| freeze 默认 | watchdog off；hunter off |
| 每笔可解释 | `watchdog_state` / `event_source` 入 trades |

```bash
python -m maga7.tools.run_watchdog_acceptance
python -m maga7.tools.run_washout_reclaim_hunter_scoreboard \
  --out maga7/results/watchdog/hunter_washout_reclaim_v2_opp
```

**L1 / L2 Verdict: `ACCEPT_RESEARCH`**（freeze 默认仍 off，待 ops 签字）。  
过拟合边界见总览 §6。
