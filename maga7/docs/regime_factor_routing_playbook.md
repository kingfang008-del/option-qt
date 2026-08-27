# 因果因子 → Regime 动作（应用手册）

> 回答：能否用因子分析决定何时用什么策略？**能，且应只用因果因子驱动软动作。**  
> 配套：[`regime_router_research.md`](regime_router_research.md) · [`watchdog_stack_architecture.md`](watchdog_stack_architecture.md) · [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)

## 1. 原则

| 做 | 不做 |
|----|------|
| 10:30 前可算因子 → `NORMAL / DEGRADE / HALT / HUNT` | 因子切换整套开仓公式 |
| 默认永远 L0 基线 | 用 Jul 单窗侦查兵当目标臂 |
| 双窗 vs「永远 L0」≥95% 才接线 | PCA/聚类事后贴标签再配策略 |
| 一天最多动 1～2 个旋钮 | 重训未过闸的 ML router 当主开关 |

**若行情回到 2–6 牛市：** 因子不触发 → 自动落在 `NORMAL`（纯基线）。无需人工「切回牛市策略」。

## 2. 固化因子 → 动作（已验收规则）

时钟：默认 **asof=10:30**（halt/wash 窗可到 10:00）。

| 状态 | 因果规则（因子） | 动作 | 专家 |
|------|------------------|------|------|
| **NORMAL** | 无命中 | 基线 Rule-A 照常 | — |
| **DEGRADE** | `reclaim_disp55`：低开收回 + 色散门槛 | DN×0.5（QQQ>open 陷阱） | `rebound_trap_dn` |
| **HALT** | `washout_and_reclaim`：开盘洗盘广度 + 收回 | 当日禁 Rule-A UP/DN | `washout_gate_halt` |
| **HUNT** | detector `washout_reclaim`（单票） | 短窗第二条开仓偏 UP | `hunt_washout_reclaim` |

优先级：**HALT > DEGRADE > HUNT > NORMAL**（`common/watchdog.py`）。

配置：

- L1：`CONFIG/watchdog/degrade_halt_v1.json`
- L1+L2：`…_hunter_washout_reclaim_v1.json` / research profile `…peer3_watchdog_v1.json`
- 专家旋钮：`CONFIG/regime_router/experts_v1.json`
- 特征表（研究用）：`results/regime_router/router_dataset_v2.parquet`（`build_regime_router_dataset`）

## 3. 双窗证据（锁参）

### 3.1 干净对照（推荐，2026-07-25，1s→1m 重建后）

同一 `peer3_v1`：只开关 L1（hunter off）。  
窗：弱 **Feb–Apr** · 强 **May–Jul23**。  
产物：`results/watchdog/l0_vs_l1_clean_jul23`。

| 窗 | L0 ret | L1 ret | vs L0 | 触发 |
|----|--------|--------|-------|------|
| strong_may_jul | 12.43 | 12.43 | **1.000** | halt×1（账本无差） |
| weak_feb_apr | 2.845 | 2.890 | **1.016** | degrade×1 |

裁决：**`ACCEPT_RESEARCH`**（双窗 ≥95%；触发仍极窄）。

### 3.2 历史 / 易混结果

- 旧 `results/watchdog/acceptance`（至 Jul17）：亦 ACCEPT，但用旧 1m 缓存。  
- `acceptance_jul23`：**REJECT_OR_REVIEW** —— **不可信为 L1 结论**：对比的是两个不同 profile（`peer3_v1` 已开 WD+Hunt，且 trade 旋钮与 `…_watchdog_v1` 不一致）。  
- `washout_gate_scoreboard_jul23`：各 router 变体 vs 相同（基线 profile 已带 WD），不反映纯 overlay。

ML `regime_router`：**保持关闭**。

## 4. 日常怎么用

1. **Live/Shadow**：开 research baseline + Watchdog（L1；Hunt 按 ops 开关）。  
2. **每日只看状态日志**：`NORMAL` / `DEGRADE` / `HALT` / `HUNT` + `event_source`。  
3. **卫星袖** `qqq_open_cont`：独立因子（开盘续作），不进 Watchdog 状态机，但可并行。  
4. **新因子**：先写入 `router_dataset` → 只映射到上表四态之一 → 再跑双窗；禁止直接挂新开仓臂。

## 5. 复跑 / 延窗注意

```bash
python -m maga7.tools.run_watchdog_acceptance \
  --strong-end 2026-07-23 \
  --out maga7/results/watchdog/acceptance_jul23

python -m maga7.tools.run_washout_gate_scoreboard \
  --strong-end 2026-07-23 \
  --out maga7/results/regime_router/washout_gate_scoreboard_jul23
```

依赖 profile `stock_root`（1m 月文件，Rule-A 信号）。  
若 `train_data/spnq_train` 缺失，replay 会 **0 日**。已可用秒级事实源重建：

```bash
PYTHONPATH=. python -m maga7.tools.build_spnq_train_from_1s \
  --start-date 2026-01-01 --end-date 2026-07-23 --force
```

源：`/mnt/s990/data/raw_1s/stocks` → 左标签 RTH 1m → `~/train_data/spnq_train/{SYM}/{YYYY-MM}.parquet`。  
元数据：`~/train_data/spnq_train/BUILD_FROM_1S.json`。注意 **VIXY 1s 覆盖不全**（重建时可能缺部分月份）。

## 6. 下一刀（按序）

1. ~~恢复 1m / 延窗~~：已重建 + 干净 L0 vs L1 至 Jul23 **ACCEPT_RESEARCH**  
2. Shadow 确认 L1/L2 触发与 `event_source`（勿用混 profile 的 `acceptance_jul23` 当否决）  
3. **不要**为「判断牛/暴」重开 ML router 或 stock_flow 类臂  
4. 验收工具若再跑，应用 **同 profile 只拨 WD**（见 `l0_vs_l1_clean_jul23`），或先对齐两个 profile 的 trade 旋钮  
