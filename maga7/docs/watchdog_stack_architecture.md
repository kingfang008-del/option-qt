# Mag7 新架构落盘：Baseline · Watchdog · Hunter

> 状态日期：2026-07-24（Hunt 已进 research_baseline；窄专家注册表见下）  
> 角色：**研究栈文档**（freeze / 生产默认仍关闭部分 overlay）  
> 时钟与 OMS 契约仍以 [`current_architecture.md`](current_architecture.md) 为准。  
> **完善优化路线**：[`watchdog_optimization_roadmap.md`](watchdog_optimization_roadmap.md)  
> **窄形态专家升级**：[`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)

## 1. 一句话

**Rule-A 基线只读不动**；坏日用 Watchdog 降级/停开；早盘窄机会走 Hunter 槽。  
三层分离，禁止把特例行情拧回全局信号参数。

## 2. 分层（当前研究栈）

```text
┌─────────────────────────────────────────────────────────────┐
│  L0  Baseline (research_baseline causal core)               │
│      open_ladder OTM5 · peer3 · QQQ align · extend_mtm      │
│      window 10:30–14:00                                     │
├─────────────────────────────────────────────────────────────┤
│  L1  Watchdog Degrade / Halt   (P1.1: research 默认 on)     │
│      Degrade: reclaim_disp55 → DN×0.5 when QQQ>open         │
│      Halt:    Mag7 广度 washout_and_reclaim → 全日禁开仓     │
├─────────────────────────────────────────────────────────────┤
│  L2  Hunter 短窗槽   (washout_reclaim v2 ACCEPT_RESEARCH)   │
│      单票深洗→收回开盘 · 日≤1 · mutex symbol_dir            │
│      + allow_baseline_opposite（不挡反向 Rule-A）           │
│      research_baseline 上 hunter.enabled=true（P2 升线）    │
├─────────────────────────────────────────────────────────────┤
│  Registry  窄形态专家目录（非第二基线）                     │
│      CONFIG/narrow_experts/catalog_v1.json                  │
│      开仓类候选（CORE DN sync / AM）QUOTE_REJECT → 默认 off │
└─────────────────────────────────────────────────────────────┘
```

优先级：**HALT > DEGRADE > HUNT > NORMAL(baseline)**。  
Halt 日默认不 arm Hunt（`block_when_halt=true`）。

| 层 | Profile / Config | 验收 |
|----|-----------------|------|
| L0+L1+L2 | `…_extend_mtm_full_day_peer3_v1.json` | 研究基线，见 [`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md) |
| L1 对照 | 关 `hunter.enabled` / `…_watchdog_v1.json` | 双窗 ≥95% L0；May–Jul ~108% |
| L2 细则 | washout_reclaim + opp | 见 [`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md) |
| 窄专家注册 | `narrow_experts.catalog_path` | [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md) |

否决检测器（槽位保留，勿升）：`orb_fractal`、`early_mf`。  
Hunter 细则：[`hunter_washout_reclaim_research.md`](hunter_washout_reclaim_research.md)。  
状态机细节：[`watchdog_architecture.md`](watchdog_architecture.md)。

## 3. 代码与观测

| 模块 | 职责 |
|------|------|
| `common/watchdog.py` | 状态机、规则、Hunt 候选、TTL |
| `common/orb_open.py` | washout / ORB / **washout_reclaim** |
| `common/replay.py` | `begin_day`、Hunt 注入、mutex / opposite、打标 |
| `live/scanner.py` | 日切评估 + meta；Degrade 缩仓已接 OMS |

Trades / daily：`watchdog_state`、`watchdog_reason`、`route`、`event_source`（`baseline`|`hunt`）。  
Summary：`n_hunt_*`、`watchdog_state_counts`、`n_hunt_mutex_skip`。

## 4. 开关纪律（防「研究默认泄漏」）

| 开关 | research_baseline | 备注 |
|------|-------------------|------|
| `watchdog.enabled` | **true**（P1.1） | L0-only 对照可临时关 |
| `hunter.enabled` | **true**（P2） | washout_reclaim；新 detector 默认 off |
| `narrow_experts` | **registry on** | 目录启用；QUOTE_REJECT 开仓臂仍 off |
| `tcn_gate` / `lgbm_bouncer` / `regime_router` | **false** | 各自研究线 |
| Rule-A `window_start` | **10:30** | 禁止为吃早盘而改基线窗 |

升线顺序：L1 → L2（已进研究基线）→ 新开仓专家须 **quote 双窗 PASS** 才挂进 registry/ACCEPT。

```bash
# L1 验收
python -m maga7.tools.run_watchdog_acceptance

# L2 scoreboard（含 L1）
python -m maga7.tools.run_washout_reclaim_hunter_scoreboard \
  --out maga7/results/watchdog/hunter_washout_reclaim_v2_opp
```

## 5. 与 Halt / Hunt 的极性

| 规则 | 粒度 | 动作 |
|------|------|------|
| Halt `washout_and_reclaim` | Mag7 **广度**深洗 + QQQ 假收回 | **停手**（如 07-17） |
| Hunt `washout_reclaim` | **单票**深洗后收回开盘 | **短窗做多**（非 Halt 日） |

二者名字相近、极性相反；配置分文件，勿混用阈值。

## 6. 过拟合：能压多少、压不住什么

### 6.1 架构已经做的（有助于降拟合）

1. **基线冻结**：特例不回写 Rule-A 全局旋钮（避免「为一日改十年」）。  
2. **窄触发 / 低次数**：Halt 强窗约 1 次；Hunt 强窗约 12 / 弱约 5——不是天天开火。  
3. **双窗门槛**：强（May–Jul）与弱（Feb–Apr）均 ≥95% freeze，否决单窗炫技。  
4. **否决清单**：`orb_fractal` / `early_mf` / 宽 washout 门等已明确 REJECT，减少「再拧一版」。  
5. **因果 1m + 可审计字段**：每笔可追溯 `event_source`，便于抓占坑/假 V。  
6. **分层验收**：L1 不依赖 L2；L2 增量可单独关掉。

### 6.2 做不到「最大限度」——诚实边界

| 风险 | 现状 |
|------|------|
| **同窗调参** | `wash_drop=1.5%`、`opp` 等在 **同一** May–Jul / Feb–Apr 上消融选出，不是真正 hold-out |
| **事后修补** | `allow_baseline_opposite` 直接针对已解剖的 06-26 / 07-02，有故事拟合成分 |
| **样本薄** | Hunt 笔数极少；一笔 −50% 期权即可左右观感 |
| **池外故事** | MU 07-17 激励了形态，但 **不在** Mag7 freeze 池 |
| **Live 未证** | Hunter 实盘注入仍弱于 replay；纸面/实盘分布漂移未验证 |
| **未做** | Walk-forward / 更早年份 OOS、符号置换、参数扰动稳定性、费用敏感度系统扫描 |

**结论**：新架构是 **结构化约束 + 研究闸门**，比继续拧 Rule-A **更抗一类过拟合**；  
**不是**统计学意义上的「最大限度降过拟合」。升线前至少补：

1. 额外 OOS 窗（如 2025H2 或 2026-01）不调参只评分；  
2. Hunt 参数邻域稳定性（wd 1.2–1.8%、有/无 opp）；  
3. 纸面会话：Halt / Hunt 触发率与损益是否同量级；  
4. Hunter 保持 off，直到 P2/OOS + ops 签字。

## 7. 当前推荐姿态

| 用途 | 开什么 |
|------|--------|
| 日常研究 / shadow | **L0+L1+L2**（research_baseline）；L0-only 对照临时关 `watchdog.enabled` |
| 坏日防护对照 | L0 + **L1**（关 `hunter.enabled`） |
| 新形态试点 | 先写 `catalog_v1.json`；quote PASS 前 **禁止**默认注入 |

### 7.1 收益速查（防忘：~1200% 是 L2 对照窗，非无脑复利预期）

同窗 May–Jul（→07-17），锁参 2026-07-18：

| 档 | total_ret | vs L0 | research_baseline？ |
|----|----------:|------:|---------------------|
| L0 | +810% | 100% | 对照（关 watchdog） |
| L1 | +875% | ~108% | 子集 |
| **L2 Hunt v2** | **+1255%** | ~155% | **是**（与 L1 同开） |

完整表与分月：[`research_full_day_peer3_baseline.md`](research_full_day_peer3_baseline.md)。  
L2 细则：[`hunter_washout_reclaim_research.md`](hunter_washout_reclaim_research.md)。  
窄专家队列：[`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)。

明细结果：

- L1 验收：`results/watchdog/acceptance/`  
- L1 基线 replay：`results/research_extend_mtm_full_day_peer3_l1_may_jul/`  
- L1 流式对拍：`results/parity_l1_watchdog_*`（见 [`replay_stream_parity.md`](replay_stream_parity.md)）  
- L2：`results/watchdog/hunter_washout_reclaim_v2_opp/`  
- 消融：`results/watchdog/hunter_washout_reclaim_v2/ablation_v2.csv`  
- L2 邻域：`results/watchdog/hunter_wd_neighborhood/` → `PASS_NEIGHBORHOOD`  
- L2 流式对拍：`results/parity_l2_hunter_washout_reclaim_20260501_0717/` → ok  
- **P0 hold-out**：`results/watchdog/holdout_p0/` → Verdict `PASS_P0`（L2 OOS 弱于 L1）  
- **升线总闸**：[`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md) → L2 已进 research_baseline；生产 freeze 另议