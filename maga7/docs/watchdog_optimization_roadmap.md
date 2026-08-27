# 基于 Watchdog 三层栈的完善优化路线

> 配套：[`watchdog_stack_architecture.md`](watchdog_stack_architecture.md)  
> 日期：2026-07-18  
> 原则：**先证伪、再升线；先 L1 后 L2；禁止回拧 L0 全局旋钮。**

## 0. 现状坐标（不要重新发明）

| 层 | 状态 | 该不该继续「拧参数」 |
|----|------|----------------------|
| L0 Baseline | 研究基线已冻 | **否**——特例进 overlay，不进 Rule-A |
| L1 Degrade/Halt | `ACCEPT_RESEARCH`，freeze off | 窄修阈值 / 纸面验证，勿加宽触发 |
| L2 Hunt washout_reclaim v2 | `ACCEPT_RESEARCH`，freeze off | 先 OOS 与稳定性，再谈新 detector |
| 已 REJECT | orb_fractal / early_mf / 宽 washout 门 | **勿复活**作默认 |

当前最大缺口不是「再多一个信号」，而是：**同窗调参未 OOS、Live 未证、Hunt 样本薄**。

---

## 1. 优化优先级（按性价比）

```text
P0  防拟合闸门（OOS / 邻域 / 冻结纪律）     ← PASS_P0（1 月空通过）
P1  L1 升线准备（流式对拍 → 研究开 L1）   ← 流式对拍已过
P2  L2 质量（假 V、占坑残余、退出）
P3  Live 接线（Hunt 注入 + 观测）
P3.5 持仓 Watchdog（QQQ 冲击中途平仓）——见 [`hold_watchdog_research.md`](hold_watchdog_research.md)；默认 off
P3.6 成交价毒性早切 trade_toxic——**已升** research_baseline；见 [`trade_mark_toxic_path_research.md`](trade_mark_toxic_path_research.md)
P4  新专家 / 扩池（MU 等）——仅 P0–P2 过后再开
```

研究升线以 **Offline + 流式对拍** 为准；Paper 是上实盘前工程门，非 P0/P1 必选项。

### 明确不做（除非双窗 + OOS 全过）

- 改 L0 `window_start` 吃早盘  
- 全局拧 `streak` / `sl_mult` / peer 门槛「兼顾 Hunter」  
- 把 Halt 与 Hunt 阈值揉成一条规则  
- 为单日故事加符号黑名单（如 ban AMD）当默认  

---

## 2. P0 — 防拟合闸门（完善的第一刀）

目标：回答「这套 overlay 是不是只在调参窗好看」。

| # | 动作 | 验收 |
|---|------|------|
| P0.1 | **Hold-out 评分**：固定现参，跑 2026-01 与/或 2025H2，**禁止再调** | L1、L2 各自 vs 同期 freeze ≥90%（可略宽于 95%） |
| P0.2 | **邻域扫描**：`wash_drop` ∈ {1.2%, 1.5%, 1.8%} × opp on/off | 最优点邻域不出现断崖（弱窗不跌破 90%） |
| P0.3 | **符号置换**：Mag7 去一只 / 换 GOOGL 权重敏感 | Hunt 增量不依赖单一名字 |
| P0.4 | **费用/滑点敏感**：entry/exit frac ±0.1 | 强窗增量不翻转符号 |

产出建议：`results/watchdog/oos_<tag>/` + 总览表写回本文件「P0 结果」节。  
**P0 不过 → 不谈 L1/L2 签字。**

### P0 结果（2026-07-18，锁参）

工具：`python -m maga7.tools.run_watchdog_holdout_p0`  
明细：`results/watchdog/holdout_p0/`

| window | L0 | L1 vs L0 | L2 vs L0 | 触发 |
|--------|-----|----------|----------|------|
| 2026-01 | +47.4% | **100%**（同 L0） | **100%**（同 L0） | Halt×0 · Hunt×0 |
| 2025 H2 | +27.7% | **191%** | **172%** | Halt×1 · Hunt×10 |

**Verdict: `PASS_P0`**（门槛 ≥90% vs L0）。

解读（重要）：

1. **门过了，但 1 月是空通过**：overlay 完全未触发，不能证明 L1/L2 在近端 hold-out 有增量，只能证明「无伤害」。  
2. **2025H2 才是有效压力测试**：L1 明显抬升（Halt 护基线）；旧 L2 弱于 L1（Hunt 笔均约 −5.4%）。  
3. **升线（2026-07-18）**：按「Feb–Apr vs L0 提升」并入旧 L2（+1255% 退出）；2025H2 / P2.1 打折不挡基线。

---

## 3. P1 — L1 完善（坏日防护先落地）

L1 已是最干净增量（触发极窄、双窗稳）。完善顺序：

| # | 动作 | 说明 |
|---|------|------|
| P1.0 | **流式对拍** L1（Hunter off）offline ↔ stream | **已过**（见下） |
| P1.1 | 研究基线开 **仅 L1**（Hunter off） | **已做**；May–Jul L1 **+875%** vs L0 +810%（~108%）。**勿与 L2 +1255% 混淆**（见 baseline 文档速查表） |
| P1.2 | 观测看板：`watchdog_state` 日频、Halt/Degrade 次数 | 触发率异常升高 = 规则漂移 |
| P1.3 | Halt 假阳性回顾（按季） | 只允许**收紧** breadth/drop，禁止放宽到天天 Halt |
| P1.4 | 实盘前工程门（Paper / Live） | 非研究签字必选项；上实盘前再开 |

### P1.0 流式对拍结果（2026-07-18）

```bash
python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_watchdog_v1.json \
  --scheme single --stock-source stock_1s \
  --start-date 2026-05-01 --end-date 2026-05-15 \
  --tag parity_l1_watchdog_20260501_15
```

| tag | period | n | ok |
|-----|--------|---|----|
| `parity_l1_watchdog_20260501_15` | 05-01..05-15 | 13 | true |
| `parity_l1_watchdog_20260528` | 05-28 | 1 | true |
| `parity_l1_watchdog_20260714_17` | 07-14..07-17 | 4 | true |

明细：`results/parity_l1_watchdog_*`；工程说明见 `replay_stream_parity.md`。  
下一步：观测 L1 触发率（P1.2）+ 假阳性回顾（P1.3）。Hunter 仍 off。

L1 已进 research_baseline；L2 Hunt 亦已进（见 P2 / baseline 文档）。新开仓形态走 P4.1 窄专家注册表，勿再拧 L0。

---

## 4. P2 — L2 质量（在升线前榨干假 V）

v2 已修「占坑挡反向」；残余问题是 **假收回本身仍亏**（如 07-02 Hunt −53%）。

### 验证闸门进度（2026-07-18）

见 [`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md)：

| 闸门 | 状态 |
|------|------|
| 流式对拍 | **PASS**（P2.1：May–Jul 60 笔） |
| 邻域 wd×opp | **PASS_NEIGHBORHOOD**（禁止 wd→1.2%） |
| OOS vs L1 | **观察**（2025H2 不作硬否决） |
| 升 baseline | **YES**（旧 L2 T30+extend；门槛=Feb–Apr vs L0 提升） |

| # | 方向 | 约束 |
|---|------|------|
| P2.1 | 退出：Hunt 专用更短 hold / 更紧 SL / 早盘 MAE | **已落盘** `hold20_noext`；见 [`hunt_exit_ablation_p21.md`](hunt_exit_ablation_p21.md) |
| P2.2 | 确认：`hold_confirm` / `reclaim_buffer` 邻域再验 | **邻域 G3 已做**；确认棒仍属可选加严 |
| P2.3 | 预算：Hunt 日亏损熔断（如 hunt 腿 −30% 当日禁第二腿） | 防单笔打穿 |
| P2.4 | 归因仪表：Hunt 日 `delta_vs_base` 分布 | 持续监控「增量来自少数日」风险 |

新 detector（洗盘结构 2.0、OFI 等）排在 **P2.1–P2.3 无效之后**，且走同一双窗 + P0 流程。

---

## 5. P3 — Live / 工程对齐

| # | 动作 | 2026-07-24 |
|---|------|------------|
| P3.1 | Scanner Hunt 注入 | **已做**（`_schedule_hunts` / `drain_hunts`；见 checklist） |
| P3.2 | OMS：`event_source=hunt` + Hunt `position_frac` | **已补**（POSITION_OPEN 字段；meta.position_frac→`_size`） |
| P3.3 | Dash Watchdog / Hunt 卡 | **已补**（`load_watchdog_hunt`） |
| P3.4 | 日终 session vs offline Hunt | **工具** `run_hunt_session_eod_align`；待 Shadow 实跑 |

无 Shadow 证据前，研究数字仍不能当实盘预期。
---

## 6. P4 — 扩池与新专家（可选）

仅当 P0–P2 过线：

| 方向 | 做法 |
|------|------|
| MU / 高波动卫星池 | **独立 profile**，勿塞进 Mag7 freeze 符号表 |
| 洗盘→reclaim 变体 | 新 `detector=` 名，默认 off，走 scoreboard |
| 与事件日历联动 | Halt/Hunt 在事件日额外收缩，不改 L0 信号 |

### P4.1 窄形态专家注册表（2026-07-24 升级）

见 [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md) · `CONFIG/narrow_experts/catalog_v1.json`。

| 状态 | 含义 |
|------|------|
| 脊骨已挂 | L1 degrade/halt + L2 Hunt washout_reclaim（research_baseline） |
| 研究队列 | `core_dn_sync`（trades PASS / **quote REJECT**）、AM delayed/HF sleeves |
| 纪律 | 默认基线；开仓类专家必须 quote 双窗 PASS 才接线；禁止拧 L0 追近端 |

下一刀：Live 对齐 Hunt（P3）；`core_dn_sync` 已做 quote 覆盖诊断 → **仍 REJECT**（lag 稀疏 + FillSpec sync 杀边；放宽 lag 双窗仍亏）。不是再开 AM 秒级主策略。
---

## 7. 建议执行节奏

### 已完成（2026-07-18）

P0.1–P0.2、P1.0–P1.1、P2 升线闸门、旧 L2 并入 `peer3_v1` — 见 [`l2_hunter_validation_gates.md`](l2_hunter_validation_gates.md)。

### 下一轮（2026-07-19 起，不动 peer3 参数）

详细勾选表：[`l2_next_acceptance_checklist.md`](l2_next_acceptance_checklist.md)

| 周 | 焦点 | 退出标准 |
|----|------|----------|
| W1 | **Round A** P3：Scanner/OMS/Dash Hunt 对齐 | 历史 Hunt 日候选一致；meta 可过滤 |
| W2 | A 周报 + **Round B** P2.3 日熔断消融 | Shadow 不空跑；双窗 ≥95% 现 L2 才议升 |
| W3 | **Round C** P0.3/P0.4 符号+费用敏感 | `PASS_SENSITIVITY` 或降级观察 |

---

## 8. 决策树（每次想「再优化」时用）

```text
问题来自坏日多亏？ ──是──► 只动 L1（收紧 Halt/Degrade）
         │
         否
         ▼
问题来自早盘错过 / Hunt 亏？ ──是──► 只动 L2（退出/确认/预算）
         │
         否
         ▼
问题来自 10:30 后 Rule-A？ ──是──► L0 研究线（另开 profile，双窗验收）
         │
         否
         ▼
样本外变差？ ──是──► 停改参，缩回 L0 或仅 L1
```

---

## 9. 一句话路线

**研究基线已是 L0+L1+L2；下一刀是 Live 对齐 → Hunt 日预算 → 窄专家 quote 闸（CORE DN sync），不拧 L0 信号旋钮。**  
详见 [`l2_next_acceptance_checklist.md`](l2_next_acceptance_checklist.md) · [`narrow_expert_routing_upgrade.md`](narrow_expert_routing_upgrade.md)。
