# qqq_btc 路线图与模型集群设计

> **定位**: 在 [ARCHITECTURE.md](./ARCHITECTURE.md) 原则之上,定义 **QQQ 单路径闭环 → MAG7 模型集群 → 组合资本分配** 的分阶段里程碑。  
> **原则**: 先跑通 QQQ,再横向复制;决策逻辑复用 `common/`,组合层独立新增,不污染单标的 replay 契约。

---

## 1. 目标形态(终局)

```text
                    ┌─────────────────────────────────────┐
                    │  L3 组合层 Portfolio (新增)          │
                    │  初始权重 · 总敞口 · 动态加减 · 组合止损 │
                    └──────────────────┬──────────────────┘
                                       │ 允许交易? 名义上限?
         ┌─────────────┬───────────────┼───────────────┬─────────────┐
         ▼             ▼               ▼               ▼             ▼
    ┌─────────┐  ┌─────────┐    ┌─────────┐    ┌─────────┐  ┌─────────┐
    │ QQQ     │  │ NVDA    │    │ AAPL    │    │ ...     │  │ MAG7×N  │
    │ ckpt_0  │  │ ckpt_1  │    │ ckpt_2  │    │         │  │ ckpt_n  │
    └────┬────┘  └────┬────┘    └────┬────┘    └────┬────┘  └────┬────┘
         │ L2 同一套   │              │              │            │
         │ entry_decision + exit_rails + ReplaySession (每标的独立 config) │
         └─────────────┴──────────────┴──────────────┴────────────┘
                                       │
                    ┌──────────────────▼──────────────────┐
                    │  L1 推理: 每标的一个模型(或共享底座+分头) │
                    └─────────────────────────────────────┘
```

**不在模型里做的事**: 组合权重、标的间竞争、总 delta 上限 —— 全部在 **L3**。

---

## 2. 三层架构(设计契约)

| 层 | 职责 | 模块归属 | 变更频率 |
|----|------|----------|----------|
| **L1 模型** | bar → net_edge / q10 / 多腿头 | `qqq_btc/model/`, 每标的 checkpoint | 训练迭代 |
| **L2 单标的决策** | 阈值、spread、q10、exit rails、fill | `qqq_btc/common/*`, `qqq/{symbol}/config.py` | 标定后较稳定 |
| **L3 组合** | 初始分配、实时加减、组合风控 | **待建** `qqq_btc/portfolio/` | 集群阶段 |

### 2.1 L2 不变量(所有标的共用实现)

以下模块 **禁止** 按标的复制逻辑,只允许 **参数化 config**:

- `common/fill_model.py` — 默认 0.775;MAG7 可 per-symbol 校准后写入各 `config`
- `common/entry_decision.py` — `choose_entry(replay_cfg, ...)`
- `common/exit_rails.py` — `check_exit(rails, ...)`
- `common/replay_session.py` — 状态机;live/replay 同一实现

### 2.2 L3 新增组件(集群阶段)

| 组件 | 职责 |
|------|------|
| `PortfolioConfig` | 标的列表、初始权重、总敞口上限、组合 daily_loss_stop |
| `PortfolioGate` | 分钟级:某标的是否允许新开仓、名义上限倍率 |
| `PortfolioAllocator` | 静态权重 + (可选)edge 强度 / 波动率动态调节 |
| `CorrelationGuard` | QQQ 与 MAG7 同向叠加时的隐性杠杆上限 |

**接入点**: OMS `_process_alpha_frame` **之前** — L3 过滤/缩放后,仍走现有 V0 + entry_bridge + 单标的 state。

### 2.3 运行时拓扑(集群目标)

```text
FCS (多 symbol batch)
  → SE: 多 checkpoint 推理 → ALPHA_FRAME (items[] 多标的)
  → OMS: L3 PortfolioGate → L2 per-symbol decide_entry → IBKR
```

与现网一致:**双引擎三进程**;不恢复 Monolith。SE 可演进为单进程批推理或每标的 async worker,OMS 保持单实例管组合状态。

### 2.4 与 New_Pro 分工

| 组件 | 维护策略 |
|------|----------|
| `New_Pro/baseline_qqq/` | **冻结运行时**: FCS / OMS 宿主 / Dashboard / compat |
| `qqq_btc/` | **主开发**: 标签、训练、replay、L2、L3、live patch |
| 策略 V0 | 仅前置门控 + gate trace;入场/出场语义以 qqq_btc 为准 |

---

## 3. 里程碑总览

```text
Phase 0 ──► Phase 1 ──► Phase 2 ──► Phase 3 ──► Phase 4 ──► Phase 5
 QQQ闭环     QQQ实盘     单名复制     组合v0      模型集群     动态分配
 (G0–G3)     稳定运维    (Pilot)     (L3)        (N ckpt)    (v1)
```

| Phase | 名称 | 核心交付 | 验收门 |
|-------|------|----------|--------|
| **0** | QQQ 单路径闭环 | 标签→训练→G2 replay→live 接线 | **G0–G3** |
| **1** | QQQ 实盘稳定 | shadow 对拍、rails 标定、运维工具 | **G3+** |
| **2** | 单标的复制 Pilot | 1 只 MAG7 (建议 NVDA) 全链路模板 | **G2′** |
| **3** | 组合层 v0 | 固定权重 + 总敞口 + 多仓 OMS | **G4** |
| **4** | 模型集群 | N 模型并行推理 + 统一 ALPHA_FRAME | **G4+** |
| **5** | 动态资本 v1 | edge/波动率调节权重 + 相关性护栏 | **G5** |

---

## 4. Phase 0 — QQQ 单路径闭环 【当前主战场】

> 详细命令见 [EXECUTION_PLAN.md](./EXECUTION_PLAN.md)。本节只定义 **里程碑与通过标准**。

### M0.1 训练数据 (G0)

| 项 | 内容 |
|----|------|
| **目标** | 带 fill 价双腿标签的 LMDB,strict sanity 通过 |
| **交付** | `label_pipeline` 报告、`build_lmdb`、按日 val 切分 |
| **通过** | net_std>0; time/trend 特征齐全; 无静默标签 fallback |
| **不做** | MAG7 数据、组合标签 |

### M0.2 模型训练 (G1)

| 项 | 内容 |
|----|------|
| **目标** | v2 双流 + q10 + (可选) call/put 头 |
| **交付** | `checkpoints_qqq_net_edge_v2/best.pth` |
| **通过** | val IC>0; q10 覆盖率≈10%; 分位数单调 |
| **不做** | 多标的联合训练 |

### M0.3 Strict replay (G2) ★ 模型验收唯一标准

| 项 | 内容 |
|----|------|
| **目标** | 0.775 fill + entry_delay + exit_rails 下 PnL 可解释 |
| **交付** | `run_replay` / `run_event_replay` 报告; `calibrate_rails` 初值 |
| **通过** | fill 口径 PnL>0; 去 best-2-day 仍为正; L1/L2 差异可解释 |
| **不做** | mid 口径回测; 截面排序 |

### M0.4 Live 接线 (G3 工具就绪 → 跑数)

| 项 | 内容 |
|----|------|
| **目标** | 双引擎 + entry/exit bridge + fill 0.775 + tick disaster_only |
| **交付** | `run_live_*_qqq.py`; `minimal_stack.env`; New_Pro 分层 + compat |
| **已完成** | oms_integration patch; strategy_entry/exit_bridge; gate_trace_stats |
| **待跑数** | REALTIME_DRY ≥2 周 shadow CSV |

### M0.5 G3 通过(Phase 0 出口)

| 指标 | 阈值 |
|------|------|
| feature parity (FCS enrich) | pass_rate > 0.95 |
| fill spread_frac median | 0.75–0.80 (目标 0.775) |
| exit reason 分布 vs L1 replay | JS 散度 ≤ 0.35 |
| 无 CRITICAL 手工干预项 | FCS/SE slow_feature 一致; checkpoint fail-fast |

**Phase 0 出口定义**: G3 指标连续 5 个交易日达标,且 QQQ 单仓 replay→live 无未解释偏差。

---

## 5. Phase 1 — QQQ 实盘稳定

### M1.1 Rails 与阈值标定

- `calibrate_rails.py` 用真实 infer parquet 重标 `EXIT_RAILS` / 分时段 `entry_threshold_schedule`
- 文档化标定前后 replay PnL 变化

### M1.2 Fill 闭环

- 从 `fill_audit.csv` 统计实际 spread_frac 分布
- 若 median 偏离 0.775 >0.03,更新 `qqq/config.py` FILL_MODEL 并 **重跑标签+G2**

### M1.3 运维可观测

- Dashboard 系统全景 Tab + gate_trace_stats 日报告
- 告警: 连续 N  bar 无 ALPHA_FRAME / fill 中位数漂移 / disaster_stop 触发

### M1.4 小资金或 dry-run 连续运行

- 2 周无 state 漂移、无 ghost position、无 exit 口径回归

**Phase 1 出口**: 运维 runbook 齐全; 团队可仅通过 qqq_btc 文档重启全栈。

---

## 6. Phase 2 — 单标的复制 Pilot (1× MAG7)

> **目的**: 验证「模板可复制」,而非一次上 7 只。建议首只: **NVDA**(流动性好、option 深)。

### M2.1 数据与锚点

| 交付 | 说明 |
|------|------|
| `mag7/nvda/anchor.py` 或 generalize `qqq/anchor.py` | 0DTE / weekly 策略按流动性定 |
| NVDA label_pipeline + LMDB | 独立 val 切分 |
| per-symbol `slow_feature_*.json` | 可共享 v2 结构,换 symbol_map |

### M2.2 训练与 G2′

- 独立 checkpoint `checkpoints_nvda_net_edge_v1/best.pth`
- **G2′**: 与 QQQ 相同 strict 标准,**独立** PnL 阈值(不混池)

### M2.3 Live 影子(单标的追加)

- SE: 第二路 infer 或 batch 内第二 symbol (仍 **MAX_POSITIONS 全局=1** 做 shadow)
- parity_audit 按 symbol 分文件

**Phase 2 出口**: NVDA 单独 G2′ 通过; QQQ G3 不退化; 复制 checklist 文档化(≤3 天可复制下一只)。

**明确不做**: 组合权重、同时持有 QQQ+NVDA。

---

## 7. Phase 3 — 组合层 v0 (L3)

### M3.1 设计冻结

| 交付 | 内容 |
|------|------|
| `portfolio/config.py` | `PortfolioConfig`: symbols, weights, max_gross_exposure |
| `portfolio/gate.py` | `allow_entry(symbol, session_bar, portfolio_state) -> bool` |
| `portfolio/state.py` | 组合 PnL、各标的已用名义、当日 trade count |

### M3.2 OMS 集成

- `_process_alpha_frame` 前调用 `PortfolioGate`
- `MAX_POSITIONS` > 1,每标的独立 `SymbolState`,组合级 `daily_loss_stop`
- 初始权重:**静态** (例: QQQ 50% + NVDA 50% 名义预算)

### M3.3 Replay 组合回测

- 新工具 `run_portfolio_replay.py`: 多 parquet 按时间对齐,组合 MTM
- **G4**: 组合 replay 与「单标的 replay 加权求和」在固定权重下 PnL 一致(无双重计数 bug)

**Phase 3 出口**: G4 通过; dry-run 可同时持 2 标的; 组合 daily_loss_stop 可触发停开新仓。

**明确不做**: 动态调权、相关性模型。

---

## 8. Phase 4 — 模型集群 (N ckpt)

### M4.1 推理层

| 方案 | 适用 |
|------|------|
| A. SE 单进程 batch 多 symbol | symbol 数 ≤8,GPU 够 |
| B. SE 子进程 per symbol | 隔离 crash; ops 复杂 |

交付: 统一 `ALPHA_FRAME.items[]`,每 item 带 `symbol`, `net_edge`, `net_edge_q10`, `checkpoint_id`。

### M4.2 配置注册表

```text
qqq_btc/registry/symbols.yaml
  QQQ:  { config: qqq/config.py,  checkpoint: .../qqq_best.pth,  weight: 0.35 }
  NVDA: { config: mag7/nvda/config.py, checkpoint: .../nvda_best.pth, weight: 0.15 }
  ...
```

### M4.3 MAG7 批量复制

- 按 Phase 2 checklist 并行数据/训练
- 每只 **独立 G2′** 通过才接入 registry

**Phase 4 出口**: registry 内 ≥3 标的同时 shadow; ALPHA_FRAME 多 item 稳定; OMS 多 state 无串仓。

---

## 9. Phase 5 — 动态资本分配 v1

### M5.1 规则型调权(先不做 ML)

- edge 强度映射名义倍率: `size_mult = clamp(edge / threshold, 0.5, 1.5)`
- 连亏降权 / 盈利恢复: 与单标的 `loss_streak` 类似,**组合级**再套一层

### M5.2 CorrelationGuard

- 同向 CALL 持仓时: `sum(abs(delta_notional)) < cap`
- QQQ 与 MAG7 高相关日,限制同时加仓

### M5.3 组合 replay 验证 (G5)

- 动态调权策略在 strict replay 上优于静态权重(或波动率 adjusted 后 Sharpe 更高)
- 最大回撤不超过静态方案 × 1.2

**Phase 5 出口**: G5 通过; live 小资金组合运行 4 周。

---

## 10. 验收门汇总

| 门 | Phase | 含义 | 关键指标 |
|----|-------|------|----------|
| **G0** | 0 | 标签/LMDB | net_std, sanity_check |
| **G1** | 0 | 模型 | val IC, q10 |
| **G2** | 0 | QQQ strict replay | fill PnL, 去 best-2-day |
| **G3** | 0→1 | Live parity | feature/fill/exit 分布 |
| **G2′** | 2 | 单 MAG7 replay | 同 G2,per symbol |
| **G4** | 3 | 组合 replay | 加权一致、无串仓 |
| **G5** | 5 | 动态组合 | vs 静态权重 |

---

## 11. 风险登记(跨 Phase)

| 风险 | 缓解 |
|------|------|
| QQQ 与 MAG7 同向隐性加杠杆 | Phase 3 起 CorrelationGuard; 组合 gross cap |
| 每标的 fill 差异 | per-symbol FILL_MODEL; 独立 fill_audit 分 symbol |
| 多模型 SE 延迟 | batch 推理; 非关键 symbol 降频 |
| legacy V0 与 replay 分叉 | 强制 QQQ_BTC_LIVE + entry/exit bridge; 禁改 StrategyCore 语义 |
| 过早上集群 | **硬顺序**: G3 → G2′ → G4,不可跳过 |

---

## 12. 当前状态快照 (2026-07)

| 项 | 状态 |
|----|------|
| Phase 0 M0.4 代码接线 | ✅ entry/exit bridge, OMS patch, New_Pro 分层 |
| Phase 0 M0.1–0.3 | 🔧 依赖训练数据与 G2 跑数 |
| Phase 0 M0.5 G3 | ⏳ 需 2 周 shadow |
| Phase 1–5 | 📋 本文档定义,未启动 |

**下一步(建议)**:

1. 完成 QQQ **G2** strict replay 报告并存档  
2. 启动 **G3** REALTIME_DRY shadow  
3. G3 通过后启动 Phase 2 NVDA Pilot 数据清单  

---

## 13. 相关文档

- [ARCHITECTURE.md](./ARCHITECTURE.md) — 原则、双引擎拓扑、fill 单一真相源  
- [ARCHITECTURE.md §2.6](./ARCHITECTURE.md#26-端到端时序bar-close--fill) — bar close → fill 端到端时序  
- [PARITY_CHECKLIST.md](./PARITY_CHECKLIST.md) — G3 shadow 逐日/逐周核对表  
- [EXECUTION_PLAN.md](./EXECUTION_PLAN.md) — Phase 0 命令与文件索引  
- [New_Pro/baseline_qqq/docs/LAYOUT.md](../New_Pro/baseline_qqq/docs/LAYOUT.md) — 运行时目录分层  
