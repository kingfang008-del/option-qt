# 状态与因果完整性：技术升级方案

> 日期：2026-07-15  
> 状态：**方案定稿（本轮不改代码）**  
> 前置审查：架构风险 Canvas、四域只读审查（数据/执行/模型治理/运维）  
> 相关文档：  
> - `qqq_btc/docs/replay_version_lineage_and_result_reconciliation.md`  
> - `qqq_btc/docs/strategy_profile_replay_stream_parity.md`  
> - `qqq_btc/docs/honest_3gate_live_parity_handoff.md`  
> - `qqq_btc/docs/v4_july_w1_stream_parity_final_aligned_20260714.md`

---

## 1. 一句话结论

输入层（标的 / DTE / 点差 / 趋势）已经挡住大部分客观不利因素后，下一阶段的主战场不是继续加策略规则，而是把系统升级到：

```text
正确决策 → 恰好执行一次 → 崩溃后可恢复到同一事实 → 版本与特征契约可证明
```

本方案把七类已知缺陷收敛为 **五条架构主线** 与 **四阶段落地**。  
本轮只定技术方案与验收标准，**不实施代码改动**。

---

## 2. 问题收敛：七项缺陷 → 五条主线

| 用户指出的缺陷 | 收敛主线 | 严重度 |
|---|---|---|
| OMS 重启丢失真实持仓 | **A. 单一状态权威** | P0 |
| Redis 丢消息 + 重复下单 | **B. 端到端幂等与可靠投递** | P0 |
| 紧急退出首次失败后锁死 | **B + A（退出 intent 状态机）** | P0 |
| 日损熔断 / 腿锁未完整持久化 | **C. 可恢复风控状态机** | P0 |
| IV / 暖机 / 时间标签未对齐 | **D. 唯一特征与时间契约** | P0 |
| 配置 SHA / 回放状态 / 模型晋升未真正冻结 | **E. 不可变发布与晋升门禁** | P0 |
| 生产代码树 / 部署守护 / 告警分裂 | **E + 运维底座** | P0/P1 |

说明：A/B/C 解决“会不会错单”；D/E 解决“就算不错单，结果是否可审计、可复现、可晋升”。

---

## 3. 目标架构（升级后应长什么样）

### 3.1 事实源分层

```text
Broker（IBKR）     = 持仓 / 订单 / 成交的最终物理事实
PG Transaction Log = 系统内部权威：intent → submit → fill → position snapshot → risk state
Redis Stream       = 传输层（非权威）；只存消息与投影，不存“真相”
Memory             = 热缓存；重启后必须可从 PG + Broker 重建
Shadow CSV/Audit   = 审计副本；不能替代 broker/PG 对账
```

硬规则：

1. **禁止**把 Redis 或进程内存当成持仓权威。  
2. **禁止**在处理失败后无条件 `XACK` 且无 DLQ。  
3. **禁止**只持久化 `day_pnl` 而不持久化 `day_halted` / 冷却 / 腿锁 / 退出 pending。  
4. **禁止**用“profile 文件名 / 部分 SHA”冒充完整策略冻结。

### 3.2 决策与执行边界（保持现有正确方向）

```text
AI 模型     → 只产出 CALL/PUT edge（排序/评分）
确定性规则 → 时段、仓位、退出、冷却、熔断、regime
执行层     → 把“已批准 intent”可靠落地，不重新发明策略语义
```

本方案**不改变**“AI 不做仓位/退出主控”的边界；只保证该边界下的因果链完整。

### 3.3 发布单元（Release ID）

每次可运行产物必须绑定：

```text
release_id = hash(
  git_commit + dirty_diff_hash,
  strategy_profile_json,
  base_ReplayConfig_asdict,      # 含 config.py 继承字段
  rule_profiles.json,
  resolved_env_effective,        # resolve_replay_cfg() 最终值
  checkpoint_sha256,
  feature_schema_hash,
  frozen_norm_sha256,
  infer_or_feature_root_hash
)
```

同 `release_id` ⇒ 同行为；不同 `release_id` ⇒ 禁止横向比较 KPI。

---

## 4. 五条主线的技术方案

### 主线 A — 单一状态权威（持仓 / 订单）

#### 现状问题

- OMS `disable_db_save=True`，运行时不写 PG 持仓快照。  
- 重启依赖内存重建 + 延迟 reconciler，易幽灵空仓 / 重复开仓。  
- Ghost exit 可在本地清零而 broker 仍有仓。

#### 目标设计

1. **PositionAuthority**
   - 运行中：OMS 每次 OPEN/PARTIAL/CLOSE 成交确认后，原子写入 PG `symbol_state` + `position_events`。  
   - 启动时：`Broker.reconcile → PG.snapshot → Memory`；三者冲突时以 Broker 为准，并写 `divergence_event`。  
2. **OrderAuthority**
   - 所有 intent 进入 `oms_intent` 表（含 `idempotency_key`、`intent_type`、`reason`、`status`）。  
   - 订单状态带 **交易日边界**：隔夜非终态单在日切时进入 `STALE_CANCELLED` 或强制 reconcile，禁止静默复活。  
3. **禁止本地“假装平仓”**
   - Ghost / accounting 路径不得在未确认 broker 无仓时把本地 `position` 强制清零；只能进入 `RECONCILE_REQUIRED` 并阻断新开仓。

#### 验收标准

| ID | 验收 |
|---|---|
| A1 | 持仓中 `kill -9` OMS → 重启后 30s 内本地仓位与 broker 一致 |
| A2 | 重启后不得出现第二次 BUY 同一逻辑仓位（除非 broker 确认为空） |
| A3 | 人为制造 `open_fill_confirmed=False` + broker 有仓 → 系统进入 reconcile 阻断，而不是本地清零后继续开仓 |
| A4 | 隔夜残留 pending order 不会在次日自动变成可执行 intent |

#### 不做什么

- 不引入分布式多 OMS 写；继续单写者锁。  
- 不把 Redis `oms:live_positions` 升级为权威。

---

### 主线 B — 端到端幂等与可靠投递

#### 现状问题

- 异常处理后仍 `XACK` → 丢信号。  
- 重启消费 PEL，但进程内 dedupe 清空 → 重复下单。  
- `qqq_btc_tick_exit_pending` 置位后失败无回滚，叠加 Ghost B 60s 可永久锁死秒级退出。

#### 目标设计

1. **消息生命周期**
   ```text
   RECEIVED → VALIDATED → INTENT_PERSISTED → SUBMITTED → FILLED/FAILED → ACK
   ```
   - 仅在 `INTENT_PERSISTED` 或明确进入 DLQ 后才允许 ACK。  
   - 处理异常：NACK / 重试计数 / DLQ；禁止“为防死转而无条件 ACK”。  
2. **持久化幂等键**
   - `idempotency_key = hash(symbol, trading_day, side, reason_family, frame_id或entry_bar)`  
   - 存 PG；跨重启有效，替代纯内存 4096 环缓冲作为唯一防重手段。  
3. **退出 Intent 状态机**
   - `tick_exit_pending` / `risk_exit_pending` 升级为：
     ```text
     EXIT_REQUESTED → EXIT_SUBMITTED → EXIT_FILLED
                  ↘ EXIT_REJECTED / EXIT_BLOCKED → 可重试（带 backoff）
     ```
   - Ghost B / stale guard 拦截时必须写 `EXIT_BLOCKED` 并安排重试，不得永久 silence。  
   - `DISASTER` / `TICK_FAST_HARD` 进入 urgent 白名单，可绕过 60s Ghost B（或在 60s 内改走强制 flatten 路径）。

#### 验收标准

| ID | 验收 |
|---|---|
| B1 | 注入处理异常：消息进入 DLQ 或可观测重试，不静默消失 |
| B2 | 未 ACK 崩溃重启：同一 BUY 只产生一笔有效 intent（幂等命中） |
| B3 | 开仓 30s 内 disaster 首次被拦截：后续仍能在 ≤N 秒内再次触发退出（N 可配置，默认 5） |
| B4 | 审计能看到完整 intent 状态转移，不只看到“尝试过一次” |

#### 不做什么

- 不改成 exactly-once 分布式事务中间件；先用 **at-least-once + 持久幂等**。  
- 不在本阶段重写整个 Redis 拓扑。

---

### 主线 C — 可恢复风控状态机

#### 现状问题

- pickle 恢复 `day_pnl` / `trades_today`，但不恢复 `day_halted`、连亏冷却、腿锁、tick 冷却、structure veto。  
- 日内重启可绕过日损熔断。  
- 部分平仓每次 close 递增 `trades_today`，可能提前耗尽日交易次数。

#### 目标设计

1. **RiskState schema（版本化）**
   ```text
   trading_day, day_pnl, day_halted,
   trades_today, trade_groups_today,
   loss_streak, streak_cooldown_until_ts,
   leg_lock_until_ts, tick_stop_cooldown_until_ts,
   tick_stopped_legs, loss_legs_today,
   put_structure_veto_until, vixy_open_shock_regime_active,
   entry_quantile_buffer_meta, schema_version
   ```
2. **持久化介质**
   - 短中期：PG JSONB（推荐）或原子写 JSON + fsync；淘汰无锁 pickle 作为唯一方案。  
   - 每次 `record_trade_close` / 熔断跳变 / 日切 → 同步落盘。  
   - 启动时：若 `day_pnl <= daily_loss_stop` 则 **重算并强制** `day_halted=True`（即使旧快照缺字段）。  
3. **交易计数语义**
   - `trades_today` 改为逻辑交易组（entry→final flat），部分平仓不重复计数。  
4. **与 OMS cooldown 单一化**
   - 明确 governor 为 qqq_btc 策略风控权威；legacy OMS circuit breaker 只作兜底或逐步对齐 reason token。

#### 验收标准

| ID | 验收 |
|---|---|
| C1 | 触发日损熔断 → kill 重启 → `blocked_for_entry` 仍为 true |
| C2 | 腿锁 / tick 冷却 / structure veto 重启后保持 |
| C3 | 一笔逻辑交易两次 partial exit → `trade_groups_today` 只 +1 |
| C4 | RiskState schema 升版时旧文件可迁移或 fail-closed，不静默丢字段 |

#### 不做什么

- 不在本阶段引入跨机器共享风控集群。  
- 不把研究用临时 env 覆盖写进默认 RiskState。

---

### 主线 D — 唯一特征与时间契约

#### 现状问题

- Gate1 期权 IV 族仍大面积 offline/live 不一致。  
- carryover 390 / 29 / 0 分裂；Gate2 跳过 trend 列掩盖问题。  
- FCS start-label vs 离线 end-label 双轨；新工具易忘 +60s。  
- frozen norm vs rolling `stats_update_interval` 默认不一致。  
- New_Pro / production 双份 FCS 源码。

#### 目标设计

1. **FeatureContract v1（单一文档 + 机器可读 JSON）**
   - 时间标签：统一 **end-label** 为交易/审计语义；FCS 内部 start-label 必须经唯一 adapter 转换。  
   - carryover：生产固定一种（建议：deep warmup + 全 RTH lookback，与训练一致），禁止脚本各写各的。  
   - IV：固定 BSM 输入（`T_label=end`、`IV_PRICE_MODE=close`、单腿有效平均规则）与动量/散度窗口定义。  
   - norm：生产只允许 frozen_norm；rolling 仅诊断；`stats_update_interval` 若启用必须 =1。  
2. **单一 FCS 运行时源**
   - 明确 `New_Pro/.../feature_compute_service_v8.py` 为生产；`production/` 副本只读或删除路径入口。  
3. **Gate1/2 进入发布门禁**
   - 正式 run 禁止 `SKIP_GATES=1` / `FORCE_GATE3=1`。  
   - Gate2 不得永久 SKIP 关键 trend 列；若暂时 skip，必须在 manifest 记为 `KNOWN_GAP` 并阻断生产晋升。

#### 验收标准

| ID | 验收 |
|---|---|
| D1 | July W1 Gate1：IV 关键列 pass_rate 达约定阈值（先定目标，例如 ≥0.90 corr 且 fail_rate≤10%） |
| D2 | 开盘 120 分钟 `trend_fit_*` / `adx_*` 在 deep warmup 下与离线 max abs err 低于阈值 |
| D3 | 任意新 consumer 未走 end-label adapter → 测试失败 |
| D4 | 进程启动日志打印 FeatureContract 版本与 FCS import 路径；路径不符则拒绝启动 |

#### 不做什么

- 不在本方案内重训模型来“迁就”错误 IV。  
- 不把 greek-parity / feature5m 金标重新引入生产诚实路径。

---

### 主线 E — 不可变发布、晋升与运维底座

#### 现状问题

- profile SHA 不绑定 `config.py` 基座；env 覆盖不入账。  
- 离线 VX 分段 replay 在段边界重置跨日状态。  
- `weekly_finetune` 晋升链路与 honest/profile/parity 脱钩。  
- New_Pro vs production 测试树分裂；nohup 无守护；canary 未接 deploy；密钥硬编码。

#### 目标设计

1. **真正冻结**
   - `strategy_profile_sha256 = hash(profile_json + base_ReplayConfig + rule_profiles)`。  
   - 每次 run 写 `deploy_manifest.json` / `run_manifest.json`：含 `release_id`、effective env、ckpt hash、git dirty。  
   - dirty tree 默认拒绝正式 KPI / 拒绝 `LIVE_TRADE=1`（可用显式 override，但记入审计）。  
2. **连续状态离线引擎**
   - 新增 `continuous_state` replay 模式：全月单 `ReplaySession`，regime 切换只换门控，不重建 session。  
   - 分段 stitch 降级为研究工具，不得作为生产可比基线。  
3. **模型晋升门禁**
   ```text
   train → honest OOS（profile + LIVE_REPLAY）
        → offline_live_aligned
        → Gate1–3 stream（parity profile）
        → paper/shadow 成交对账
        → promote
   ```
   - `weekly_finetune` 不得再用默认 `REPLAY` + 非 honest 特征根直接晋升。  
4. **运维底座（最小可用）**
   - 单一 runtime import 路径进入 CI。  
   - systemd / supervisor：`Restart=on-failure`；进程退出告警。  
   - deploy 强制 canary PASS；最小告警：进程存活、Redis lag、最后成交/信号时间。  
   - 密钥迁出源码；轮换已泄露 key。

#### 验收标准

| ID | 验收 |
|---|---|
| E1 | 改 `config.py` 一处门控但不改 profile 文件 → SHA 变化或 run 被拒 |
| E2 | continuous_state 与 stitch 在含 OPEN↔TREND 月的逐日交易差异有报告 |
| E3 | weekly_finetune gate 失败若未跑 honest+profile → 不能 promote |
| E4 | `LIVE_TRADE=1` 且 canary 缺失/过期 → deploy 退出码非 0 |
| E5 | CI 跑的是实盘实际 import 的 OMS/FCS 路径，而不是旁路 `production/` 副本 |

---

## 5. 四阶段落地（仍属方案，不含本轮实现）

### 阶段 0 — 故障证明（0–3 天，只验证不优化策略）

目标：用实验把 P0 组合故障变成可复现票据。

| 实验 | 目的 |
|---|---|
| 重启三角 | 持仓中杀 OMS，核对 broker/PG/Redis/内存/audit |
| 消息故障 | 异常 ACK vs PEL 重放 |
| 退出锁死 | 30s 内 disaster + Ghost B |
| 风控恢复 | 日损熔断后重启 |

产出：每项实验的期望行为 vs 实际行为记录；作为阶段 1 的验收基线。  
**冻结**：暂停按单日亏损改策略规则；暂停自动模型晋升。

### 阶段 1 — 不错单（1–2 周）

优先实现主线 **A + B + C**：

1. 恢复/规范化 PG 持久化与启动 reconcile。  
2. 持久幂等键 + 失败不 ACK/进 DLQ。  
3. 退出 intent 可重试；urgent 退出绕过错误保护。  
4. RiskState 完整持久化与重算。

门禁：阶段 0 四项实验全部由失败变为通过，才允许扩大 paper 交易。

### 阶段 2 — 可证明（2–4 周）

主线 **D + E（发布冻结）**：

1. FeatureContract + Gate1 IV 闭环。  
2. release_id / 真 profile SHA。  
3. continuous_state 离线引擎。  
4. weekly_finetune 接入 honest+profile 门禁。

门禁：同 `release_id` 的 offline vs stream Gate1–3 PASS；禁止 SKIP_GATES 出正式结论。

### 阶段 3 — 可运维（并行，2–4 周）

主线 **E（运维底座）**：

1. 合并/冻结单一 runtime。  
2. systemd + 最小告警 + canary 强制。  
3. 密钥治理与备份演练。  
4. paper → shadow → limited-live 分级晋升 runbook。

门禁：无外部告警不得 `LIVE_TRADE=1`；无 deploy_manifest 不得宣称生产版本。

---

## 6. 优先级排序（实施时严格按此序）

```text
1. A1/A2 持仓权威与重启恢复
2. B1/B2 消息 ACK/幂等
3. B3    紧急退出可重试
4. C1/C2 风控完整持久化
5. D1/D2 FeatureContract + IV/暖机
6. E1/E3 release_id + 晋升门禁
7. E4/E5 部署守护与 CI 路径统一
```

理由：先保证“崩溃不会错单”，再保证“数字可复现”，最后保证“人可以睡着”。

---

## 7. 与 AI 模型验证的关系（边界）

本方案**不证明**模型能稳定盈利；它只保证：

- 模型 edge 进入系统后的因果链完整；  
- 规则与模型版本可冻结；  
- 后续才能公平做「仅规则 vs 模型+规则」消融与 shadow。

模型侧独立验证（walk-forward、edge 校准、收益集中度）应在阶段 1 完成后再启动，避免在错单系统上优化 alpha。

---

## 8. 明确非目标（本轮方案刻意不做）

1. 继续为 Jul13 等单日加新策略门控。  
2. 重写全部 OMS / 换交易中间件。  
3. 多区域高可用、多 OMS 集群。  
4. 用 LLM 介入实时下单。  
5. 把 mock fill 的 shadow PnL 直接当实盘收益承诺。

---

## 9. 交付物清单（进入实现阶段时应产出）

| 交付物 | 说明 |
|---|---|
| 本文件 | 技术方案权威（本文） |
| `FeatureContract` JSON | 时间/IV/norm/carryover 机器可读契约 |
| `RiskState` schema | 版本化风控快照 |
| `oms_intent` / `position_events` 设计 | PG 表或等价事件日志 |
| `release_id` 生成器 | profile + base config + env + ckpt 绑定 |
| 故障实验脚本 | 阶段 0 四项可重复运行 |
| 部署 runbook | canary、systemd、告警、回滚 |

---

## 10. 决策待确认（实现前需你拍板）

1. **持仓权威**：Broker 优先 + PG 事件日志（推荐）还是 PG 优先 + Broker 校验？  
2. **RiskState 介质**：PG JSONB 还是本地原子 JSON？  
3. **紧急退出**：DISASTER 是否允许开仓后立即绕过 Ghost B 60s？  
4. **生产 FCS 唯一源**：确认锁定 `New_Pro` 并废弃 `production/baseline` 入口？  
5. **正式 KPI**：是否立即禁止 dirty tree / `SKIP_GATES` 出正式结论？

---

## 11. 最终表述

```text
升级主题：状态与因果完整性
升级目标：从“能回放盈利”升级到“可证明地执行”
实施原则：先不错单，再可复现，后可运维；本轮只定方案，不改代码
```

确认第 10 节五项决策后，再进入阶段 0 故障实验与阶段 1 实现排期。
