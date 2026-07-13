# QQQ 短 DTE 升级路线架构

> 日期：2026-07-13  
> 状态：路线图（指导后续实现，非已上线规格）  
> 相关备忘：[`preprocess/download/LOCK_MAP_OLD_V2_MEMO.md`](../preprocess/download/LOCK_MAP_OLD_V2_MEMO.md)

本文固化 2026-07 一轮锁约 / 0DTE 重训 / 1DTE ladder 讨论的结论，作为后续升级的**分层架构**与**实施边界**。

---

## 1. 一句话结论

| 议题 | 结论 |
|------|------|
| **真 0DTE** | 先搁置。开盘锁约 + 30min 持有 + 1DTE 式护栏不可用；θ / gamma 难控。 |
| **主路径** | **trading 1DTE**（与 V4 / `standard_old_v2` 母体一致）。 |
| **升级核心** | **特征日锁 4 约（连续推理）** + **开盘 ±2% ladder 下载** + **盘中 value_score 选腿（放大赚 / 减少亏）**。 |
| **不要做的事** | 盘中更换特征用的 4 约；用全天 δ 前视去「对齐」实盘；期望 0DTE 平移 V4 六月高收益。 |

---

## 2. 背景与实证摘要

### 2.1 锁约血缘（为何 IC 会腰斩）

- V4 高 IC 母体 `standard_old_v2`：**≈99.7% trading 1DTE**，不是日历 0DTE。
- 旧锁约：全天原始 `|δ|`（有前视）→ 离线可学、IC 高。
- 现网开盘窗锁约：实盘合法，但行权价中位常差 ~$2 → 结构特征 corr ~0.35–0.50 → Val IC 约腰斩。
- 细节见 `LOCK_MAP_OLD_V2_MEMO.md`。

### 2.2 真 0DTE 重训（为何先放着）

隔离管线：`dte0_options_old_lock_databento` → `builds/v4_true_0dte_old_lock` → V4 同款训练。

| 指标 | 共用 1DTE 规则 | 真 0DTE 专用规则 (`config_true_0dte`) |
|------|----------------|--------------------------------------|
| Val acct@25% | −78% | +131%（笔数少） |
| Test acct@25% | +45% | +42% |
| Test MDD | −56% | −12% |
| 2026-06 日账 | — | 约 **+1.3%**（几乎不可用） |

结构原因：

1. 开盘锁死 OCC，与 0DTE 快速漂移的 ATM 不匹配。  
2. 标签路径 ≤−28% 占比 ~30%（1DTE ~8%）→ 硬止损打穿或放宽后噪声更大。  
3. 下半日 θ 加速贬值；提前收工仍难做出稳定六月收益。  
4. IC 可以尚可（~0.22），但 **PnL 可控性远差于 1DTE**（V4 六月 replay 曾到 ~98% 量级）。

产物（归档，非生产）：

- 配置：`qqq_btc/qqq/config_true_0dte.py`（`PROFILE=true_0dte`）  
- 1DTE 族配置：`qqq_btc/qqq/config.py`（`PROFILE=1dte_family`）  
- 评测：`eval_test_set.py --strategy-config qqq_btc.qqq.config_true_0dte`

### 2.3 全链分钟数据烟雾验证（0DTE 动态定位）

数据：`/home/kingfang007/data/new_option_data_s3/QQQ`（全日全约分钟 bar）。  
样例日 `2026-06-09`：开盘 ±2% ladder + 波动加速选约 vs 开盘 ATM。

- 动态定位 15min 均收益略正；开盘 ATM 在大趋势日大量失效 / 无报价。  
- 说明「池子要大」有道理，但 **0DTE 仍难控**；同思路应优先落在 **1DTE**。

---

## 3. 目标架构：三层分离

表面矛盾（「特征要连续 4 约」vs「盘中动态选约」）在分层后消失：

```text
开盘 09:30–09:40（无全日前视）
 ├─ Primary 4-bucket（日锁）──► 全天连续特征 ──► 模型推理（做不做 / 哪一边）
 └─ Ladder ±2%（日锁下载池）──► 仅下单时 value_score 选腿（买哪一张）
                                      │
信号触发 ─────────────────────────────┴──► 选中腿盘口成交
```

| 层 | 合约集合 | 连续？ | 盘中动态？ | 职责 |
|----|----------|--------|------------|------|
| **L0 下载** | 开盘 spot±2%（或 N 档）1DTE ladder | 有行情即可 | 否（开盘定池） | 与离线 `1dte_api_ladder` 同口径覆盖 |
| **L1 特征 / 推理** | Primary 4-bucket ⊂ ladder | **必须** | **否** | V4 结构特征 + net_edge |
| **L2 执行** | 同侧 ladder 子集 | 单笔路径 | **是** | value_score：放大赚、减少亏 |

### 3.1 语义（给产品 / 交易）

- **模型**：不变的是信号（方向与是否开仓）。  
- **选腿**：同样信号下，少买点差差 / 权利金过贵的腿（减亏），偶尔选到弹性更好的腿（放大赚）。  
- **不是**另一套 alpha；选腿应仍靠近 primary 同侧，避免信号与标的脱节。

### 3.2 硬边界（禁止）

1. 盘中更换 **特征用** 的 4 约（会打断 rolling / 与训练日锁假设冲突）。  
2. 用全天 δ / 收盘信息决定「开盘锁定约」（离线前视，实盘不可用）。  
3. 实盘到期用 strict 日历 0DTE 去配 V4 1DTE 权（家族错配）。  
4. 把 0DTE 专用护栏与 1DTE `config.py` 再合并回一套。

---

## 4. 与现有产物的对齐

| 产物 | 角色 |
|------|------|
| `~/train_data/locked_targets_map_from_standard_old_v2.parquet` | 离线复现 V4 IC 的权威 4-bucket map |
| `preprocess/download/step1_build_target_map_old.py` | 旧锁约（仅研究复现） |
| `preprocess/download/step1_build_target_map.py` + `anchor_qqq_1dte_4bucket.json` | 开盘窗 / trading 1DTE 4-bucket（生产对齐重训） |
| `preprocess/download/build_qqq_1dte_api_ladder_map.py` | **开盘 ladder map**（已有） |
| `~/train_data/locked_targets_map_1dte_api_ladder.parquet` | 示例：每日 8 约 PUT K00–03 + CALL K00–03 |
| 5DTE `dynamic_ladder` + `value_score` | 标签/执行动态选腿参考实现 |
| `qqq_btc/tools/score_option_contract_value.py` | value_score 工具 |

推荐关系：

```text
Primary 4 = 从 ladder 派生的「开盘 ATM/OTM 目标 δ」四约
         或与开盘 4-bucket step1 结果取交，保证 ∈ 下载池

Ladder     = 1dte_api_ladder（可扩展训练全区间、±2% 或固定档数）
```

不必再发明「多套互相独立的完整 4 组合」语义；ladder 已覆盖附近 ATM/OTM，下载与选腿更简单。

---

## 5. 分阶段实施路线

### Phase 0 — 冻结认知（已完成讨论）

- [x] 确认 old_v2 ≈ trading 1DTE  
- [x] 真 0DTE 重训 + 专用规则 → 可用性不足，归档  
- [x] 明确 L1 日锁特征 vs L2 动态选腿  
- [ ] 本文档入库并在 memo / ARCHITECTURE 互链

### Phase 1 — 下载与锁约宇宙（工程优先）

目标：实盘与离线下载同一合约池。

1. 扩展 `1dte_api_ladder` 覆盖 V4 训练期（至少 2023-04→近期）。  
2. 参数固化：`lock_minute≈09:40`、`trading_dte=1`、band=`±2%` 或 `n_put/n_call` 档。  
3. Primary 4 从 ladder / 开盘 δ 派生，**写入同一天 map 的标记列**（如 `is_primary_feature=1`）。  
4. step2 只下载 ladder；特征管线只读 primary 四约。

验收：

- 每日 ladder 完整；primary ⊂ ladder  
- 周五到期 = 下一交易日（trading 1DTE）  
- 与旧 4-bucket 的 strike 差分布可监控（shadow）

### Phase 2 — 执行选腿（增益层）

目标：不改模型输入的前提下改善成交。

1. 复用 / 移植 5DTE `value_score`（点差、premium%、短窗弹性、量；**禁止**未来最优桶）。  
2. 信号触发后：同侧 ladder 内选腿；默认 fallback = primary 执行腿。  
3. Replay：同一 `net_edge` 序列，对比「固定 primary 腿」vs「value_score 腿」的 acct / MDD / 点差拖累。

验收：

- 同信号下 MDD 不恶化；亏损单平均拖累下降或盈利单右尾改善  
- 选腿与 primary 的 `|Δstrike|` 有上限（例如 ≤$2–3）

### Phase 3 — 开盘锁重训（train≈serve）

目标：生产模型与开盘合法锁约同分布。

1. 用 Phase 1 的 primary 4 重建 day_iv → 特征 → 标签 → LMDB。  
2. 按 V4 同切分重训；Val/Test IC 与 replay 单独建基线（预期低于旧前视血缘）。  
3. 实盘只上这一支权；旧 V4 权仅作研究对照。

验收：

- 开盘锁特征 vs 旧锁约特征的 shadow gap 报告  
- 实盘 / strict replay 同一 `config`（`1dte_family`）

### Phase 4 — 可选（不阻塞主线）

- 多面板并行 4 约（开盘多套连续特征，面板间选型）——等于新特征族，需单独立项。  
- 0DTE 动态 locator：仅研究；若重启须独立护栏与极短持有，不共用 `config.py`。  
- 真 0DTE 归档实验保留脚本与结果目录，不进入生产 checklist。

---

## 6. 配置与代码归属

| 用途 | 模块 / 路径 |
|------|-------------|
| 1DTE 生产 / 研究主规则 | `qqq_btc.qqq.config`（`PROFILE=1dte_family`） |
| 真 0DTE 归档规则 | `qqq_btc.qqq.config_true_0dte`（`PROFILE=true_0dte`） |
| 评测选规则 | `eval_test_set.py --strategy-config ...` |
| 锁约备忘 | `preprocess/download/LOCK_MAP_OLD_V2_MEMO.md` |
| 本路线图 | `qqq_btc/docs/1dte_ladder_upgrade_architecture.md` |

---

## 7. 决策记录（ADR 风格）

**ADR-1：主路径选 trading 1DTE，搁置真 0DTE**  
理由：流动性好不等于可控；六月真 0DTE 专用规则后账户仍接近打平，与 V4 1DTE 高收益不可比。

**ADR-2：特征日锁，执行可选腿**  
理由：V4 推理依赖连续 4 约结构特征；动态换特征约破坏训练假设。选腿只优化执行质量。

**ADR-3：下载对齐 `1dte_api_ladder`，不另建「多套 4 组合」**  
理由：±2% ladder 已覆盖附近 ATM/OTM；与现有 map/下载脚本一致，复杂度更低。

**ADR-4：离线高 IC 与生产对齐拆开**  
理由：全天前视锁约不能上实盘；生产必须开盘锁重训或接受 gap，禁止混用。

---

## 8. 修订记录

- 2026-07-13：初稿。汇总 0DTE 负向结论、1DTE ladder 三层架构、分阶段路线与 ADR。
