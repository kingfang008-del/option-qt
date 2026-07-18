# Regime Router：主基线 + 坏日专家路由

## 问题陈述

Freeze 基线（`full_day_peer3` + extend_mtm）在**大多数交易日有效**；近年试过的全局补丁（结构硬门 / LGBM / TCN / MAE 早切）都呈现**水床**：修好个别日、砍掉强窗利润。

正确拆法不是再改「唯一规则」，而是：

1. **默认永远跑基线**  
2. 离线标出基线搞不定的日子 → 按失败形态聚类成少数 **day_type**  
3. 每个 type 只挂**窄专家策略**（相对 freeze 的小 diff）  
4. 用**因果市场状态**训练 **Router**，决定今天走 `baseline` 还是某个专家  
5. 验收：强窗 ≥95% 基线 **且** 坏日簇有改善；误路由成本受控

## 一年坏日扫描（2025-07-01 → 2026-07-17）

数据：`results/regime_router/baseline_daily_scan.csv`  
口径：交易日 `day_ret ≤ −3%` 且 `n>0` → **40** 天（全样本约 225 日 / 143 有成交日）。

| day_type（事后按成交归因） | n | 含义 |
|---------------------------|---|------|
| `up_toxic` | 18 | 主要亏在 UP 期权 |
| `dn_toxic` | 12 | 主要亏在 DN 期权 |
| `rebound_trap_dn` | 5 | 10:30 前 QQQ>开且离 LOD 反弹，当日 DN 亏损（含 **2026-07-17**） |
| `other_loss` / `wide_chop` | 5 | 杂类 |

明细特征：`results/regime_router/bad_day_features.csv`。

要点：

- 坏日**不是一种行情**；07-17 只是 `rebound_trap_dn` 子集。  
- 10:30 的 QQQ gap/bounce/range 在 bad vs 大赢日上**均值几乎分不开** → 路由特征要加 breadth / VIXY / Mag7 同步，且允许 **日内触发**（首笔 Rule-A 前重估），不能只靠「QQQ 红绿」。  
- 理论上若能完美抹平 ≤−3% 日，曲线会好看很多——但那是上帝视角；Router 只能用因果信息，目标是**提高命中坏簇、压低误伤好日**。

## 架构

```
              morning / pre-entry state
                         │
                  ┌──────▼──────┐
                  │   Router    │   P(type) or argmax
                  └──────┬──────┘
             ┌───────────┼───────────┐
             ▼           ▼           ▼
        baseline     expert_A    expert_B
        (freeze)  rebound_trap  dn_toxic …
```

### Router（要训的部分）

- **输入（因果）**：截至路由时刻（建议默认窗口起点 10:30，或「当日第一笔候选前」）  
  - QQQ：gap、from_prev、mf10、range、bounce_from_lod、above_open/vwap  
  - Mag7：同向 Rule-A 广度、peer 对齐数、平均 from_prev、vol_z  
  - 可选：VIXY z、隔夜期货/前一日 day_ret  
- **输出**：`baseline | rebound_trap_dn | dn_toxic | up_toxic | …`（先 3～4 类，含 baseline）  
- **标签**：仅离线用——由基线当日成交归因（上表），**禁止**用 EOD 股价标签直接当路由标签而不经基线失败定义  
- **训练**：walk-forward（例：train≤2026-04 / valid May–Jul）；类别极不均衡 → 先做 **「需专家 vs baseline」二分类**，再在专家内分子类  
- **动作**：`route` 只切换当日 `trade/regime` 覆盖层，不改信号生成核心（Rule-A / TopK 可先不动）

### Expert（规则层，先手写再学）

相对 freeze 的**小 diff**，每个只服务一类坏日：

| expert | 触发意图 | 建议覆盖（研究开关） |
|--------|----------|----------------------|
| `rebound_trap_dn` | 07-17 类低开反抽假空 | `scale_dn_if_qqq_above_open=0.5` 或 DN size↓；**不要**全局硬挡 DN |
| `dn_toxic` | DN 单边有毒 | 提高 DN peer / 暂停 DN / 缩仓 |
| `up_toxic` | UP 假突破 | 对称处理 UP（先缩仓） |
| `orb_open`（研究） | 09:30–10:00 洗盘后分形高点突破 UP | **REJECT 开仓**；见 [`orb_open_expert_research.md`](orb_open_expert_research.md) |
| `washout_gate`（研究） | 开盘多标的 washout → 缩/禁基线开仓 | 宽阈值 REJECT；窄门 `washout_and_reclaim`+halt 为候选（同文档） |

> **架构升级**：上述专家现由 [`watchdog_architecture.md`](watchdog_architecture.md) 统一调度（NORMAL/DEGRADE/HALT/HUNT）。`regime_router` 仍可桥接为 degrade-only。

专家 **默认不启用**；仅 Router 高置信才切换。低置信 → 强制 baseline。

### 防再水床的硬约束

1. Router `p_special < p_min` → baseline（宁漏专家、勿误伤）  
2. 每个专家单独双窗 scoreboard；**禁止**未过验收写入 freeze  
3. 专家改动预算：最多动 regime/size/exit 中的 **1～2 个旋钮**  
4. 线上可解释：日志打 `route=… p=…` 与当日覆盖项

## 落地步骤

| Step | 产物 | 状态 |
|------|------|------|
| 1 | `scan_baseline_hard_days.py` | ✅ |
| 2 | `day_type_labels.csv` + experts JSON | ✅ |
| 3 | Oracle scoreboard（上帝路由） | ✅ 有边 |
| 4 | 训因果 Router | 下一刀 |
| 5 | 双窗验收 ≥95% 基线 + 坏簇改善 | 待 Router |

## Oracle 结果（2026-07-18）— **上限成立，值得训 Router**

接入：`profile.regime_router` → 按日覆盖 `Mag7RegimeGate.cfg`（当日结束还原）。

专家（软，`CONFIG/regime_router/experts_v1.json`）：

| day_type | 覆盖 |
|----------|------|
| `rebound_trap_dn` | `scale_dn_if_qqq_above_open=0.5` |
| `dn_toxic` | `direction_size_scale.DN=0.5` |
| `up_toxic` | `direction_size_scale.UP=0.5` |

| window | variant | total_ret | vs base | MaxDD | 07-17 | 坏簇 mean day_ret |
|--------|---------|-----------|---------|-------|-------|-------------------|
| May–Jul | baseline | +810% | — | −13.2% | −6.6% | −6.6% |
| May–Jul | **oracle soft** | **+977%** | **121%** | **−8.1%** | −3.3% | **−2.5%** |
| Feb–Apr | baseline | +140% | — | −28.9% | — | −5.9% |
| Feb–Apr | **oracle soft** | **+246%** | **176%** | **−22.1%** | — | **−3.2%** |

激进硬挡上限（`experts_v1_aggressive.json`，仅作天花板）：May–Jul → +1238% / 07-17=0；弱窗 +368%。说明 type 信息价值很大，但上线应先追软专家。

明细：`results/regime_router/oracle_scoreboard/`。

## 与已否决工作的关系

| 尝试 | 角色 |
|------|------|
| 全局 LGBM/TCN/MAE_CUT | 否决为默认层；可降级为某一专家内部旋钮 |
| `scale_dn_if_qqq_above_open` | 已挂进 `rebound_trap_dn` 专家 |
| freeze baseline | **永远是 default 路由** |

## 命令

```bash
python -m maga7.tools.scan_baseline_hard_days
python -m maga7.tools.build_regime_router_labels
python -m maga7.tools.run_regime_router_oracle
# 激进上限
python -m maga7.tools.run_regime_router_oracle \
  --experts maga7/CONFIG/regime_router/experts_v1_aggressive.json \
  --out maga7/results/regime_router/oracle_scoreboard_aggressive
```

## 预测路由（2026-07-18）— Oracle ✅ / 因果 ML ❌（暂不升）

| 层级 | 结果 | 结论 |
|------|------|------|
| **Oracle**（上帝 day_type） | 强窗 121% / 弱窗 176% 基线，DD 改善 | **专家有边，架构正确** |
| 多类 LGBM | valid 专家 recall=0 | 子类样本太稀（rebound 全样本仅 5） |
| 二分类 need_expert | valid AUC≈0.24，几乎从不触发 | 10:30 特征分不开「将亏日」 |
| 因果规则 rebound（above_open∧bounce≥1.2%） | 07-17 −6.6%→−3.3%；May–Jul **89%** 基线 | 召回部分 rebound，但误触发伤强窗 |

产物：

| Path | 角色 |
|------|------|
| `tools/build_regime_router_dataset.py` | 10:30 因果特征表 |
| `tools/train_regime_router.py` / `_binary.py` | 多类 / 二分类 |
| `tools/run_regime_router_oracle.py` | Oracle scoreboard |
| `tools/run_regime_router_binary_predicted.py` | 预测路由 scoreboard |
| `results/regime_router/oracle_scoreboard/` | Oracle 明细 |
| `results/regime_router/rule_rebound_scoreboard/` | 规则 rebound |

## 续：加厚特征 + rebound 专训（2026-07-18）

v2 特征：`router_dataset_v2.parquet`（隔夜 `qqq_open_vs_prev` / `low_open_reclaim`、前 1–3 日、VIXY、Mag7 分散度）。

| 方法 | May–Jul vs base | Feb–Apr vs base | 07-17 | 结论 |
|------|-----------------|-----------------|-------|------|
| Oracle rebound-only | **104%** | **104%** | −3.3% | 单专家上限干净 |
| LGBM rebound | 100%（不触发） | 100% | −6.6% | 正样本仅 5，仍不可学 |
| 宽规则 reclaim+bounce | 85–89% | ~100–107% | −3.3% | 误伤 6 月 DN 大胜日 |
| **`reclaim_disp55` 因果规则** | **104%** | **104%** | **−3.3%** | **首个双窗不掉水床的可上线路由** |

### 规则 `reclaim_disp55`（10:30 因果）

同时满足：

1. QQQ **低开收回**：`open < prev_close` 且 `px_1030 > open`  
2. `bounce_from_lod ≥ 0.8%`  
3. Mag7 `frac_above_open ≤ 0.55`（反抽日个股分化，非全面翻红）

命中 → 挂 `rebound_trap_dn` 专家（`scale_dn_if_qqq_above_open=0.5`）。

Replay 接入（**默认 off**）：

```json
"regime_router": {
  "enabled": false,
  "mode": "rule",
  "rule": "reclaim_disp55",
  "asof": "10:30",
  "experts_path": "maga7/CONFIG/regime_router/experts_v1.json"
}
```

验证：`results/regime_router/rule_mode_verify/`（强/弱窗各触发 1 日，与 Oracle rebound 对齐）。

### 状态

- **架构与专家**：成立（Oracle 大边；rebound 单专家小边但干净）  
- **可上线候选**：`reclaim_disp55` 规则路由（双窗各 +4% 相对收益、07-17 半损、MaxDD 不变）  
- **仍不升 freeze**：边不大，再观察更多交易日后再决定是否 `enabled=true`  
- **ML Router**：暂搁置，等标签积累或更长样本

### 命令

```bash
python -m maga7.tools.build_regime_router_dataset   # → router_dataset_v2
python -m maga7.tools.train_regime_router_rebound
python -m maga7.tools.run_regime_router_rebound_scoreboard
# 规则模式（profile.regime_router.mode=rule）
```
