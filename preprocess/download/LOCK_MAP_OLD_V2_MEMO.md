# V4 / standard_old_v2 锁约备忘

> 日期：2026-07-12  
> 目的：固化「旧锁约 vs 现网 step1」调查结论，避免再次用错 lock map 把 IC 训崩。

## 一句话结论

**离线复现 V4 / IC≈0.2 必须用旧锁约逻辑（或直接用 old_v2 导出的 map）。**  
旧锁约带**全天 δ 前视**，但对训练有效：模型能学到稳定规律。  
现网 `step1_build_target_map.py` 对**实盘/无前视**更正确，但对 **V4 离线血缘是错误数据源**——会把 Val IC 打到大约一半（~0.27 → ~0.13）。

---

## 关键产物路径

| 角色 | 路径 |
|------|------|
| 旧锁约脚本（冻结 main 首版） | `preprocess/download/step1_build_target_map_old.py` |
| 旧锁约配置 | `preprocess/CONFIG/anchor_qqq_old_v2.json` |
| 现网锁约（实盘向） | `preprocess/download/step1_build_target_map.py` |
| dayiv 父代 | `~/train_data/_bak_pre4c/dayiv_old_dirs/standard_old_v2` |
| 从 dayiv 反导的权威 map | `~/train_data/locked_targets_map_from_standard_old_v2.parquet` |
| 首版 vs old_v2 错日清单 | `~/train_data/bak_lineage_reproduce/old_v2_vs_first_step1_mismatch_days.csv` |
| 错 bucket 明细 | `~/train_data/bak_lineage_reproduce/old_v2_vs_first_step1_mismatch_pairs.csv` |
| 血缘 JSON | `~/train_data/_bak_pre4c/LINEAGE_CRACKED.json` |

main 首版 commit：`64b0532`（2026-07-03，`preprocess/download/step1_build_target_map.py` 首次入仓）。

---

## old_v2 锁的到底是什么 DTE？

对 `standard_old_v2` 全部 3126 个 `(date, bucket)`：

| 口径 | 结果 |
|------|------|
| **trading DTE = 1** | **3116 / 3126（99.68%）** |
| trading DTE = 0 | 3（2025-01-08） |
| trading DTE = 2 | 7（2024-03-08、2026-03-06） |

日历日：周一–周四多为 cal=1；**周五多为 cal=3（到周一）**，交易日仍是 1DTE。

因此：名义上的「0dte 配置」实际母体是 **trading-1DTE 四约**。

---

## 旧 step1 为什么叫 0dte，却锁成 1dte？

首版两件事叠在一起：

1. **配置**：`front_allowed_dte=[0,1,2]`，`front_prefer_dte=0`（有 0 锁 0，否则退到 1/2）。
2. **DTE 计算 bug**：`expiration` naive 先 `tz_localize("UTC")` 再转 NY → 日期往前偏一天。  
   真实 trading-1DTE 合约大量被标成 **buggy_dte=0**，于是 prefer=0 正好锁中它们；周五常落到 allowed 的 2。

对 old_v2 的 OCC 到期反算首版 buggy DTE：约 **80% 为 0，19% 为 2，1% 为 1**。

后来改成 strict 0DTE / trading DTE / 禁止乱退档，是为修这个「假 0dte」，但也因此和 old_v2 **分家**。

---

## 旧锁约 vs 现网锁约：算法差异

| 点 | 旧版（`step1_build_target_map_old.py`） | 现网（`step1_build_target_map.py`） |
|----|----------------------------------------|-------------------------------------|
| DTE | legacy：naive exp→UTC（默认可复现）；可选 trading | `compute_dte_series`，常配 trading + strict 0/1 |
| 快照 | **全天所有 bar** 的供应商原始 `\|delta\|` | 开盘窗 `lock_window_minutes=10` |
| delta | 不重算 | `_recompute_open_deltas` + CALL 可用 PUT 合成 |
| 配置 | `[0,1,2]` prefer 0 | strict 单 DTE，无退档 |
| 校验 | 无 | `validate_contract_exists` 等 |
| 前视 | **有**（用盘中信息选开盘锁定约） | **无**（开盘后不换） |

用首版逻辑对拍 old_v2（当前 `nq_options_day_iv`）：

- 同 ticker ≈ **82.6%**，整天全对 ≈ **460/789**
- 残差 ~17% 高度集中在 **2026-04~06**（重处理 nq 链之后），更早多为差 $1 档噪声

现网 1DTE-4bucket 对拍 old_v2：同 ticker 仅 ≈ **18%**（到期仍 ~99% 一致）。

---

## 为什么换锁约会让 IC 掉一半？

**不是到期日错了，是同一天 1DTE 上锁错了行权价 → 截面结构特征换源。**

### 合约几何（old_v2 vs 现网 1dte4b）

- 同 ticker ~18%，同到期 ~99.7%
- 错约时 `|Δstrike|` 中位 **$2**
- 旧约用全天 δ，日末更贴目标 δ；新约贴开盘现货，日末 `|δ|-target` 更差

### 特征层（真正打 IC 的通道）

V4 期权侧约 14 列；alpha 主要来自**四约截面**，不是单点 IV：

- `options_struc_skew` / `options_flow_skew`
- `options_pcr_volume` / `options_vw_imbalance`
- `options_iv_momentum` / `options_iv_divergence` / `options_gamma_accel`
- 以及 vw greeks / spread

实测（旧血缘特征 vs 新锁约重建）：

| 量 | 旧↔新 |
|----|------|
| 单约 IV 路径 corr（错 ticker） | 中位 ~0.93（看起来还行） |
| 日级 skew（callATM−putATM）corr | ~**0.47** |
| 14 个期权特征 median corr | ~**0.35–0.50** |
| 最差列量级 | pcr / δ / gamma_accel / struc_skew 可到 0.1–0.3 |

点 IV 还像，**结构信号已是另一套过程** → 模型学的「结构因子 ↔ 远期收益」映射失效 → Val IC 腰斩是预期结果。  
正股特征基本不受锁约影响；滚动归一化还会放大期权 raw 的漂移。

### 训练对照（量级）

| 血缘 | Val best IC（量级） |
|------|---------------------|
| old_v2 / 旧锁约重训 | ~0.27–0.28 |
| 新锁约 / prefer_primary 等重建 | ~0.13 一带 |

---

## 如何理解「带前视但能学到规律」

1. **前视存在**：全天 δ 选约会偏向「当天曾当过 ATM、更活跃」的行权价，与当日路径耦合，可能抬高离线 IC。
2. **仍是可复现规律**：同一套（有偏）锁约规则在 train/val/test 上一致时，模型学到的是该采样规则下的条件分布，不是随机噪声。
3. **工程取舍**：
   - **离线冲 IC / 复现 V4** → 用旧锁约（接受前视）。
   - **实盘无前视对齐** → 用现网 step1；不要指望直接继承 V4 那条 IC，需单独重训与验收。

不要把「现网更干净」误当成「现网更能复现历史 IC」。

---

## 操作建议

1. 离线 V4 管线：  
   - 优先直接用 `locked_targets_map_from_standard_old_v2.parquet` 或 dayiv `standard_old_v2`；  
   - 或跑 `step1_build_target_map_old.py --dte-mode legacy`（默认）。
2. 不要用现网 `step1_build_target_map.py` 的输出去重建「要对齐 bak_train / V4 IC」的特征。
3. 若 `nq_options_day_iv` 再重算，预期 2026 春季附近与 old_v2 的 ticker 重合会继续变差；权威源仍是 dayiv / 已导出 map，而不是重跑 step1。
4. 现网 step1 与 `step1_build_target_map_old.py` **并行保留**，文档和脚本入口写清用途，避免混用。
5. 实盘锁约见下一节（**不能**搬全天前视；开盘锁是正确约束）。

---

## 实盘应该怎么锁约？

### 硬约束

实盘**不可能**复用离线旧版的「全天 δ 选约」——那是用未来信息决定开盘锁定约。  
开盘前 / 开盘初快照锁约，是实盘唯一合法口径。现网开盘窗逻辑方向是对的。

当前实盘入口（`New_Pro/.../ibkr_connector_v8._find_contracts`）：

- 用当日现货 + IB 链 + model Greeks（或近似 δ）在开盘附近选 4 bucket
- 到期选择走 `anchor` profile（现默认 `anchor_qqq_0dte.json` = **strict 日历 0DTE**）
- 与 V4 训练母体（**trading 1DTE + 全天前视选行权价**）本来就不是同一分布

### 三角矛盾（必须显式承认）

| 目标 | 锁约方式 |
|------|----------|
| 复现离线高 IC | 旧锁约（全天前视 + 实质 1DTE） |
| 实盘合法 | 开盘快照，无前视 |
| train≈serve | 离线也必须用开盘快照重训 |

三者不能同时满足。实盘只能选后两行的组合，或接受「高 IC 研究模型 ≠ 实盘模型」。

### 推荐分层

**A. 实盘执行层（现在就该改的）**

1. **到期：锁 trading 1DTE，不要用 strict 0DTE。**  
   - 与 old_v2 / V4 特征家族一致（周五锁周一到期，而不是当日到期）。  
   - 日历 DTE 在周五会把 1DTE 标成 3；实盘应用 **trading DTE** 或显式「下一交易日到期」。  
2. **时点：继续开盘锁**（建议 09:30–09:40 有稳定报价后再定稿，仍无全日前视）。  
3. **行权价：开盘快照 `|δ|→{0.5,0.25}`**（现逻辑可保留）；不要盘中按 δ 换主力约（否则特征与「日锁」假设又漂）。  
4. **当天锁定后不再换约**（与离线 day-lock 一致）。

**B. 模型层（二选一，不要混）**

| 路线 | 做法 | 预期 |
|------|------|------|
| **研究 / 复现 IC** | 特征+训练继续用旧锁约 dayiv；实盘另议 | 离线 IC 高；实盘有 train/serve gap |
| **生产对齐（推荐中期）** | 用「开盘窗 + trading 1DTE」重跑 step1→特征→重训，实盘同一规则 | IC 上限低于 V4 旧血缘，但盘上可兑现 |

不要用「旧锁约训出来的权」配「开盘 0DTE 锁约」硬上实盘，还期望 IC 接近 0.2——结构特征源已经换了。

**C. 缓解 train/serve gap 的过渡手段（可选）**

- 开盘锁 1DTE 四约后，用同一套特征管线做 **shadow**：对比若用「旧规则回放当日」会锁哪些约、特征差多少（只监控，不交易）。  
- 执行仍只交易开盘锁的约；用 shadow 量化 gap，决定要不要上「开盘锁重训」模型。  
- 不建议：盘中按实时 δ 换约去「追」旧锁约——那是另一类前视/不稳定，且和日频锁定特征定义冲突。

### 实盘 checklist（短）

- [ ] profile = trading **1DTE** 4-bucket（非 strict 0DTE）  
- [ ] DTE = trading sessions（或「下一交易日到期」），尤其周五  
- [ ] 09:30+ 短窗快照 δ 锁约，锁完不换  
- [ ] 上线模型要么是「开盘锁重训」版，要么接受 V4 旧权的分布偏移并单独验收实盘 IC/PnL  
- [ ] 离线高 IC 复现继续只用 `step1_build_target_map_old.py` / old_v2 map

---

## 相关命令备忘

```bash
# 旧锁约（对齐 old_v2 口径）——仅离线复现
python preprocess/download/step1_build_target_map_old.py \
  --start-date 2023-03-28 --end-date 2026-06-30

# 现网开盘窗锁约（实盘同口径的离线重建；非 V4 旧血缘）
python preprocess/download/step1_build_target_map.py \
  --profile qqq_1dte --start-date 2023-03-28 --end-date 2026-06-30
```

---

## 修订记录

- 2026-07-12：初稿。确认 old_v2≈trading-1DTE；首版 UTC+0/1/2 退档复现 ~83% ticker；现网开盘窗锁约导致结构特征 corr~0.4、IC 腰斩；落地 `step1_build_target_map_old.py`。
- 2026-07-12：补充「实盘锁约」——开盘锁合法；应对齐 trading 1DTE；生产模型需开盘锁重训或接受 gap。
