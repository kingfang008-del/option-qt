# Replay 版本谱系、口径差异与结果对账

> 更新时间：2026-07-14  
> 目的：结束 V4、FT56、honest replay、stream parity、Jul13 实验和跨月临时回放之间的口径混用。  
> 本文不重新优化参数，只回答：每个数字从哪里来、用了什么配方、能否横向比较。  
>  
> **全版本结果总表（含 4/5/6 月 V4+VX 金标 +214/+175/+68、各版本改动清单）**：  
> [`qqq_btc/docs/replay_versions_full_catalog.md`](./replay_versions_full_catalog.md)
>
> **当前统一评级与流式对拍优先级以本文 §14 为准**；旧章节保留历史推导，
> 若评级冲突，以 §14 的证据核验结果覆盖。

---

## 1. 结论先说

### 1.1 当前应保留的 V4 冻结基线

V4 `open30 + bounce_cut + lock45` 的官方冻结结果是：

| 日期 | acct25 日收益 |
|---|---:|
| 2026-07-01 | -0.42% |
| 2026-07-02 | +16.24% |
| 2026-07-06 | +7.08% |
| 2026-07-07 | +25.61% |
| 2026-07-08 | +4.92% |
| 2026-07-09 | +1.07% |
| 2026-07-10 | 不纳入冻结可比区间 |
| **累计复利** | **+65.09%** |

精确累计值：`0.650874054000236`，12 笔。

权威产物：

- `qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json`
- recipe：`open30 + bounce_cut + thesis_lock_leg_bars=45 + honest causal put_gate + LIVE q10=-0.2`

**这条结果没有被最近的临时回放推翻。**

但它的身份是 **V4、无 VX、分钟级离线 Jul W1 金标**，不是当前
FT56+VX+tick 的实盘收益承诺。问“原 V4 离线策略是否复现”看 `+65.09%`；
问“当前实盘大概怎样”应看后文的流式 `+19.23%` 或
FT56 honest+VX 离线 `+22.33%`。

### 1.2 最近的跨月结果不是新版 honest 基线

以下结果：

- V4：2026-06-01 至 2026-07-13 累计 `+16.51%`
- FT56：同期累计 `+128.12%`
- V4 6 月 `-25.13%`、7 月截至 13 日 `+55.63%`
- FT56 6 月 `+24.46%`、7 月截至 13 日 `+83.29%`

来自：

- `qqq_btc/results/june_july_daily_replay_20260714/`

它是一个**临时 Python 命令生成的混合诊断回放**，不是
`train_ft56_julw1_honest_kpi.sh` 的正式产物，也没有独立可复跑脚本。

该结果应标记为：

```text
DIAGNOSTIC_MIXED_INPUTS / 非正式 / 不得替代冻结基线
```

---

## 2. 什么才叫 honest replay

本项目约定的 July W1 honest KPI 包含以下四层：

| 层 | 正确口径 |
|---|---|
| 模型特征 | `july_w1_v4_honest_openwin/quote_features_test` |
| 期权成交数据 | 同一 honest root 下的 `options_1m` |
| PUT gate | raw 1min `vix_level`，时间戳 `+1m` 后 `merge_asof(backward)` |
| 策略 | `LIVE_REPLAY`，并显式覆盖 `edge_q10_floor=-0.2` |

标准入口：

```bash
SKIP_TRAIN=1 \
  bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
```

标准脚本只把 raw1 的 `vix_level` 作为因果 `put_gate` 挂入 infer：

```python
s["timestamp"] = s["timestamp"] + pd.Timedelta(minutes=1)
out["put_gate"] = merge_asof(..., direction="backward")["put_gate"]
```

它**不会**用另一个 raw parquet 覆盖 infer 中已经生成的
`trend_fit_ret_30m`、`trend_fit_r2_30m`、`open30_ret` 等上下文字段。

详细定义：

- `qqq_btc/docs/honest_live_kpi_finetune_replay.md`
- `qqq_btc/tools/train_ft56_julw1_honest_kpi.sh`

---

## 3. V4 `open30 / bounce / lock45` 三个版本

三组结果使用相同的 V4 honest infer、相同因果 put gate、相同 LIVE replay
主口径，差别只在退出和再入场规则。

| 版本 | 规则差异 | acct25 | 主要现象 |
|---|---|---:|---|
| 仅 `open30` | 早盘 PUT 必须满足 open30 结构 | +60.28%（约 +60.3%） | 挡掉 Jul1 09:46 的错误 PUT，但保留 10:47 的后续亏单 |
| `open30 + bounce_cut` | 增加 `SPOT_THESIS` 提前证伪退出 | +52.63%（约 +52.6%） | 单笔减亏，但提前释放仓位后又开出更差的同腿交易 |
| `open30 + bounce_cut + lock45` | `SPOT_THESIS` 后同腿锁 45 分钟 | **+65.09%** | 阻止证伪后立即重入，同时保留稍晚的有效趋势机会 |

### 3.1 `open30` 做了什么

关键配置：

```text
put_early_open30_max_min = 0.0
```

含义：早盘 PUT 需要开盘结构曾经翻红；Jul1 09:46 的阴跌无结构 PUT
被整笔挡掉。

它把当时的 V4 周收益从原始约 `+54.9%` 提高到约 `+60.3%`。

### 3.2 为什么单加 bounce 反而从 +60.3% 降到 +52.6%

`bounce_cut` 对已有亏单本身有效：

- Jul1 10:47 PUT：约 `-18%` 提前收窄到约 `-10.5%`
- Jul6 10:58 PUT：约 `-13%` 提前收窄到约 `-9.0%`

但退出后立刻恢复空仓状态，允许同腿再次入场：

- Jul1 后续出现 11:01 PUT，约 `-22%`
- 减掉的亏损被新的重入亏损重新吃掉

因此 `bounce_cut` 不是无效，而是缺少与它配套的再入场治理。

### 3.3 为什么 lock45 能提高到 +65.09%

关键配置：

```text
thesis_lock_leg_bars = 45
```

只有 `SPOT_THESIS` 证伪退出后，同一腿才进入 45 分钟锁定：

- 挡住 Jul1 11:01 的差 PUT
- 不锁全日，因此仍能保留 Jul1 稍晚交易
- Jul6 早盘减亏后，仍能接到午后的有效大 PUT

这就是 `+65.09%` 的完整机制，不是“只加了一个止损”。

---

## 4. V4 +65.09% 的精确日收益

来源：

`qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json`

| 日期 | 笔数 | 日收益 | 主要退出 |
|---|---:|---:|---|
| Jul1 | 2 | -0.4248% | `SPOT_THESIS`、`STEP_PROTECT` |
| Jul2 | 1 | +16.2397% | `MAX_HOLD` |
| Jul6 | 2 | +7.0778% | `SPOT_THESIS`、`TRAILING` |
| Jul7 | 2 | +25.6063% | `MAX_HOLD`、`STEP_PROTECT` |
| Jul8 | 3 | +4.9200% | `STEP_PROTECT` ×3 |
| Jul9 | 2 | +1.0739% | `STEP_PROTECT`、`TIME_STOP` |
| Jul10 | — | 不纳入冻结可比区间 | 最终流式对拍按 Jul1–9 比较 |

累计使用账户复利，而不是日收益简单相加：

```text
Π(1 + daily_acct25) - 1 = 65.0874%
```

---

## 5. FT56 的几个数字为什么也不一样

FT56 不是 V4 +65.09% 的同义版本。至少存在以下独立快照：

| 名称 | acct25 | 配方/状态 | 是否可与 V4 +65.09 直接比 |
|---|---:|---|---|
| FT56 hardcap 冻结结果 | +49.00% | `end_bar=240`、`max_hold=55`、`q10=-0.2`；生成时尚未使用后来完整的 open30+bounce+lock45 路径 | 否 |
| honest KPI compare 中的 FT56 | +58.80% | 与当时 V4 current rails 同跑，15 笔 | 可以作为当时同快照模型对照 |
| 全局 `min_dual_leg_edge_gap=0.001` 后 | +18.19% | Jul13 补丁被全局应用，误杀 Jul7 等盈利交易 | 否，属于受损配置 |
| 关闭全局 gap 的消融 | 约 +54.58% / +58.80% 等 | 取决于是否同时关闭 early cross gate、是否复用冻结 infer/context | 只能作为消融 |
| 最近临时 isolated replay | +66.44% | 临时覆盖多个 raw context 字段，非标准脚本 | 否 |
| 最近跨月 mixed replay | +128.12% | 6–7 月拼接、连续状态、不同 infer lineage | 否 |

### 5.1 FT56 +49.00% 是什么

来源：

`qqq_btc/results/ft56_julw1_end240_hold55_hardcap_20260713/summary.json`

关键配置：

```json
{
  "session_entry_end_bar": 240,
  "max_hold_bars": 55,
  "edge_q10_floor": -0.2
}
```

14 笔，累计 `+49.004985%`。

该文件的 Jul1 仍包含 09:46 PUT `-22.62%`，说明它不是后来
`open30 + bounce_cut + lock45` 的同一策略快照。

### 5.2 FT56 +18.19% 为什么明显变差

当时把：

```text
min_dual_leg_edge_gap = 0.001
```

全局应用到每一天。该门控原本用于阻止 Jul13 的双腿低置信交易，
但也删除了正常日的 FT56 盈利路径，尤其是 Jul7 的大 PUT。

因此 `+18.19%` 是 Jul13 定向补丁污染全局历史后的受损结果，
不是 FT56 的原始能力。

当前代码已把全局 gap 关闭，并尝试仅在因果
`OPEN_SHOCK_CHOP` 状态中启用局部门控；但这个新状态机还没有完成
统一 honest 全历史验收，不能直接宣布为新生产基线。

### 5.3 当前“更贴近实盘”的版本是另一条谱系

权威对照板：

`qqq_btc/docs/replay_version_kpi_board.md`

| 优先级 | 版本 | July W1 acct25 | 与 +65.09 的主要差异 |
|---:|---|---:|---|
| 1 | FT56 诚实流式对拍 | **+19.23%** | FT56、FCS+OMS、1s tick、VX 日切 |
| 2 | FT56 honest+VX 离线 | **+22.33%** | FT56、诚实特征、分钟执行、VX/quarantine |
| 3 | FT56 honest 旧版无 VX | +58.80% | FT56、分钟执行、无 CHOP 日切 |
| 4 | V4 offline-live-aligned + VX | +55.67% | V4、分钟执行，但 VX 将 Jul8/9 判为 CHOP 空仓 |
| 5 | V4 无 VX 冻结金标 | +65.09% | V4、分钟执行、open30+bounce+lock45，无 VX |

因此同时存在两个都正确、但回答不同问题的数字：

```text
V4 原策略离线金标：+65.09%
当前 FT56 实盘预期：流式约 +19.23%，同模型离线约 +22.33%
```

不能用后者宣布前者“失效”，也不能用前者承诺当前生产收益。

### 5.4 为什么启用 VX 后 V4 从 +65.09% 变成 +55.67%

V4+VX 版本保留了 Jul1、Jul2、Jul6、Jul7 的同一日收益路径，
但 Jul8、Jul9 被 VX selector 判为 `CHOP_NO_TRADE`：

| 日期 | V4 无 VX 冻结 | V4 + VX |
|---|---:|---:|
| Jul1 | -0.42% | -0.42% |
| Jul2 | +16.24% | +16.24% |
| Jul6 | +7.08% | +7.08% |
| Jul7 | +25.61% | +25.61% |
| Jul8 | +4.92% | 0.00% |
| Jul9 | +1.07% | 0.00% |
| 累计 | **+65.09%** | **+55.67%** |

这不是计算错误，而是 VX 日切主动删除了两天交易。

相关产物：

- 流式：`qqq_btc/results/july_w1_ft56_honest_live_parity_20260714/`
- FT56 honest+VX：`qqq_btc/results/offline_live_aligned/ft56_julw1_honest_live_aligned/summary.json`
- V4+VX：`qqq_btc/results/offline_live_aligned/jun_jul_live_aligned/summary.json`

### 5.5 已完成的流式对拍版本

之前确实完成过多轮流式对拍。它们检验的是“同口径 offline 与
FCS/OMS stream 是否一致”，不能只保留最终实盘收益数字。

#### 流式版本 S1：frozen norm 早期同口径对拍

| 项 | 结果 |
|---|---:|
| FT56 offline（同 frozen npz） | -13.66% |
| Redis stream | -13.22% |
| 差异 | 约 0.44 个百分点 |

这轮策略收益差，但证明了在相同 frozen normalization 下，
offline 与 stream 可以基本对齐。它同时证明 rolling 特征的
`+37.7%` 不能直接拿来当 frozen stream 目标。

来源：

- `qqq_btc/results/july_w1_replay_baseline_25pct.md`
- `qqq_btc/results/july_w1_ft56_4c_stream/`

#### 流式版本 S2：FT56 honest 三闸门中间版本

在 honest FCS 特征和 Gate1/2 对齐后，早期全周流式约 `+10%`。
后续消融发现，直接重新开启 feature5m/regime gold 并不能恢复历史
开卷结果，反而会恶化。这轮用于确认问题主要位于决策门控和腿选择，
不是 raw feature 完全算错。

主要产物：

- `qqq_btc/results/july_w1_ft56_honest_3gate_week_g12pass/`

#### 流式版本 S3：V4 `open30+bounce+lock45` 直接对拍

这是与 V4 `+65.09%` **最直接可比**的一轮：

| 区间 | 流式 | 离线 | 差异 |
|---|---:|---:|---:|
| Jul1–9 | **+63.66%，13 笔** | **+65.09%，12 笔** | -1.43 个百分点 |

主要结论：

- Jul1、Jul2、Jul6、Jul8 的决策和退出路径基本贴合
- 剩余主要差异在 Jul7：
  - 流式 CALL edge：`0.199515`
  - 流式动态阈值：`0.199217`
  - 离线对应 edge：约 `0.194557`
  - 因此流式多开一笔 CALL
- 这是临界 edge 的特征/标签微差，不再是 bounce-cut 缺字段等大范围接线错误

Jul10 加入流式后全段为 `+57.76% / 15笔`，但最终对账明确将 Jul10
排除出 `+65.09%` 的冻结可比区间，所以正式 parity 结论使用 Jul1–9。

来源：

- 完整过程文档：`qqq_btc/docs/v4_july_w1_stream_parity_final_aligned_20260714.md`
- 流式目录：`qqq_btc/results/july_w1_v4_stream_final_aligned/`
- 汇总：`.../stream_summary_paired.json`
- 离线：`qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json`
- 信号差异：
  - `.../signal_diff_20260707.json`
  - `.../signal_diff_20260710.json`

#### 流式版本 S4：当前 FT56+VX+tick

| 项 | 结果 |
|---|---:|
| FT56 honest+VX 离线 | +22.33%，9 笔 |
| FT56 FCS+OMS+tick 流式 | +19.23%，7 笔 |

这是当前更贴近生产执行的版本；与 S3 的区别是模型改成 FT56，并启用
VX 日切和 tick exits。因此：

```text
S3 回答：原 V4 +65.09 离线配方能否被流式复现？
答案：基本可以，Jul1–9 为 +63.66%，剩余 1.43pp。

S4 回答：当前 FT56+VX+tick 生产栈的收益路径是什么？
答案：流式约 +19.23%，同模型分钟离线约 +22.33%。
```

两轮都有效，但不能把 S4 的 `+19.23%` 当成 S3 对拍失败。

---

## 6. Jul13 实验版本

使用 fresh Polygon 1s→1m 数据和 old-lock 合约重建后，基础结果曾是：

| 模型 | Jul13 基础结果 |
|---|---:|
| V4 | -4.72% |
| FT56 | -2.73% |

盘中 VIXY/QQQ 状态实验曾得到：

| 模型 | 实验结果 |
|---|---:|
| V4 | -0.24% |
| FT56 | +0.27% |

但需要修正解释：

- V4 `-0.24%` 已能由 `OPEN_SHOCK_CHOP` 的 PUT 管理复现
- FT56 `+0.27%` 的实验同时保留了全局 `gap=0.001`
- 全局 gap 正是导致 FT56 Jul W1 从正常水平跌到约 `+18%` 的原因
- 把 gap 改为纯状态内生效后，最新 isolated 结果是 FT56 Jul13
  约 `-2.86%`，因为 09:45 的 CALL 发生在 10:04 状态确认之前

所以，之前“Jul13 FT56 已改善到 +0.27%，且不伤历史”这个表述不完整；
它混入了一个会伤历史的全局 gate。

---

## 7. 最近跨月 `+16.51% / +128.12%` 到底是什么

结果目录：

`qqq_btc/results/june_july_daily_replay_20260714/`

### 7.1 实际使用的 infer 文件

V4：

- 6 月：`v4_databento_rebuild_apr_jun/test_infer.parquet`
- Jul1–10：`v4_jul_w1_fixed5m_infer/test_infer.parquet`
- Jul13：`july10_13_polygon_fresh_old_lock/v4/test_infer.parquet`

FT56：

- 6 月：`ft56_replay_2026apr/test_infer.parquet`
- Jul1–10：`ft56_julw1_intraday_regime_eval/test_infer.parquet`
- Jul13：`july10_13_polygon_fresh_old_lock/ft56/test_infer.parquet`

### 7.2 它与正式 honest 脚本的关键差异

#### 差异 A：不是统一的特征与期权数据 lineage

6 月、Jul W1、Jul13 分别来自不同重建目录；Jul13 还使用 old-lock
fresh Polygon 数据。它们不是一次统一 infer 任务的连续产物。

#### 差异 B：临时命令覆盖了多个上下文字段

正式 honest 脚本只附加：

```text
put_gate = raw1.vix_level（+1m 因果）
```

临时跨月命令却从 raw parquet 覆盖了：

```text
vix_level
spot_ret_15bar
vix_ret_15bar
trend_fit_ret_30m
trend_fit_r2_30m
open30_ret
open30_peak_dd
```

这会改变 open30、趋势、cross-asset 和新 VIXY 状态门控的判断，
因此交易路径已不再等于冻结 infer 对应的上下文。

#### 差异 C：使用了当前工作区代码，而不是 +65.09 的配置快照

当前 `qqq_btc/qqq/config.py` 比 +65.09 生成时多了或改变了：

- `call_spent_*`
- `put_early_cross_confirm_*`
- `OPEN_SHOCK_CHOP` 盘中状态
- `min_dual_leg_edge_gap=None`（不再全局启用）
- 跨日 PUT quarantine
- 跨日 all-leg defense

因此即使输入文件相同，当前代码也不能自动代表冻结策略。

#### 差异 D：连续状态跨越 6 月进入 7 月

临时命令把 6 月和 7 月拼成同一个 `ReplaySession`：

- `entry_quantile` 缓冲从 6 月持续到 7 月
- 跨日 quarantine / all-leg defense 会继承
- 前日腿收益和账户收益会影响下一日

而 V4 `+65.09%` 是 July W1 独立验收，不是先跑完整 6 月后再进入 7 月。

#### 差异 E：日期范围不同

V4 `+65.09%`：

```text
冻结可比区间 Jul1–Jul9；不含 Jul10、Jul13
```

临时跨月 July：

```text
Jul1–Jul13；包含 Jul13 -8.73%
```

临时结果拆开后：

```text
临时混合口径 Jul1–10：+70.51%
再复利 Jul13 -8.73%：降为 +55.63%
```

因此 `+55.63%` 不是同一 Jul W1 从 `+65.09%` 无故下降，
而是“混合字段下的 Jul1–10 +70.51%”再加入 Jul13 亏损后的结果。

#### 差异 F：V4 缺少 6 月 18 日

V4 6 月 infer 没有 2026-06-18；FT56 有该日输入但无交易。
两者有效 session 数分别为 28 和 29，已经不是严格等样本比较。

### 7.3 跨月总收益只是月收益复利

V4：

```text
(1 - 25.13%) × (1 + 55.63%) - 1 = +16.51%
```

FT56：

```text
(1 + 24.46%) × (1 + 83.29%) - 1 = +128.12%
```

数学计算没有错；错误在于把这组混合输入结果称为正式 honest replay，
并拿它替换 V4 `+65.09%`。

---

## 8. 为什么会出现这么多看起来矛盾的数字

收益数字至少由以下维度共同决定，任意一项不同都不能直接横比：

1. 模型 checkpoint：V4 / FT56 / 其他 finetune
2. feature root：honest / databento / bak / rebuilt
3. 归一化：rolling / frozen daily / frozen upto May
4. option quote：Databento / Polygon / degraded 1m
5. 合约锁定：4-contract / 8-contract / old lock / new lock
6. put gate：infer 内特征 / raw1 +1m 因果 / VIXY stream buffer
7. replay 配置：`REPLAY` / `LIVE_REPLAY`
8. q10：`-0.25` / `-0.2`
9. rails：max hold、bounce、thesis lock
10. entry gates：open30、cross confirm、dual-leg gap、VX profile
11. 日期：Jul1–8 / Jul1–10 / Jul1–13 / Jun–Jul
12. session 状态：单段冷启动 / June 预热 / 连续跨日
13. tick exits：分钟 rails / 秒级 disaster exits
14. 代码快照：生成产物时的 commit 与当前未提交代码

只写“V4 replay”或“FT56 replay”不足以唯一确定一个结果。

---

## 9. 结果分级：以后哪些数字可以使用

### A. `OFFICIAL_HONEST_FROZEN`

可以作为策略验收基线，必须有：

- 独立脚本
- 完整 manifest
- checkpoint
- feature / option / raw gate 路径
- 日期范围
- config 快照
- git commit
- daily 和 trades 文件

当前可保留：

- V4 open30+bounce+lock45：`+65.09%`
- FT56 hardcap：`+49.00%`，但明确它是历史调参快照，不是当前 VX 实盘预期

### B. `CONTROLLED_ABLATION`

只改一个字段，其他输入完全冻结。例如：

- open30 only `+60.3%`
- bounce no lock `+52.6%`
- bounce lock45 `+65.1%`

### C. `STREAM_PARITY`

用于判断 offline 与 Redis/OMS 是否同信号、同成交、同退出。
不能把 stream 的 tick 结果直接替换分钟 offline KPI。

### D. `DIAGNOSTIC_MIXED_INPUTS`

用于排查问题，不进入基线表：

- `june_july_daily_replay_20260714`
- 临时 raw context 覆盖 replay
- 同时改变 gate 与模型的 grid search

---

## 10. 后续唯一推荐的对比流程

### 10.1 重跑 July W1 V4 / FT56

只使用：

```bash
SKIP_TRAIN=1 \
  CKPT_FT=checkpoint/checkpoints_qqq_ft56_julw1/best.pth \
  bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh
```

并将输出写入新目录，禁止覆盖冻结目录。

### 10.2 若要正式比较 6 月 + 7 月

目前没有一条已经验收的“6–7 月完整 honest 一键脚本”。

必须先建立新脚本，要求：

1. V4 与 FT56 使用同一个 honest feature root
2. 使用同一套锁约和 option quote
3. 对整个 6–7 月重新 infer，不能拼接不同历史产物
4. 只附加 raw1 `put_gate`，不覆盖其他 infer context
5. 明确运行两种模式：
   - `monthly_reset`：每月独立验收
   - `continuous_state`：6 月状态连续进入 7 月
6. 保存 `config.json`，不能直接读取未来会变化的当前默认值而不落盘
7. 输出每日日收益、交易和 first divergence

在这条脚本完成之前，`+16.51% / +128.12%` 只能保留为诊断记录。

---

## 11. 权威文件索引

| 内容 | 路径 |
|---|---|
| V4 +65.09 冻结结果 | `qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json` |
| honest KPI 定义 | `qqq_btc/docs/honest_live_kpi_finetune_replay.md` |
| honest 一键脚本 | `qqq_btc/tools/train_ft56_julw1_honest_kpi.sh` |
| FT56 +49 hardcap | `qqq_btc/results/ft56_julw1_end240_hold55_hardcap_20260713/summary.json` |
| 同快照 V4/FT56 honest compare | `qqq_btc/results/ft56_julw1_honest_kpi_compare/summary.json` |
| 全版本目录（含 4–6 月金标） | `qqq_btc/docs/replay_versions_full_catalog.md` |
| 当前 replay 版本对照板（摘要） | `qqq_btc/docs/replay_version_kpi_board.md` |
| FT56 honest+VX 离线 | `qqq_btc/results/offline_live_aligned/ft56_julw1_honest_live_aligned/summary.json` |
| FT56 诚实流式 | `qqq_btc/results/july_w1_ft56_honest_live_parity_20260714/` |
| V4 +65 流式对拍完整过程 | `qqq_btc/docs/v4_july_w1_stream_parity_final_aligned_20260714.md` |
| V4 +65 直接流式对拍 | `qqq_btc/results/july_w1_v4_stream_final_aligned/` |
| V4+VX 离线 | `qqq_btc/results/offline_live_aligned/jun_jul_live_aligned/summary.json` |
| Jul13 VIXY grid | `qqq_btc/results/july10_13_polygon_fresh_old_lock/vixy_open_shock_regime_grid.csv` |
| 最近 isolated 临时 replay | `qqq_btc/results/vixy_open_shock_integrated_replay_20260714/summary.json` |
| 最近跨月 mixed replay | `qqq_btc/results/june_july_daily_replay_20260714/` |

---

## 12. 最终口径

在新的统一 6–7 月 honest 脚本完成前：

```text
V4 July W1 官方策略基线：
open30 + bounce_cut + lock45
acct25 = +65.0874%
```

```text
FT56 hardcap 历史基线：
acct25 = +49.0050%
注意：不是同一个 open30+bounce+lock45 策略快照
```

```text
june_july_daily_replay_20260714：
诊断性混合回放
不得用于替换或否定上述冻结基线
```

---

## 13. 统一 Strategy Profile（2026-07-14）

后续新增离线 replay 与流式对拍，应使用同一个版本化 strategy profile，并在两边
manifest 核对 `strategy_profile_id + strategy_profile_sha256`。

- V4 live-aligned：`qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json`
- FT56 honest logic parity（tick off）：`qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_parity_v1.json`
- FT56 production parity（tick enabled）：`qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_production_v1.json`
- 使用与验收说明：`qqq_btc/docs/strategy_profile_replay_stream_parity.md`

首次 profile 化回归结果：

- V4 Jul W1：`acct25=+55.67%`，7 笔（与原 offline live-aligned 结果一致）
- FT56 parity / production profile Jul W1：均为 `acct25=+22.33%`，9 笔，
  使用 `post_gatefix` infer + VX selector；离线引擎不执行 tick，因此两份 profile
  只在流式执行层分叉。

这两个结果的 profile、解析后 ReplayConfig、输入路径和 git provenance 均已写入各自
manifest；它们与旧的 `+65.09%`（无 VX 日切基线）不是同一 recipe。

---

## 14. 统一离线 Replay 现状、评级与流式对拍队列

> 本节合并另一轮 replay 审计、此前 Apr–Jul 结果、profile 化复跑和已有流式结果。
> 评级时间：2026-07-14。它是当前选择 replay 基线的主入口。

### 14.1 先定义“更好”

“收益更高”不等于“版本更好”。统一使用三条互相独立的轴：

1. **证据等级 E（Evidence）**
   - `E4`：冻结输入、独立入口、完整 manifest/profile hash、可从产物直接复核。
   - `E3`：主要输入和脚本可复现，但生成时 dirty，或仍缺一项完整冻结证据。
   - `E2`：历史 summary/消融可复核，但配置快照、窗口或 provenance 不完整。
   - `E1`：临时脚本、混合输入、字段覆盖或结果只能从叙述推导。
   - `E0`：开卷、无产物或不可复现。
2. **实盘贴近度 L（Live proximity）**
   - `L4`：FCS + OMS + tick 流式，三闸门通过。
   - `L3`：同生产模型、honest 特征、现门控；差别主要是分钟执行。
   - `L2`：门控接近，但模型、selector 或执行层仍不同。
   - `L1`：诚实历史基线/单变量消融，不代表现生产 recipe。
   - `L0`：old_lock、mixed、开卷或受损配置。
3. **流式优先级 P**
   - `P0`：当前必须重点对拍。
   - `P1`：用于闭合关键模型/门控问题。
   - `P2`：有研究价值，但不是当前上线阻塞项。
   - `P3`：归档或禁止进入正式对拍。

acct@25% 只作为**结果字段**记录，不参与 E/L 评分。不同模型、窗口、特征 lineage
之间禁止按收益排序。

### 14.2 当前主版本统一评级

| ID | 版本 | acct@25% | E | L | P | 统一结论 |
|---|---|---:|---:|---:|---:|---|
| **B** | FT56 honest + VX 离线 profile | **+22.33% / 9笔** | **E3** | **L3** | **P0** | 当前最佳离线生产基线；现 profile 的 `tick_exits=off` 适合先做逻辑对拍 |
| **S4** | FT56 + VX + tick 流式（profile 前） | **+19.23% / 7笔** | **E3** | **L4** | **P0** | 当前最贴近执行；三闸门 PASS，但不是现 `tick_exits=off` profile 的同 SHA run |
| **D7** | V4 honest + VX 离线 profile | **+55.67% / 7笔** | **E3** | **L2** | **P1** | V4 门控基线；适合隔离“模型 V4→FT56”影响 |
| **V0** | V4 honest、无 VX、open30+bounce+lock45 | **+65.09% / 12笔** | **E3** | **L1** | **P0** | 最高可信收益迁移主线；与 S3 做正式 gated 闭环 |
| **S3** | V0 历史流式、无 VX、tick off | **+63.66% / 13笔（Jul1–9）** | **E2** | **L3** | **P0** | 交易链已接近；`SKIP_GATES=1`，需升级为同 SHA PASS |
| **C-FT56** | FT56 honest、无 VX | **+58.80% / 15笔** | **E2** | **L1** | **P2** | 同快照模型对照；缺现生产 VX 日切 |
| **H** | FT56 hardcap 历史快照 | **+49.00% / 14笔** | **E2** | **L1** | **P3** | 配方不完整且不是现 profile，归档 |
| **D4–D6** | V4 + VX，Apr–Jun old_lock | **+214.84 / +175.84 / +68.41%** | **E2** | **L0** | **P3** | selector 上界研究；禁止当资金目标或流式金标 |
| **V1/V2** | V4 open30 / bounce 单变量消融 | +60.28 / +52.63% | **E2** | **L1** | **P3** | 解释 rails 机制，不单独流式 |
| **P60** | put_q10 + start_bar=60 过严消融 | 0% / +2.61% | **E2** | **L0** | **P3** | 有效反例：门控过严，不推进 |
| **G** | 全局 gap 污染 FT56 | ~+18.19% | **E1** | **L0** | **P3** | 已废弃，禁止作为模型评价 |
| **X** | Jun–Jul mixed 拼接 | V4 +16.51 / FT56 +128.12% | **E1** | **L0** | **P3** | 混合 lineage 诊断结果，禁止进入基线 |
| **I** | VIXY isolated 临时 replay | ~+66.44% | **E1** | **L0** | **P3** | 覆盖 context 的临时实验，禁止引用为正式收益 |
| **J13** | Jul13 old_lock 单日实验 | 0%～−4.8% | **E1** | **L0** | **P3** | 仅用于定位 OPEN_SHOCK/gap 时序 |

当前没有 `E4`：profile 化结果已经记录完整 SHA 和 resolved config，但 manifest
生成于 dirty working tree，且模型/外部数据尚未打成不可变数据版本。把当前分支、
checkpoint、feature root 和 VX 数据冻结后，V0/B/D7 才能升级为 `E4`。

### 14.3 哪些结果“更好”

按不同目标分别回答：

- **用于当前生产决策：B 最好。** 它使用 FT56、honest 输入、VX/quarantine 和统一
  profile，分钟级 `+22.33%`；已有流式同族 S4 为 `+19.23%`，但执行层启用了 tick，
  不是当前 `tick_exits=off` profile 的逐字段同版结果。
- **用于当前实际执行估计：S4 最好。** 它是唯一三闸门 PASS 的 FT56+VX+tick 结果，
  但旧产物生成早于统一 profile hash，因此应重跑后再升级为正式冻结证据。
- **用于判断 V4+VX 门控效果：D7 最好。** Jul 输入 honest、已有 profile manifest；
  它不能替代 FT56 生产预期。
- **用于迁移最高可信收益：V0 最好。** `+65.09%` 是无 VX honest 金标；历史 S3
  已做到 `+63.66%`，因此它应与 B/S4 并列 P0，但仍不能直接等同未来实盘收益。
- **纯收益最高的 D4/D5 并不是最佳版本。** 4–6 月使用 old_lock 非 honest 特征，
  因此只说明该历史输入下 selector 的乐观上界。

### 14.4 流式对拍优先级

#### P0-B：FT56 B ↔ 流式两阶段闭环（生产控制组）

统一使用：

```text
strategy_profile_id = ft56_honest_vx_parity_v1
profile tick exits  = off
offline baseline    = +22.33% / 9 trades
existing S4         = +19.23% / 7 trades / tick enabled / Gate1-3 PASS
```

现有 S4 是生产执行证据，但不是当前 parity profile 的同 SHA 证据。下一次应分两步：

1. **逻辑对拍**：直接用 `ft56_honest_vx_parity_v1`（tick off）跑流式，先隔离
   feature、entry、分钟 rails 和成交差异；
2. **生产对拍**：使用已冻结的 `ft56_honest_vx_production_v1`，再跑
   `disaster_only` tick 模式，对比 tick 带来的增量差异。

两步都必须核对：

1. offline/stream 的 `strategy_profile_sha256` 完全一致；
2. checkpoint、frozen norm、honest root、VX term 和日期窗口一致；
3. Gate-1 raw、Gate-2 normalized、Gate-3 trade 全部 PASS；
4. manifest 记录 env override；正式 run 不允许未记录覆盖；
5. 生产对拍逐日解释 9→7 笔和约 `−3.10pp` 差异，重点是 tick stop/leg lock
   与成交时序；逻辑对拍则不应把 tick 差异混入。

这是当前唯一直接服务 shadow/生产决策的重点对拍。

#### P1-A：V4+VX profile D7 ↔ 新 V4 流式

目标不是追求 `+55.67%`，而是控制 selector/执行后隔离模型差异：

```text
D7 V4+VX offline = +55.67% / 7 trades
B  FT56+VX offline = +22.33% / 9 trades
```

现有 `july_w1_v4_stream_final_aligned` 不能直接充当 D7 对拍：

- stream summary 是 `+57.76% / 15笔 / Jul1–10`；
- Gate-1、Gate-2 在历史命令中被 `SKIP_GATES=1` 跳过；
- 该 run 没有使用 D7 的同一 VX profile recipe。

因此需要用 `v4_vx_live_aligned_v1` 重新流式运行并落同 SHA manifest。

#### P0-A：V0 ↔ S3 最高可信收益迁移

文档记录 Jul1–9 `+63.66% / 13笔` 对离线 `+65.09% / 12笔`，但现有
`stream_summary_paired.json` 的直接 headline 是包含 Jul10 的 `+57.76% / 15笔`，
且 Gate-1/2 未执行。`+63.66%` 可作为历史分析结论，但还不是完整机器可验收产物。

历史结果已经表明它最有希望把高离线收益迁移到流式；正式重跑的目的，是把
`UNGATED` 提升为三闸门 PASS，并用统一 profile SHA 消除历史 env 分叉。

#### P2/P3：暂不流式

- C/H/V1/V2：历史无 VX 或 rails 消融，离线解释已足够。
- D4–D6：先重建 Apr–Jun honest 特征和冻结输入，再讨论流式；直接对拍 old_lock
  没有生产意义。
- X/I/G/J13/P60：诊断、污染或反例，禁止进入正式流式 KPI。

### 14.5 当前离线 Replay 的真实现状

1. **July 已形成可用主线**：V4/FT56 各有统一 profile，结果分别
   `+55.67%`、`+22.33%`，具备成对流式运行入口。
2. **Apr–Jun 仍没有统一 honest 生产基线**：`+214/+175/+68` 的输入是 old_lock；
   这些数字可用于历史 selector 消融，不能外推 live。
3. **V0 已升级为独立盈利迁移主线**：不直接替代 FT56 生产控制组，但应通过正式
   gated stream 判断其高收益能否迁移；C/H 仍只作历史基线。
4. **mixed/isolated 结果已完成降级**：`+128%`、`+66%` 不再进入正式排名。
5. **版本管理已有 profile/hash，但尚未完全冻结**：下一步关键不是继续增加 replay
   数字，而是 clean commit + immutable input lineage + 同 SHA stream run。

### 14.6 建议执行顺序

```text
1. 用 `v4_honest_v0_parity_v1` 正式闭合 V0↔S3（P0-A，高收益迁移）
2. 用 FT56 parity profile（tick off）重跑流式逻辑对拍（P0-B，生产控制）
3. 用 `ft56_honest_vx_production_v1` 复跑 S4 执行对拍（P0）
4. 重跑 V4+VX profile 流式（P1-A），隔离模型差异
5. Apr–Jun 先重建 honest 输入，再产生新离线基线
6. 不再为 X/I/G/D4–D6 直接安排流式对拍
```

### 14.7 权威证据路径

| 对象 | 路径 |
|---|---|
| B profile 离线 | `qqq_btc/results/offline_live_aligned/ft56_profile_v1_july/summary.json` |
| B profile manifest | `qqq_btc/results/offline_live_aligned/ft56_profile_v1_july/manifest.json` |
| FT56 production profile 离线 | `qqq_btc/results/offline_live_aligned/ft56_production_profile_v1_july/summary.json` |
| FT56 production profile manifest | `qqq_btc/results/offline_live_aligned/ft56_production_profile_v1_july/manifest.json` |
| S4 流式 PASS | `qqq_btc/results/july_w1_ft56_honest_live_parity_20260714/stream_summary_paired.json` |
| D7 profile 离线 | `qqq_btc/results/offline_live_aligned/v4_profile_v1_july/summary.json` |
| D7 profile manifest | `qqq_btc/results/offline_live_aligned/v4_profile_v1_july/manifest.json` |
| V0 冻结离线 | `qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json` |
| V4 现有 ungated 流式 | `qqq_btc/results/july_w1_v4_stream_final_aligned/stream_summary_paired.json` |
| D4–D6 old_lock | `qqq_btc/results/offline_live_aligned/apr_may_jun_live_aligned/summary.json` |
| X mixed | `qqq_btc/results/june_july_daily_replay_20260714/summary.json` |

---

## 15. V0 `+65.09%` 历史流式对拍同步与正式闭环方案

完整历史记录已同步纳入主谱系：

- `qqq_btc/docs/v4_july_w1_stream_parity_final_aligned_20260714.md`
- 产物：`qqq_btc/results/july_w1_v4_stream_final_aligned/`

### 15.1 历史 S3 实际已经做到什么

历史命令通过 env 临时拼出 V0 配方：

```text
checkpoint = V4 best.pth
selector   = off
tick exits = off
q10        = -0.2
recipe     = open30 + bounce_cut + lock45
```

可比区间必须是 Jul1–9：

| 结果 | acct@25% | 笔数 | 状态 |
|---|---:|---:|---|
| V0 离线 | **+65.0874%** | 12 | 冻结 baseline |
| S3 流式 Jul1–9 | **+63.6575%** | 13 | 交易链基本对齐 |
| 差异 | **−1.4299pp** | +1 | 主要为 Jul7 临界 CALL |
| S3 流式 Jul1–10 | +57.7590% | 15 | 含不可比 Jul10，仅 diagnostic |

`stream_summary_paired.json` 的 headline 是 Jul1–10 `+57.76%`；从
`trades_detail` 排除 Jul10 两笔后得到 Jul1–9 `+63.66%`。两者不矛盾。

### 15.2 Gate 状态的正确解释

历史命令设置了 `SKIP_GATES=1`。因此：

```text
gate1_raw.pass  = false
gate2_norm.pass = false
parity_status   = UNGATED
```

这里的 `false` 是“没有执行”，不是“实际比较失败”。所以历史证据可以证明
交易链高度接近，但不能标为三闸门正式 PASS。

### 15.3 已经解决的核心流式分叉

这轮历史工作不是简单改配置，而是补齐了 V0 实时接线：

1. `vwap_log_return`、同标签 `spot_close` 进入 live context；
2. bounce 输入在 V0 前置门控前持续记录，并按分钟去重；
3. `entry_spot`、持仓期 spot closes 跨分钟持久化；
4. 开仓时携带入场前 momentum 历史；
5. Jul1 PUT 恢复为 `sb77 → sb80 SPOT_THESIS`；
6. signal/replay 工具统一到 `LIVE_REPLAY + q10=-0.2`。

这些修复已在当前 live 代码中，正式重跑可以直接复用，不需要重新实现 bounce 链。

### 15.4 当前已固化的 V0 profile

新增：

```text
qqq_btc/CONFIG/strategy_profiles/v4_honest_v0_parity_v1.json
```

关键口径：

```text
model              = V4
infer              = v4_jul_w1_fixed5m_infer/test_infer.parquet
honest raw1        = July W1 honest openwin
base               = LIVE_REPLAY
edge_q10_floor     = -0.2
selector           = off
cross-day defenses = off
tick exits         = off
```

同 profile 离线回归已完成：

```text
output  = qqq_btc/results/offline_live_aligned/v4_honest_v0_parity_v1_offline/
acct25  = +65.09%
trades  = 12
profile = TREND_PUT_OK 7 days
```

这证明 profile 没有把 V0 错接到 D7 的 VX/CHOP 路径。

### 15.5 正式流式入口

新增严格包装脚本：

```bash
bash qqq_btc/tools/run_v4_v0_stream_parity.sh
```

它固定：

- `v4_honest_v0_parity_v1`
- 仅 Jul1、2、6、7、8、9
- 清理可能覆盖 profile 的 `QQQ_BTC_*`
- 禁止 `SKIP_GATES` / `FORCE_GATE3`
- 调用现有 honest 三闸门 FCS→Redis→OMS 链

可指定新输出目录：

```bash
V0_STREAM_OUT_DIR="$PWD/qqq_btc/results/july_w1_v4_v0_gated_20260714" \
  bash qqq_btc/tools/run_v4_v0_stream_parity.sh
```

### 15.6 正式验收条件

| 层 | 必须满足 |
|---|---|
| 版本 | offline/stream `strategy_profile_id` 和 SHA 完全相同 |
| Gate-1 | raw feature compare 实际执行并 PASS |
| Gate-2 | normalized feature compare 实际执行并 PASS |
| Gate-3 | `parity_status=PASS`，不是 `UNGATED` |
| 窗口 | 只统计 Jul1–9，不包含 Jul10 |
| 收益 | 流式与 +65.0874% 的差异建议不超过 ±2pp |
| 笔数 | 12±1；额外交易必须逐笔解释 |
| 关键路径 | Jul1 SPOT_THESIS、Jul2 大 PUT、Jul6 bounce/午后 PUT、Jul7 大 PUT 对齐 |
| 证据 | manifest 包含 git、dirty、profile snapshot、checkpoint/input 元数据 |

### 15.7 首个重点残差

历史 S3 的主要差异是 Jul7 `sb217 CALL`：

```text
stream edge ≈ 0.199515
dynamic th  ≈ 0.199217  → 入场
offline edge≈ 0.194557  → 不入场
trade result = -7.23%
```

正式 Gate1/2 跑完后：

- 若 raw/norm 不过：先修 FCS 与 honest parquet 的特征偏差；
- 若 Gate1/2 通过但该 edge 仍不同：检查模型输入标签、timestamp 和动态分位状态；
- 不应先增加新 gate 强行删除该笔，否则会再次把“特征差异”伪装成“策略优化”。

### 15.8 实施判断

V0 不再只是“值得尝试的高收益离线版本”。已有历史 S3 证明其关键交易路径能进入
FCS/OMS，Jul1–9 仅差 `1.43pp`。当前缺口主要是**证据冻结**：

```text
旧状态：高收益基本复现，但 UNGATED + env 配方
目标状态：同 profile SHA + Gate1/2/3 PASS + Jul1–9 独立 summary
```

因此 V0↔S3 应与 FT56 生产控制组并列 P0，并优先安排正式 gated 重跑。
