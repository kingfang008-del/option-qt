# Replay 全版本目录：改了什么 × 结果是什么

> 更新：2026-07-14  
> 本文把已知离线 / 流式 replay **按版本列全**：配方改动、数字、产物路径、能否当实盘预期。  
> 详细对账叙事见：`qqq_btc/docs/replay_version_lineage_and_result_reconciliation.md`  
> **统一 E/L/P 评级和流式优先级以谱系文档 §14 为准。**  
> **禁止**把不同版本的 acct% 直接横比高低。

---

## 0. 一眼总表

| ID | 名称 | 窗口 | acct@25% | 笔数 | 贴近实盘 | 分级 |
|---|---|---|---:|---:|---|---|
| **S4** | FT56 诚实流式 + VX + tick | Jul W1 | **+19.23%** | 7 | **最贴近** | STREAM_PARITY |
| **B** | FT56 honest 离线 + VX + quarantine | Jul W1 | **+22.33%** | 9 | 离线最贴近 | LIVE_ALIGNED |
| **C** | FT56 honest 离线（无 VX） | Jul1–10 | +58.80% | 15 | 特征诚实，缺日切 | HISTORICAL |
| **H** | FT56 hardcap 历史快照 | Jul1–10 | +49.00% | 14 | 历史调参，非现生产 | OFFICIAL_HONEST_FROZEN* |
| **V0** | V4 open30+bounce+lock45（无 VX） | Jul W1 | **+65.09%** | 12 | V4 离线金标，非现 FT56 栈 | OFFICIAL_HONEST_FROZEN |
| **V1** | V4 open30 only | Jul W1 | +60.28% | — | 消融 | CONTROLLED_ABLATION |
| **V2** | V4 open30+bounce（无 lock） | Jul W1 | +52.63% | — | 消融 | CONTROLLED_ABLATION |
| **D7** | V4 + VX + quarantine | Jul W1 | +55.67% | 7 | 门控像现生产，模型仍是 V4 | LIVE_ALIGNED_V4 |
| **D4** | V4 + VX + quarantine | 2026-04 | **+214.84%** | 57 | old_lock，偏乐观 | LIVE_ALIGNED_V4 |
| **D5** | 同上 | 2026-05 | **+175.84%** | 49 | 同上 | LIVE_ALIGNED_V4 |
| **D6** | 同上 | 2026-06 | **+68.41%** | 34 | 同上 | LIVE_ALIGNED_V4 |
| **S3** | V4 流式对拍（对 V0，历史 UNGATED） | Jul1–9 | +63.66% | 13 | 已基本复现，待正式 Gate | STREAM_PARITY_DIAGNOSTIC |
| **G** | 全局 gap=0.001 污染后 FT56 | Jul W1 | ~+18.19% | — | 受损配置，已废弃 | DAMAGED |
| **P60** | put_q10 + start_bar=60 过严消融 | Jul W1 | FT56 0% / V4 +2.6% | 0/1 | 消融，非默认 | CONTROLLED_ABLATION |
| **J13** | Jul13 FT56 old_lock + 因果 VIX 修复 | 2026-07-13 | 0%～−4.8%（视 gap） | 0–2 | 单日实验 | DIAGNOSTIC |
| **X** | 6–7 月 mixed 临时拼接 | 6/1–7/13 | V4 +16.5% / FT56 +128% | 60/57 | **非正式** | DIAGNOSTIC_MIXED |

\* hardcap 可归档，但不是 open30+bounce+lock45 同快照。

---

## 1. 贴近实盘怎么排

```text
1) S4 流式 +19.23%     ← 报「实盘大概怎样」用这个
2) B  离线 FT56+VX +22.33%
3) C  旧 honest 无 VX +58.8%   ← 只作历史对照
4) D7 V4+VX +55.67% / D4–D6 四月金标  ← 门控研究，不作资金预期
5) V0 +65.09%                  ← V4 策略冻结金标，回答「原 V4 离线」
```

---

## 2. 各版本详表（改动 → 结果 → 路径）

### S4 — FT56 诚实流式对拍（当前生产栈）

**相对基线改了什么**

| 维度 | 内容 |
|---|---|
| 模型 | FT56 `checkpoints_qqq_ft56_julw1/best.pth` |
| 特征 | FCS 自算 + `frozen_norm_qqq_daily.npz`；无 greek-parity / 无 regime gold |
| put_gate | 实盘 `vixy_z` |
| 日切 | `QQQ_BTC_RULE_PROFILE_SELECTOR=vx`（CHOP/TREND/…） |
| 执行 | Redis 1s + OMS mock；含 `TICK_FAST_HARD` / leg_lock |
| 相对 V0 | 换模型、加 VX、换 tick 出场 |

**结果（Jul W1）**：`+19.23%`，7 笔，hit 71.4%，PUT6/CALL1

| 日期 | leg | sb | exit | net |
|---|---|---:|---|---:|
| 07-01 | PUT | 77 | TICK_FAST_HARD | −18.36% |
| 07-02 | CALL | 19 | STEP_PROTECT | +11.41% |
| 07-02 | PUT | 60 | MAX_HOLD | +65.00% |
| 07-02 | PUT | 187 | MAX_HOLD | +4.81% |
| 07-07 | PUT | 43 | STEP_PROTECT | +10.00% |
| 07-07 | PUT | 81 | EARLY_STOP | −14.11% |
| 07-10 | PUT | 61 | TRAILING | +17.74% |

日@25%：7/1 −4.59% → 7/2 +21.0% → 7/7 −1.12% → 7/10 +4.44%；7/8–9 CHOP=0。

**路径 / 入口**

- 目录：`qqq_btc/results/july_w1_ft56_honest_live_parity_20260714/`
- 汇总：`.../stream_summary_paired.json`
- 脚本：`bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh`

---

### B — FT56 honest 离线 + VX + quarantine

**相对 C（旧 honest）改了什么**

| 改动 | 说明 |
|---|---|
| + VX selector | 7/8–9 → `CHOP_NO_TRADE` |
| + PUT quarantine | 前日 PUT sleeve≤−2% 且 VX slope≥6% |
| LIVE_REPLAY 现金标 | 含 early cross-confirm 等（工作区当前默认） |
| 仍用 | honest openwin 特征 + 因果 1m put_gate |

**结果（Jul W1）**：regime `+22.33%` / 9 笔；纯 TREND baseline `+26.28%` / 14 笔

成交摘要：7/1 PUT@77/−10.5% + @191/+9.1%；7/2 CALL@19、PUT@60/+65%、PUT@176；7/6 多笔 CALL/PUT 互有盈亏。

**路径**：`qqq_btc/results/offline_live_aligned/ft56_julw1_honest_live_aligned/summary.json`

---

### C — FT56 / V4 honest KPI compare（无 VX）

**改了什么**：诚实特征 + 因果 put_gate + `LIVE_REPLAY` q10=-0.2 + 当时 EXIT_RAILS；**未**开 VX/CHOP。

**结果**

| 模型 | 窗口 | acct@25% | 笔数 | hit |
|---|---|---:|---:|---:|
| FT56 | jul1–10 | **+58.80%** | 15 | 66.7% |
| V4 | jul1–10 | **+65.09%** | 12 | 75.0% |
| FT56 | jul1–8 | +57.11% | 13 | — |
| V4 | jul1–8 | +63.33% | 10 | — |

**路径**：`qqq_btc/results/ft56_julw1_honest_kpi_compare/summary.json`  
**定位**：历史同快照对照；**不是**现生产（缺 VX）最优预期。

---

### H — FT56 hardcap 历史快照

**改了什么**：`session_entry_end_bar=240`，`max_hold_bars=55`，`q10=-0.2`；**尚未**走完整 open30+bounce+lock45。

**结果**：Jul1–10 `+49.00%`，14 笔；Jul1 仍含 09:46 PUT −22.6%。

**路径**：`qqq_btc/results/ft56_julw1_end240_hold55_hardcap_20260713/summary.json`

---

### V0 / V1 / V2 — V4 open30 / bounce / lock45 消融族

**共同前提**：V4 honest infer + 因果 put_gate + LIVE q10=-0.2。

| ID | 相对改动 | acct@25% |
|---|---|---:|
| V1 | 仅 `put_early_open30_max_min=0`（挡无结构早盘 PUT） | +60.28% |
| V2 | + `bounce_cut` / `SPOT_THESIS` 提前证伪退出 | +52.63% |
| **V0** | + `thesis_lock_leg_bars=45`（证伪后同腿锁 45 分钟） | **+65.09%** |

**V0 日收益（冻结）**

| 日期 | day@25% |
|---|---:|
| 07-01 | −0.42% |
| 07-02 | +16.24% |
| 07-06 | +7.08% |
| 07-07 | +25.61% |
| 07-08 | +4.92% |
| 07-09 | +1.07% |
| **累计** | **+65.09%**（12 笔；07-10 不纳入冻结可比） |

**路径**：`qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json`  
**身份**：V4 官方离线策略基线；**不是**当前 FT56+VX+tick 收益承诺。

---

### D4 / D5 / D6 / D7 — V4 offline_live_aligned（VX 金标配方）

**相对 V0 改了什么**

| 改动 | 说明 |
|---|---|
| + VX 日切 | `selector_source=vx` → OPEN_DEFENSE / CHOP / TREND |
| + PUT quarantine | −2% PUT sleeve + vx_slope≥6% |
| LIVE_REPLAY 当前默认 | early cross-confirm 等 |
| 4–6 月特征 | `*_v4_old_lock`（**非** honest openwin） |
| 7 月特征 | `july_w1_v4_honest_openwin`（仅 W1） |
| 模型 | 仍为 **V4** infer |

**入口**：`bash qqq_btc/tools/replay_offline_live_aligned.sh`  
（引擎：`replay_offline_live_aligned.py` → `run_month()`；带 `manifest.json` 版本）

#### D4–D6：2026-04 / 05 / 06

目录：`qqq_btc/results/offline_live_aligned/apr_may_jun_live_aligned/`

| 月 | acct@25% | 笔数 | MDD | profile | vs 纯 TREND |
|---|---:|---:|---:|---|---:|
| **2026-04** | **+214.84%** | 57 | −6.81% | OPEN_DEFENSE 6 / TREND 15 | +67.5pp |
| **2026-05** | **+175.84%** | 49 | −8.40% | TREND 19 / CHOP 1 | +111.3pp |
| **2026-06** | **+68.41%** | 34 | −13.03% | TREND 16 / CHOP 5 | +28.5pp |

#### D7：2026-07 W1

目录：`qqq_btc/results/offline_live_aligned/jun_jul_live_aligned/`  
（另有同配方 Jul 汇总：`qqq_btc/results/july2026_offline_live_aligned/summary.json`）

| 月 | acct@25% | 笔数 | MDD | profile | vs 纯 TREND |
|---|---:|---:|---:|---|---:|
| 2026-06 | +68.41% | 34 | −13.03% | 同上 | +28.5pp |
| **2026-07 W1** | **+55.67%** | 7 | −0.42% | TREND 5 / CHOP 2 | −9.4pp |

为何 V0 +65.09% → D7 +55.67%：Jul8/Jul9 被 CHOP 置空（日收益路径 Jul1–7 与 V0 一致）。

| 日期 | V0 无 VX | D7 + VX |
|---|---:|---:|
| 07-01 | −0.42% | −0.42% |
| 07-02 | +16.24% | +16.24% |
| 07-06 | +7.08% | +7.08% |
| 07-07 | +25.61% | +25.61% |
| 07-08 | +4.92% | **0%（CHOP）** |
| 07-09 | +1.07% | **0%（CHOP）** |
| 累计 | +65.09% | +55.67% |

**git（本批 manifest）**：`bc83432`（working tree dirty）。

---

### S3 — V4 流式对拍（验证 V0）

**改了什么**：把 V0 配方接到 FCS/OMS 流式，检验能否复现 +65.09%。

**历史分析结果**：Jul1–9 流式 **+63.66%** / 13 笔 vs 离线
**+65.09%** / 12 笔（差 −1.43pp，主要 Jul7 临界 edge）。

证据限制：现有 `stream_summary_paired.json` 的直接 headline 是包含 Jul10 的
`+57.76% / 15笔`；历史命令用了 `SKIP_GATES=1`，所以 Gate-1/2 状态为 false
表示**未执行**，不是实测失败。`+63.66%` 尚未形成独立 gated summary，因此按
谱系文档 §15 归入 P0 高收益迁移闭环，而不是当前正式 PASS 基线。

**路径**：`qqq_btc/results/july_w1_v4_stream_final_aligned/`

**当前正式入口**

```bash
# 同 profile 离线应复现 +65.09%
python qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/v4_honest_v0_parity_v1.json \
  --out-name v4_honest_v0_parity_v1_offline

# Jul1–9 三闸门流式；包装器禁止 SKIP_GATES/FORCE_GATE3
bash qqq_btc/tools/run_v4_v0_stream_parity.sh
```

---

### S1 / S2 — 更早流式对拍（过程版本）

| ID | 要点 | 结果量级 |
|---|---|---|
| S1 | frozen norm 早期同口径；证 offline≈stream | 双方约 −13%（对齐成功，策略差） |
| S2 | honest 三闸门中间版 | 全周流式约 +10%；确认问题在决策/腿选择 |

产物示例：`july_w1_ft56_4c_stream/`、`july_w1_ft56_honest_3gate_week_g12pass/`。

---

### G — 全局 `min_dual_leg_edge_gap=0.001`（已废弃污染）

**改了什么**：把为 Jul13 设计的双腿 gap 门控 **全局**套到历史日。

**结果**：FT56 Jul W1 掉到约 **+18.19%**（误杀 Jul7 等）。  
**现状**：全局 gap 已关；仅考虑 OPEN_SHOCK 局部（未完成全历史验收）。

---

### P60 — put_q10 + start_bar=60 过严消融

**改了什么**：PUT 也吃 q10 + 入场起点提到 bar60。

**结果**：FT56 **0 笔 / 0%**；V4 仅 **+2.61%** / 1 笔。  
**路径**：`qqq_btc/results/ft56_julw1_honest_kpi_post_gatefix/summary.json`

---

### J13 — Jul13 单日实验

**改了什么**：补 VIXY/`vix_level` 因果后重 infer；试 gap / early cross-confirm。

| 子版本 | acct@25% | 笔数 |
|---|---:|---:|
| gap=0.001（某次） | 0% 或约 −4.8% | 0–1 |
| 无 gap oracle | 更差（误开 CALL） | 2 |

**路径**：`qqq_btc/results/ft56_jul13_old_lock_massive/`；补数文档：`docs/patch_vixy_features.md`

---

### X — 6–7 月 mixed 临时拼接（非正式）

**改了什么**：临时 Python 拼不同 infer lineage、覆盖多个 context 字段、跨月连续状态。

**结果**：V4 6/1–7/13 **+16.51%**（60 笔）；FT56 **+128.12%**（57 笔）。  
**分级**：`DIAGNOSTIC_MIXED_INPUTS` — **不得**替代 V0/H/S4。  
**路径**：`qqq_btc/results/june_july_daily_replay_20260714/`

---

## 3. 配方对照矩阵（核心开关）

| 版本 | 模型 | 诚实特征 | 因果 put_gate | LIVE | VX/CHOP | quarantine | 执行 |
|---|---|---|---|---|---|---|---|
| S4 | FT56 | 流式 | vixy_z | ✓ | ✓ | ✓(live) | tick |
| B | FT56 | ✓ | ✓ | ✓ | ✓ | ✓ | 分钟 |
| C | FT56/V4 | ✓ | ✓ | ✓ | ✗ | ✗ | 分钟 |
| H | FT56 | ✓* | ✓* | ✓ | ✗ | ✗ | 分钟 |
| V0 | V4 | ✓ | ✓ | ✓ | ✗ | ✗ | 分钟 |
| D4–D7 | V4 | 7月✓ / 4–6 old_lock | ✓ | ✓ | ✓ | ✓ | 分钟 |
| X | 混 | 混 | 部分 | ✓ | ? | ? | 分钟 |

---

## 4. 复现命令

```bash
# B: FT56 honest+VX profile 离线
python qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_parity_v1.json \
  --out-name ft56_honest_vx_parity_v1_offline

# S4: FT56 production profile 流式（tick enabled，最贴实盘）
HONEST_OUT_DIR="$PWD/qqq_btc/results/july_w1_ft56_honest_live_parity_YYYYMMDD" \
QQQ_BTC_STRATEGY_PROFILE="$PWD/qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_production_v1.json" \
  bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh

# D4–D7 V4+VX 离线（带 profile manifest）
bash qqq_btc/tools/replay_offline_live_aligned.sh \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json \
  --months 2026-04,2026-05,2026-06 --out-name apr_may_jun_live_aligned
bash qqq_btc/tools/replay_offline_live_aligned.sh \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json \
  --months 2026-06,2026-07 --out-name jun_jul_live_aligned

# V0 / C 类 honest KPI（标准一键；默认无强制 VX）
SKIP_TRAIN=1 bash qqq_btc/tools/train_ft56_julw1_honest_kpi.sh

# 只看离线脚本 provenance
python qqq_btc/tools/replay_offline_live_aligned.py --print-version
```

研究消融引擎（**默认不是** D 金标）：

```bash
python qqq_btc/tools/replay_regime_profiles_apr_jul.py --skip-build
# 默认 selector=vixy、quarantine 关 —— 与 D4–D7 不同
```

---

## 5. 权威产物索引

| 内容 | 路径 |
|---|---|
| 本文（全版本目录） | `qqq_btc/docs/replay_versions_full_catalog.md` |
| 谱系对账长文 | `qqq_btc/docs/replay_version_lineage_and_result_reconciliation.md` |
| V0 +65.09 | `qqq_btc/results/v4_jul_w1_honest_kpi_replay/summary.json` |
| C +58.8 / +65.09 compare | `qqq_btc/results/ft56_julw1_honest_kpi_compare/summary.json` |
| H +49 | `qqq_btc/results/ft56_julw1_end240_hold55_hardcap_20260713/summary.json` |
| D4–D6 | `qqq_btc/results/offline_live_aligned/apr_may_jun_live_aligned/` |
| D6–D7 | `qqq_btc/results/offline_live_aligned/jun_jul_live_aligned/` |
| B FT56+VX | `qqq_btc/results/offline_live_aligned/ft56_julw1_honest_live_aligned/` |
| S4 流式 | `qqq_btc/results/july_w1_ft56_honest_live_parity_20260714/` |
| S3 V4 流式 | `qqq_btc/results/july_w1_v4_stream_final_aligned/` |
| X mixed | `qqq_btc/results/june_july_daily_replay_20260714/` |
| 诚实 KPI 定义 | `qqq_btc/docs/honest_live_kpi_finetune_replay.md` |
| VIXY 补数 | `qqq_btc/docs/patch_vixy_features.md` |

---

## 6. 使用纪律（短）

1. 报实盘预期 → **只认 S4（+19%）**，辅以 B（+22%）。  
2. 报原 V4 离线策略 → **认 V0（+65.09%）**；S3 证明可流式复现。  
3. D4–D6 的 +214/+175/+68 → **V4+VX+old_lock 上界**，不作资金目标。  
4. C 的 +58.8% → 无 VX 的旧 honest，不作现生产最优。  
5. X 的 +128% → 诊断垃圾，禁止进基线。  
6. 正式 run 必须留 `manifest` / summary，并写清 git dirty。
