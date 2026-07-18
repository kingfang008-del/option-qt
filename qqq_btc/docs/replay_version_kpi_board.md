# 离线 / 流式 Replay 版本对照（摘要）

> 更新日期：2026-07-14  
> **完整目录（含 4/5/6 月 +214/+175/+68、各版本改动与成交明细）**：  
> [`qqq_btc/docs/replay_versions_full_catalog.md`](./replay_versions_full_catalog.md)  
> 谱系对账长文：[`replay_version_lineage_and_result_reconciliation.md`](./replay_version_lineage_and_result_reconciliation.md)  
> **统一证据等级、实盘贴近度与流式优先级见谱系文档 §14。**  
> 不要把不同版本的 acct% 直接横比高低。

---

## 1. 先看结论：对拍优先级与生产基线

| 优先级 | 版本 | July W1 acct@25% | 贴近实盘？ | 用途 |
|---:|---|---:|---|---|
| **1A** | **V0 V4 honest 无 VX → S3** | **离线 +65.09% / 历史流式 +63.66%** | 最高可信收益迁移；历史流式为 UNGATED | 正式 gated 重跑 |
| **1B** | **诚实流式对拍** FT56 | **+19.23%** | **最贴当前生产**（FCS+OMS+tick） | 生产控制组 |
| **2** | **FT56 honest 离线** + VX/quarantine | **+22.33%** | **当前生产离线基线** | 同模型/诚实特征 |
| 3 | V4 offline_live_aligned（7月 W1） | +55.67% | 门控像，模型仍是 V4 | 隔离模型/VX |
| 4 | FT56 honest 离线（旧，无 VX） | +58.80% | 特征诚实，但缺日切 | 历史对照 |
| 5 | V4 offline_live_aligned（4–6 old_lock） | +214 / +175 / +68 | **不贴近实盘预期** | 上界参考，禁止当目标 |

**一句话**

- 问「最高可信收益能否迁移到实时」→ 正式闭合 **V0 +65.09% ↔ S3 +63.66%**。  
- 问「当前生产大概怎样」→ 看 **FT56 流式 +19%**，其次离线 **+22%**。  
- 问「V4 + VX 金标长什么样」→ 看 **4–6 月 +214/+175/+68、7W1 +55%**，不要当实盘。

---

## 2. 版本定义（配方清单）

### A. 诚实流式对拍（最贴近实盘）

| 项 | 值 |
|---|---|
| 脚本 | `qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh` |
| 模型 | `checkpoint/checkpoints_qqq_ft56_julw1/best.pth` |
| 特征 | FCS 自算 + `frozen_norm_qqq_daily.npz`（无 greek-parity / 无 regime gold） |
| put_gate | `vixy_z` |
| 日切 | `QQQ_BTC_RULE_PROFILE_SELECTOR=vx` |
| 执行 | Redis 1s stream + tick exit / leg_lock |
| 结果目录 | `qqq_btc/results/july_w1_ft56_honest_live_parity_20260714/` |
| 汇总 | `.../stream_summary_paired.json` |

**July W1：+19.23%，7 笔，hit 71.4%，PUT6 / CALL1**

| 日期 | leg | entry_sb | exit | net |
|---|---|---:|---|---:|
| 2026-07-01 | PUT | 77 | TICK_FAST_HARD | −18.36% |
| 2026-07-02 | CALL | 19 | STEP_PROTECT | +11.41% |
| 2026-07-02 | PUT | 60 | MAX_HOLD | +65.00% |
| 2026-07-02 | PUT | 187 | MAX_HOLD | +4.81% |
| 2026-07-07 | PUT | 43 | STEP_PROTECT | +10.00% |
| 2026-07-07 | PUT | 81 | EARLY_STOP | −14.11% |
| 2026-07-10 | PUT | 61 | TRAILING | +17.74% |

日收益（@25%）：7/1 −4.59% → 7/2 +21.0% → 7/7 −1.12% → 7/10 +4.44%；7/8–9 CHOP 空仓。

---

### B. FT56 honest 离线 + 现金标门控（离线最贴近）

| 项 | 值 |
|---|---|
| 模型 | FT56（同上 ckpt） |
| 特征 | `july_w1_v4_honest_openwin` |
| put_gate | raw 1min `vix_level`，`+1min` asof（因果） |
| 门控 | `LIVE_REPLAY` + `edge_q10_floor=-0.2` + **VX selector** + PUT quarantine(−2%, vx≥6%) |
| infer | `qqq_btc/results/ft56_julw1_honest_infer_fixed5m_post_gatefix/test_infer.parquet` |
| 结果 | `qqq_btc/results/offline_live_aligned/ft56_julw1_honest_live_aligned/summary.json` |

**July W1：+22.33%，9 笔，MDD −0.42%**（纯 TREND baseline +26.28% / 14 笔）

| 日期 | bar | leg | exit | net |
|---|---:|---|---|---:|
| 2026-07-01 | 77 | PUT | SPOT_THESIS | −10.53% |
| 2026-07-01 | 191 | PUT | STEP_PROTECT | +9.07% |
| 2026-07-02 | 19 | CALL | STEP_PROTECT | +9.70% |
| 2026-07-02 | 60 | PUT | MAX_HOLD | +64.96% |
| 2026-07-02 | 176 | PUT | MAX_HOLD\|NO_QUOTE | +14.22% |
| 2026-07-06 | 15 | CALL | MAX_HOLD | +22.12% |
| 2026-07-06 | 88 | PUT | SPOT_THESIS | −9.01% |
| 2026-07-06 | 135 | CALL | TIME_STOP | −1.04% |
| 2026-07-06 | 176 | CALL | EARLY_STOP | −12.60% |

与流式差距主要来自：tick 硬止损 / leg_lock、分钟填价 vs 1s、个别 bar edge 微差。

---

### C. FT56 honest 离线（旧版，无 VX）— 历史数字

| 项 | 值 |
|---|---|
| 结果 | `qqq_btc/results/ft56_julw1_honest_kpi_compare/summary.json` |
| 配方 | honest + 因果 put_gate + `LIVE_REPLAY` q10=-0.2 + EXIT_RAILS；**无 VX/CHOP** |

**July 1–10：+58.80%，15 笔，hit 66.7%**（PUT11 / CALL4）

> 特征已诚实，但缺现生产日切；**不要**再用 +58% 当现预期。

---

### D. V4 offline_live_aligned（门控金标，数据偏乐观）

| 项 | 值 |
|---|---|
| 入口 | `bash qqq_btc/tools/replay_offline_live_aligned.sh` |
| 引擎 | `qqq_btc/tools/replay_offline_live_aligned.py` → `run_month()` |
| 模型 | **V4** infer（非 FT56） |
| put_gate | 因果 1m `vix_level` |
| 门控 | `LIVE_REPLAY` + VX + PUT quarantine（与 B 同门控族） |
| 4–6 特征 | `*_v4_old_lock`（**非** honest openwin） |
| 7 月特征 | `july_w1_v4_honest_openwin`（仅 W1） |
| git（本批） | `bc83432`（working tree dirty） |

#### D1. 2026-04 / 05 / 06

目录：`qqq_btc/results/offline_live_aligned/apr_may_jun_live_aligned/`

| 月 | acct@25% | 笔数 | MDD | profiles | vs 纯 TREND |
|---|---:|---:|---:|---|---:|
| 2026-04 | **+214.84%** | 57 | −6.81% | OPEN_DEFENSE 6 / TREND 15 | +67.5pp |
| 2026-05 | **+175.84%** | 49 | −8.40% | TREND 19 / CHOP 1 | +111.3pp |
| 2026-06 | **+68.41%** | 34 | −13.03% | TREND 16 / CHOP 5 | +28.5pp |

#### D2. 2026-06 / 07(W1)

目录：`qqq_btc/results/offline_live_aligned/jun_jul_live_aligned/`

| 月 | acct@25% | 笔数 | MDD | profiles | vs 纯 TREND |
|---|---:|---:|---:|---|---:|
| 2026-06 | +68.41% | 34 | −13.03% | 同上 | +28.5pp |
| 2026-07 W1 | **+55.67%** | 7 | −0.42% | TREND 5 / CHOP 2 | −9.4pp |

7 月日明细（V4+VX）：

| 日期 | n | day@25% | cum@25% | legs |
|---|---:|---:|---:|---|
| 2026-07-01 | 2 | −0.42% | −0.42% | PUT2 |
| 2026-07-02 | 1 | +16.24% | +15.75% | PUT1 |
| 2026-07-06 | 2 | +7.08% | +23.94% | PUT2 |
| 2026-07-07 | 2 | +25.61% | +55.67% | PUT1 CALL1 |

---

## 3. July W1 横比（同一窗口）

| 版本 | acct@25% | 笔数 | 模型 | 诚实特征 | VX | 执行 |
|---|---:|---:|---|---|---|---|
| 流式对拍 | **+19.23%** | 7 | FT56 | 流式自算 | ✓ | tick/OMS |
| FT56 honest+VX 离线 | **+22.33%** | 9 | FT56 | ✓ | ✓ | 分钟 replay |
| FT56 honest 旧（无 VX） | +58.80% | 15 | FT56 | ✓ | ✗ | 分钟 |
| V4+VX 离线 | +55.67% | 7 | V4 | ✓(W1) | ✓ | 分钟 |

---

## 4. 怎么复现

```bash
# V0/S3：最高可信收益正式流式闭环
python qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/v4_honest_v0_parity_v1.json \
  --out-name v4_honest_v0_parity_v1_offline
bash qqq_btc/tools/run_v4_v0_stream_parity.sh

# A) FT56 honest+VX 离线（同 profile）
python qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_parity_v1.json \
  --out-name ft56_honest_vx_parity_v1_offline

# B) 诚实生产流式对拍（tick enabled）
HONEST_OUT_DIR="$PWD/qqq_btc/results/july_w1_ft56_honest_live_parity_YYYYMMDD" \
QQQ_BTC_STRATEGY_PROFILE="$PWD/qqq_btc/CONFIG/strategy_profiles/ft56_honest_vx_production_v1.json" \
  bash qqq_btc/tools/restart_ft56_july_w1_honest_live_parity.sh

# C) V4 门控离线（带 profile manifest）
bash qqq_btc/tools/replay_offline_live_aligned.sh --months 2026-04,2026-05,2026-06 \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json \
  --out-name apr_may_jun_live_aligned
bash qqq_btc/tools/replay_offline_live_aligned.sh --months 2026-06,2026-07 \
  --strategy-profile qqq_btc/CONFIG/strategy_profiles/v4_vx_live_aligned_v1.json \
  --out-name jun_jul_live_aligned

# 只看版本 provenance
python qqq_btc/tools/replay_offline_live_aligned.py --print-version
```

FT56 honest+VX 离线当前结果在  
`qqq_btc/results/offline_live_aligned/ft56_profile_v1_july/summary.json`。  
现在由 `ft56_honest_vx_parity_v1.json` 直接指定 FT56 infer/raw1；不再需要临时
`--model ft56` 分支。

研究消融（默认 **不是** 现金标）仍用：

```bash
python qqq_btc/tools/replay_regime_profiles_apr_jul.py --skip-build
# 注意：默认 selector=vixy、quarantine 关
```

---

## 5. 使用纪律

1. **对外/对内报「实盘预期」**：只用 **A（流式）** 或 **B（FT56 honest+VX）**。  
2. **V4 +214/+175/+68**：只作门控/selector 相对效果，不作资金预期。  
3. **+58% FT56**：旧 honest、无 VX，归档即可。  
4. 每次正式 run 保留 `manifest.json`（git commit / dirty / infer mtime / gates）。  
5. 最新 V4 离线别名：`qqq_btc/results/offline_live_aligned/LATEST`。

---

## 6. 相关文档

- `qqq_btc/docs/honest_live_kpi_finetune_replay.md` — 诚实 KPI 定义  
- `qqq_btc/docs/patch_vixy_features.md` — VIXY / vix_level 补数  
- `qqq_btc/docs/week_regime_match_finetune.md` — 周 regime / 微调窗  
