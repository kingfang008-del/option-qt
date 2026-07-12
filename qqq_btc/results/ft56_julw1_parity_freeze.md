# F56（5–6 微调）结果 + July W1 实时对拍冻结记录

> 冻结日期：2026-07-12  
> 目的：把「5–6 bak 微调出的 F56」与「July W1 金标/流式对拍通过」的可复核口径落盘，并与「长窗现管线 prefer_primary」问题分开，避免混为一谈。

---

## 1. 两套问题必须分开

| 问题 | 结论 |
|------|------|
| **A. F56 近窗金标链**（本文） | 在 bak 5–6 微调 + July W1 金标特征/门控下，W1 @25% 可达约 **+51%**；部分流式对拍 `overall_pass=True` |
| **B. 长窗现管线 prefer_primary** | 报价/覆盖可对齐 bak，但 `options_*` 归一化面断代；同协议重训/微调 Val IC 天花板约 **0.11–0.13**，到不了 0.2 |

**B 不否定 A**；**A 也不等于生产已可日更**。生产日更若走 B 的特征面，F56 权重会漂。

---

## 2. F56 模型（5–6 微调）

| 项 | 值 |
|----|----|
| 脚本 | `qqq_btc/tools/train_ft56_julw1.sh` |
| 政策 | `qqq_btc/CONFIG/weekly_finetune_policy.json` |
| init | `checkpoint/checkpoints_qqq_v4/best.pth` |
| 特征根 | `~/train_data/_bak_pre4c/quote_features_test_QQQ` |
| train | 2026-05 + 2026-06 |
| val | 2026-06 |
| 产出 ckpt | `checkpoint/checkpoints_qqq_ft56_julw1/best.pth` |
| Val IC（train.log） | 最佳约 **0.297** |
| 对照 summary | `qqq_btc/results/ft56_julw1_compare/summary.json` |

说明：此处 IC≈0.3 是 **bak_test 上的 Val IC**，不是 prefer_primary 重建特征上的 IC。

---

## 3. July W1 收益口径（含 early-vix）

### 3.1 推理金标

- Infer：`qqq_btc/results/ft56_julw1_with_vix/test_infer.parquet`
- Replay summary（默认门控当时）：约 **+37.7%** / 10 笔 / IC≈0.079  
  → `qqq_btc/results/ft56_julw1_with_vix/replay_summary.json`

### 3.2 早盘 PUT 网格（已写入默认 REPLAY）

网格工具：`qqq_btc/tools/grid_early_put_filters.py`  
结果：`qqq_btc/results/early_put_filter_grid_julw1/summary.json`

推荐（已启用）：`put_early_session_bar=30` + `put_early_vix_min=0.6`

| 样本 | 无过滤 | +early vix | 变化 |
|------|--------|------------|------|
| July W1 | +35.8% / 11 笔 | **+51.1%** / 13 笔 | **+15.3pp** |
| July1 | −7.2%（早盘 HARD_STOP） | **+2.1%**（午后 3 笔） | 早盘大亏消除 |
| Apr–Jun | −42.5% | −42.5% | 0pp |
| 2025H2 | −66.4% | −63.4% | +3pp |

配置落点：`qqq_btc/qqq/config.py`（`REPLAY.put_early_*`）  
逻辑落点：`qqq_btc/common/entry_decision.py`（`choose_entry`）

网格里 `open30>0` / `range≥0.002` 同效，语义弱于 VIX，故只启用 **vix-only**。

### 3.3 July 特征并非「纯 bak 七月」

流式/金标脚本明确依赖混合链（示例：`run_july_w1_ft56_4c_stream_rolling.sh`）：

| 组件 | 路径/设定 |
|------|-----------|
| 权重 | `checkpoints_qqq_ft56_julw1/best.pth` |
| 7 月离线特征 | `july_w1_v4_databento` / `july_w1_v4_experiment` |
| rolling 暖启动 | **bak 6 月已归一化** `.../_bak_pre4c/.../2026-06.parquet` |
| put_gate 5min | 离线 `quote_features_test` 的 `vix_level` |
| regime 金标 | 离线 1min 特征（避免 SE 短历史早翻转） |

因此：**+51% 证明的是「F56 + 该金标链 + early-vix」**，不是「任意现管线日更特征自动同收益」。

---

## 4. 实时对拍：冻结为通过的结果

目录：`qqq_btc/results/july_w1_ft56_4c_stream_rolling/`

门控：`debug_slow_vs_offline_normed_features`（流式 debug_slow vs 离线已 norm 特征）

### 4.1 记为「通过」的对拍（`overall_pass=True`）

| 文件 | offline 金标 | n_feats | 备注 |
|------|-------------|--------|------|
| `feat_parity_step1_dbnative3.json` | `july_w1_v4_databento/.../quote_features_test_clean/.../2026-07.parquet` | 32 | 2026-07-01 pass_rate=1.0 |
| `feat_parity_vs_clean_fullexit.json` | 同上 databento clean | 32 | 全日口径 |
| `feat_parity_step1_greek6.json` | `july_w1_v4_experiment/.../quote_features_test_clean/.../2026-07.parquet` | 32 | greek 对齐后通过 |

### 4.2 明确未通过（勿当生产绿灯）

大量 `feat_parity_step1*.json` / `*_vs_test*.json` 为 `overall_pass=False`，失败项常见整组 **`options_*`**。  
含义：只有在 **指定 clean 金标 + 对应流式设定** 下对拍才绿；不等于与 bak_test 或任意重建特征一致。

### 4.3 相关入口脚本（复现用）

- `qqq_btc/tools/run_july_w1_ft56_4c_stream_rolling.sh` — PG deep warmup + live rolling  
- `qqq_btc/tools/run_july_w1_ft56_4c_stream.sh` — Redis 流式基线  
- `qqq_btc/tools/compare_debug_slow_offline.py` — 对拍工具  

---

## 5. 逻辑影响检查（相对旧默认行为）

本记录相关、且会改变「之前逻辑」的改动：

| 改动 | 影响 | 是否破坏旧 V4 默认 replay |
|------|------|---------------------------|
| `REPLAY.put_early_session_bar/vix_min` 默认开启 | 早盘 PUT 更严；W1 改善，Apr–Jun 中性 | **有意改变**进场行为（加过滤，非关旧门控） |
| `entry_decision.choose_entry` 增加 early-vix/open30/range | Live/offline 共用进场 | 同上；open30/range 默认仍为 `None`，仅 vix 生效 |
| `FEATURE_CONFIG_PATH`：`v2` → **`v4`** | 与 V4/F56 ckpt 42 列对齐；去掉 v2 多出的 spot/trend 列 | **会影响**仍指向 v2 特征表的旧入口；F56/V4 应用 v4 |
| call/put `softplus`（backbone） | 对齐归档 V4 infer | 修复双头错位；旧「裸 logit」路径不再默认 |
| bak June 作 rolling seed（脚本级） | 仅 July W1 复现脚本 | 不改训练默认；但生产若无等价续写会漂 |

**未纳入本冻结结论、且与「长窗现管线」相关的实验**（勿与 A 混淆）：

- prefer_primary 重建 / 长窗微调：`builds/0dte_prefer_primary`，Val IC≈0.11  
- 多 seed 重训：最高≈0.132 后中断 → `qqq_btc/results/prefer_primary_multiseed/summary_interrupted.json`

---

## 6. 生产含义（冻结时结论）

1. **可以固化为「研究/金标复现基线」**：F56 ckpt + July W1 金标链 + early-vix + 上述 pass 对拍文件。  
2. **尚不足直接宣称生产日更安全**：日更特征必须能在 **不依赖手工 bak seed / 离线 regime 金标** 的条件下，持续通过对拍与 OOS 门禁。  
3. 长窗 prefer_primary IC 崩塌说明：**现管线特征面 ≠ bak 微调面**；上 F56 前先钉特征契约，或接受在现管线上重训并门禁。

---

## 7. 关键路径速查

```
ckpt:     checkpoint/checkpoints_qqq_ft56_julw1/best.pth
train:    qqq_btc/tools/train_ft56_julw1.sh
bak feat: ~/train_data/_bak_pre4c/quote_features_test_QQQ
infer:    qqq_btc/results/ft56_julw1_with_vix/test_infer.parquet
early:    qqq_btc/results/early_put_filter_grid_julw1/summary.json
parity:   qqq_btc/results/july_w1_ft56_4c_stream_rolling/feat_parity_step1_dbnative3.json
          qqq_btc/results/july_w1_ft56_4c_stream_rolling/feat_parity_vs_clean_fullexit.json
          qqq_btc/results/july_w1_ft56_4c_stream_rolling/feat_parity_step1_greek6.json
config:   qqq_btc/qqq/config.py  (put_early_*, FEATURE_CONFIG_PATH→v4)
```
