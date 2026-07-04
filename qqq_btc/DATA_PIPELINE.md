# QQQ 慢通道训练数据流水线（qqq_btc v2）

本文记录从期权报价下载到 LMDB 的**正确顺序**与命令。配置统一使用：

`qqq_btc/CONFIG/slow_feature_qqq_v2.json`

> 快通道已弃用。慢通道为 **5 分钟步进采样**，特征仍是 **1min 微观 + 5min regime** 双塔，不是纯 5min。

---

## 总览

```
[A] 目标合约 map
[B] Databento 秒级 quote → 分钟聚合
[C] IV / Greeks（日频）→ 月频期权特征
[D] feature_merge（正股 + 期权 → quote_features_raw）
[E] 主标签 process_labels_file（必需）
[F] 可选双腿 label_pipeline
[G] split → train / val / test
[H] rolling_norm（分 stage）
[I] build_lmdb
```

**切分与归一化顺序不可颠倒：必须先 split，再 rolling_norm。**

| 错误顺序 | 正确顺序 |
|----------|----------|
| norm → split → LMDB | **split → norm → LMDB** |

原因：`apply_rolling_norm_standalone.py` 读取的是已存在的 `quote_features_{train,val,test}`；且各 stage 独立滚动，避免 train→val 统计泄漏。

---

## A. 目标合约 map

```bash
python preprocess/download/step1_build_target_map.py \
  --profile qqq_0dte \
  --start-date 2023-03-28 \
  --end-date 2026-06-30 \
  --output ~/train_data/locked_targets_map_0dte.parquet
```

依赖日频源：`~/train_data/nq_options_day_iv/QQQ/QQQ_YYYY-MM-DD.parquet`。

`step2` **只下载 map 里已有的日期**，不会按 `--date-from` 自动造新日。

---

## B. Databento 秒级 quote → 1 分钟

硬约束：`MIN_DATE=2023-03-28`（更早只有 `cbbo-1m` 分钟级，已禁止）。

Schema 自动选择：

| 日期 | Schema | 说明 |
|------|--------|------|
| ≥ 2025-02-20 | `cbbo-1s` | 原生 1 秒 |
| 2023-03-28 ~ 2025-02-19 | `cmbp-1` | tick 重采样到 1s |
| &lt; 2023-03-28 | 跳过 | — |

```bash
export DATABENTO_API_KEY=db-xxxxxxxx

# 秒级（断点续传：已存在日期跳过）
python preprocess/download/step2_databento_second_sniper_v1.py

# 聚合到 1 分钟
python preprocess/download/step3_databento_aggregate_1s_to_1m.py
```

| 路径 | 内容 |
|------|------|
| `/mnt/s990/data/raw_1s/options_databento/` | 秒级 quote |
| `/mnt/s990/data/raw_1m/options_databento/` | 分钟 OHLC + quote |

增量补新日：先更新 map，再 `--date-from`；**同天合约变更**需 `--force` 或删文件重下。

---

## C. IV / Greeks 与月频期权特征

```bash
# 日频 IV + Greeks（读 1m databento + 正股 1min）
python preprocess/ask_bid/option_cac_day_vectorized_day.py

# 日 → 月（供 options_locked / feature_merge）
python preprocess/ask_bid/iv_day2month.py   # 或 iv_day2month_spnq.py，按你现用脚本
python preprocess/ask_bid/options_locked_feature.py
```

RFR：`~/risk_free_rates.parquet`，脚本会按期权日期从 FRED `DGS3MO` 刷新；利率为小数（约 0.03–0.05）。

输出示例：

- `~/train_data/quote_options_day_iv/QQQ/standard/`
- `~/train_data/quote_options_monthly_iv/` 或 bucketed 月频宽表

---

## D. feature_merge

```bash
# CONFIG_FILE 已指向 slow_feature_qqq_v2.json
python preprocess/ask_bid/feature_merge_option_raw.py
```

默认产出：`~/train_data/quote_features_raw/{SYMBOL}/regular/09:30-16:00/{1min,5min}/YYYY-MM.parquet`

要点：

- `resample_freq` 含 **1min 与 5min**（都要算）
- 模型特征见配置 `features`（带 `resolution` 标签）
- **OHLC 不进模型 features 列表**，但 `feature_merge` 会额外落盘 `open/high/low/close/volume/vwap`，供标签使用
- `calc: "raw"` 的 time/trend 特征不进 rolling_norm

已弃用：`process_option_labels_file_vectorized` / `add_option_labels_data()`（旧 ATM Call DTE7–60 标签，与 v2 无关）。

---

## E. 主标签（必需）——期权 fill 价，不是正股收益

**必须用 `label_pipeline.py`**，不要用 `process_labels_file` / `write_main_labels.py` 当主标签。

后者用正股 `close` + 固定 cost floor，会把 ~76% 标签截成 0，与 replay 的期权 ROI 阈值（1.5%）完全错位。

正确路径：从 `quote_options_day_iv` 取 **CALL ATM (bucket2) / PUT ATM (bucket0)** 的 bid/ask，按 fill 模型算权利金净收益：

```
entry @ t+1min (fill_frac=0.775 买)
exit  @ t+1+30min (fill_frac=0.775 卖)   # hold_bars=30,与 trend_fit_30m 对齐
net = exit_fill/entry_fill - 1 - commission
```

> 原 hold=5(6min) 对 0DTE 过短：theta+点差主导、正股方向噪声被杠杆放大，test IC≈0。
> 拉长到 30min 后 val IC≈0.23、test IC≈0.10（见 `checkpoints_qqq_v2_h30`）。

```bash
# 对每个 stage 的 1min 特征目录原地写标签（可在 rolling_norm 之后）
for stage in train val test; do
  python qqq_btc/tools/label_pipeline.py \
    --input ~/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min \
    --output ~/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min \
    --symbol QQQ \
    --anchor-config qqq_btc/CONFIG/anchor_qqq_0dte.json \
    --report /tmp/label_report_${stage}.json
done
```

产出列：

```
label_return_fwd_net / gross / execution_cost / direction_net / label_net_valid
label_call_return_fwd_net / label_put_return_fwd_net / label_straddle_*
exec_call_bid/ask/mid  exec_put_bid/ask/mid
```

验收：`avg net_std` 应在 **0.1–0.5** 量级，`|net|≥0.015` 占比应 **>50%**（不是接近 0）。

依赖：`~/train_data/quote_options_day_iv/QQQ/standard/QQQ_YYYY-MM-DD.parquet`（含 `bucket_id`+bid/ask）。

---

## F. 双腿标签

`label_pipeline` **已同时写入** call/put/straddle 列，无需另一步。  
`loss_weights.call_put_edge` / `straddle_edge` 在有这些列时自动生效。

---

## G. 按月切分 train / val / test

```bash
python preprocess/ask_bid/split_raw_features.py
```

当前默认（`split_raw_features.py` 内可改）：

| Stage | 月份范围 |
|-------|----------|
| train | 2023-03 ~ 2025-12 |
| val | 2026-01 ~ 2026-03 |
| test | 2026-04 ~ 2026-06 |

| 目录 |
|------|
| `~/train_data/quote_features_train` |
| `~/train_data/quote_features_val` |
| `~/train_data/quote_features_test` |

源：`~/train_data/quote_features_raw`。按文件名 `YYYY-MM.parquet` 整月拷贝。

---

## H. 滚动归一化

```bash
export FEATURE_CONFIG="/home/kingfang007/文档/GitHub/option-qt/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
python preprocess/ask_bid/apply_rolling_norm_standalone.py
```

- 必须已存在 `quote_features_{train,val,test}`
- **必须与 LMDB 使用同一份 `slow_feature_qqq_v2.json`**（默认路径是旧的 `~/notebook/train/slow_feature.json`，务必 export）
- 跳过 `calc: "raw"` 与 `label_*`、OHLC
- **原地覆盖** parquet

---

## I. 建 LMDB

使用 **`qqq_btc/tools/build_lmdb.py`**，不要用旧的 `s0_create_slow_channel_lmdb_alpha.py`（旧脚本只打包 `direction/return_fwd/volatility/event`，缺 net/cost，与 `LMDBAlphaDataset` 不兼容）。

```bash
python qqq_btc/tools/build_lmdb.py \
  --feature-root ~/train_data/quote_features_train \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --symbol-map qqq_btc/CONFIG/symbol_map.json \
  --output ~/train_data/lmdb/train_qqq.lmdb \
  --symbols QQQ

python qqq_btc/tools/build_lmdb.py \
  --feature-root ~/train_data/quote_features_val \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --symbol-map qqq_btc/CONFIG/symbol_map.json \
  --output ~/train_data/lmdb/val_qqq.lmdb \
  --symbols QQQ

python qqq_btc/tools/build_lmdb.py \
  --feature-root ~/train_data/quote_features_test \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --symbol-map qqq_btc/CONFIG/symbol_map.json \
  --output ~/train_data/lmdb/test_qqq.lmdb \
  --symbols QQQ
```

窗口：`WINDOW_1M=30`，`WINDOW_5M=6`，`WINDOW_STEP=5`（与 `slow_channel` 一致）。

必需标签：

```
label_return_fwd_net
label_return_fwd_gross
label_execution_cost
label_direction_net
```

缺列时报错时，先跑 **E. label_pipeline**。

推荐权重：`checkpoints_qqq_v2_h30/best.pth`（hold=30min fill 标签）。

入场 / 退出（val 上 `calibrate_rails` 重标，见 `/tmp/rails_h30_suggestion.json`）：

- 入场：`net_edge >= 0.03` 且 `q10 > -0.20`；开仓截止 `session_bar<=330`(15:00)；日最多 4 笔
- 孵化期 15 bar：仅 `hard_stop=-28%`，无 soft/ladder/trailing
- 之后：`soft=-20%`，ladder 从 20% 起锁利，`max_hold=45`
- tick 灾难轨保持紧：`tick_fast_hard=-20%`，`disaster=-35%`

评估除全样本 IC 外，应看 **Top5/10/20% edge bar 的标签均值与 hit**（稀疏脉冲信号）。

---

## 一键命令清单（特征已就绪时）

```bash
# 1) 切分
python preprocess/ask_bid/split_raw_features.py

# 2) 归一化
export FEATURE_CONFIG="/home/kingfang007/文档/GitHub/option-qt/qqq_btc/CONFIG/slow_feature_qqq_v2.json"
python preprocess/ask_bid/apply_rolling_norm_standalone.py

# 3) 期权 fill 标签（必需；可在 norm 之后）
for stage in train val test; do
  python qqq_btc/tools/label_pipeline.py \
    --input ~/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min \
    --output ~/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min \
    --symbol QQQ \
    --anchor-config qqq_btc/CONFIG/anchor_qqq_0dte.json
done

# 4) LMDB
for stage in train val test; do
  python qqq_btc/tools/build_lmdb.py \
    --feature-root ~/train_data/quote_features_${stage} \
    --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
    --symbol-map qqq_btc/CONFIG/symbol_map.json \
    --output ~/train_data/lmdb/${stage}_qqq.lmdb \
    --symbols QQQ
done

# 5) 训练
python -m qqq_btc.model.train --mode pretrain \
  --config qqq_btc/CONFIG/slow_feature_qqq_v2.json \
  --data-root ~/train_data/lmdb \
  --train-lmdb train_qqq.lmdb --val-lmdbs val_qqq.lmdb \
  --checkpoint-dir checkpoints_qqq_v2_fill
```

---

## 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| `缺必需标签 label_return_fwd_net` | 未跑期权 fill 标签 | `label_pipeline.py` |
| `No price column` | 特征文件无 OHLC | 已支持从 `spnq_train_resampled` 回补；重跑 feature_merge 会自带 OHLC |
| rolling_norm 目录不存在 | 先 norm 后 split | 先 split |
| LMDB 缺 net 标签 | 用了 `s0_create_slow_channel_lmdb_alpha.py` | 改用 `build_lmdb.py` |
| 归一化列不对 | `FEATURE_CONFIG` 未指向 v2 | export 正确路径 |
| Databento 只有 1 天任务 | map 里没有那些日期 | 先 step1 扩 map |
| 同天改合约不重下 | 按日文件跳过 | `--force` 或删日文件 |

---

## 相关文件

| 角色 | 路径 |
|------|------|
| 特征配置 | `qqq_btc/CONFIG/slow_feature_qqq_v2.json` |
| 特征 merge | `preprocess/ask_bid/feature_merge_option_raw.py` |
| 主标签 | `qqq_btc/tools/label_pipeline.py`（期权 fill） |
| 双腿标签 | `qqq_btc/tools/label_pipeline.py` |
| 切分 | `preprocess/ask_bid/split_raw_features.py` |
| 归一化 | `preprocess/ask_bid/apply_rolling_norm_standalone.py` |
| LMDB | `qqq_btc/tools/build_lmdb.py` |
| 数据集 | `qqq_btc/model/dataset.py` (`LMDBAlphaDataset`) |
| 训练执行总览 | [EXECUTION_PLAN.md](./EXECUTION_PLAN.md) |
