# stock_options — single-stock option modeling path

`stock_options` is the independent path for NVDA/TSLA/MAG7-style equity option
models. It reuses the **same V4 dual-stream TFT** (`qqq_btc.model.train` /
`DualStreamAlphaNet`) as QQQ, but keeps data profiles, anchors, thresholds,
checkpoints, and replay reports separate from the QQQ 0DTE path.

## Initial Scope

- Pilot symbols: `NVDA`, `TSLA`
- **Model**: V4 TFT（双流 `net_edge` / q10），**不是** state-gate 规则模型
- **Non-0DTE mainline（当前）**: trading `dte∈{1,2}` only — QQQ 0DTE 已验证不稳定，
  个股训练 **禁用当日到期**；独立构建目录 `~/train_data/builds/stock_non0dte/`
- Modeling shape: **one shared TFT per symbol** + `day_of_week`（静态）+
  `stock_dte` / `stock_expiry_weekday` / `time_to_expiry_norm` 特征；
  replay 阈值按 `(symbol, dte, weekday)` 标定 — **不要**按 weekday 各训一个 TFT
- Legacy weekly-DTE / short-DTE(含0) paths: secondary
- State-gate / micro stability tools: optional diagnostics only

## Non-0DTE TFT（推荐）

```bash
# 锁约（已排除 0DTE，prefer trading-1）
python preprocess/download/step1_build_target_map_old.py \
  --config preprocess/CONFIG/anchor_stock_non0dte_old_lock.json \
  --dte-mode trading --symbols NVDA,TSLA \
  --start-date 2026-02-02 --end-date 2026-03-18 \
  --raw-dir ~/train_data/nq_options_day_iv \
  --output ~/train_data/locked_targets_map_stock_non0dte_old_lock.parquet

# 特征 + LMDB + 训练（不改旧代码；新脚本）
bash stock_options/tools/train_stock_non0dte_tft_v4.sh NVDA
bash stock_options/tools/train_stock_non0dte_tft_v4.sh TSLA
```

## Directory Layout

```text
stock_options/
  CONFIG/
    anchor_stock_short_dte.json
    slow_feature_stock_short_dte.json   # derived from qqq v4 + weekday/DTE cols
    anchor_stock_weekly_dte.json
    symbol_map_stock.json
  common/
    short_dte_config.py
    weekly_dte_config.py
  nvda/ config_short_dte.py
  tsla/ config_short_dte.py
  tools/
    rebuild_short_dte_pipeline.py       # TFT entry
    feature_merge_short_dte.py
    report_mag7_short_dte_weekday_coverage.py
    train_stock_short_dte_tft_v4.sh
```

## Quick start (short-DTE → V4 TFT)

```bash
# Show paths + confirm model_family=tft_dual_stream_v4
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step show

# Generate slow_feature_stock_short_dte.json from QQQ v4 + weekday/DTE features
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step feature-config

# Weekday×DTE coverage (uses existing locked map)
python stock_options/tools/rebuild_short_dte_pipeline.py --step weekday-report --probe-micro

# After day_iv / monthly / bucketed exist for the symbol:
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step feature-merge
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step filter
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step label
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step lmdb
python stock_options/tools/rebuild_short_dte_pipeline.py --symbol NVDA --step train --epochs 20

# Or wrapper:
bash stock_options/tools/train_stock_short_dte_tft_v4.sh NVDA
```

Research window default: **2026-02-02** onward.  
TFT split (thin but honest): train→2026-04, val→2026-05, test→2026-06.

## Design Rules

1. Do not import `qqq_btc.qqq.config` or QQQ 0DTE thresholds.
2. Train with `python -m qqq_btc.model.train` (same backbone as QQQ V4).
3. Validate every result by `symbol`, `weekday`, and `dte`; aggregate PnL is not
   enough.
4. Mon/Wed sample depth is still short; Friday must not silently dominate
   validation — slice reports by `expiry_weekday` / `trade_weekday`.
