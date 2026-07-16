# maga7 — Mag7 Rule-A money-flow Top2 short-DTE scalp

规则策略路径（非 TFT）。布局对齐 `stock_options/`，成交模型复用 `qqq_btc.common.fill_model`。

**默认 profile（`CONFIG/mf10_top2_v1.json`）**：`only_reenter_after_win=true` + regime **仅 QQQ `from_prev` 对齐**（VIX/put 闸默认关）。正式 Jan–Jul：`maga7/results/jan_jul_m5c_qqq_onlywin`。

版本对比与消融见 [`docs/jan_jul_replay_versions.md`](docs/jan_jul_replay_versions.md)；**开盘阶梯 + concurrent + mf_flip 当前结论**见 [`docs/open_ladder_live_package_results.md`](docs/open_ladder_live_package_results.md)。命名 profile 在 [`CONFIG/strategy_profiles/`](CONFIG/strategy_profiles/)（`catalog.json`）。

## 策略摘要

- 信号：规则 A（mf10 streak≥8，|from_prev|≥2%，vol_z≥1，10:30–14:00）
- 选股：当日 Top2（最早触发）
- 选约：默认 day_lock ATM；研究可用 `open_ladder`（开盘阶梯 OTM5）
- 出场：TP 1.6x / SL 0.4x / 超时 30m；可选 `exit_mode=mf_flip`（mf10 翻向提前平）
- 仓位：旧 topk 均分；研究可用 concurrent（独处满袖套，最多 2 腿）
- 成交：点差 frac=0.8（`bid + 0.8*(ask-bid)` 买入）

## 目录

```text
maga7/
  CONFIG/mf10_top2_v1.json
  common/   signals, fills, replay, stream_engine, bar_agg, config
  live/     scanner (1s→1m 或直接 1m)
  tools/    prepare_jan_jul_data, run_replay_offline, run_stream_parity,
            run_scanner_shadow, run_scanner_from_1s, run_oms_dry_run,
            run_oms_live_stub, run_sensitivity_grid, run_regime_ablation
  live/     scanner, oms_dry, oms_stub
  common/   signals, fills, replay, stream_engine, bar_agg, regime, config
  results/  回测与对拍产物
  docs/
```

## 数据准备（2026-01-02 ~ 2026-07-13）

```bash
export MASSIVE_API_KEY=...   # 或 POLYGON_API_KEY
cd /path/to/option-qt
export PYTHONPATH=$PWD

# 报告缺口 → 锁约 → 下载/复用 1s quote
python -m maga7.tools.prepare_jan_jul_data --step all --max-workers 12
```

说明：`spnq_train` 在约 **2026-03-19 ~ 2026-04-30** 无正股 → 该窗无法算 day_iv/锁约，需另补正股后再跑。

产物：

- 锁约：`~/train_data/locked_targets_map_maga7_mf10_jan_jul.parquet`
- 1s：`/mnt/s990/data/raw_1s/maga7_mf10_old_lock/{SYM}/`

## Offline replay

```bash
python -m maga7.tools.run_replay_offline --scheme single
python -m maga7.tools.run_replay_offline --scheme m5_circuit

# 规则敏感性（streak / from_prev / vol_z）
python -m maga7.tools.run_sensitivity_grid --start-date 2026-01-02 --end-date 2026-07-13

# QQQ/VIXY regime 闸门消融（m5_circuit）
python -m maga7.tools.run_regime_ablation --scheme m5_circuit
```

## 流式对拍

```bash
python -m maga7.tools.run_stream_parity --scheme single
# exit 0 = 逐笔 key/收益一致
```

流式引擎按时间推进 1m 正股 bar，因果触发 Rule-A / TopK，再用同一套 1s fill 闭合交易，与 offline batch 对拍。

## Dashboard

```bash
streamlit run qqq_btc/dashboard/qqq_btc_dash.py
# Sidebar → Board → Mag7
```

## Scanner shadow（不下单）

```bash
# S1：历史 1m
python -m maga7.tools.run_scanner_shadow --start-date 2026-05-06 --end-date 2026-05-10

# S2：正股 1s → 因果聚合 1m（实盘 ingest 对齐；Rule-A 仍按 1m）
python -m maga7.tools.run_scanner_from_1s --start-date 2026-05-06 --end-date 2026-05-10

# S3：OMS dry-run（fill 0.8 限价 + 1s 闭合，不下真单）
python -m maga7.tools.run_oms_dry_run --start-date 2026-05-06 --end-date 2026-05-10 --compare-offline

# S4：OMS stub / shadow（小仓 + fill_audit；可选 --redis）
MAG7_MAX_QTY=1 python -m maga7.tools.run_oms_live_stub \
  --start-date 2026-05-06 --end-date 2026-05-10 --compare-offline
```

实盘拓扑见 `docs/scanner_oms_integration.md`（1s 正股 → 1m 信号 → OMS 1s quote，不经 QQQ TFT）。
