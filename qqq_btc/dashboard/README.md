# qqq_btc dashboard

`qqq_btc_dash.py` is a focused Streamlit dashboard for the QQQ_BTC path.

Recommended workflow (avoid logic forks):

1. **Download** — lock 4 contracts from open → quote → day_iv  
2. **Offline Replay** — continuous features + norm (data-side), then live-aligned replay; tune `strategy_profiles`  
3. **Stream Parity** — same profile, Gate1/2/3 vs offline baseline  

Also: **Live** for Redis / OMS / shadow audit.

## Boards

| Board | 用途 |
|---|---|
| **Live** | 实盘准备（刷新 frozen → warmup → shadow/dry/live）+ 链路观察 |
| **Download** | 开盘价锁约 → quote → 1m/day_iv（**不含**特征/归一化） |
| 默认 quote 根 | `/mnt/s990/data/raw_1s/dte1_options_old_lock`（`QQQ/*.parquet`） |
| 缺数比对 | `~/train_data/spnq_train_resampled`（股价 QQQ/VIXY 1min） |
| **Offline Replay** | 特征+归一化 + 离线日收益/诊断（QQQ TFT） |
| **Stream Parity** | 导出 frozen → 三闸门对拍 → 结果（QQQ TFT） |
| **Mag7** | 多标的 Rule-A Top2：offline / 流式对拍 / 数据准备 / Scanner→OMS（**非 TFT**） |

### Live

对拍通过后：

1. **刷新 frozen**：从过去 `quote_features_raw` 导出（`--upto-date=昨收` 或月冻）→ `frozen_norm_qqq_daily.npz`
2. **Warmup**：deploy 脚本检查 PG/历史分钟（Deep Warmup）——与 frozen 是两件事
3. **启动**：Shadow（不下单）→ Dry →（确认后）Live  
   `FCS_FROZEN_NORM_PATH` 指向刚导出的 `.npz`；**没有** Offline 金标可对，但归一化提取与对拍相同

脚本：`prepare_ft56_shadow_live.sh` / `deploy_ft56_julw1_live.sh`

| **Offline Replay** | 离线日收益、诊断、一键 replay；与流式共用 profile |
| **Stream Parity** | 三闸门流式对拍结果与 catalog 配方 |

### Download

- `preprocess/download/step1_lock_4bucket_from_open.py`
- `preprocess/download/backfill_warmup_check.py`（相对今天缺数 + 目标区间预热）
- `preprocess/download/run_backfill_open_lock_pipeline.py`（本页只跑 lock / download，不跑 `--features`）
- **W1 股价标签门禁**：自动检测 `spnq_train_resampled` 是否右标签（首根 09:31）；
  一键纠正：`qqq_btc.common.bar_label_convention`（Massive raw 左标签保持不动）

一键：检查缺数 · 标签纠正 · 仅锁约 · 下载+day_iv · 停止任务。

特征与归一化：**不要**在 Download 页做。下载完整后，在数据管线生成连续 `quote_features_raw`，再 rolling/frozen 归一化，供 Offline / Stream 使用。

### Offline Replay

- 结果根：`qqq_btc/results/offline_live_aligned/`
- 引擎：`qqq_btc/tools/replay_offline_live_aligned.py`
- Catalog：`qqq_btc/CONFIG/parity_board_catalog.json`（`offline_cmd` / `offline_result`）
- Profiles：`qqq_btc/CONFIG/strategy_profiles/*.json`
- **默认配方**：`v4_honest_jul1_13_old_lock`（W1 infer/raw 01–10 + old_lock Jul13）
- W1 冻结对拍：catalog 里选 `v4_honest_v0_gated`（`july_w1_v4_honest_openwin`）
- 股价 July 右标签修复：`qqq_btc/tools/fix_qqq_july_right_label_1min.py`
  （左标签 09:30 会导致 `close_log_return` 相对 W1 提前 1 分钟）

页面两段：

1. **特征 + 归一化**：一键 `--features --norm-mode {rolling|frozen|none}`（从 Download 移出，避免分叉）
2. **Offline Replay**：月度 KPI、每日 PnL、best/worst、segments、启动/停止 replay


### Stream Parity

流程固化：

1. **导出 frozen**：从 Offline 的 `quote_features_raw`（`--upto-month` 通常为对拍月前一月）导出 `.npz`
2. **触发对拍**：`FROZEN_NORM` + `HONEST_FEAT_ROOT` + `OFFLINE_RAW/NORM` + strategy profile → catalog `stream_script`
3. **看结果**：Gate1/2/3、trades、Δ vs offline

- Catalog：`qqq_btc/CONFIG/parity_board_catalog.json`
- 默认 frozen 输出：`qqq_btc/CONFIG/frozen_norm_dash_stream.npz`
- 作业日志：`qqq_btc/results/_dash_stream_jobs/`

Run:

```bash
python qqq_btc/tools/run_dashboard_qqq.py
```

or:

```bash
streamlit run qqq_btc/dashboard/qqq_btc_dash.py --server.port 8502
```

Useful environment variables:

- `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`
- `QQQ_BTC_DASH_PORT`, `QQQ_BTC_DASH_HOST`
- `QQQ_BTC_DASH_SYMBOLS` (comma-separated, max 5; default `QQQ,NVDA`)
- `QQQ_BTC_FILL_AUDIT_PATH`
- `QQQ_BTC_LIVE`
- `MASSIVE_API_KEY` / `POLYGON_API_KEY`（Download 锁约与 quote）

The Live board does not write trading controls or Redis state.
