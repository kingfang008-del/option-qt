# VIXY / vix_level 缺数补写

> 脚本：`qqq_btc/tools/patch_vixy_features.py`  
> 典型故障：2026-07-13 `put_gate` / `vix_level` 全 0 → PUT 被系统性挡掉或误放行

---

## 1. 根因

常见链路：

1. `spnq_train_resampled/VIXY/.../1min/{ym}.parquet` 只更新了 OHLCV
2. 未重跑 `generate_vix_level_global`（或等价滚动 z）
3. `quote_features_{raw,test}` 合并到的仍是旧/空 `vix_level`
4. 离线 put_gate（`vix_level` +1min asof）与实盘 `vixy_z` 一起失效

诊断信号：

| 指标 | 正常 | 塌缩 |
|---|---|---|
| `vix_level` nunique（全日） | 数百 | 1 或接近 0 |
| `vix_level` std | >0.2 | ≈0 |
| `vix_proxy_close` mean | ~VIXY 价格 | 0 / NaN |

---

## 2. 补数步骤（脚本已封装）

```bash
# A. 只检查
python qqq_btc/tools/patch_vixy_features.py \
  --ym 2026-07 --day 2026-07-13 --check-only \
  --feature-root ~/train_data/jul13_v4_old_lock_massive/quote_features_raw

# B. 重算 VIXY resampled，并回写 raw / test
python qqq_btc/tools/patch_vixy_features.py \
  --ym 2026-07 \
  --feature-root ~/train_data/<exp>/quote_features_raw \
  --feature-root ~/train_data/<exp>/quote_features_test \
  --apply-frozen-on-test

# C. 只补某一天
python qqq_btc/tools/patch_vixy_features.py \
  --ym 2026-07 --day 2026-07-13 \
  --feature-root ~/train_data/<exp>/quote_features_raw
```

脚本行为：

1. 用近 `--history-months`（默认 7）个月 VIXY 分钟序列重算  
   `vix_proxy_close / vix_z / vix_level / vixy_detrended_level / is_vix_jump`
2. 回写 `~/train_data/spnq_train_resampled/VIXY/.../{res}/{ym}.parquet`
3. `merge_asof(backward)` 注入目标特征 parquet
4. 首次改写前备份 `*.bak_vixpatch`
5. `--apply-frozen-on-test` 时，仅对路径含 `quote_features_test` 的文件做 `vix_level` frozen_norm

---

## 3. 补完后必做

```bash
# 1) 用补后 raw 重新因果 put_gate + infer（若 edge 列也依赖 vix 特征，需重 infer）
# 2) 诚实 KPI / regime replay
# 3) 若是流式对拍日：重跑 honest live parity，勿沿用旧 FCS debug 缓存
```

Jul13 案例：先补 `jul13_v4_old_lock_massive` 的 raw/test，再 FT56 infer + LIVE 门控 replay。

---

## 4. 与「全管线重建」的边界

| 场景 | 用本脚本 | 用完整 rebuild |
|---|---|---|
| 仅 VIX 列坏、OHLCV/期权特征完好 | ✅ | 过重 |
| 期权 quote / IV / 锁约也坏 | ❌ | ✅ `rebuild_july_w1_honest_openwin_features.sh` 等 |
| 5min VIXY close 常数塌缩 | 可先本脚本 `--res 5min`；历史一次性修复见 `fix_vixy_5min_july2026.py` | — |

---

## 5. 实盘注意

- 模型特征仍走 FCS 在线 `vix_level`；门控默认 `QQQ_BTC_PUT_GATE_MODE=vixy_z`
- 补离线 parquet **不会**自动修好实盘 PG/Redis 历史；实盘需保证 VIXY 分钟预热（`ensure_fcs_warmup_ready.py`）
- 确认层（QQQ/VIXY 15m）与模型特征层分开；确认层将来可 shadow VIX 指数，但模型列切换需重训
