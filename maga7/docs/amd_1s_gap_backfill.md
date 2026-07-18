# AMD 数据缺口补齐（2026-04~07）

> 承接 [`missed_movers_apr_jul.md`](missed_movers_apr_jul.md)：`eligible_topk_but_no_fill` 里 AMD 占多数。

## 1. 正股 1s（已补）

扫描：`maga7/results/stock_1s_gaps_apr_jul/gaps.csv`

| 标的 | 缺口（相对 1m 交易日） | 状态 |
|------|------------------------|------|
| **AMD** | **17** 日 | **已全部下载** → `/mnt/s990/data/raw_1s/stocks/AMD/` |
| NVDA | 1 日（05-19，事件停牌日） | 已补 |
| 其余 Mag7 / QQQ | 0 | — |

AMD 缺口日：

```
2026-05-04,05,11,12,18,19,26
2026-06-01,02,08,09,15,22,23,29
2026-07-06,07
```

多为 **周一 / 周二**（及 Memorial Day 后的 05-26）。

复跑：

```bash
python preprocess/download/download_stock_1s.py \
  --symbols AMD --start-date 2026-05-04 --end-date 2026-07-07
```

## 2. 为何仍无法在这些日成交 AMD

Freeze 锁约只允许 **DTE ∈ {0,1,2}**。  
对这些 AMD 缺口日查 `option_1m` / `day_iv`：

| 现象 | 结果 |
|------|------|
| `min_dte` | 全为 **3 或 4** |
| `has_012` | **False（17/17）** |
| `lock_symbol_day_open(..., allowed_dte=[0,1,2])` | **0 行** |
| 放宽到 `[0,1,2,3,4]`（07-06） | 可锁出 12 行（front_dte=4） |

因此：原先锁约表里这些日 **根本没有 AMD 行**（不是漏 merge），根因是 **短到期期权链在本地数据里不存在**，不是单纯缺正股 1s。

07-06 对照：同日 TSLA 等有完整 quote；AMD 无 0/1/2 DTE → TopK 名义有 AMD，resolve → `no_lock` / 无成交。

## 3. 对策略影响的判断

| 动作 | 效果 |
|------|------|
| 补正股 1s | ✅ 完成；改善 1s 衍生特征 / 未来扩链基础 |
| 在现口径下指望 AMD 这些日进成交 | ❌ 仍不可能（无 0/1/2 DTE 链） |
| 临时放宽 `allowed_dte` 到 3–4 | 可锁约，但偏离 freeze（更长 DTE，希腊/流动性不同）→ **不做默认** |
| 从 S3/Polygon 重拉 AMD 短到期 minute/day_iv | 若上游有 daily options，值得单独开下载任务 |

## 4. 建议下一步（未做）

1. 查 Polygon/S3 上这些 Mon/Tue 是否真有 AMD 0DTE/周度链；若有 → 补 `option_1m` + `day_iv` → 再 lock → sniper 1s quote。  
2. 若上游也没有短链 → 记为 **标的/日历结构性缺口**，扫描里把 AMD 这些日从「策略漏抓」改判为 **数据不可交易**。  
3. Freeze profile **不改** `allowed_dte`。

## 5. 文件

| 路径 | 说明 |
|------|------|
| `results/stock_1s_gaps_apr_jul/gaps.csv` | 全 universe 缺口清单 |
| `results/stock_1s_gaps_apr_jul/amd_lock_merge.json` | 锁约合并尝试记录（gap 日新增 0 行） |
| 锁约备份 | `~/train_data/locked_targets_map_maga7_googl_open_ladder_atm5otm_jan_jul.parquet.bak_amd_gap_20260718` |
