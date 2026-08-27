# July TFT Replay 锁约纠正

**日期：** 2026-07-22

## 问题

先前 `run_july_full_frozen_replay.sh` 使用的  
`/mnt/s990/data/raw_1s/dte1_options_old_lock` **July 已被 open_4bucket 开盘锁污染**，  
与 V4/FT56 训练用的 **全天 δ 前视 old_style** 不一致，KPI 不可信。

## 三类锁约（勿混）

| 类型 | 选约时刻 | 代表 | 是否训练一致 |
|------|----------|------|--------------|
| A. old_style foresight | 全天 δ 前视 | `locked_targets_map_old_style_trading_1dte*.parquet` | **是** |
| B. open_4bucket | 09:30 开盘价推 δ | `locked_targets_map_open_4bucket.parquet` | 否（现盘） |
| C. maga7 open_ladder OTM3/5 | 09:30 spot 阶梯 | `..._open_ladder_atm5otm_...` | 否（执行车辆对照） |

## 已重跑（车辆 C：maga7 1DTE ATM+OTM{rung}）

工具：`python qqq_btc/tools/run_july_otm_ladder_tft_replay.py --otm-rung 3|5`

范围：2026-07-01..07-13（OTM5 本地 1s 覆盖上限）

| 车辆 | 模型 | VX acct25 | 笔数 | MaxDD |
|------|------|----------:|-----:|------:|
| OTM3 | V4 | **+3.57%** | 15 | −7.93% |
| OTM3 | FT56 | **−7.08%** | 8 | −12.52% |
| OTM5 | V4 | **+11.82%** | 13 | −8.16% |
| OTM5 | FT56 | **−6.20%** | 8 | −12.52% |
| （污染 open_4bucket 全月） | V4 | +36.40% | 17 | −8.25% |
| （污染 open_4bucket 全月） | FT56 | +26.82% | 15 | −6.13% |

读数：换成 maga7 开盘阶梯后，之前 open_4bucket 的 +36%/+27% **大幅回落**；FT56 在 OTM3/5 上为负。这印证「锁约错了 KPI 虚高」。

产物：
- `.../v4_vx_jul_ladder_otm3_v1_offline/` / `.../ft56_vx_jul_ladder_otm3_v1_offline/`
- `.../v4_vx_jul_ladder_otm5_v1_offline/` / `.../ft56_vx_jul_ladder_otm5_v1_offline/`

## 训练一致路径（A）— 待补下载

前视 map 已用 nq day_iv 重建（7 日）：

`~/train_data/locked_targets_map_old_style_trading_1dte_jul2026.parquet`  
日期：07-01,07,10,13–16（nq 缺 02/06/08/09 等）

下一步（需有效 `MASSIVE_API_KEY`）：

```bash
python preprocess/download/step2_polygon_second_sniper_v1.py \
  --target-map ~/train_data/locked_targets_map_old_style_trading_1dte_jul2026.parquet \
  --output-dir /mnt/s990/data/raw_1s/dte1_options_old_style_jul2026 \
  --symbols QQQ --start-date 2026-07-01 --end-date 2026-07-16 --force

RAW1S=/mnt/s990/data/raw_1s/dte1_options_old_style_jul2026 \
LOCK_MAP=~/train_data/locked_targets_map_old_style_trading_1dte_jul2026.parquet \
EXP_OVERRIDE=~/train_data/july_v4_old_style \
bash qqq_btc/tools/build_v4_old_lock_month.sh 2026-07
```

然后再对 V4 / FT56 做 frozen + VX replay。

协议入口（Base 冻结 / 近月微调 / July 前向，**勿** val=1–6）：

见 [`tft_base_ft_july_forward_protocol.md`](tft_base_ft_july_forward_protocol.md)  
脚本：`bash qqq_btc/tools/run_tft_base_ft_july_forward.sh check|download_july|build_july|forward`
