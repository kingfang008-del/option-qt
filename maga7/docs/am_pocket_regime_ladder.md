# AM 口袋：trade-only + 成情动态阶梯退出

> 工具：`scan_am_pocket_regime_ladder_v2.py` · `scan_am_pocket_stock_up_grid.py`  
> 产物：`research_am_pocket_regime_ladder_v2b` · `research_am_pocket_stock_up_grid`  
> 口径：**不用历史 quote**；mark = 期权成交 last ±1% slip

## 冻结候选（v2b + stock_up 抠参）

```text
entry:   vd_soft (no_b_up pockets)
mark:    trade-last slip 1%
ladder:  CHOP = TP8/SL15/h240
         TREND = 50%@+8% → runner hard TP20 / SL15 / h480 / BE after scale
         IMPULSE = 33%@+8% → runner hard TP30 / SL15 / h600 / BE after scale
stock_up (CHOP/TREND→IMPULSE):
         confirm_sec = 30
         stock_min   = 20bp   (signed underlying)
         opt_min     = 1%     (weak / often non-binding)
size:    20% / max5 / cooldown 10m
```

相对 `fixed_tp8`（vd_soft n=65）：

| 变体 | disc | blind | capture |
|------|-----:|------:|--------:|
| fixed_tp8 | +33.2% | +21.0% | 0.093 |
| prior stock_up (45s / 15bp / 3%) | +43.6% | +21.0% | 0.105 |
| **best grid (30s / 20bp / 1%)** | **+43.2%** | **+21.0%** | **0.107** |

`promote = STOCK_UP_c30_s0.002_o0.01`（捕获优先；disc 与 prior 同阶）

## stock_up 网格要点

- 144 格：`confirm∈{30,45,60,90}` × `stock_min∈{5…50bp}` × `opt_min∈{1…6%}`
- **opt_min 在优胜区几乎无信息**（同 confirm+stock 下 o=1%…6% 结果相同）→ 真正闸是 **正股确认速度/幅度**
- 捕获最优：**30s × 20bp**（升档 17/65）
- 复利最优：**45s × 15–20bp**（升档 19/65，disc +43.6% 略高、cap 略低）
- 过松（stock 5–10bp 或 confirm 90s）升档过多 → disc/cap 回落

## 裁决

1. 离线继续只认 trade。  
2. 动态阶梯 + stock_up **成立**；默认改为 **30s / 20bp / opt≥1%**。  
3. 若更看重 disc 复利，可改用 45s / 15bp（prior）。  
4. champ 硬门（35 笔）上仍与 TP8 打平——增益主要在 **vd_soft 扩样**。

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pocket_stock_up_grid \
  --entry vd_soft --tag research_am_pocket_stock_up_grid

PYTHONPATH=. python -m maga7.tools.scan_am_pocket_regime_ladder_v2 \
  --tag research_am_pocket_regime_ladder_v2b
```

## 后续：持仓特征退出

见 [`am_pocket_hold_feature_exit.md`](am_pocket_hold_feature_exit.md)。  
同入口上 stall/mom/soft-floor 把 capture 抬到 **~0.15**（未达 0.20）；disc 可到 +50%+。  
`promote = CAPTURE_LIFT_ride_combo_st15_f03_tp50`（research only）。
