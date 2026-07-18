# 趋势纯度 v2：分时噪声 / 路径效率

相对 v1（\|fp\|+peer 把 AMD 排在 META 之上），本版用**路径形态**打分。

## 特征（近 `path_window` 根 1m，默认 20）

| 分量 | 权重 | 含义 |
|---|---:|---|
| `path_eff` | 35% | \|净收益\| / Σ\|bar 收益\|（直线≈1，震荡≈0） |
| `range_eff` | 25% | \|收-开\| / (高-低) |
| `dir_frac` | 20% | 同向 K 线占比 |
| `adverse_ok` | 15% | 1 − 最大逆势回撤 / 50bp |
| QQQ 同号 | 5% | 轻量对齐 |

开关：`trend_purity_features=efficiency`（另有 `momentum` / `hybrid`）。

脚本：`maga7/tools/run_trend_purity_eff_ablation.py`  
产物：`results/trend_purity_eff_ablation_extend_mtm_peer3_may_jul/`

## 诊断：排序已纠正

| 信号 | mom 分 | **eff 分** |
|---|---:|---:|
| 07-09 AMD UP（抢跑亏） | 0.80 | **0.23** |
| 07-09 META UP（真趋势） | 0.63 | **0.73** |
| 07-07 TSLA DN | 0.64 | 0.47 |
| 07-07 AMD DN | 0.73 | 0.26 |

## 全期消融（May–Jul）

| 变体 | total_ret | MaxDD | scaled | 说明 |
|---|---:|---:|---:|---|
| extend_mtm_only | **+401%** | -16.2% | 0 | 底仓 |
| mom_cont（v1） | +381% | -16.2% | 7 | 缩不到 AMD |
| **eff_cont** | +233% | -16.0% | 44 | AMD×0.35，但赢家也被砍 |
| eff_cont_soft | +313% | -15.5% | 29 | 稍温和 |
| hybrid_cont | +336% | -15.6% | 32 | |
| confirm1_eff | +217% | -13.9% | 41 | |
| **full_day** | **+673%** | -12.2% | 0 | 仍是收益王 |
| full_day_eff | +387% | **-10.1%** | 38 | DD 更好，收益大削 |
| full_day_confirm1_eff | +339% | **-9.7%** | 37 | DD 最好 |

`eff_cont` 对 07-09 AMD：`pur=0.23 → size 0.20→0.07`（缩仓生效）。  
同日仍误伤 07-08 TSLA 赢家（×0.78 / w30 时更惨 ×0.35）。

## 结论

1. **特征方向对了**：效率分能区分 AMD 噪声 vs META 干净趋势。  
2. **仍非收益帕累托**：约 40+/53 笔被缩，赢家一并缩小，全期收益从 +401% 掉到 ~+230%。  
3. 叠 `full_day` 可把 MaxDD 压到约 **-10%**，但相对纯 `full_day` 少约 280pp 收益。  
4. **研究可留作风险旋钮**（偏爱 DD）；默认生产仍建议 `full_day` ± `confirm_1_mf`，不必默认开 efficiency 缩仓。
