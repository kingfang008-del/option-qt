# VRP-lite soft prior（买方侧）— 双窗否决

**状态：REJECT（2026-07-24）**  
不是 Short Strangle / 卖 VRP；只在 **买期权** 时，用 QQQ `IV − RV` 过富作为缩仓/跳过 prior。

## 定义

| 项 | 值 |
|----|-----|
| IV | QQQ `bucketed_v7` `options_struc_atm_iv` @ 10:30 |
| RV | QQQ RTH last print，根 `/mnt/s990/data/raw_1s/stocks`（**不用**左标签 `spnq_train`） |
| rich | 因果滚动分位 ≥70% 且 `VRP ≥ 0` |
| 动作 | `scale50`（×0.5）或 `skip` |

代码：`maga7/common/vrp_prior.py` · 消融：`tools/run_vrp_soft_prior_ablation.py`  
Replay：`trade.vrp_size_scale` + `stock_by` from 1s。

## 双窗（research_baseline，1s→1m）

| window | variant | total_ret | maxdd | n | vs_off |
|--------|---------|-----------|-------|---|--------|
| May–Jul9 | off | **+15.29** | −5.1% | 48 | — |
| May–Jul9 | scale50 | +13.22 | −5.1% | 48 | **−2.06** |
| May–Jul9 | skip | +11.30 | −5.1% | 40 | **−3.98** |
| Jan–Mar | off | +0.96 | −18.4% | 56 | — |
| Jan–Mar | scale50 | +0.84 | −14.4% | 56 | −0.12 |
| Jan–Mar | skip | +0.62 | −13.3% | 32 | −0.34 |

Rich 日：强窗 10 / 弱窗 22（diag 见结果目录 `vrp_diag.json`）。

## 裁决

- 强窗 **明确伤收益**；弱窗仅改善 maxDD，不能补偿。  
- **不挂脊骨**；catalog `vrp_soft_prior = REJECT`。  
- 曲面仍可作 hold/path 特征；全局 QQQ VRP→size 不作默认 risk expert。

产物：`/mnt/s990/data/maga7/results/research_vrp_soft_prior_dual_v1/`
