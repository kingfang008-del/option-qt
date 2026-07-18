# 入场确认棒（entry_confirm_bars）

Rule-A（+peer/regime）触发后，再等 N 根 1m；仅当 mf/streak 仍与方向同号才成交。  
成交时钟 = `confirm_ts + bar_delay`（**不**做全日 TopK 推迟拍卖）。

## 机制

| 键 | 含义 |
|---|---|
| `trade.entry_confirm_bars` | 确认等待分钟数（1/2/3/…） |
| `trade.entry_confirm_mode` | `mf`（默认）/ `streak` / `both` |

peer / regime 仍按 **开火时刻** 判定；确认只挡「刚触发就翻面」的假火。

脚本：`maga7/tools/run_entry_confirm_ablation.py`  
产物：`results/entry_confirm_ablation_extend_mtm_peer3_may_jul/`

## 消融（May–Jul，extend_mtm_only peer3）

| 变体 | total_ret | MaxDD | n | block | 07-07~09 |
|---|---:|---:|---:|---:|---:|
| **extend_mtm_only** | **+401.1%** | -16.2% | 53 | 0 | -0.50 |
| **confirm_1_mf** | +376.6% | **-12.7%** | 50 | 3 | **-0.15** |
| confirm_2_mf | +220.3% | -15.7% | 49 | 4 | -0.28 |
| confirm_3_mf | +267.9% | -18.7% | 48 | 5 | -0.26 |
| confirm_5_mf | +59.8% | -39.3% | 42 | 12 | -0.60 |
| confirm_2_streak / both | 同 confirm_2_mf | | | | |
| **full_day** | **+673.3%** | **-12.2%** | 44 | 0 | -0.50 |
| full_day+confirm_2 | +343.5% | -15.7% | 42 | 2 | -0.28 |

## 07-07~09

| | 底仓 | confirm_1_mf |
|---|---|---|
| 07-07 NVDA | **-30.6%** | **被挡**（1 根后 mf 已不对齐） |
| 07-08 META/TSLA | -25.8% / +24.4% | -33.2% / +27.9% |
| 07-09 AMD | -18.4% | -10.0% |

确认棒成功砍掉 07-07 假 DN；07-08/09 仍在，且略推迟入场会改写期权路径。

## 结论

1. **`confirm_1_mf` 是目前唯一「像那么回事」的入场质量开关**：MaxDD -16.2%→-12.7%，focus 改善，收益只小幅回撤（+401%→+377%）。  
2. N≥2 误伤加重，全期明显变差；streak/both 与 mf 几乎同效。  
3. **勿叠 full_day**：会吃掉日历优势。相对 full_day（+673% / -12.2%），单确认棒不是收益帕累托。  
4. 研究候选可记：`entry_confirm_bars=1, mode=mf`；是否升格取决于能否在更长样本上稳住 DD，且不伤 Jan–Apr 等区间。

与拍卖对比：确认棒保留「早信号时刻附近成交」，只多等 1 分钟，因此不像 commit 那样毁掉全期。

趋势纯度缩仓（特征版负面）：[`trend_purity_sizing_research.md`](trend_purity_sizing_research.md)。
