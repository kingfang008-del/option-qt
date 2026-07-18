# 后到信号挤仓（displace_on_later）

针对 07-07~09「早假信号占坑、午后大行情进不来」做的研究机制。

## 机制

| 键 | 含义 |
|---|---|
| `trade.displace_on_later` | 仓位已满时，允许后到的合格 Rule-A **强制平掉最老持仓** 再入场 |
| `trade.displace_universe` | 默认 `all_first`：事件宇宙=全日各标的首个 Rule-A（不再只吃 TopK 最早 2 个） |
| `trade.displace_score` | `none`：有空位/挤出即可；`abs_from_prev`：要求新信号 \|from_prev\| ≥ 旧 × ratio |
| `trade.late_signal_universe_all_first` | 只扩大宇宙、**不挤仓**（后信号仅在前仓自然平后吃空位） |

实现要点（offline replay）：

1. `simulate_trade(..., force_exit_ts=)` → 原因 `DISPLACE`
2. 满仓时挑 **最早入场且仍未平** 的仓，按 `force_exit_ts=新入场时刻` 重算盈亏并回写权益
3. Live / stream **尚未接线**（先看 offline 是否值得）

脚本：`maga7/tools/run_displace_later_ablation.py`  
产物：`maga7/results/displace_later_ablation_extend_mtm_peer3_may_jul/`

## 消融（May–Jul，extend_mtm_only peer3）

| 变体 | total_ret | MaxDD | n | n_displace | 07-07~09 ret 和 |
|---|---:|---:|---:|---:|---:|
| **extend_mtm_only** | **+401.1%** | **-16.2%** | 53 | 0 | -0.50 |
| all_first_no_displace | +206.1% | -38.6% | 103 | 0 | -1.03 |
| displace_oldest | +136.7% | -44.5% | 106 | 6 | -1.03 |
| displace_fp | +158.5% | -44.5% | 104 | 3 | -1.03 |
| **full_day** | **+673.3%** | **-12.2%** | 44 | 0 | -0.50 |
| full_day_displace | +311.3% | -29.2% | 93 | 5 | -1.03 |

## 为何救不了 07-07~09

时间线（delay=60s）：

| 日 | 早仓 | 自然平 | 「大行情」信号 | 是否重叠 |
|---|---|---|---|---|
| 07-07 | NVDA DN@10:36 | **11:06 T+30** | TSLA DN@11:22 | **否** → 无需挤，直接空位进 TSLA |
| 07-09 | AMD UP@10:34 | **11:04 T+30** | META UP@12:22 | **否** → 空位进 META |
| 07-08 | META/TSLA 早盘 DN | 正常持有 | NVDA UP@12:58 | 被 **QQQ regime** 挡，挤仓也进不来 |

扩大宇宙后实际成交：

- 07-07：NVDA **-30.6%** + TSLA **-14.7%**（真趋势标的期权窗内仍亏）
- 07-09：AMD **-18.4%** + META **-8.0%**（全日大涨，但入场后 30m 期权仍亏）

结论：**坑位不是主因**；即便吃到「大行情」标的，持仓窗期权收益也未必正。全样本 6 次 DISPLACE 还曾截断赢家（如 06-09 TSLA）。

## 建议

- **不上生产 / 不叠 full_day**。相对 `extend_mtm_only` 与 `full_day` 均严格更差。
- 07-07~09：更快认错已测（负面，见 [`early_cut_mtm_floor_mf_flip_research.md`](early_cut_mtm_floor_mf_flip_research.md)）；下一步更应挖 **入场质量 / 日级过滤**。
- 代码保留作研究开关；默认关闭。
