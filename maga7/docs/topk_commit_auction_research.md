# 延迟提交 TopK（commit 拍卖）

针对「滑动窗口命中了真趋势，但最早信号占坑」：在 `commit_tod` 前只收集候选，提交时刻按 score 取 TopK，**在 commit 时钟入场**（不是事后挤仓）。

## 机制

| 键 | 含义 |
|---|---|
| `trade.topk_commit_tod` | 如 `11:30`；此前不交易 |
| `trade.topk_rank` | `abs_from_prev`（默认）/ `peer_n` / `peer_fp` |
| `trade.topk_post_commit_fill` | commit 后若还有并发空位，允许更晚信号补位（默认 true） |

开启后事件宇宙自动用 **all_first**（各标的当日首个 Rule-A）。  
入场时刻 = `commit_tod + bar_delay`（选型信息 ≤ commit，因果成立）。

脚本：`maga7/tools/run_topk_commit_ablation.py`  
产物：`results/topk_commit_ablation_extend_mtm_peer3_may_jul/`

## 消融（May–Jul，extend_mtm_only peer3）

| 变体 | total_ret | MaxDD | n | 07-07~09 |
|---|---:|---:|---:|---:|
| **extend_mtm_only** | **+401.1%** | **-16.2%** | 53 | -0.50 |
| commit_1100_fp | -2.4% | -40.1% | 95 | -0.94 |
| commit_1130_fp | -41.3% | -56.4% | 89 | -0.69 |
| commit_1200_fp | -29.4% | -44.1% | 80 | -0.97 |
| commit_1230_fp | -52.8% | -60.7% | 75 | **+0.20** |
| commit_1130_peer_fp | -43.4% | -60.1% | 89 | -0.88 |
| commit_1130_fp_nofill | -25.4% | -44.4% | 58 | -0.72 |
| **full_day** | **+673.3%** | **-12.2%** | 44 | -0.50 |
| full_day+commit_1130 | -2.6% | -38.1% | 77 | -0.69 |

## 07-07~09：选型对了，兑现仍差 / 全期被毁

| 日 | 底仓 | commit_1130_fp | 说明 |
|---|---|---|---|
| 07-07 | NVDA **-30.6%** | **TSLA -19.0%** | 拍卖选对了 TSLA，推迟入场后 put 仍亏 |
| 07-08 | META/TSLA | AMD/META 更差 | 推迟毁掉原 TSLA +24% 赢家 |
| 07-09 | AMD -18.4% | AMD 略好 + META fill 仍 -8% | 12:30 才能把 META 选进拍卖池 |

`commit_1230` 的 focus 转正，主要靠 07-08 META 极晚入场碰巧 TP，**不是稳定机制**；全期 -53%。

## 结论

1. **拍卖能改选型**（07-07 从 NVDA→TSLA），证明「占坑」叙事在选型层成立。  
2. **推迟入场代价过大**：策略收益大量来自早盘 T+30/T+45 窗口；commit≥11:00 把全期打成负期望。  
3. **post_commit_fill** 再次引入 all_first 噪声，nofill 仍远逊底仓。  
4. **不升格**；勿叠 full_day。

若还要挖「质量」：应在 **不推迟入场时刻** 的前提下过滤假火（确认棒 / 微结构弃权），而不是把成交时钟往后拖。  
已测确认棒：[`entry_confirm_bars_research.md`](entry_confirm_bars_research.md)（`confirm_1_mf` 略有苗头）。
