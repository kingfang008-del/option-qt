# 2026-07-07~09：大行情标的 vs 为何选中亏损单

底仓：`extend_mtm_only`（peer3）。这三日 **不在 `full_day` 事件黑名单内**，日历禁入挡不住。

实际成交与日收益：

| 日期 | 成交 | opt ret | 当日最大行情（未吃到/吃错） |
|---|---|---|---|
| 07-07 | NVDA DN | **-30.6%** | 真趋势是 **TSLA DN -3.5%**；NVDA 全日反成 **+2.7%** |
| 07-08 | META DN / TSLA DN | -25.8% / **+24.4%** | 真大行情是 **NVDA UP +4.0%**（信号晚且 QQQ 不对齐） |
| 07-09 | AMD UP | **-18.4%** | 真大行情是 **META UP +8.9%**（信号太晚，TopK 已满） |

## 结论（为什么选中亏损标的）

1. **07-07：最早信号抓错方向（假 DN）**  
   - TopK 按时间取最早：NVDA DN@10:35（peer=4，QQQ 仍红）→ 成交。  
   - 持仓 30 分钟内 NVDA **反弹 +0.64%**，全日收 **+2.7%** → put 大亏。  
   - 真正顺势的大跌是 **TSLA DN**（全日 -3.5%），但 RuleA 到 **11:21** 才触发，已是 TopK #3，进不来。

2. **07-08：方向对但期权仍亏 + 错过主升**  
   - 早盘 DN 池：META / TSLA 最先过 peer → 成交；TSLA 靠 T+45 赚到，META 持仓窗微反弹导致 put 亏。  
   - 全日最大涨幅 **NVDA +4%** 的 UP 信号要到 **12:58**，且当时 **QQQ from_prev 未对齐** → 被 regime 挡掉。

3. **07-09：抢跑弱趋势，错过强趋势**  
   - AMD UP@10:33 最早进 TopK；全日虽 +1.5%，但持仓窗 **-0.22%**（早盘冲高回落）→ call 亏。  
   - **META +8.9%** 要到 UP@12:21 才 RuleA，已是 #3；AMZN/MSFT/AAPL 大涨但窗内根本没出 RuleA。

共性：系统优化的是 **「最早满足 RuleA+peer3」**，不是 **「当日最大动量」**。  
强行情常出现更晚，或早盘假方向占用 TopK 名额。

---

## 分日明细

## 2026-07-07
QQQ day=-0.79%  fp@10:35=-2.32%
| symbol | day% | max_up% | max_dn% | morn% | aft% | RuleA | peer | qqq_ok | EOD |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| TSLA | -3.54 | +0.30 | -3.71 | -0.56 | -1.14 | DN@11:21 | 3 | True | DN |
| NVDA | +2.67 | +3.37 | -0.27 | +1.54 | +1.08 | DN@10:35 | 4 | True | UP |
| META | +1.57 | +3.05 | -0.36 | -0.87 | +1.71 | - | - | None | UP |
| AAPL | -1.28 | +0.09 | -1.29 | +0.93 | -1.13 | - | - | None | DN |
| MSFT | -0.89 | +0.71 | -0.94 | -0.04 | -1.29 | - | - | None | DN |
| AMD | -0.48 | +0.96 | -3.00 | +0.84 | +0.08 | DN@10:46 | 5 | True | FLAT |
| GOOGL | -0.35 | +1.22 | -0.71 | -0.44 | -0.56 | - | - | None | FLAT |
| AMZN | -0.13 | +1.03 | -1.41 | -0.49 | +0.91 | - | - | None | FLAT |

Eligible (RuleA+peer3) by time → TopK takes first 2:
  #1 DN@10:35 NVDA day=+2.67% EOD=UP  **DIR vs EOD mismatch**
  #2 DN@10:46 AMD day=-0.48% EOD=FLAT
  #3 DN@11:21 TSLA day=-3.54% EOD=DN

Actual picks:
  10:36 NVDA DN opt=-0.306 T+30 peer=4 | hold stock=+0.64% QQQ=+0.13% | EOD day=+2.67% (UP)

Big movers (|day|≥1.2%) — why not / why lose:
  TSLA day=-3.54% sig=DN@11:21 → 有信号但 TopK 已满（排名#3）
  NVDA day=+2.67% sig=DN@10:35 → 被选中; 信号方向与全日收盘相反; 持仓晨段已在反弹
  META day=+1.57% sig=- → 无 RuleA（窗内 streak/门槛未达标）
  AAPL day=-1.28% sig=- → 无 RuleA（窗内 streak/门槛未达标）

## 2026-07-08
QQQ day=+0.72%  fp@10:35=-0.41%
| symbol | day% | max_up% | max_dn% | morn% | aft% | RuleA | peer | qqq_ok | EOD |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| NVDA | +3.95 | +4.43 | -0.08 | +0.24 | +3.33 | UP@12:58 | 6 | False | UP |
| GOOGL | -1.17 | +0.00 | -2.15 | -0.47 | +0.33 | - | - | None | DN |
| META | -1.09 | +0.00 | -1.87 | -0.17 | -0.11 | DN@10:41 | 5 | True | DN |
| AAPL | +1.01 | +1.48 | -0.97 | +0.38 | +0.97 | - | - | None | UP |
| AMD | +1.01 | +1.84 | -2.63 | -2.32 | +2.68 | DN@11:20 | 3 | True | UP |
| TSLA | -0.87 | +0.33 | -1.69 | -0.44 | +0.06 | DN@10:43 | 6 | True | DN |
| MSFT | +0.16 | +0.61 | -0.29 | +0.11 | +0.06 | - | - | None | FLAT |
| AMZN | +0.00 | +0.26 | -1.22 | -0.70 | +1.06 | DN@11:30 | 7 | True | FLAT |

Eligible (RuleA+peer3) by time → TopK takes first 2:
  #1 DN@10:41 META day=-1.09% EOD=DN
  #2 DN@10:43 TSLA day=-0.87% EOD=DN
  #3 DN@11:20 AMD day=+1.01% EOD=UP  **DIR vs EOD mismatch**
  #4 DN@11:30 AMZN day=+0.00% EOD=FLAT
  #5 UP@12:58 NVDA day=+3.95% EOD=UP

Actual picks:
  10:42 META DN opt=-0.258 T+30 peer=5 | hold stock=+0.27% QQQ=-0.17% | EOD day=-1.09% (DN)
  10:44 TSLA DN opt=+0.244 T+45 peer=6 | hold stock=-0.50% QQQ=-0.67% | EOD day=-0.87% (DN)

Big movers (|day|≥1.2%) — why not / why lose:
  NVDA day=+3.95% sig=UP@12:58 → QQQ from_prev 不对齐（且过晚）

## 2026-07-09
QQQ day=+0.86%  fp@10:35=+1.06%
| symbol | day% | max_up% | max_dn% | morn% | aft% | RuleA | peer | qqq_ok | EOD |
|---|---:|---:|---:|---:|---:|---|---:|---|---|
| META | +8.90 | +9.18 | -0.07 | +2.39 | +3.82 | UP@12:21 | 3 | True | UP |
| AMZN | +3.20 | +3.32 | -0.30 | +0.40 | +2.24 | - | - | None | UP |
| TSLA | +3.08 | +3.38 | -0.73 | +1.43 | +1.78 | UP@12:46 | 3 | True | UP |
| MSFT | +2.64 | +2.69 | -0.21 | +0.93 | +1.16 | - | - | None | UP |
| AAPL | +2.18 | +2.26 | -0.20 | +0.85 | +0.40 | - | - | None | UP |
| AMD | +1.51 | +3.83 | +0.00 | +0.46 | -1.93 | UP@10:33 | 5 | True | UP |
| GOOGL | +0.66 | +0.72 | -1.45 | -0.41 | +1.71 | DN@11:46 | 4 | False | UP |
| NVDA | -0.25 | +0.63 | -2.04 | +0.93 | +0.50 | - | - | None | FLAT |

Eligible (RuleA+peer3) by time → TopK takes first 2:
  #1 UP@10:33 AMD day=+1.51% EOD=UP
  #2 DN@11:46 GOOGL day=+0.66% EOD=UP（qqq_ok=False，实盘会被 qqq_align 挡）
  #3 UP@12:21 META day=+8.90% EOD=UP
  #4 UP@12:46 TSLA day=+3.08% EOD=UP

Actual picks:
  10:34 AMD UP opt=-0.184 T+30 peer=5 | hold stock=-0.22% QQQ=+0.27% | EOD day=+1.51% (UP)

Big movers (|day|≥1.2%) — why not / why lose:
  META day=+8.90% sig=UP@12:21 → 有信号但 TopK 已满（排名#3）
  AMZN day=+3.20% sig=- → 无 RuleA（窗内 streak/门槛未达标）
  TSLA day=+3.08% sig=UP@12:46 → 有信号但 TopK 已满（排名#4）
  MSFT day=+2.64% sig=- → 无 RuleA（窗内 streak/门槛未达标）
  AAPL day=+2.18% sig=- → 无 RuleA（窗内 streak/门槛未达标）
  AMD day=+1.51% sig=UP@10:33 → 被选中（持仓窗回撤）

## 亏损单持仓窗（股 vs 信号）
- 2026-07-07 NVDA DN opt=-0.306: entry≈191.55 fp=-2.08% → exit≈192.77（持仓 +0.64%），全日收涨
- 2026-07-08 META DN opt=-0.258: entry≈601.72 → exit≈603.33（持仓 +0.27%），方向对但 put 仍亏
- 2026-07-09 AMD UP opt=-0.184: entry≈556.77 → exit≈555.54（持仓 -0.22%），随后日线仍收涨、META 主升浪在午后

相关：[`event_calendar_full_day.md`](event_calendar_full_day.md)（事件禁入不覆盖这三日）。  
后到信号挤仓消融（结论负面）：[`displace_later_research.md`](displace_later_research.md)。  
更快认错 mtm_floor/mf_flip（局部止血、全局负面）：[`early_cut_mtm_floor_mf_flip_research.md`](early_cut_mtm_floor_mf_flip_research.md)。  
延迟提交 TopK 拍卖（选型可改善、全期大幅负面）：[`topk_commit_auction_research.md`](topk_commit_auction_research.md)。  
入场确认棒（`confirm_1_mf` 略改善 DD、挡 07-07 NVDA）：[`entry_confirm_bars_research.md`](entry_confirm_bars_research.md)。  
趋势纯度缩仓 v1（开火时亏单分反而高）：[`trend_purity_sizing_research.md`](trend_purity_sizing_research.md)。  
趋势纯度 v2 路径效率（排序纠正、全期仍伤收益）：[`trend_purity_efficiency_research.md`](trend_purity_efficiency_research.md)。  
简单剔 AMD / 加重 NVDA·META（剔 AMD 全期更差）：[`symbol_exclude_amd_research.md`](symbol_exclude_amd_research.md)。
