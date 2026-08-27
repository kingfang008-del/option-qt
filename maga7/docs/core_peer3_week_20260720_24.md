# CORE peer3：近一周 replay × 市场状态（波动回调）

> Profile：`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`  
> 产物：`results/research_core_peer3_week_20260720_24/`  
> 窗：**2026-07-20 → 07-24**（5 个交易日）  
> 说明：日历「过去一周」含 07-27/28/29，但 **stock_1s 仅到 07-24**，后三日未入 replay。

## 1. Replay（先看账）

| 路径 | n 笔 | total_ret | MaxDD | 胜率 |
|------|-----:|----------:|------:|-----:|
| **Offline（主账）** | **1** | **−2.6%** | −2.6% | 0% |
| Stream | 3 | −11.4% | −11.4% | 0% |

Offline 唯一成交：

| 日 | 标的 | 向 | 退出 | 单笔 ret | size |
|----|------|----|------|--------:|-----:|
| 07-22 | MSFT | DN | T+30 | **−12.8%** | 0.20 |

Stream 多出 2 笔（parity **ok=false**）：07-21 AMD UP −25.9%、07-24 TSLA DN −17.7%（均 T+30）。  
近一周分析以 **offline** 为准；stream 差额另记对拍债，不混进基线归因。

漏斗（offline）：

- TopK 信号 10 → 成交 **1**
- 挡枪：`range_stall`×2、`fo_lod_chase`×2、`regime`×2、`overnight_gap`×1、`peer`×1、`stock_path_confirm`×1
- Hunt：**0**；Watchdog 日态无触发
- 07-22 `event_symbol_blackout`：TSLA/GOOGL（AH earnings）

→ 回调周里 CORE **大多空手**；进场的那一笔也没赚到。

## 2. 市场状态（波动回调）

相对前一周高点（QQQ high≈724），07-24 收盘 **684（约 −5.5%）**；滚动峰回撤约 **−4.9%**。

| 标的 | 07-20..24 日收益连乘 | 日均 range | 备注 |
|------|--------------------:|----------:|------|
| QQQ | **−1.6%** | 1.3% | 周中反抽后继续阴跌 |
| TSLA | **−11.0%** | 4.6% | 回调核心 |
| META | −6.9% | 2.8% | |
| AMZN / MSFT / GOOGL | −3~−4% | ~2–3% | |
| NVDA | +1.0% | 3.1% | 22 日大阳对冲 |
| AAPL | +2.1% | 2.4% | 24 日强 |
| AMD | ≈0 | **5.4%** | 高波动横盘 |

广度：5 天里 Mag7 日红票数 5–8/8；**07-23 全红**。  
前一周（07-13..17）QQQ 已先跌约 −2%，本周是 **下跌延续 + 个股发散**，不是趋势市那类「齐涨吃期权」环境。

## 3. 对齐解读

1. **行情侧**：确认处于波动回调期（QQQ 离近高 −5%+，广度差，TSLA 重挫）。  
2. **策略侧**：基线门控在回调里 **大量拒单**（stall / FO-LOD chase / regime / gap）→ 成交极稀；稀成交还亏 → 近一周 **拖累净值，但幅度远小于「满仓挨打」**。  
3. **与 May–Jul 大账对比**：强窗靠趋势段堆复利；这种回调切片本来就不贡献 alpha，问题是 **稀成交是否错过反向机会、以及 T+30 是否在回调里系统性砍亏不够/砍早**。  
4. **数据债**：07-27..29 需补 stock_1s/quote 后再续跑，才能覆盖「完整过去一周」。

## 4. 逐项推进（已做）

### ① 补 07-27..29 → **BLOCKED**

- `stock_1s` / quote：**无**；环境无 `MASSIVE_API_KEY` / `POLYGON_API_KEY`
- Live 代理（**非** stock_1s offline 对拍，仅作参考）：
  - 07-27 `fused_replay_core_baseline_0727`：n=6，**+18.3%**（5TP/1SL）
  - 07-27 `core_only` / 07-28 fused：**0 笔 / 0%**
- 要真正扩窗：

```bash
export MASSIVE_API_KEY=...
python preprocess/download/download_stock_1s.py \
  --symbols QQQ,NVDA,TSLA,AAPL,AMZN,META,MSFT,AMD,GOOGL,VIXY \
  --start-date 2026-07-27 --end-date 2026-07-29
# 再补 open-lock quote 后重跑 stream_parity
```

### ② TopK 被拒拆解 → **门控对 T+30 大致正确**

| | n |
|--|--:|
| TopK | 10 |
| 成交 | 1（MSFT DN） |
| 未成交 | 9 |

- **9/9** 未成交的全日 `fav_day ≥ +1%`（方向对）  
- **0/9** 未成交的 `fav_30m ≥ +0.5%`（T+30 正股几乎不顺）  
- → 用「全日大跌/大涨」衡量会误判错过；CORE 是 **T+30 持仓**，挡单与 30m 现实一致  
- 已定位 regime：`qqq_align_up` 挡 2 笔（07-22 AMD UP、07-24 AAPL UP）  
- 其余计数：`range_stall`×2、`fo_lod_chase`×2、`overnight_gap`×1、`peer`×1、`path_confirm`×1  

产物：`topk_signals.csv` / `topk_vs_day_oracle.csv` / `step2_topk_reject_summary.json`

### ③ MSFT DN 07-22 路径 → **延长持仓救不了；fail-fast 略减损**

实际：**T+30 −12.8%**。入场后正股 2h 内几乎不给 DN（最大顺风约 0.3%）。

| 方案 | ret | 备注 |
|------|----:|------|
| 实际 T+30 | **−12.8%** | 基线 |
| quote clock 45m | −24.0% | 更差 |
| quote clock 60m | +0.8% | 偶然翻平，不可当规则 |
| fail-fast SL8%@~10m | **−10.2%** | 略好于 T+30 |

结论：这笔是 **方向对、节奏错**；优先看短失败止损/确认 abort，不是加长 hold。

产物：`msft_0722_path_alts.csv` / `step3_msft_path_summary.json`

## 5. 行情形态：头一小时跌到位 → 横盘（已验证）

用户判断与本周 1s 一致：

| Mag7（07-20..24） | 中位 |
|-------------------|-----:|
| \|开盘→10:30\| | **1.07%** |
| \|10:30→收盘\| | **0.51%** |
| 全日位移落在 H1 的比例（中位） | **~82%** |
| \|H1\| > \|午后\| 的 day×标的 | **68%** |
| 10:30 后 \|收益\| &lt; 50bp（横盘） | **50%** |

大跌例子（H1 已完成大部分）：TSLA 07-23 H1 **−5.0%** / 其后 −1.4%；MSFT 07-22 H1 **−2.5%** / 其后 **+0.13%**（正是那笔 T+30 亏损单）。

TopK 全部 **≥10:30**；其中 **5/10** 明确是「H1 已顺风 ≥1%，午后 &lt;50bp」——信号打在横盘段。

→ 回调周 CORE 不是「门控太严错过全日」，而是 **默认 10:30 席位机制天生偏晚**：趋势段还能吃延续，**早盘一次性定价日**就会买到横盘期权衰减。

产物：`session_frontload_stats.csv` / `topk_vs_h1_frontload.csv` / `frontload_thesis_summary.json`

## 6. 综合

回调周 offline **−2.6%** = 稀成交 + 晚入场吃横盘；被拒 TopK 用全日看像错过，用 **30m/H1 视角**则门控合理、问题在 **时钟**。  
策略含义（未实施）：识别「H1 已定价」日 → CORE 降权/停手，或另开 **09:30–10:30** 短持袖；不要为全日方向放松 stall。  
数据债：API 补 07-27..29 后复核同一 frontload 形态。

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.run_stream_parity \
  --profile maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json \
  --scheme single --stock-source stock_1s \
  --start-date 2026-07-20 --end-date 2026-07-24 \
  --tag research_core_peer3_week_20260720_24
```
