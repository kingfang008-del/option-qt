# 2026-04~07：大行情漏抓扫描（freeze baseline，TCN 关闭）

> Profile：`single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1`  
> 工具：`python -m maga7.tools.scan_missed_movers --start-date 2026-04-01 --end-date 2026-07-16`  
> 产出：`maga7/results/missed_movers_apr_jul/`

TCN 保持 **enabled=false**。本扫描只回答：哪些大行情没吃到、为什么、有没有「整体提收益 / 降亏损」的抓手。

---

## 1. 口径

| 项 | 定义 |
|---|---|
| 大行情 | 标的 RTH \|day_ret\| ≥ **2%**（universe 8 票） |
| 抓住 | 当日对该标的有同向成交（`traded_dir == EOD`） |
| 对照成交 | 同 profile 离线 replay（Apr–Jul 独立起盘） |

Apr–Jul 基线（本窗起盘）：**n=77**，total_ret≈**+1779%**，MaxDD≈**−24.6%**，win≈63.6%。  
（May–Jul 连续窗仍是 freeze 公布的 +875% / −13.2%。）

---

## 2. 总量

| 指标 | 值 |
|---|---:|
| 交易日 | 73 |
| 大行情 symbol-day | 155 |
| 抓住 | 40（**25.8%**） |
| 漏抓 | 115 |
| 「当日最大行情」未成交天数 | **45** |
| 亏损日且最大行情未成交 | **12** |
| 大行情上做反方向 | 3 |

### 漏抓原因（symbol-day，\|day\|≥2%）

| miss_reason | n | 含义 |
|---|---:|---|
| `no_rule_a` | 36 | 信号窗内根本没有 Rule-A |
| `regime:qqq_align_*` | 31 | 有 Rule-A，但 QQQ 对齐挡掉 |
| `eligible_topk_but_no_fill` | 20 | 扫描侧认为可进 TopK，实际无成交（见 §3） |
| `topk_full_rank3+` | 22 | 有资格但来得晚，坑位已满 |
| `peer_fail` | 3 | peer3 不够 |
| `traded_wrong_dir` | 3 | 吃到了，但是假方向 |

\|day\|≥4% 时抓住率升到 ~37%，说明**更大的行情并不是更容易按现规则吃到**，仍大量卡在 regime / TopK / 无信号。

---

## 3. 关键机制：TopK 坑位被「挡掉的 #2」空转

Replay 默认 `event_sigs = topk`（按 **Rule-A 时间最早** 取 2 个，**先于** peer/regime 过滤）。

典型：**2026-07-09**

| 顺序 | 信号 | 结果 |
|---|---|---|
| #1 | AMD UP@10:33 | 成交 → **−18.4%**（弱趋势） |
| #2 | GOOGL DN@11:46 | 进了 TopK，但 **QQQ 不对齐 → 入场被挡** |
| （未进 TopK） | **META UP@12:21** | 全日 **+8.9%**，扫描侧 eligible，但 **根本不在 top2 事件流** |
| #3 | TSLA UP@12:46 | TopK 已满 |

→ 不是「META 太晚排第 3」，而是 **#2 名额被后来会被挡掉的 GOOGL 占住，且不会回填**。

同类「TopK 有人被 regime 挡 + 后面仍有同向大行情候选」见  
`results/missed_movers_apr_jul/topk_slot_wasted.csv`。

### 粗暴 `all_first` 不可用

| 窗 | baseline topk | all_first |
|---|---:|---:|
| May–Jul | **+874.7%** / −13.2% / n=48 | +496% / −22.7% / n=94 |
| Apr–Jul | +1779% / −24.6% / n=77 | +1078% / −25.0% / n=136 |

打开全宇宙会灌入噪音，**整体变差**。需要的是 **「TopK 候选被挡时顺延下一个」**，不是 all_first。

---

## 4. 其它高价值漏抓簇

### 4.1 数据缺口（AMD）— 已查清

`eligible_topk_but_no_fill` 里 **AMD=14/20**。详见 [`amd_1s_gap_backfill.md`](amd_1s_gap_backfill.md)。

- **正股 1s**：17 个缺口日 **已补齐**。  
- **真正挡成交的**：这些日本地 `option_1m`/`day_iv` **没有 DTE 0/1/2**（min_dte=3~4）→ 开盘锁约锁不出 AMD → 即使有正股 1s 也无法按 freeze 口径成交。  
- 07-06 AMD +4.5% TopK#1 仍吃不到，属 **短链数据/可交易性** 问题，不是 TopK 逻辑单独能修。

### 4.2 事件黑名单日上的大行情

halt 日仍出现 \|day\|≥2%：05-19/20/21、06-12/16/17/18、07-10。  
例如 **06-16 AMD −6.8%**（TopK#1 本可做 DN）被 `full_day` 禁掉——这是已知日历代价。

### 4.3 QQQ regime 挡住「方向其实对」的大行情

regime 挡掉且 `dir_match_eod=True`：**30** 次（如 04-02 AMD +6.2%、07-08 NVDA +3.9%）。  
松绑 QQQ 可能抓到这些，但历史 ablation 表明容易引入更多假突破——**不要无扫描直接关对齐**。

### 4.4 亏损日 ∩ 最大行情未吃到（优先复盘）

| 日期 | 成交（亏） | 当日最大行情 | 机制摘要 |
|---|---|---|---|
| 04-06 | AAPL −62% | TSLA −3.2% | TSLA 被 QQQ 挡；AAPL 假方向 |
| 04-21 | MSFT −46% | AAPL −2.2% | TopK 选错弱票 |
| 07-07 | NVDA DN −31% | TSLA −3.5% | 最早假 DN；真跌晚到 |
| 07-09 | AMD −18% | **META +8.9%** | TopK 坑位空转（§3） |
| 04-02 | TSLA −14% | AMD +6.2% | AMD 被 QQQ 挡 |
| 07-08 | META−26% / TSLA+24% | NVDA +3.9% | NVDA 被 QQQ 挡 |

完整表：`loss_vs_mover.csv`。

---

## 5. 对「提收益 / 降亏损」的判断

| 方向 | 预期 | 建议 |
|---|---|---|
| TopK **blocked-backfill**（挡掉则顺延） | 专打 07-09 META 类；不扩大噪音宇宙 | **优先做小 ablation** |
| 补齐 AMD（及缺口股）1s | 直接兑现已进 TopK 的大行情 | 数据工程，低争议 |
| `all_first` / 关 QQQ | 扫描上像能多抓，replay **整体变差** | **不做** |
| 事件日历微调 | 夺回 06-16 等，但可能放回更差事件日 | 单日 ROI 评估后再动 |
| displace_later / 晚信号挤仓 | 已有研究；对 topk_full 有关 | 与 backfill 二选一或组合 |

**结论（本轮）**：Apr–Jul 大行情抓住率约 1/4；最干净的候选杠杆是  
**（1）TopK 被挡回填** + **（2）AMD 1s 缺口**。  
不要为了漏抓去开 TCN 或 all_first。

### 回填 A/B（已做）

见 [`topk_backfill_research.md`](topk_backfill_research.md)。  
`topk_backfill_on_block=true`：May–Jul 收益 +1084%（高于 +875%），但 MaxDD 变差；Feb–Apr 变差。  
07-09 META 成功顺延，期权窗内仍 **−8%**。**不进 freeze。**

---

## 6. 文件

| 路径 | 内容 |
|---|---|
| `results/missed_movers_apr_jul/symbol_day.csv` | 每日每票行情 + RuleA/peer/regime |
| `daily_rank.csv` | 每日最大行情 vs 成交 |
| `missed_big.csv` | \|day\|≥2% 及 miss_reason |
| `loss_vs_mover.csv` | 亏损日 vs 最大行情 |
| `topk_slot_wasted.csv` | TopK 被挡空转日 |
| `baseline_trades.csv` / `baseline_daily.csv` | 本窗 freeze 成交 |
| `tools/scan_missed_movers.py` | 可复跑 |
