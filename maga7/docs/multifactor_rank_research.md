# 多因子重排（研究）— 能否把 META/NVDA 打进 top2

**日期：** 2026-07-18  
**状态：** research only · **默认 off** · **不进基线**  
**问题：** 宏观单因子不够；用多因子截面打分，盘中能否把 07-08 NVDA / 07-09 META 排进 UP top2？

## 因子（因果 1m）

| 因子 | 含义 | 默认权重 |
|------|------|--------:|
| `fp` | from_prev | 0.25 |
| `accel` | fp − fp<sub>t−15m</sub> | 0.25 |
| `rs` | fp − 池内中位 fp | 0.20 |
| `vol_x` | cum$ / 同刻 10 日中位 | 0.15 |
| `qqq_div` | fp − QQQ fp | 0.10 |
| `reclaim` | 相对当日最低 fp 的修复幅度 | 0.05 |

入池软门：`|fp|≥0.5%`（`fp_gate`）。得分 = 截面 winsorized z 的加权和。  
代码：`maga7/common/multifactor_rank.py`  
扫描：`python -m maga7.tools.scan_multifactor_top2`  
产物：`maga7/results/multifactor_top2_scan/summary.json`

## Focus 结果

| 日 | 标的 | 进 top2？ | 首次进 top2 | vs Rule-A | regime | 备注 |
|----|------|:--------:|-------------|-----------|--------|------|
| **07-08** | **NVDA UP** | **是** | **10:53（#1）** | Rule-A 12:58（早 **125m**） | **QQQ 挡** | 12:30 仍为 UP#1；能定位，**过不了现门** |
| **07-09** | **META UP** | **是** | **11:41（#2）** | Rule-A 12:21（早 **40m**） | **ok** | 12:00/12:30 仍 #2（AMD #1）；13:00 被 TSLA 挤到 #3 |

对照（最早 Rule-A TopK vs 午后多因子）：

| 日 | Rule-A earliest top2 | 多因子 UP@12:30 |
|----|----------------------|-----------------|
| 07-08 | META DN / TSLA DN | **NVDA** |
| 07-09 | AMD UP / GOOGL DN | **AMD / META** |

## 解读

1. **比纯宏观放量强**：META 不再只是「看见但排第 3」——在 11:41 起就能稳定占 UP **#2**；NVDA 也能在午前被标成 UP 第一（宏观放量做不到）。  
2. **仍不等于基线能成交**：  
   - NVDA：多因子选得出，**QQQ align 仍挡**。  
   - META：选得出且过 regime，但要进成交还需 **改 TopK 规则**（重排或被挡回填），且期权兑现另验。  
3. **首次 top2 可能偏早、池子很薄**（例如仅 1 票过 `fp_gate` 时分数退化）——落地时应加「连续 N 分钟维持 top2」或抬高 `fp_gate`。

## 方案 B 已实现：`topk_mf_backfill_on_block`

开关（默认 **false**）：`trade.topk_mf_backfill_on_block=true`

| | time backfill（旧） | **mf backfill（新）** |
|---|---|---|
| 事件流 | 全部 first-Rule-A 按时间 | **先 earliest TopK**，再剩余票按多因子分排序 |
| 名额 | 过闸才占席 | 同左 |
| 弱窗 | 易灌噪音 | 更严，少伤弱窗 |

```bash
python -m maga7.tools.run_topk_mf_backfill_ablation \
  --out maga7/results/topk_mf_backfill_ablation
```

### 双窗 scoreboard（当前 L2 基线 profile，2026-07-18）

| variant | May–Jul vs base | Feb–Apr vs base | focus 07-07..10 |
|---------|----------------:|----------------:|-----------------|
| baseline | 100%（+1255%） | 100% | −10.5%（4 笔） |
| time_backfill | 99.4% | **87% FAIL** | −11.9%（+META −8%） |
| **mf_backfill** | **98.9%** | **98.1%** | −11.9%（+META −8%） |

**Verdict: `PASS_CANDIDATE`（仅 mf_backfill）** — 双窗 ≥95% vs 同档 L2；time_backfill 弱窗仍 REJECT。

Focus：mf/time 均顺延成交 **07-09 META（约 −8%）**；**07-08 NVDA 仍无**（TopK 两席已被早盘 DN 填满，不是回填问题，是 QQQ/占坑）。

### 升线？

仍 **默认不写进** `peer3_v1`：focus 段多一笔仍亏；stream 未接 mf 排序；需你显式确认再开。

## 和非目标

- 不为此关 QQQ（NVDA 故事日单独开例外要另开研究）。  
- 不把「进 top2 / 回填成交」当成「期权必赚」。  
- 不与 L2 Hunt 混用阈值。
