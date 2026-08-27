# TopK blocked-backfill（研究开关）

针对「最早 TopK 里有人被 regime/peer 挡掉 → 坑位空转、后面大行情进不来」（典型 07-09 META）。

> **注意：此处 backfill ≠ 补数据。**  
> 数据补齐（如 AMD 1s）见 [`amd_1s_gap_backfill.md`](amd_1s_gap_backfill.md)。  
> 本开关是 **TopK 名额分配策略**：挡掉的票是否仍占用当日 `top_k` 席位。

## 开关

`trade.topk_backfill_on_block`（或 `signal.topk_backfill_on_block`）默认 **false**。

| | 关（baseline） | 开（backfill） |
|---|---|---|
| 事件流 | 当日 earliest TopK（`rank_by` 默认按时间） | 当日全部 first-Rule-A（按时间） |
| 名额 | 进入事件流即占坑；被 regime/peer 挡掉也占掉 | **通过** regime / peer / mf_idio / tcn / confirm 后才占 1 席 |
| 挡掉的票 | 占名额 → 后面的票进不来 | **不占名额** → 顺延下一个能过闸的 |
| 报价/sim 失败 | 占名额 | 仍占名额（避免「无 1s → 无限往后捞」） |
| 每日上限 | `top_k` | 仍最多 `top_k`；**不是**无上限 `all_first` |

Freeze profile **不打开**。实现：`maga7/common/replay.py`（`topk_backfill_on_block`）。

### 一句话

- **关**：最早进 TopK 的票若被 regime/peer 挡掉，名额照样被占掉，后面的票进不来。  
- **开**：被挡的不占名额，顺延下一个能过闸的；每天仍最多 `top_k` 笔。

## Scoreboard（相对 freeze，TCN off）

| window | variant | total_ret | MaxDD | n | backfill 笔 |
|--------|---------|-----------|-------|---|------------|
| May–Jul | baseline | **+874.7%** | **−13.2%** | 48 | 0 |
| May–Jul | backfill | +1083.8% | −17.7% | 64 | 16 |
| Feb–Apr | baseline | **+140.0%** | **−28.9%** | 75 | 0 |
| Feb–Apr | backfill | +97.5% | −31.0% | 78 | 5 |
| Apr–Jul | baseline | +1779% | −24.6% | 77 | 0 |
| Apr–Jul | backfill | +2183% | −24.6% | 93 | 16 |

明细：`maga7/results/topk_backfill_ablation/`。

### 07-09 验证

| | AMD | META |
|---|---|---|
| baseline | −18.4% | （未进事件流） |
| backfill | −18.4% | **−8.0%**（顺延成交） |

大行情进场了，但持仓窗内期权仍亏——说明「吃到日线大票 ≠ 策略窗内赚钱」。

## 结论（旧 freeze 窗）

**REJECT for freeze。**

- May–Jul 收益更高，但 MaxDD 变差（−13.2% → −17.7%）  
- Feb–Apr 收益与回撤双差  
- 与「≥95% 强市 + 弱市不伤」的采纳条不一致  

保持 pluggable；若再做，应加更严触发（例如仅当原 TopK 成员当日确实被 regime 挡）或与 displace 分数门合并，而不是默认开启。

相关：成交额重排 TopK 见 [`topk_dollar_vol_research.md`](topk_dollar_vol_research.md)。

---

## 2026-07-23 复验（S1 research_baseline）

基线已含 S1 soft `stock_path_confirm`。工具：`python -m maga7.tools.run_topk_backfill_accept`  
产物：`/mnt/s990/data/maga7/results/topk_backfill_accept_s1_apr_jul_jan_mar_v1/`

| window | PRE (top2) | BF (top2+backfill) | TOP3 | BF keep |
|--------|------------:|-------------------:|-----:|--------:|
| 强 Apr–Jul→07-22 | +4386% / −14.1% / n=79 | +4116% / −14.1% / n=91（bf=25） | +3398% / −17.3% / n=100 | **0.94** |
| 弱 Jan–Mar | +104.6% / −17.8% / n=55 | +89.9% / **−23.8%** / n=58（bf=9） | +88.1% / −20.2% / n=64 | 变差 |
| 七月 07-01..22 | **+90.7%** / −7.6% / n=15 | +57.9% / −12.7% / n=18 | +61.4% / −17.3% / n=18 | **0.64 fail** |
| 孤立 07-22 | AMD −1.5%（漏 NVDA） | **AMD+NVDA +5.1%**（抓住 NVDA TP） | 同 BF | — |

### 决策

| 臂 | 决策 | 理由 |
|----|------|------|
| **BF** | **`REJECT_FOR_BASELINE`** | 强窗 keep≈0.94 过线，但弱窗收益/回撤双差，七月 keep≈0.64≪0.95；仅孤立 07-22 修好漏抓 |
| **TOP3** | **`REJECT_FOR_BASELINE`** | 强窗 keep≈0.77 不过；弱/七月均差于 PRE |

**加仓位**（`position_frac`↑）不在本表：它不改变 TopK 座位，治不好漏抓。

### 实务含义

- **漏抓最有价值票**：BF / TOP3 在孤立 07-22 都能补上 NVDA；连续窗里 07-22 路径依赖不同（常成交 MSFT DN），不会自动复现「去 AMD → NVDA」。
- **升 research / freeze 基线：否** —— 与旧结论一致。
- **可留作 overlay / 日终诊断开关**；下一步若再做，优先「仅当原 TopK 被 regime 挡才顺延」的窄触发，而不是无条件 backfill 或盲升 Top3。

## 复跑

```bash
# 双窗验收（推荐）
python -m maga7.tools.run_topk_backfill_accept

# 单窗手搓
python - <<'PY'
from copy import deepcopy
from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
p = load_profile("maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json")
p["date_range"] = {"start": "2026-05-01", "end": "2026-07-16"}
p.setdefault("trade", {})["topk_backfill_on_block"] = True
print(run_offline_replay(p, scheme="single")["summary"])
PY
```
