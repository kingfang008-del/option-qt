# TopK 引入成交额（dollar volume）重排

承接：大行情往往伴随最大成交额 → 是否用成交额替代 / 辅助 `rank_by=earliest`。

> 名额顺延开关（`topk_backfill_on_block`）见 [`topk_backfill_research.md`](topk_backfill_research.md)，**不是补数据**。

## 机制

`signal.rank_by`（`maga7/common/signals.py::build_topk_signals`）：

| 值 | 含义 | 因果性 |
|---|---|---|
| `earliest`（freeze 默认） | 按 `sig_ts` 取前 `top_k` | 完全因果 |
| `dollar_vol` | 信号时刻 **session 成交额** ∑(close×volume) 从高到低取 TopK | 日终批重排（入场时刻仍用原 `sig_ts`；选型相对 online earliest 有轻度前视） |
| `dollar_vol_10` | 信号前 10 分钟成交额 | 同上 |
| `abs_from_prev` | \|from_prev\| | 批重排 |
| `fp_x_dvol` | \|from_prev\| × session 成交额 | 批重排 |
| `cs_dollar_vol` | 信号时刻在 **universe 截面** 成交额排名 ≤ `top_k`，再按时间取 TopK | **完全因果** |

成交额字段在 `first_rule_a_day` 写入：`dollar_vol` / `dollar_vol_10`。

## Scoreboard（freeze 底仓，TCN off，无 backfill）

| window | rank_by | total_ret | MaxDD | n | win | exp |
|--------|---------|----------:|------:|--:|----:|----:|
| May–Jul | **earliest** | **+874.7%** | **−13.2%** | 48 | 67% | +29% |
| May–Jul | dollar_vol | +380% | −15.7% | 57 | 54% | +18% |
| May–Jul | dollar_vol_10 | +449% | −19.4% | 54 | 56% | +20% |
| May–Jul | fp_x_dvol | +288% | −15.7% | 57 | 54% | +17% |
| May–Jul | abs_from_prev | +304% | −18.3% | 57 | 51% | +15% |
| May–Jul | cs_dollar_vol | +387% | −13.0% | 43 | 60% | +22% |
| Feb–Apr | **earliest** | **+140%** | **−28.9%** | 75 | 45% | +7.4% |
| Feb–Apr | dollar_vol | +35% | −32.5% | 79 | 38% | +2.6% |
| Feb–Apr | cs_dollar_vol | +73% | **−16.2%** | 45 | 47% | +9.9% |

叠 `topk_backfill_on_block=true` 时事件流变为 all_first，`rank_by` 基本失效，收益回到 backfill 曲线（May–Jul ~+1084% / DD−17.7%）。

明细：`maga7/results/topk_dollar_vol_ablation/`。

## 结论

1. **「大行情 = 大成交额」观察成立，但不等于更好的 TopK 目标函数。** 成交额重排会换进更多高换手名字，却打掉 earliest 抓到的早段期权窗口。  
2. **因果截面门控 `cs_dollar_vol`**：MaxDD 不差（May–Jul −13.0%，Feb–Apr 显著改善至 −16%），但收益腰斩级落后 earliest。  
3. **不升格**；freeze 保持 `rank_by=earliest`。  
4. 若继续挖流动性，更合理的是 **软特征**（确认 / size scale），而不是用成交额替换时间序 TopK。  
5. **2026-07-23 S1 复验**：`dollar_vol` / `cs_dollar_vol` 仍 `REJECT_FOR_BASELINE`；软加仓 `trade.dvol_size_scale` 双窗过线（`PROMOTE_LIQ_RESEARCH`）但不改座位、不修 07-22 NVDA 漏抓 —— 见 [`dvol_liq_soft_research.md`](dvol_liq_soft_research.md)。

## 复跑

```bash
python - <<'PY'
from copy import deepcopy
from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
p = load_profile("maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json")
p["date_range"] = {"start": "2026-05-01", "end": "2026-07-16"}
p["signal"]["rank_by"] = "cs_dollar_vol"  # or dollar_vol / fp_x_dvol
print(run_offline_replay(p, scheme="single")["summary"])
PY
```
