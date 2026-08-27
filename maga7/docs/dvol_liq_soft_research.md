# 流动性排序 / 软加仓（S1 research_baseline）

**日期：** 2026-07-23  
**问题：** Top2 earliest 漏掉高价值大票（如 07-22 NVDA）；点名加权 NVDA 过拟合。  
**两条腿：**

| 腿 | 旋钮 | 是否改座位 |
|----|------|------------|
| 流动性排序 | `signal.rank_by=dollar_vol` / `cs_dollar_vol` | **是** |
| 软加仓 | `trade.dvol_size_scale`（因果截面 $vol 名次 → size 乘数） | **否** |

实现：`maga7/common/dvol_size_scale.py` · replay / stream_engine 已接线。  
验收：`python -m maga7.tools.run_liq_seat_accept`  
产物：`/mnt/s990/data/maga7/results/liq_seat_accept_s1_apr_jul_jan_mar_v1/`

软加仓默认形态（只抬不砍）：

```json
"dvol_size_scale": {
  "enabled": true,
  "mode": "cs_rank",
  "scales": {"1": 1.25, "2": 1.15},
  "default_scale": 1.0,
  "min_scale": 1.0,
  "max_scale": 1.25
}
```

## Scoreboard（相对 PRE = S1 + earliest top2）

| window | PRE | DVOL | CS_DVOL | **SOFT** | DVOL_SOFT |
|--------|-----|------|--------|----------|-----------|
| 强 Apr–Jul→22 | +4386% / −14.1% | +1802% | +1408% | **+5552% / −13.9%** | +2436% |
| 弱 Jan–Mar | +105% / −17.8% | +73% / −15.6% | +99% / **−8.3%** | **+112% / −18.4%** | +101% / −17.3% |
| 七月 | +90.7% | +92.7% | +98.4% | **+100.0%** | +109.0% |
| 孤立 07-22 | AMD −1.5%（漏 NVDA） | **NVDA+AMD +12.5%** | AMD+NVDA +5.1% | 仍 AMD −1.5% | NVDA+AMD +15.8% |

## 决策

| 臂 | 决策 | 说明 |
|----|------|------|
| **SOFT** | **`PROMOTE_LIQ_RESEARCH`** | 强 keep≈1.27，弱收益↑，七月 keep≥0.95；**不改座位**，故 07-22 仍漏 NVDA |
| DVOL / CS_DVOL / DVOL_SOFT | **`REJECT_FOR_BASELINE`** | 孤立日能抓 NVDA，但强窗 keep≪0.85（与旧 [`topk_dollar_vol_research.md`](topk_dollar_vol_research.md) 一致） |

### 解读

1. **软加仓**：对「已经进 TopK 且当日截面流动性靠前」的票加码（rk1×1.25 / rk2×1.15）。七月有 5 笔被抬；07-22 的 AMD 当时截面 rk=3 → 不加。  
2. **流动性重排座位**：能修漏抓，但强窗代价过大，不升基线。  
3. **漏抓 vs 加仓是两件事**：SOFT 改善整段复利，不保证吃到第三枪大票；要修漏抓仍看窄触发 backfill / 别的座位策略。

## 基线态度

- **2026-07-23：SOFT 已写入 research_baseline**  
  `single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json` → `trade.dvol_size_scale.enabled=true`  
  `research_revision=2026-07-23_s1_dvol_soft`  
- **production freeze（googl_peer3_v1）仍不打开**  
- 座位类（dollar_vol 重排 / seat_score_gate / backfill）保持关；见 [`seat_score_gate_research.md`](seat_score_gate_research.md)

相关：[`topk_backfill_research.md`](topk_backfill_research.md) · [`topk_dollar_vol_research.md`](topk_dollar_vol_research.md)
