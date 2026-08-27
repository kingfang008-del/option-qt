# Seat score gate（低分不占坑）

**日期：** 2026-07-23  
**动机：** earliest Top2 早枪占坑 → 漏掉后面高价值票（07-22 NVDA）。软加仓不改座位；裸 backfill / Top3 / dollar_vol 重排双窗不过。  
**机制：** 事件流改为当日 `all_first`（按时间），**评分不过则 SKIP 且不消耗** `top_k` 席位；每天仍最多 `top_k` 笔成功入场。

```json
"trade": {
  "seat_score_gate": {
    "enabled": true,
    "mode": "cs_dvol_max_rank",
    "max_rank": 2
  }
}
```

| mode | 含义 |
|------|------|
| `cs_dvol_max_rank` | 因果截面 session $vol 名次 ≤ `max_rank` 才占坑（主候选） |
| `vol_z_min` | `vol_z ≥ min_vol_z` |
| `fp_x_vz_min` | `\|from_prev\| × vol_z ≥ min_fp_x_vz` |

实现：`maga7/common/seat_score_gate.py` · offline `replay.py`（开启时自动 `all_first` + fill-cap）。  
**Live/stream 尚未接线。**

验收：`python -m maga7.tools.run_seat_score_gate_accept`  
产物：`/mnt/s990/data/maga7/results/seat_score_gate_accept_s1_apr_jul_jan_mar_v1/`

## Scoreboard（S1 research_baseline）

| window | PRE | GATE_CS (rk≤2) | GATE_VZ (≥1.5) | GATE_FPV (≥0.03) |
|--------|-----|----------------|----------------|------------------|
| 强 Apr–Jul→22 | +4386% / n=79 | **+626%** / n=41（skip=105） | +474% / n=55 | +1350% / n=77 |
| 弱 Jan–Mar | +105% / −17.8% | +27% / **−10.5%** | +89% / −13.5% | +119% / −19.4% |
| 七月 | **+90.7%** | +82.9% | +28.3% | +22.1% |
| 孤立 07-22 | AMD −1.5%（漏 NVDA） | **NVDA +13.3%** ✓ | 仍 AMD | AMD+NVDA +5.1% ✓ |

## 决策

| 臂 | 决策 | 说明 |
|----|------|------|
| **GATE_CS** | **`REJECT_FOR_BASELINE`** | 孤立 07-22 **抓住 NVDA**；强窗 keep≈0.14，七月 keep≈0.91≪0.95 |
| GATE_VZ | `REJECT_FOR_BASELINE` | 07-22 仍漏 NVDA（AMD vol_z 更高） |
| GATE_FPV | `REJECT_FOR_BASELINE` | 能抓 NVDA，强/七月仍差 |

### 解读

1. **方向对：**「低流动性早枪不占坑」在孤立 07-22 正好放出 NVDA。  
2. **全样本过狠：** `cs_dvol≤2` 在强窗砍掉一半成交，复利崩塌——和 `rank_by=cs_dollar_vol` 同类代价。  
3. **连续七月窗**里 07-22 路径不同，GATE_CS 未必复现「单日 NVDA」（状态依赖）。  
4. **下一步若再挖：** 更窄触发（例如仅当原 earliest-TopK 成员 `cs_rk>2` 才启用 gate / 或仅 10:30–11:30），不要无条件全天 `max_rank=2`。

## 基线态度

- **不写入 research_baseline / freeze**  
- 保留为研究开关；与 [`topk_backfill_research.md`](topk_backfill_research.md) · [`dvol_liq_soft_research.md`](dvol_liq_soft_research.md) 并列

---

## 2026-07-23 窄触发（续）

旋钮扩展：

| 键 | 含义 |
|----|------|
| `when` | `always` / `topk_weak` / `morning` / `topk_weak_morning` |
| `apply_to` | `all`（所有候选打分）/ **`topk_members`**（只弃 earliest TopK；后面补位免检） |
| `tod_start`/`tod_end` | morning 窗，默认 10:30–11:30 |

主推窄版语义（对应「前两枪分低就丢、后面补」）：

```json
"seat_score_gate": {
  "enabled": true,
  "mode": "cs_dvol_max_rank",
  "max_rank": 2,
  "when": "always",
  "apply_to": "topk_members"
}
```

验收：`python -m maga7.tools.run_seat_score_gate_accept`  
产物：`/mnt/s990/data/maga7/results/seat_score_gate_skip_topk_accept_s1_v1/`

### Scoreboard（窄版）

| window | PRE | GATE_CS (all) | **SKIP_TOPK** | SKIP_TOPK_AM | SKIP_TOPK_RK3 |
|--------|-----|---------------|---------------|--------------|---------------|
| 强 | +4386% | +626% | **+434%** | +506% | +1055% |
| 弱 | +105% | +27% | +47% | +47% | +44% |
| 七月 | +90.7% | +82.9% | +49.2% | +49.2% | +47.8% |
| 孤立 07-22 | 漏 NVDA | **NVDA +13.3%** | **NVDA +13.3%** | 同左 | AMD+NVDA +5.1% |

### 决策（窄版）

全部 **`REJECT_FOR_BASELINE`**。

| 发现 | 含义 |
|------|------|
| `when=topk_weak` 几乎≈`always` | earliest TopK 里经常有人 `cs_rk>2`，日级开关收不窄 |
| **`apply_to=topk_members` 仍伤强窗** | 弃掉的早枪里有不少强窗赢家；补进来的后枪净贡献为负 |
| 孤立 07-22 仍可修 | 机制方向对，但流动性名次不是好的「弱枪」定义 |

### 下一步（若再挖）

1. **换评分**：不要用 `cs_dvol`，改试短窗 path 质量 / peer 对齐 / 入场后 1–3 分钟 soft abort（持仓后快速丢弃）  
2. **或接受漏抓**：基线保持 earliest；用已过线的 **`dvol_size_scale` 软加仓** 抬高已进场的流动性票  
3. Live/stream 仍未接线 seat_score_gate
