# Event / News / Gap-trap：Jul10–23 主动救援

> 工具：`tools/run_event_news_triple_ablation.py`  
> 产物：`results/event_news_triple_v1`（日历 oracle）· `results/event_news_triple_v1_gap`（隔夜 gap）

## 1. 日历事实

| 毒日 | 票 | Finnhub / live 日历 |
|------|----|--------------------|
| 07-10 | META UP | **无**（META 财报 07-29） |
| 07-20/21 | AMD UP | **无** |
| 07-22 | MSFT DN | **无**（当日是 GOOGL/TSLA AH） |

peer3 已开 `feb_jul_aapl_ceo` + `hard_risk`，但对上述日期 **零命中**。

## 2. 日历消融（`event_news_triple_v1`）

| arm | weak | mid | jul |
|-----|------|-----|-----|
| ORACLE_SYM | 1.00 | 1.00 | **2.35** |
| ORACLE_FULL | 1.00 | 1.00 | **2.35** |
| FH_JUL22 / FH_EARN_JUL | 1.00 | 1.00 | 1.00 |
| CAL_OFF | 0.65 | 0.87 | 1.00 |

**裁决：`ORACLE_CEILING`。** 事后禁票上界巨大，但 **无可接线日历源**。Finnhub 真实行帮不上。

## 3. 因果代理：隔夜顺势 gap（`overnight_gap_gate`）

规则：`fav_gap = gap(UP) | −gap(DN)`，`fav_gap ≥ max_fav_gap` → block/scale。  
探针：META/AMD 毒单 fav_gap≈+4.3–4.6%；MSFT≈flat（本门漏网）。

| arm | weak | mid | jul | 裁决 |
|-----|------|-----|-----|------|
| **GAP04_BLOCK** | **1.01** | **0.865** | **2.17** | **GAP_TRAP_LIFT** |
| GAP04_SCALE | 0.98 | 0.93 | 1.57 | keep 更好、jul 较弱 |
| GAP035_BLOCK | 1.01 | 0.89 | 1.76 | 抬 Jul 较少 |

Jul 硬拦：META 07-10 + AMD 07-20/21（MSFT 仍在）。mid 误伤约 1 笔赢家（keep 刚好 ≥0.85）。

## 4. 标准双窗（`overnight_gap_dual_v1`，2026-07-25）

窗：weak `Feb1–Apr30` · strong `May1–Jul23`（peer3 OFF vs GAP04）。

| arm | weak vsOFF | strong vsOFF | Jul10–23 口袋 |
|-----|------------|--------------|---------------|
| GAP04_BLOCK | **0.876** | **0.950** | 0.33→**0.45** |
| GAP04_SCALE | 0.921 | 0.978 | 0.33→0.39 |

**裁决：`DUAL_FAIL` — 不接线。**  
Jul 口袋确实变好，但强窗整体仍低于 OFF（5–7 月误伤 > Jul 救援）；弱窗 keep 也不到 0.95。

## 5. UP-only 高开降级（`overnight_gap_up_degrade_dual_v1`）

规则：`gap≥thr` 且 `dir=UP` → `size×scale`（DN 不动）。

| arm | weak | strong | Jul10–23 |
|-----|------|--------|----------|
| UP04_X50 | 0.934 | 0.978 | 0.33→0.39 |
| UP04_X25 | 0.901 | 0.964 | →0.42 |
| UP035_X50 | 0.934 | **0.992** | →0.39 |
| UP05_X50 | 0.934 | 0.934 | 无变化 |

连续账本效果：07-20 −5.83%→−2.92%，07-21 −2.59%→−1.29%；但 07-01 META 赢家 +14.5%→+7.2%。

**裁决：仍 `DUAL_FAIL`。** 最近的是 UP035_X50（强窗几乎打平仍未 >1）。

## 6. 高开 + adverse-vol 确认（`gap_adv_confirm_dual_v3`）

探针：同为 UP gap≥4%，赢家（META 07-01 / 04-08）进场 adv≈0.39–0.45；AMD 07-20/21 tox adv≈0.55–0.56（信号后约 60s 才热）。

规则：`up_only & gap≥thr & adv120≥amin` → scale/block；`lag_seconds=60`。

**Bugfix（replay）：** `require_adv_share` 时必须加载 stock 1s（原先只在 `adverse_vol_share` / `entry_adv_vol` 开启时加载 → v1/v2 全 `sc=0`）。

| arm | weak vsOFF | strong vsOFF | Jul10–23 | 命中 |
|-----|------------|--------------|----------|------|
| **G4_A55_BLK** | **0.860** | **1.094** | 0.33→**0.45** | **AMD 07-20/21 不开仓**；弱窗 META 04-08 不开 |
| G4_A55_X50 | 0.931 | 1.017 | →0.35 | 同条件×0.5（已弃用为推荐） |
| G4_A55_X25 | 0.896 | 1.055 | →0.40 | 更狠缩仓 |
| G4_A52_X50 / G35_A52_X50 | 0.931 | 1.017 | →0.35 | 与 X50 同级 |

要点：
- **推荐动作：硬拦（`mode=block`），不缩仓。** 07-20/21 类型直接 skip。
- **META 07-01 赢家未触碰**（adv 不够热）。
- 弱窗误伤 **META 04-08**（keep 0.86，够 research 不够 wire）。
- MSFT 07-22 仍漏网（非高开）。

**裁决：`DUAL_PASS_RESEARCH`（best=`G4_A55_BLK`）。**  
**2026-07-26：已接线 research baseline**（`mode=block`）。弱窗 keep 0.86 < 0.95 wire 条，属主动接受误伤 META 04-08。

## 7. 接线纪律

- **日历 oracle：不接线**（无因果源）。
- **纯 `overnight_gap_gate`（含 UP-only 降级）：`REJECT`。**
- **gap+adv 确认：`DUAL_PASS_RESEARCH` → 2026-07-26 已 WIRE research baseline（`mode=block`）；禁止再推 scale。**
- 禁止 Jul 单独拧阈值后宣称 PASS。

```bash
# 日历上界
PYTHONPATH=. python -m maga7.tools.run_event_news_triple_ablation \
  --out maga7/results/event_news_triple_v1

# gap 因果臂
PYTHONPATH=. python -m maga7.tools.run_event_news_triple_ablation \
  --arms PRE,GAP04_BLOCK,GAP04_SCALE,GAP035_BLOCK \
  --out maga7/results/event_news_triple_v1_gap

# gap+adv 双窗（需 replay 1s fix）
PYTHONPATH=. python -m maga7.tools.run_gap_adv_confirm_dual \
  --out maga7/results/gap_adv_confirm_dual_v3
```
