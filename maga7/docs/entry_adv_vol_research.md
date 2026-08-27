# Entry Adverse Volume Share（入场毒性 / adverse）

> 主动入场门：信号时钟前 1s adverse volume share。配套：`common/adverse_vol_share.py` · `tools/run_entry_adv_vol_triple_ablation.py`  
> 出场侧 `trade_toxic` 已升线；本门正交（挡开仓 vs 截 hold）。

## 1. 规则

在拟入场时刻（可选 `lag_seconds`）计算：

```
share = vol_on_adverse_ticks / (adverse + favorable)   # flat 不计
```

`share ≥ max_share` → `block` 或 `scale`。可选 `dirs` / TOD。

## 2. Jul 探针（2026-07-25）

PRE Jul10–23 毒单形态：**多数是顺势进场后反转**（META/MSFT 入场前 stock adverse-ret 为负/接近 0，share 偏冷）。

| 毒单 | entry share w120 | 会被 0.55 打到？ |
|------|------------------|------------------|
| META UP 07-10 | 0.33 | 否 |
| AMD UP 07-20 | 0.56 | 是 |
| AMD UP 07-21 | 0.55 | 边界 |
| MSFT DN 07-22 | 0.46 | 否 |

→ 本门最多修 AMD 口袋，修不了整段 Jul。

## 3. 三窗消融（`entry_adv_vol_triple_v1`）

```bash
PYTHONPATH=. python -m maga7.tools.run_entry_adv_vol_triple_ablation \
  --out maga7/results/entry_adv_vol_triple_v1
```

| arm | weak | mid | jul |
|-----|------|-----|-----|
| SCALE55_W120 | 0.92 | **0.74** | **1.18** |
| BLOCK55_W120 | 1.07 | 0.62 | 1.20 |
| SCALE50_* | ≤0.75 | ≤0.74 | 1.26 |
| *_LAG60 | ≤0.38 | ≤0.46 | 1.37–1.58 |

**裁决：`JUL_ONLY_PARTIAL` — 不接线。**  
Jul 有小幅抬升（主要 scale/block AMD），但 mid keep 全面 &lt;0.85；`lag60` 因推迟入场更伤先验窗。

## 4. 纪律

- Freeze / spine 保持 `entry_adv_vol` OFF。
- 勿对 Jul 单独降 `max_share` 或开 lag 后宣称 PASS。
- Jul META/MSFT 需别的主动门（事件/新闻，或接受出场 `TRADE_TOX` 已截 META）。
