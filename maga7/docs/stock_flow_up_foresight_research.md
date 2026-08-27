# Stock-flow UP 前视（安静牛 / 续作假设）

> 对照已淘汰的 DN 臂 [`stock_flow_opt_research.md`](stock_flow_opt_research.md)。  
> 工具：`tools/scan_stock_flow_up_foresight.py`  
> 产物：`results/research_stock_flow_up_foresight_feb_jun_jul`

## 协议

| 窗 | 角色 | 日期 |
|----|------|------|
| discover | **唯一选参** | 2026-02-01 … 2026-04-30 |
| holdout | 只报告 | 2026-06-01 … 2026-06-30 |
| blind | 盲测 | 2026-07-10 … 2026-07-23 |

假设：大涨波前 `up_vol_share` / 正短窗 ret 相对对照抬升 → ATM call。

## 前视

| 窗 | distill_n | 最佳 lift@0.55 | 评 |
|----|-----------|----------------|-----|
| discover | **0** | **0.85**（&lt;1） | 涨波前 up_share **低于**对照 |
| holdout | 0 | ≈1.00 | 无抬升 |
| blind | 0 | 1.30 但 wave_n 小且 frac&lt;0.45 | 不 distill |

裁决：**`NO_DISTILL_ON_DISCOVER`** / `FORESIGHT_NO_DISTILL`。

## 因果（附属）

discover 有 1 格软 PASS：`up_d0.005_f120_sh0.6_tp0.1_sl0.2`（n=165 win≈63% add≈+9.4% day_win≈58%）。

**转移失败**（同一格）：

| 窗 | n | win | add | day_win |
|----|---|-----|-----|---------|
| discover | 165 | 63.0% | +9.4% | 58% |
| holdout Jun | 73 | 56.2% | **−16.5%** | 37% |
| blind Jul | 36 | 58.3% | **−2.7%** | 40% |

holdout 自己另有一格 PASS，与 discover 冠军不同 → 不构成可迁移规则。

## 结论

- **2–6 上「正股上冲量 → 买 call」前视未成立**（发现窗 lift&lt;1）。  
- 弱因果口袋不能迁移到 Jun / Jul。  
- **不升格、不接线**；与 DN `stock_flow_opt` 一并停在研究旁路。  
- 安静牛段更应依赖已有 **`qqq_open_cont` + Hunt + L0`**，而不是对称镜像一条 stock-flow UP 侦查兵。
