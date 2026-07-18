# dyn_trail：快慢窗动态跟踪离场（研究）

针对「静态 T30/T45 时钟不完美」的设想：用 `mf_fast`（3m）+ `mf_slow`（mf10）+ 慢窗斜率做动态离场，废除固定 30 分钟。

## 实现

`exit_mode=dyn_trail`（别名 `mf_dual`），见 `maga7/common/replay.py::simulate_trade`。

| 规则 | 触发 | reason |
|------|------|--------|
| 快窗逆转 | ≥`dyn_min_hold`（默认 5m），快窗连续 `dyn_fast_opp_bars` 根反向，慢窗斜率&lt;0，可选跌破入场 1m low/high | `FAST_REVERSAL` |
| 动能耗尽 | ≥`dyn_trail_start`（默认 15m），快窗 ≤ 当日因果分位 `dyn_fast_pct`，慢窗斜率&lt;0 | `MOM_EXHAUST` |
| 趋势死亡 | ≥trail_start，慢窗有利 MF&lt;0 | `TREND_DEAD` |
| 硬终点 | `dyn_max_hold_minutes`（默认 60） | `T+60` |

Freeze profile **不打开**。产物：`results/dyn_trail_ablation/`。

## Scoreboard（相对 freeze，TCN off，无 backfill）

### May–Jul

| variant | total_ret | MaxDD | win | exp | med hold | FR/ME/TD |
|---------|----------:|------:|----:|----:|---------:|----------|
| **baseline T30→T45** | **+874.7%** | **−13.2%** | 67% | +29% | 30m | 0/0/0 |
| dyn_v1_spec | +94.6% | −14.3% | 45% | +8% | 12m | 24/3/10 |
| dyn_soft_late | +134.1% | −17.3% | 55% | +13% | 15m | 19/2/13 |
| dyn_v1_nops（无价格破位） | +60.8% | −17.7% | 40% | +6% | 7m | 39/1/0 |

### Feb–Apr

| variant | total_ret | MaxDD |
|---------|----------:|------:|
| **baseline** | **+140%** | **−28.9%** |
| dyn_v1_spec | +24% | −31.3% |
| dyn_soft_late | +32% | −27.2% |

## 结论

1. **直觉对、落地错**：静态时钟确实粗糙，但快慢 MF 噪声大，软离场与已 REJECT 的 `mf_flip` / `mtm_floor` 同类——大量误杀赢家。
2. **并未出现「拿 60–90m」**：中位持仓被压到 7–15m；几乎到不了硬终点。
3. **不升格**。若还要挖「让利润跑」，优先在 **T30 已盈利** 前提下延长（现有 `hold_extend`），而不是用 MF 状态机替代时钟。
