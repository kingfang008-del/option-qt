# mf10 + 快速伴生窗（early_on_mf_fast）

研究开关：在 **10 分钟 mf10 streak** 仍为主路径的前提下，用更短的滑动窗（默认 3/5 分钟）在 streak 未满 `streak_min` 时提前开火。

## 机制

| 键 | 含义 |
|---|---|
| `mf_window=10` | 主窗（不变） |
| `mf_fast_window=3\|5` | 伴生快窗（`net$` 滚动和；别名列 `mf_short`） |
| `early_on_mf_fast=true` | 开启提前路径 |
| `streak_min_fast=5\|6` | 须 `< streak_min`（基线为 8） |
| `require_mf_short_align` | **另一条路**：满 streak 时再要求快窗同向（偏过滤，非提前） |

提前开火条件：`streak_min_fast ≤ streak < streak_min` 且 `mf_fast` 与方向同号，且 cum / from_prev / vol_z 门照旧。

实现：`maga7/common/signals.py`（batch + `StreamSignalState`）；replay / stream / export 已接线。

## 消融（May–Jul peer3，delay=60）

```bash
python -m maga7.tools.run_mf_fast_ablation \
  --tag mf_fast_early_ablation_mag7_googl_peer3_may_jul
```

产物：`maga7/results/mf_fast_early_ablation_mag7_googl_peer3_may_jul/`

| 变体 | ret | MaxDD | n | 重叠笔数中提前 |
|---|---|---|---|---|
| baseline_mf10 | +356.6% | -16.5% | 50 | — |
| early_mf3_s5 | +330.8% | -20.2% | 54 | 15/45，均提前 ~1.4min |
| early_mf3_s6 | +282.6% | -19.6% | 54 | 15/47，~1.2min |
| early_mf5_s5 | +310.1% | -15.7% | 54 | 15/45，~4.6min |
| early_mf5_s6 | +350.2% | **-13.3%** | 54 | 14/46，~1.2min |

结论（研究态，**未升格基线**）：

- 快窗确实能让约 1/3 重叠信号更早（多数仍同刻；中位提前 0）。
- 收益最高仍是纯 mf10；过早开火（尤其 mf3）会拉低 ret、加大 MaxDD。
- `early_mf5_s6` 最接近基线收益，且 MaxDD 更好，可作后续与 `hold_extend` 叠用的候选。

## 配置示例（研究 profile 覆盖）

```json
"signal": {
  "mf_window": 10,
  "streak_min": 8,
  "mf_fast_window": 5,
  "early_on_mf_fast": true,
  "streak_min_fast": 6
}
```
