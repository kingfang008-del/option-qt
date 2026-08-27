# from_open 延伸帽（追高/超买）

**日期：** 2026-07-23  
**状态：** 双窗验收完成 · **不升 research baseline**（最佳臂仅 `OVERLAY_ONLY`）  
**旋钮：** `trade.from_open_gate`（默认关）  
**代码：** `maga7/common/from_open_gate.py` · replay 已接线  
**验收：** `maga7/tools/run_from_open_gate_accept.py`  
**产物：** `/mnt/s990/data/maga7/results/from_open_gate_accept_s1_apr_jul_jan_mar_v1/`

## 动机

07-22 AMD 直觉「涨太多还亏」：相对**昨收** `from_prev≈+2.1%`（刚过 Rule-A），相对**开盘** `from_open≈+4~5.5%`。  
既有 `max_from_prev_abs` 挡不住；见 [`entry_quality_false_break_research.md`](entry_quality_false_break_research.md)。

## 旋钮

```json
"trade": {
  "from_open_gate": {
    "enabled": false,
    "max_abs": 0.04,
    "mode": "block",
    "scale": 0.5,
    "same_sign_only": true
  }
}
```

| 字段 | 含义 |
|------|------|
| `max_abs` | `\|close_asof / day_open − 1\|` 超阈触发 |
| `mode=block` | 硬拒（不占成交；**不**自动补下一席，除非另开 backfill） |
| `mode=scale` | 软缩仓 ×`scale`（默认 0.5），座位不变 |
| `same_sign_only` | 仅同向延伸（UP 且 from_open>0 / DN 且 <0） |

## 双窗结果（vs S1 PRE，含 dvol soft）

Pass：`strong keep≥0.85` **且** `weak(ret↑|MaxDD↑)` **且** `july keep≥0.95`

| 臂 | strong | keep | weak | july | keep | 决策 |
|----|-------:|-----:|-----:|-----:|-----:|------|
| PRE | +5552% | 100% | +112% | +100% | 100% | — |
| HARD_035 | +5501% | **99%** | +109% | +96% | **96%** | **OVERLAY_ONLY** |
| HARD_040 | +5360% | 97% | +101% | +93% | 93% | REJECT（july） |
| HARD_045 | +5360% | 97% | +101% | +93% | 93% | REJECT（july） |
| SOFT_035 | +5370% | 97% | +110% | +98% | 98% | OVERLAY_ONLY |
| SOFT_040 | +5456% | **98%** | +106% | +97% | 97% | OVERLAY_ONLY |
| SOFT_045 | +5456% | 98% | +106% | +97% | 97% | OVERLAY_ONLY |

无臂弱窗 ret↑ 或 MaxDD↑ → **无 `PROMOTE_FROM_OPEN_RESEARCH`**。

### 07-22 孤立日（lookback 伪影）

| 臂 | 成交 | day_ret |
|----|------|--------:|
| PRE | AMD UP | −1.46% |
| HARD_* | （拒 AMD） | 0 |
| SOFT_* | AMD ×0.5 | −0.73% |

连续窗（Jul / Apr–Jul）里同一天实际是 **MSFT DN**（`from_open≈−2.5%`，本门不触发）：孤立 `start=end=07-22` 时 MSFT 缺历史特征被滤掉，AMD 才露出来。评价「滤 AMD」请以**连续 replay**为准，勿只看单日包。

### July 误伤样例（HARD_035）

| 日 | 标的 | from_open | PRE ret | 备注 |
|----|------|----------:|--------:|------|
| 07-02 | TSLA DN | −6.5% | **+28%** | 硬拒砍赢家 |
| 07-09 | AMD UP | +3.9% | −14% | 硬拒挡亏损 |

## Verdict

1. **不进 research baseline / 不进 freeze** — 弱窗无改善；硬拒在 July 砍到 07-02 TSLA。  
2. **若只要压孤立追高亏**：`SOFT_040`（或 `HARD_035` overlay）可研究级叠加；优先软缩仓，少动座位。  
3. **与 `|fp|` 帽同族风险**：入场延伸与强趋势赢家重叠，硬门易水床。  
4. 毒性更宜继续挖**入场后路径**（见 [`wave_confirm_spec.md`](wave_confirm_spec.md) / toxic path），而非再收紧 Rule-A。

## 复现

```bash
PYTHONPATH=. python maga7/tools/run_from_open_gate_accept.py \
  --out /mnt/s990/data/maga7/results/from_open_gate_accept_s1_apr_jul_jan_mar_v1
```
