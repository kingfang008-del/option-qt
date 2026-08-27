# chop_gate：平静震荡 / 混向噪声日 overlay

**日期：** 2026-07-23  
**状态：** 双窗验收完成 · 最佳臂 **`OVERLAY_ONLY`**（弱窗持平、未 lift）→ **不默认升 research baseline**  
**旋钮：** `profile.chop_gate`（默认关）  
**代码：** `maga7/common/chop_gate.py` · replay 已接线  
**验收：** `maga7/tools/run_chop_gate_accept.py`  
**产物：** `/mnt/s990/data/maga7/results/chop_gate_accept_s1_apr_jul_jan_mar_v1/`

## 动机

7/10 后趋势脉冲变稀、连胜变短；Watchdog 只管 washout/reclaim **毒日**，平静「指数不走、个股乱晃」的假突破日常落在 NORMAL。  
目标：因果识别后 **软缩仓**（优先）或停开，压 Jul21–22 类尾亏，且强窗 keep≥0.85。

## 特征（RTH，asof=10:30）

| 字段 | 含义 |
|------|------|
| `q_am` | QQQ close/open − 1 |
| `q_rng` | QQQ (high−low)/open（可选腿，默认关） |
| `frac_above` | Mag7 close>open 占比 |
| `med_abs` | Mag7 \|from_open\| 中位数 |

> 必须用 **RTH 1m**（`load_stock_month_files`）。盘前低点会把 `q_rng` 虚高，早期 `wide_mix` 阈值因此失效。

## 默认规则 `stock_noise`

```json
"chop_gate": {
  "enabled": false,
  "asof": "10:30",
  "rule": "stock_noise",
  "mode": "scale",
  "scale": 0.5,
  "q_am_max": 0.005,
  "q_rng_min": 0.0,
  "frac_above_lo": 0.35,
  "frac_above_hi": 0.50,
  "med_abs_min": 0.01
}
```

含义：QQQ 几乎不漂移，但个股已有 ≥1% 中位绝对位移且上涨广度偏弱 → 混向噪声 / 假突破燃料。

`mode=block` 全日禁基线；与 `state_gate` / Watchdog **并联**（不改座位逻辑）。

## 双窗结果（vs S1 PRE）

| 臂 | strong | keep | weak | july | chop Jul10–22 | 决策 |
|----|-------:|-----:|-----:|-----:|--------------:|------|
| PRE | +5552% | 100% | +112% | +100% | +25% | — |
| **SOFT_NOISE** | +5313% | **96%** | +112%（平） | **+108%** | **+30%** | **OVERLAY_ONLY** |
| HARD_NOISE | +5058% | 91% | +112%（平） | +116% | +35% | OVERLAY_ONLY |
| SOFT_NOISE65 | +4930% | 89% | +112% | +93% | +30% | REJECT（july） |
| SOFT_AM6 | +5313% | 96% | +112% | +108% | +30% | OVERLAY_ONLY |

无臂弱窗 ret↑/MaxDD↑ → 无 `PROMOTE_CHOP_RESEARCH`。

### July 命中（SOFT_NOISE）

连续 replay 仅标 **chop** 两日：`2026-07-21`、`2026-07-22`（AMD UP / MSFT DN 各 ×0.5）。  
07-13/14 大赢日 **不触发**（`med_abs` 不够或 frac 偏多头）。

## Verdict

1. **推荐研究 overlay：`SOFT_NOISE`** — 强窗保留 96%，July 与 Jul10+ chop 窗均改善，弱窗不伤。  
2. **暂不写进 research baseline** — 严格双窗要弱窗 lift；本臂弱窗持平。若接受「不伤弱窗 + July 改善」可手工打开。  
3. **勿用盘前校准的 `q_rng` 宽门**；`frac_hi=0.65` 会误伤 July。  
4. 与 Watchdog 分工：毒日仍 degrade/halt；本门只管平静混向噪声。

## 复现

```bash
PYTHONPATH=. python maga7/tools/run_chop_gate_accept.py \
  --out /mnt/s990/data/maga7/results/chop_gate_accept_s1_apr_jul_jan_mar_v1
```
