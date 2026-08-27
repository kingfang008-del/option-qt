# AM 换信号源：Impact → Launch-slope 多火

目标：告别 FO/VWAP 口袋密扫；找能支撑「更高笔数 + 正期望」的因果源。  
窗口：2026-05-01～07-23 双窗；mark：option trade-last slip 1%。

## 尝试顺序

### 1) Buyer-impact 稀有门 — **FAIL（密度有、边没有）**

工具：`maga7/tools/scan_am_impact_scalp_tpsl.py`  
结果：`/mnt/s990/data/maga7/results/research_am_impact_scalp_tpsl/`

| gate | 事件/日 | 最佳均笔 | 双窗 |
|---|---:|---:|:---:|
| volr2 | ~17 | ≈ −0.9% | ✗ |
| ret30_volr | ~20 | ≈ 0 | ✗ |
| impact_hi | ~31 | ≈ +0.1% | ✗ |
| burst / volz_ret | 40–70 | 负 | ✗ |

挂 ride / TP8 同样双窗为负。  
结论：冲击门能加密，但还不够当 scalp 引擎（与 `buyer_impact_1s` 文档一致）。

### 2) Launch-slope 多火 densify — **PASS**

事件：`research_launch_slope_may_jul`（允许多火，非 first-only）  
scoreboard：`research_am_impact_scalp_tpsl/launch_densify_scoreboard.csv`

| cell | exit | 笔/日 | 胜率 | 均笔 | disc | blind |
|---|---|---:|---:|---:|---:|---:|
| **all（多火）** | **tp15/sl25/h300** | **22.0** | **55%** | **+0.90%** | **+31%** | **+47%** |
| volz≥2 | tp15/sl25/h300 | 10.6 | 55% | +1.12% | +40% | +20% |
| volz≥1.5 | tp15/sl25/h300 | 11.8 | 54% | +0.91% | +56% | +22% |
| first/sym/dir/日（稀） | tp15/sl25/h300 | 5.4 | 59% | +0.88% | +18% | +8% |
| all | tp5/sl8/h60 | 28 | 50% | −0.4% | 负 | ~0 |

要点：
- **信号源已换**：正股 launch-slope rising edge，不是 FO/VWAP pocket。
- 一天 **~20 笔** 量级可达，且双窗赚钱。
- 均笔约 **+1%**（未到想象中的 +5%）；快 TP5% 在此源上反而亏。
- 要更高均笔可收紧 `volz≥2`（~11 笔/日，+1.1%）。

## promote

```text
promote = LAUNCH_MULTI_all__tp15_sl25_h300

signal: launch_slope multi-fire (open+mid AM events, all unique)
mark:   trade-last slip 1%
exit:   TP15% / SL25% / max_hold 300s
size:   10% / max5 / cd1m   (research book)
≈22 笔/交易日  win≈55%  mean≈+0.9%
disc≈+31%  blind≈+47%
Scanner: not production-wired; quote 对拍仍待做
```

备选（均笔优先）：`volz2__tp15_sl25_h300`（~11 笔/日，mean≈+1.1%）。

## 与旧 keep 对比

| | vd_soft+ride | launch multi |
|---|---|---|
| 信号 | FO/VWAP pocket | launch-slope |
| 笔/日 | ~1.1 | **~22** |
| 胜率 | 60% | 55% |
| 均笔 | +7.7% | +0.9% |
| 双窗 | ✓ | ✓ |

密度目标基本达成；单笔 5% 未达成（此源要用更宽 TP 才赚钱）。

## 复现

```bash
# impact（预期 FAIL）
PYTHONPATH=. python -m maga7.tools.scan_am_impact_scalp_tpsl \
  --tag research_am_impact_scalp_tpsl

# launch densify 见同目录 launch_densify_*.csv（基于 research_launch_slope_may_jul events）
```
