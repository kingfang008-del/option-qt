# 06-24 GOOGL tox × crowd peer=6 空洞

**Status: DUAL_FAIL — 不接线**（2026-07-27）

## 形态

| | GOOGL UP（毒） | AMZN UP（赢） |
|--|--|--|
| peer | **6** | **6** |
| chase | ~0.95 | ~0.99 |
| pre5@entry | **+15bp** | **+31bp** |
| fo | +1.2% | +2.6% |
| 期权 5m | MFE **0** / MAE **−23%** | MFE **+29%** / MAE −11% |
| 结局 | TRADE_TOX **−27%** | TP **+62%** |

当前 RS：Arm A `max_peer=5` → peer=6 **跳过**；Arm C `crowd_min_peer=7` → peer=6 **跳过**。  
空洞正好是 **peer=6**。

## 候选

`crowd_min_peer: 7 → 6`（其余 knobs 不变）：06-24 **拦 GOOGL、留 AMZN**（pre5 分界 ~25bp）。

## 双窗

| arm | weak | strong | GOOGL | AMZN |
|-----|-----:|-------:|:-----:|:----:|
| OFF (crowd7) | 1.000 | 1.000 | 在 | 在 |
| CROWD6 | **0.739** | **1.036** | 清 | 留 |
| CROWD6_FFO01 | **0.739** | **1.036** | 清 | 留 |

弱窗误杀 Apr 赢家（NVDA/MSFT/META 各约 +60% TP）→ keep 崩。  
产物：`crowd6_range_stall_dual_v1` · `crowd6_ffo01_range_stall_dual_v1`  
工具：`tools/run_crowd6_range_stall_dual.py`

## 裁决

入场横截面 **扩 crowd 到 6 不晋级**。  
06-24 GOOGL 继续交给已有 **TRADE_TOX**（持仓早期无 MFE 快砸）——进场门做不到「只杀 GOOGL 不伤弱窗」。

## 与 Hunt 半边的关系

同日 AMD Hunt 见 [`hunt_range_stall_peer_research.md`](hunt_range_stall_peer_research.md)（T10 也不接线）。  
06-24 日线靠 AMZN 仍为正；双杀是肥尾，不是日亏主因。
