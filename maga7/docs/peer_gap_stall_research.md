# Peer + overnight gap stall（大亏日主动门）

> 模块：`common/peer_gap_gate.py` · 工具：`tools/run_peer_gap_dual.py`  
> 产物：`results/peer_gap_dual_v1`  
> 基线：research spine（已含 overnight_gap BLOCK 4%+adv）

## 1. 动机

接线 overnight_gap 后，2–7 月仍大亏：

| 日 | day_ret | 形态 |
|----|---------|------|
| 04-08 | −11.1% | AAPL UP **SL−55%**，peer=3，fav_gap≈+2.0%，ffo≈+0.3% |
| 02-17 | −11.1% | GOOGL DN SL−55%，peer=6，fav_gap≈+1.9% |
| 02-18 | −8.2% | NVDA UP TOX（peer=3,gap+2%）+ AMZN UP TOX |

纯 gap/ffo 规则会误杀 META 07-01 / 多笔 NVDA 赢家。  
**可区分特征：** 弱 peer（`peer_align==3`，已是门槛最小值）× 中等顺势隔夜 gap（≥1.5%）。

## 2. 规则

`peer_align ≤ max_peer(3)` ∧ `fav_gap ≥ min_fav_gap(1.5%)` → **`mode=block`**（不缩仓）。

## 3. 双窗（`peer_gap_dual_v1`）

| arm | weak vsOFF | strong vsOFF | 裁决 |
|-----|------------|--------------|------|
| **P3_G15** | **1.438** | **1.026** | **`DUAL_PASS_WIRE`** |
| P3_G15_UP | 1.438 | 1.026 | 同（命中皆 UP） |
| P3_G15_FFO | 1.259 | 1.000 | 强窗无抬升 |
| P3_G18 | 1.303 | 1.000 | 强窗无抬升 |

命中（相对 OFF）：
- 弱：02-18 NVDA、03-25 AMD、03-31 GOOGL、**04-08 AAPL**（全亏单）
- 强：05-07 MSFT 小亏

日账：04-08 −11%→0；02-18 −8.2%→−2.7%（AMZN 仍在）；**02-17 未覆盖**。

## 4. 接线

**2026-07-26：已 WIRE research baseline** `trade.peer_gap_gate` = P3_G15。

02-17 GOOGL DN：peer 强，DN gap-stall 代理会误杀 02-27/06-10/06-26 赢家 → **暂 park**。

```bash
PYTHONPATH=. python -m maga7.tools.run_peer_gap_dual \
  --out maga7/results/peer_gap_dual_v1
```
