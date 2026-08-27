# AM 口袋：成交明细（trade-last）离线验收

> 工具：`tools/scan_am_pocket_trades_dual.py`  
> 产物：`results/research_am_pocket_trades_dual/`  
> 前置：`docs/am_pocket_combo_opt.md`、`docs/am_pocket_quote_ready.md`

## 决策

**09:30–10:00 历史 quote 不可用 → 离线闸改用期权成交明细（trade-last + 1% slip）。**  
Live/IB 仍可用实时 NBBO；不再用历史 FillSpec 卡住冻结口袋。

## 协议

| 项 | 值 |
|----|-----|
| 入场宇宙 | `no_b_up` enriched probes |
| Mark | 成交 last：买 `last×(1+slip)`，卖 `last×(1−slip)`，slip=1% |
| 组合 | @20% / max5 / cooldown 10m |
| 双窗 | `may_jul09` / `jul10_23` |
| **官方 dual** | 经济 dual：两窗 equity compound>0，disc n≥8，blind n≥3 |
| 参照 pulse dual | `_ok`（mean/add/day_win/max_hold）；blind 样本薄时可能不过 |

## 结果

| cell | n | disc 复利 | blind 复利 | blind n | econ dual | pulse dual |
|------|--:|--------:|----------:|--------:|:---------:|:----------:|
| **champ TP8/SL15/h240（冻结）** | **35** | **+44%** | **+5%** | **4** | **✓** | ✗（n&lt;6） |
| vd_soft 同退出 | 65 | — | — | — | ✓ | ✓ |
| vd+volr12 同退出 | 53 | — | — | — | ✓ | ✓ |
| champ TP10/h300 | 35 | — | — | — | ✗ | ✗ |
| champ SL12 | 35 | — | — | — | ✓* | ✗ |

\*以当次 `scoreboard.csv` 为准；冻结仍以 **champ TP8** 为准。

覆盖：**560/560** 探针有成交路径（对比 quote 中位首笔 lag≈1809s）。

## 裁决

**`PASS`（trade-mark 经济 dual）**

1. 成交明细覆盖开盘，适合做 AM 口袋离线账本。  
2. 冻结 champ 经济 dual 过；blind 仅 4 笔，**不过** pulse 风格 `min_n`——记为样本薄，不因此否决。  
3. `vd_soft` / `vd+volr12` 样本更大，pulse dual 也过；若要更稳的正式 `_ok`，可降级到 `vd+volr12`（牺牲一点复利）。  
4. 历史 quote FillSpec **不再**作为本袖离线硬闸。

## 冻结（更新）

```text
offline_mark: option trade-last (slip 1%)
pockets: no_b_up
entry: vd_soft ∩ cont60 ∩ mf100+ ∩ volr12
exit: TP8 / SL15 / h240
size: 20% / max5
acceptance: econ_dual_pass on trades
```

Profile：`CONFIG/strategy_profiles/am_pocket_vd_multi_tp8_sl15_v1.json`  
（Scanner 仍未接线；`acceptance.trade_last_dual=PASS`）

## 复现

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_pocket_trades_dual \
  --tag research_am_pocket_trades_dual
```
