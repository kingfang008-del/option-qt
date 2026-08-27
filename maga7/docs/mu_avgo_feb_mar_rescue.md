# MU/AVGO 扩池救援 2–3 月弱窗（研究）

**日期：** 2026-07-19  
**状态：** `REJECT_BASELINE`（弱窗有抬升，强窗腰斩）  
**基线：** peer3 L2 + TT1_05 + sl55 + trade_toxic（tt600d）  
**产物：** `results/mu_avgo_feb_mar_rescue/`

## 设定

| 项 | 值 |
|----|-----|
| 交易池 | Mag7+GOOGL **+ MU + AVGO** |
| peer | 仍 Mag7-only（可比） |
| TopK | 2（不变） |
| lock | `~/train_data/locked_targets_map_maga7_googl_mu_avgo_open_ladder_atm5otm_research.parquet` |
| Feb–Mar 覆盖 | MU lock **19** 日 / AVGO **31** 日（→03-18）；quote miss=0 |

## Scoreboard

| window | Mag7+GOOGL | +MU+AVGO | vs base |
|--------|----------:|---------:|--------:|
| **Feb–Mar** | +39.9% / DD −23.9% / n=53 | **+58.5%** / DD −21.9% / n=52 | **146%** |
| **May–Jul** | +1856% / DD −5.4% / n=57 | **+656%** / DD −10.6% / n=54 | **35% FAIL** |

Feb–Mar 新成交：MU 8 笔（均 ret +12%）、AVGO 6 笔（均 ret −5%）。

## 解读

1. **弱窗确实被「救」了一截**（+40%→+58%），主要来自 MU 占到 TopK 席位后的几笔赢家（如 03-05 DN TP）。  
2. **强窗不可接受**：同 TopK=2 下 MU/AVGO 挤掉 Mag7 大赢家，May–Jul 只剩基线约 **1/3**。  
3. 与历史结论一致（[`causal_single_t30_rails_baseline.md`](causal_single_t30_rails_baseline.md)）：MU/AVGO 扩池弱于 Mag7+GOOGL，**不默认并入 freeze**。

## 若还想用 MU/AVGO

不要塞进 Mag7 TopK=2，可选（均未做）：

- 独立袖仓 / 独立 TopK 预算  
- 仅弱窗日历启用（易过拟合）  
- 仅 Hunt/故事日名单  

**当前 research_baseline 符号表不改。**
