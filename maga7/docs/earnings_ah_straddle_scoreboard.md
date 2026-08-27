# Earnings AH ATM straddle scoreboard

**Status: ABANDONED（2026-07-27）** — 不做财报隔夜长波动 / 不对次日做波动预测；保留 AH 日 `earnings_ah` **symbol blackout**（禁 CORE）。样本见下，不再晋级 sleeve。

Research-only archive. Answers: after Mag7 **AH earnings**, does buying ATM call+put into the print survive **IV crush / no-move**?

## How to run

```bash
PYTHONPATH=. python -m maga7.tools.run_earnings_ah_straddle_scoreboard \
  --out /mnt/s990/data/maga7/results/earnings_ah_straddle_scoreboard_v1
```

Contract: open-ladder ATM C+P same strike（prefer `front_dte=2,1,0`）.  
Mark: AH eve RTH last mid → next open / +30m / +60m（quote_1s，缺则 option_1m `c`）。  
`em_pct ≈ eve_straddle / eve_spot`；`move_vs_em = |gap| / em_pct`。

## v1 结果（2026-07-26）

| AH 日 | 标的 | gap | EM | move/EM | straddle@open | @+30m | IV crush |
|------:|------|----:|---:|--------:|--------------:|------:|---------:|
| 05-20 | NVDA | **−0.5%** | 5.7% | **0.09×** | **−48%** | −48% | **−15%** |
| 07-22 | GOOGL | −6.1% | 6.0% | 1.02× | +25% | +27% | n/a |
| 07-22 | TSLA | −8.8% | 5.9% | **1.50×** | **+73%** | +119% | n/a |
| 07-29 | META/MSFT | — | — | — | missing（本地数据切到 ~07-24） | | |
| 07-30 | AAPL/AMZN | — | — | — | missing | | |

产物：`/mnt/s990/data/maga7/results/earnings_ah_straddle_scoreboard_v1/`  
（`scoreboard.csv` / `summary.json` / `verdict.json`）

## 解读

- **NVDA**：典型「没怎么动 + crush」→ 双边开盘即亏近半。这是 sleeve 的主风险，不是猜错方向。
- **TSLA / GOOGL**：实现跳空 ≥ 定价 EM → 双边赚钱；TSLA 是大赢家样本，**不能单独晋级**。
- 样本可用 n=3，`verdict=EDGE_HINT_NEED_MORE_SAMPLE`。回填 07-29/30 期权+股票后再判。

## 与系统的关系

- 日历：`earnings_ah` **禁 CORE 方向**（已补 07-22 TSLA/GOOGL）。
- **不**因此自动开长波动 sleeve；需更多 crush / 小跳样本 keep 过线。
