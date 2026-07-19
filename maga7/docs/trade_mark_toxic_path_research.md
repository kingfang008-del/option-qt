# 成交价标记的早期毒性路径（trade-mark toxic cut）

**日期：** 2026-07-19  
**数据：** `/mnt/s990/new_option_data_s3_trades`（OPRA trades → **1s OHLCV**，`c`=秒内 last）  
**状态：** **已升 research_baseline**（cut25 / mfe05 / min_hold60 / **max_cut=600s** / **div_mfe=6% when stock adverse&lt;0.5%**）  
**对照基线：** L2+TT1_05+sl55（trade_toxic off）  
**工具：** `run_trade_toxic_ablation` · `run_trade_toxic_asymm_ablation` · `run_trade_toxic_div_ablation`  
**产物：** `results/trade_toxic_*_dual_window*/`（以 **div06_adv05** 为当前基线对照）

## 规则（当前 baseline）

- **信号（成交 last）：** 自 **quote fill** 起累计 MFE &lt; `mfe_bypass`（默认 5%），且成交 MTM ≤ −`cut_ret`（25%）；持仓 ≥`min_hold_seconds`（60）后允许切
- **非对称窗：** 仅 fill 后 **`max_cut_seconds=600`** 内允许 TRADE_TOX（挡住慢阴跌误伤，如 06-11）
- **股价背离软 MFE：** 标的不利方向 ret &lt; `div_stock_adverse_max`（0.5%）时，MFE 门槛放宽到 `div_mfe_bypass`（6%）（啃 05-06 边界漏网）
- **MFE：** 全部成交 print 的 running max（含 min_hold 窗口内）
- **离场（quote）：** 触发时刻用既有 fill 卖出价 → reason `TRADE_TOX`
- 缺成交 path：不触发

## 接线 bug（v1 FAIL → v2 PASS）

首轮正式消融强窗仅 ~66%，根因不是规则本身，而是两处实现错误：

1. **锚点：** 成交标记用了 signal ts，而非 quote fill。`entry_confirm` / 报价空窗下，fill 前的高价 print 造成虚假 −40%+「挖坑」（05-15 AMD、07-01 META 被秒切）。
2. **MFE 窗口：** 峰值只在 `min_hold` 之后更新，持仓前 60s 的真实 MFE（如 07-13 NVDA +15%）被丢掉，bypass 失效。

修复后复跑；升线以 **v2** 为准。v1 产物仅作对照。

## 正式接线双窗 v2（`run_offline_replay`）

升线门槛：双窗 total_ret **≥95%** of sl55 off。

| variant | May–Jul ret | vs | MaxDD | tox | Feb–Apr ret | vs | MaxDD | tox |
|---------|------------:|---:|------:|----:|------------:|---:|------:|----:|
| 00_off | +1555% | 100% | −11.1% | 0 | +151% | 100% | −26.0% | 0 |
| cut20_mfe05 | +1585% | 102% | −6.6% | 8 | +108% | **71%** | −29.0% | 12 |
| **cut25_mfe05** | **+1721%** | **111%** | **−6.6%** | 5 | **+159%** | **106%** | −26.0% | 9 |
| cut30_mfe05 | +1784% | 115% | −6.6% | 3 | +135% | **90%** | −27.8% | 7 |
| cut25_mfe08 | +1403% | **90%** | −11.2% | 8 | +143% | 95% | −24.6% | 14 |

**v2 候选取舍（已由 asymm 取代）：** `cut25/mfe05/min_hold60`。

## 非对称消融（v3，相对 sl55 off）

目标：压 06-11 误伤，保留 05-11 / 06-24 真毒性切。

| variant | May–Jul vs | Feb–Apr vs | 06-11 日 | 备注 |
|---------|----------:|----------:|---------:|------|
| base cut25 | 111% | 106% | **−6.2%**（误伤） | tox=5 |
| persist30 | 115% | **82%** | −3.5% | 弱窗 FAIL |
| **max600** | **118%** | **108%** | **−3.5%** | **升线**；tox=2 |
| persist30+max600 | 115% | **85%** | −3.5% | 弱窗 FAIL |
| qconf20 | 111% | 105% | −6.2% | 无帮助 |

## 股价背离软 MFE（v4）

目标：啃 05-06（期权挖坑、股价几乎不动，但峰值 MFE 刚好 =5% 漏网）。

规则：当标的相对 fill 的 **不利方向 ret &lt; `div_stock_adverse_max`** 时，允许峰值 MFE &lt; `div_mfe_bypass`（仍要求成交 MTM≤−25%、max_cut 窗内）。股价用已加载的 1m `close`（与 replay stock_day 一致；秒级 OHLCV 结论同向）。

| variant | May–Jul vs off | vs max600 | MaxDD | 05-06 日 | Feb–Apr vs off |
|---------|---------------:|----------:|------:|---------:|---------------:|
| max600 | 118% | 100% | −6.6% | −6.6% | 108% |
| div08 + adv0.5% | **100%** | **85%** | −8.7% | −5.4% | 109% |
| **div06 + adv0.5%** | **119%** | **101%** | **−5.4%** | **−5.4%** | **109%** |
| 全局 mfe08（无股价闸） | 100% | 85% | −8.7% | −5.4% | 104% |

div08 误伤 **06-15 NVDA**（本是赢家）→ 强窗崩盘。  
**升线：`div_mfe_bypass=0.06` + `div_stock_adverse_max=0.005` + max600。**  
May–Jul **+1856%** / MaxDD **−5.4%**；Feb–Apr **+164%**。

### 强窗 focus（div06 vs off）

| 日 | 标的 | off | div06 | 备注 |
|----|------|----:|------:|------|
| 05-06 | NVDA UP | T+30 −33% | TRADE_TOX **−27%** | 日亏 −6.6%→−5.4% |
| 05-11 | TSLA DN | SL −56% | TRADE_TOX **−24%** | 保留 |
| 06-24 | GOOGL UP | SL −55% | TRADE_TOX **−27%** | 保留 |
| 06-11 | TSLA UP | T+30 −18% | T+30 **−18%** | max600 挡住 |

## 工程状态

| 项 | 状态 |
|----|------|
| `option_trades.py` / `simulate_trade` | 已接（fill 锚点 + min_hold 内 MFE + max600 + div06） |
| profile `trade_toxic` | **enabled: true**（research_baseline） |
| 单测 `tests/test_trade_toxic.py` | 通过 |
| Live/stream/OMS 成交标记 | **未做**（offline 已升；live 前需同口径） |

## 残留风险 / 下一步

- 弱窗仍有部分硬 SL 未被成交路径提前标出。
- Live 成交稀疏时 asof 滞后；上实盘前需 stream 对拍。
- 旧 TCN / 入场因子：不建议直接重开；若再挖优先持仓后路径签名（见 autopsy）。
