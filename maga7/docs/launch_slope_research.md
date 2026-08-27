# Launch-slope sleeve (research)

Detect the **steepest short-window price impulse** on 1s bars — the “first candle”
of a spike — independent of 10:30 Rule-A / peer3 freeze.

## Detector

- Module: `maga7/common/launch_slope.py`
- `ret_k = close[t]/close[t−k]−1` (k = 3/5/10s)
- Causal local peak: `ret_k` equals rolling max/min over `peak_lookback` (default 60s)
- Rising edge when `|ret_k| ≥ abs_ret_min`
- Optional confirm: second-level mf same direction, `vol_z`, `from_prev`, peer mf

## Scanner

```bash
python -m maga7.tools.scan_morning_launch_slope \
  --start-date 2026-05-01 --end-date 2026-07-17 \
  --tag research_launch_slope_may_jul
```

Sessions: `open_0930_1030`, `mid_1030_1100`.

## Option fill

```bash
python -m maga7.tools.run_morning_launch_option_fill \
  --events-tag research_launch_slope_may_jul \
  --tag research_launch_slope_option_fill_may_jul_v2
```

Exit books: `horizon` (flatten at H from fill) vs `ladder_active` (SEC_MAX=H).
Clocks are **fill-anchored** (first quote ≥ signal) because morning 0DTE books are gappy.

### May–Jul v2 highlights (pos_frac=0.10)

| Cell | Book | total_ret | win | notes |
| --- | --- | --- | --- | --- |
| mid 5s/0.2%/H180 | **horizon** | **+39.0%** | 54% | best overall |
| open 3s/0.2%/H120 peer3 | horizon | **+19.0%** | 56% | best open |
| open 3s/0.2%/H180 mf1 | ladder | +4.6% | 50% | ladder > horizon here |
| mid same | ladder | +8.7% | 47% | cuts winners vs horizon |

Promotion lean: **mid-session launch + horizon H≈180s**; open 3s/H120 as secondary. Ladder helps some open cells but clips the mid winner.

## AM sleeve freeze (2026-07-23)

三袖组合的 **AM #1** 冻结为 open-cell（与 CORE 硬互斥，不用 mid）：

| 项 | 值 |
|----|-----|
| Profile | `CONFIG/strategy_profiles/am_launch_slope_open_s3_h120_peer3_v1.json` |
| Cell | `s3_r002_h120_fp0_p3` / `horizon` |
| Signal | 09:30–**10:25**（截断 10:30 前） |
| Size | `position_frac=0.10` |
| May–Jul（历史） | **+19%** / win 56% / MaxDD −8% |
| Accept | `tools/run_am_sleeve_accept.py` → `results/am_sleeve_accept_v1` |

Rule-A morning sleeve（`morning_sleeve_early_mf5_s6_isolated_v1`）保持负对照，不升格。  
组合文档：[`sleeve_portfolio_research.md`](sleeve_portfolio_research.md)。

## Trade-last TP/SL（2026-07-24，May–Jul）

Clock H 不再作主出场。对同一批 1s launch edges，用 `new_option_data_s3_trades` last±1% slip + first-passage TP/SL：

```bash
PYTHONPATH=. python -m maga7.tools.scan_launch_slope_tpsl \
  --events-tag research_launch_slope_may_jul \
  --tag research_launch_slope_tpsl_may_jul
```

| Cell | TP/SL | n | mean | win | add | day_win | 备注 |
|------|-------|---|------|-----|-----|---------|------|
| `open_s3_r002_p2` | 15/25% | 24 | **+9.4%** | 83% | +0.23 | 85% | 最佳 add |
| `open_s3_r002_p3` | 15/25% | 22 | +8.7% | 82% | +0.20 | 84% | 与旧 AM 冻结同源入口 |
| `open_s3_r002_p2` | 15/15% | 24 | +8.3% | 75% | +0.20 | 80% | 更紧 SL，仍 +EV |
| `mid_s5_r002_fp005_vz1` | 15/25% | 31 | +4.7% | 65% | +0.16 | 60% | mid 仍弱于 open |

产物：`results/research_launch_slope_tpsl_may_jul/`（50 个过闸组合）。  
旧 profile 的 clock H120 **不要**当作已验证实盘出场。

## Dual-window OOS + quote 对拍（2026-07-24）

| Book | Window | Cell | TP/SL | n | mean | add | day_win |
|------|--------|------|-------|---|------|-----|---------|
| **trade** | May–Jul | open p2 | 15/15 | 24 | +8.3% | +0.20 | 80% |
| **trade** | Jan–Mar | open p2 | 15/15 | 36 | +0.9% | +0.04 | **54%** |
| **trade** | Jan–Mar | open p2 | 15/25 | 36 | +3.3% | +0.14 | 68% |
| **quote** | May–Jul | open p2 | 15/15 | 45 | +1.8% | +0.08 | 57% |
| **quote** | Jan–Mar | open p2 | 15/15 | 63 | **−3.7%** | **−0.19** | 38% |
| **quote** | Jan–Mar | open p3 | 15/25 | 54 | **−5.9%** | **−0.27** | 42% |

工具：
- trade OOS：`scan_launch_slope_tpsl --events-tag research_launch_slope_jan_mar_am --tag research_launch_slope_tpsl_jan_mar`
- quote 对拍：`results/research_launch_slope_tpsl_quote_dual/trade_vs_quote.csv`

**裁决：不升格。**  
1. 紧 SL（15/15）OOS day_win 掉到 ~50%。  
2. 宽 SL（15/25）trade 双窗仍 +，但 **quote FillSpec 在 Jan–Mar 全面翻负**。  
3. trade 成交数明显少于 quote（需 5s 内有 print）→ 流动性选择偏差，trade-last 虚高。  
实盘应按 quote 可成交价计；当前入口+TP/SL **未通过可执行验收**。

## Quote 可执行门 + TP/SL（2026-07-24）

入口门：`lag≤{2,3}`、`spread≤{5,8,10,15}%`、`mid≥0.05`；出场 FillSpec TP/SL。

```bash
PYTHONPATH=. python -m maga7.tools.scan_launch_slope_quote_tpsl \
  --events-tags research_launch_slope_may_jul,research_launch_slope_jan_mar_am \
  --tag research_launch_slope_quote_tpsl_dual_m05 \
  --min-mid 0.05
```

| 项 | 结果 |
|----|------|
| 双窗同时过闸 | **0**（`verdict=REJECT`） |
| May–Jul | 紧 spread 后 n 很低（宽门 resolve≈20/82）；小样本假阳常见 |
| Jan–Mar | 紧 spread（≤5–8%）偶有单窗 +add，但 day_win 多 &lt;0.55；与 May–Jul 无交集 |

模块：`common/option_quote_tpsl.py`、`tools/scan_launch_slope_quote_tpsl.py`。

## AM 09:30–10:00 trades 双窗重验（2026-07-24）

信号硬切到 **[09:30, 10:00)**（不再混 10:00–10:30），定价仍为 `new_option_data_s3_trades`：

```bash
PYTHONPATH=. python -m maga7.tools.scan_am_0930_1000_trades_dual \
  --tag research_am_0930_1000_trades_dual
```

| 项 | 结果 |
|----|------|
| dual PASS | **17**（trades 账本） |
| 最佳 add | `open_s3_r002_p2` tp15/sl25（JM +2.6%/dw65%，MJ +9.2%/dw82%） |
| 紧 SL≤15% | 仅 tp5/sl15 两条薄边过闸 |

**研究状态：** 开盘半小时在 trades 上可立住；**quote 可执行仍未过**（见上节 Dual-window OOS）。未升格。

### AM quote 对拍（硬切 09:30–10:00）

`scan_am_0930_1000_quote_dual` → `research_am_0930_1000_quote_dual/`：

| 项 | 结果 |
|----|------|
| dual PASS | **0**（`REJECT`） |
| resolve | 31/119（~26%） |
| champion p2 tp15/sl25 | JM quote **负**；MJ n&lt;15 |

**裁决：`AM_TRADES_PASS_QUOTE_REJECT`** — 与全窗 quote 对拍结论一致；不升格。  
**结论：** 加可执行过滤不能把 launch_slope 修成可交易；quote 双窗仍失败。下一刀应换入口家族，不以本 sleeve 调参为主。
