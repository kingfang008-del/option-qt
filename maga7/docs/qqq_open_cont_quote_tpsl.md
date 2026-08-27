# QQQ open_cont × quote TP/SL（research）

First sleeve to clear **causal 1s + quote FillSpec TP/SL + dual-window** after Mag7 launch_slope / morn_sec REJECT.

## Entry

- Symbol: QQQ 0DTE ATM
- Clock: **09:45** open continuation
- Direction: sign of `(px_0945 / open − 1)`
- Selective filter: `|from_open| ≥ 0.2%` (preferred over fo=0 daily)

## Exit / gates

- Buy/sell: FillSpec 0.75/0.75 on quote
- First-passage **tp=10% / sl=25%**
- Entry gates: `spread ≤ 15%`, `lag ≤ 2s`, `mid ≥ 0.05`
- `max_hold=900s` safety only

## Dual-window (fixed merge recount)

| Window | n | mean | add | day_win |
|--------|---|------|-----|---------|
| May–Jun* | 24 | +3.6% | +0.087 | 79% |
| Jan–Mar | 26 | +6.8% | +0.178 | 88% |

\*Option root `/mnt/s990/data/raw_1s/dte0_options/QQQ` ends **2026-06-30**.

```bash
PYTHONPATH=. python -m maga7.tools.scan_qqq_open_cont_quote_tpsl \
  --tag research_qqq_open_cont_quote_tpsl_dual
```

## Caveats (not promoted)

1. Hard dual PASS cluster on **sl=25%**; tighter SL fails.
2. fo=0 (trade every day) also passes — prefer fo≥0.2%.
3. Need July data + open-drift null before profile freeze.

## Trades 账本双窗（含 Jul，2026-07-24）

Quote dte0 只到 06-30；改用 `/mnt/s990/new_option_data_s3_trades`（至 **2026-07-22**）。

```bash
PYTHONPATH=. python -m maga7.tools.scan_qqq_open_cont_trades_tpsl \
  --tag research_qqq_open_cont_trades_tpsl_dual \
  --include-fade-null --clocks 09:40,09:45,09:50
```

| 项 | 结果 |
|----|------|
| 日期 | 131 天（2026-01-12…07-22）；Jul 用 OCC ATM（`trades_occ`×126） |
| dual PASS | **15**（cont **10** / fade null **5**）→ `verdict=PASS` |
| Champion | **09:45** `fo≥0.2%` **tp10/sl25**：JM n=21 mean **+8.4%** dw 90%；MJ n=24 mean **+2.6%** dw 75% |
| 紧 SL | fo≥0.2% 下 **tp5/sl10、tp5/sl15、tp10/sl15** 亦双窗过闸（mean 更薄） |
| Open-drift null | **09:45 fade 不过闸**；**09:50 fade 有过闸** → 续作在 09:45 更干净，09:50 慎用 |

**研究状态（2026-07-25 Shadow 接线）：** catalog **`ACCEPT_RESEARCH`**（卫星袖，不改 Mag7 Rule-A）。  
- Research profile：`CONFIG/strategy_profiles/qqq_open_cont_0945_fo02_tp10_sl25_v1.json`  
- Offline runner：`python -m maga7.tools.run_qqq_open_cont_expert`  
- **Live/Shadow：** research_baseline 上 `qqq_open_cont.enabled=true` → `Mag7Scanner.drain_open_cont`（`live_engine` / `redis_consumer` 帧时钟）；`event_source=qqq_open_cont`；出场 `exit_simple` tp10/sl25/hold15m。  
- Jul quote 缺口仍用 trades fallback；IB 实盘需另订 QQQ 期权链（磁盘 ATM 解析服务 shadow/dry）。

## Quote vs trades 公平对照（champion，2026-07-24）

同日交集（quote∩trades∩stock，**May 窗截到 06-30**），规则固定：09:45 · fo≥0.2% · tp10/sl25。

| 账本 | Jan–Mar | May–Jun | dual |
|------|---------|---------|------|
| **quote** FillSpec（sp≤15% lag≤2） | n=24 mean **+6.4%** dw 88% | n=24 mean **+3.6%** dw 79% | **PASS** |
| **trades** last±1% | n=21 mean **+8.4%** dw 90% | n=16 mean **+0.5%** dw 69% | **PASS**（后窗更薄） |

产物：`research_qqq_open_cont_quote_vs_trades_fair/`。  
要点：开盘 quote **有**且可成交；quote 双窗更稳。trades 前窗偏乐观、后窗偏瘦（流动性选择）；含 Jul 的 trades 全窗仍正，但不能替代 quote。

## Related rejects

- Mag7 launch_slope quote: `research_launch_slope_quote_tpsl_dual_m05`
- Mag7 morn_sec quote: `research_morn_sec_quote_tpsl_dual`
- Mag7 AM 09:30–10:00 launch trades PASS / quote REJECT: `AM_TRADES_PASS_QUOTE_REJECT`
