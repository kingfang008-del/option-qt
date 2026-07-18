# 2026-07-05~06：META / NVDA「大行情没抓住」问题陈述（新 agent 分析入口）

> **用途**：把「7/5–7/6 META、NVDA 大行情没吃到」从聊天里拆成独立 brief，供新 agent 继续深挖。  
> **本文是问题与事实底稿，不是已完成结论。** 相关连亏窗见 [`jul7_9_mover_vs_pick_analysis.md`](jul7_9_mover_vs_pick_analysis.md)。

---

## 1. 用户问题（原意）

在研究底仓下，**2026-07-05 ~ 07-06** 前后感觉 **META / NVDA 有大行情但策略没抓住**。需要单独复盘：

1. 这两天（及紧邻交易日）真实日线 / 盘中最大行情是谁？幅度多少？  
2. RuleA / TopK / 实际成交分别是什么？  
3. 漏抓是因为：无信号、信号太晚、坑位被占、数据缺失、regime 挡、还是期权窗内根本赚不到？  
4. 是否和紧随其后的 **07-07~09 连亏**（NVDA/META 假方向 / 真趋势进不去）是同一机制？

---

## 2. 研究底仓与复现口径

| 项 | 值 |
|---|---|
| 因果基线 | `single_qqq_open_ladder_atm5otm_t30_rails_p20_googl_peer3_v1` |
| 研究底仓 | `extend_mtm_only`（T30→T45，MTM≥0，无 mf 确认） |
| 可选叠层 | `full_day` 事件禁入（**07-05/06 不在黑名单**） |
| Profile | `maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json` |
| 成交证据 | `maga7/results/hold_extend_ablation_mag7_googl_peer3_may_jul/extend_mtm_only/trades.csv` |
| 日线/信号扫描 | `maga7/results/mover_window_locate_may_jul/symbol_day.csv`、`daily_rank.csv` |
| 股票事实源 | `/mnt/s990/data/raw_1s/stocks/<SYM>/<SYM>_YYYY-MM-DD.parquet` |

`full_day` 黑名单只有 05-19/20/21、06-12/16/17/18，**挡不住 7 月初这几天**。

---

## 3. 日历与数据缺口（先澄清日期）

| 日期 | 说明 |
|---|---|
| **2026-07-05（日）** | **非交易日**。mover / trades / 1s 均无此日。若口语「7/5」指周末前，应对齐到 **07-02（四）或 07-03（五）**。 |
| **2026-07-03（五）** | 独立日周末前；`raw_1s/stocks` **全员缺失**（NVDA/META/TSLA/QQQ 皆无）。replay 窗常直接跳过。 |
| **2026-07-06（一）** | 正常交易日；**AMD 的 1s 缺失**（NVDA/META/TSLA/QQQ 有）。mover 表里仍可能出现 AMD（其它源），但 **stock_1s 路径无法成交 AMD**。 |
| **2026-07-07** | AMD 1s 仍缺失；NVDA/META 有。 |

→ 新 agent 分析「7/5–7/6」时，**有效交易日核心是 07-06**；周末前对照建议拉上 **07-02**。

---

## 4. 已提取事实：07-06（核心日）

### 4.1 标的日涨跌与 RuleA（mover_window）

按 `|day_ret|` 排序：

| symbol | day_ret | RuleA | RuleA ts (ET) | in_topk |
|---|---:|---|---|---|
| **TSLA** | **+6.86%** | UP | 11:01 | **是** |
| **AMD** | **+6.50%** | UP | 10:43 | **是**（但 1s 缺失，实盘/1s replay 吃不到） |
| **META** | **+3.05%** | UP | **12:18** | **否** |
| AAPL | +1.49% | 无 | - | 否 |
| AMZN | +0.81% | 无 | - | 否 |
| MSFT | -0.70% | 无 | - | 否 |
| **NVDA** | **+0.57%** | **无** | - | 否 |

`daily_rank`：当日 biggest = **TSLA**；topk 名义 = `AMD,TSLA`。

### 4.2 实际成交（extend_mtm_only / full_day 相同）

| date | symbol | dir | opt ret | reason | sig_ts |
|---|---|---|---:|---|---|
| 2026-07-06 | **TSLA** | UP | **+60.4%** | TP | 11:02 |

- **抓住了**：TSLA 大涨日（期权 TP）。  
- **没抓住 META**：有 RuleA UP@12:18，但 **未进 TopK**，当日也无 META 成交。  
- **NVDA**：全日仅 +0.57%，**无 RuleA**——在 07-06 上不应表述为「有大行情却漏抓」；更像记忆与邻近日（见 §6）混淆。  
- **AMD**：mover 显示大涨且更早 RuleA，但 **1s 数据缺失 → 策略路径无法成交**（数据问题，不是选股逻辑单独能解释）。

### 4.3 初步机制假说（待新 agent 验证）

对 **META +3% 漏抓**（§11 已验证）：

1. **TopK 早占满**：名义 topk = AMD/TSLA；META `in_topk=False`（主因）。  
2. **不是 concurrent**：`max_concurrent=2`；TSLA 11:31 已 TP，坑位在 META 信号前已空。  
3. **即使挤坑**：强制 META@12:19 → **SL −60%**（§11.2）；与 `displace_later` 全期为负一致。  
4. **不是事件黑名单**：07-06 ∉ `full_day`。

对 **NVDA**：

- 07-06 **不是** NVDA 趋势日；若用户坚持「NVDA 大行情」，应把窗口扩到 **07-07~08**（见下节）。

---

## 5. 紧邻对照：07-02（周末前最后一个完整交易日）

| symbol | day_ret | RuleA | in_topk | 实际成交 |
|---|---:|---|---|---|
| TSLA | **-7.65%** | DN@10:30 | 是 | DN，T+45 **+45%** |
| **META** | **-4.94%** | DN@**11:07** | **否** | **无** |
| AAPL | +4.75% | UP@10:31 | 否 | 无 |
| AMD | -4.18% | DN@10:30 | 是 | DN **TP +61%** |
| NVDA | -1.55% | DN@13:02（晚） | 否 | 无 |

→ 周末前同样存在 **META 有清晰日线/RuleA 但进不了 TopK** 的模式；NVDA 信号过晚且幅度小。

---

## 6. 不要和 07-07~09 混谈（但必须交叉读）

用户记忆里的「META / NVDA 大行情」在数据上更贴 **07-07~09**：

| 日期 | 策略成交（亏/赚） | 当日更大行情（未吃到或吃错） |
|---|---|---|
| 07-07 | NVDA DN **-30.6%** | 真跌更多在 **TSLA DN**；NVDA 全日反涨 |
| 07-08 | META DN **-25.8%**；TSLA DN +24% | 真大涨 **NVDA UP ~+3.7%~+4%**，信号 **12:58** 且 QQQ 不对齐 |
| 07-09 | AMD UP **-18.4%** | **META UP** 大涨，RuleA **12:21**，TopK 已满 |

完整表与路径级复盘：**[`jul7_9_mover_vs_pick_analysis.md`](jul7_9_mover_vs_pick_analysis.md)**。

**建议**：新 agent 先把 **07-06 META 漏抓** 钉死，再决定是否把「NVDA 漏抓」并入 07-08 叙事，避免日期错位。

---

## 7. 已做过、证明无效或仅部分有效的邻近实验（避免重复踩坑）

| 方向 | 文档 | 对「晚信号大行情」 |
|---|---|---|
| 后信号挤仓 `displace_on_later` | `displace_later_research.md` | 选型偶可换票，全期大亏 |
| TopK 延迟提交拍卖 | `topk_commit_auction_research.md` | 能选对票，推迟入场毁收益 |
| `entry_confirm_bars=1, mf` | `entry_confirm_bars_research.md` | 挡 07-07 假 NVDA，不解决晚 META/NVDA UP |
| 趋势纯度 / 剔 AMD | `trend_purity_*` / `symbol_exclude_amd_research.md` | 不能当默认开 |
| 事件 `full_day` | `event_calendar_full_day.md` | **与 07-05/06 无关** |

---

## 8. 新 agent 建议任务清单

1. **对齐日期**：确认用户说的「7/5–7/6」是否实指 07-06，或含 07-02 / 误指 07-07~09。  
2. **07-06 路径复盘**：用 1s 画 META / TSLA / QQQ 的 from_prev、mf10、peer、RuleA 触发与 TopK 占用时间线。  
3. **量化 META 漏抓代价**：若在 12:18 强制入场（或 TSLA 平仓后重入），期权 ret / 账户 day_ret 会怎样（单日反事实，不做全期默认）。  
4. **AMD 数据缺口**：补齐 `AMD_2026-07-06.parquet` 后，topk 是否变成先吃 AMD？会否挤掉 TSLA TP？  
5. **NVDA**：若只关心 07-05/06 → 记为「无大行情」；若关心「没吃到 NVDA 大涨」→ 转去 07-08 + `jul7_9` 文档。  
6. **产出**：在本文末尾或新文档写清「根因分类 + 是否值得改规则 + 与 07-07~09 是否同一旋钮」。

---

## 9. 快速命令（复现扫描）

```bash
# 日线 / RuleA / TopK
python - <<'PY'
import pandas as pd
sym = pd.read_csv('maga7/results/mover_window_locate_may_jul/symbol_day.csv')
print(sym[sym.date.astype(str)=='2026-07-06'].sort_values('abs_day_ret', ascending=False).to_string(index=False))
PY

# 实际成交
python - <<'PY'
import pandas as pd
tr = pd.read_csv('maga7/results/hold_extend_ablation_mag7_googl_peer3_may_jul/extend_mtm_only/trades.csv')
print(tr[tr.date.astype(str).between('2026-07-02','2026-07-09')][
    ['date','symbol','dir','ret','reason','sig_ts']
].to_string(index=False))
PY
```

---

## 10. 一句话现状（给新 agent）

**07-06：吃到了 TSLA 大涨；META +3% 有午后 RuleA 但未进 TopK；NVDA 当日几乎没动、无 RuleA。07-05 不是交易日。若问题核心是「META/NVDA 主升/主跌没吃到」，数据上更重的证据在 07-02（META DN）与 07-07~09（见姊妹文档），不要只在 07-05 上空转。**

---

## 11. 深挖结论（2026-07-17）

产出目录：`maga7/results/jul5_6_meta_nvda_deepdive/`（`summary.json`、时间线 CSV）。

### 11.1 时间线（stock_root 宇宙，与 ablation 一致）

| 时刻 (ET) | 事件 |
|---|---|
| 10:43 | **AMD** RuleA UP（fp≈+9.6%）→ TopK #1 |
| 11:01–11:02 | **TSLA** RuleA UP → TopK #2 → 成交；11:31 **TP +60.4%** |
| 12:18–12:19 | **META** RuleA UP（fp≈+2.56%，peer=3，QQQ 对齐）→ **rank #3，进不了 TopK** |
| — | **NVDA** 全日无 RuleA（day≈+0.6%） |

关键点（1s 特征，含正确 `prev_close`=07-02）：

| tod | META fp | TSLA fp | QQQ fp | peer_up |
|---|---:|---:|---:|---:|
| 11:01 | +1.1% | **+4.2%** | +1.7% | 6 |
| 12:18 | **+2.6%** | +6.3% | +1.7% | 3 |
| 12:45 | +1.6% | +6.1% | +1.6% | 0 |

> **口径陷阱**：只加载单日 1s 时 `prev_close` 会退化成开盘价，META `from_prev` 被压到 &lt;2%，会**假阴性**掉 RuleA。必须带上前日（07-02；07-03 缺失）。

### 11.2 META@12:18 强制入场反事实

| 项 | 值 |
|---|---|
| 合约 | `META260706C00597500`（0DTE ladder） |
| entry | 12:19:01 @ 2.636 |
| exit | **12:24:39 SL** @ 1.048 |
| **opt ret** | **−60.2%** |
| 持仓窗正股 | 595.20→593.81（−0.23%），信号后 fp 从 +2.56% 回落到 ~+1.5% |
| vs baseline | TSLA 单日贡献 ≈ **+12.1pp**；若再叠加 META ≈ **+0.0pp**（几乎抹平） |

**结论：07-06 漏抓 META 不是「错过利润」，而是「躲过一笔快速 SL」。** 与 `displace_later` 全期为负一致——午后挤仓在这一天也没有正期望。

### 11.3 AMD 缺口：会不会挤掉 TSLA？

| 层 | 07-06 AMD |
|---|---|
| `stock_root`（ablation / mover） | **有** → 进 TopK #1 |
| `stock_1s` | **无** `AMD_2026-07-06.parquet` |
| option 1s / open_lock | **无 quotes**；锁表 resolve → `no_lock` |
| 实际成交 | 无法成交 → 只剩 TSLA |

反事实：

1. **补齐 AMD 股票+期权**：TopK 仍是 AMD+TSLA；**不会挤掉 TSLA 的 TopK 名额**；可能多一腿 AMD（与 TSLA 重叠时第二腿 `size_frac=0.1`）。META 仍进不来。  
2. **若只用 stock_1s（无 AMD）**：TopK 变成 TSLA+**META**，replay 会成交 META 并 **SL −60%**——比现状更差。  
3. 文档初稿写「concurrent=1 占坑」不准确：配置是 `max_concurrent=2`；TSLA 11:31 已平，META 被挡纯粹是 **TopK earliest**。

### 11.4 根因分类 + 是否改规则

| 问题 | 分类 | 值得改默认规则？ |
|---|---|---|
| META 未成交 | TopK 最早占满（AMD/TSLA） | **否**——强制/挤仓当日为负 |
| META「大行情」 | 日线 +3% 有，但 **期权窗冲高回落** | 不应用日线后悔驱动入场 |
| NVDA | 无大行情 / 无 RuleA | 转 07-08（见 jul7_9） |
| AMD 未成交 | 数据缺口（1s + lock/quotes） | **值得补数据**，不是改选股 |
| 与 07-07~09 | 同属「最早 RuleA」旋钮 | 是；但 07-06 META 是该旋钮的**幸运面** |

**建议**：07-06 不必为 META 开 `displace_later` / 扩 TopK。优先补 AMD（及 07-03）stock/option 1s 以消除路径分叉；真要抓「晚到主升」应在 **07-08 NVDA / 07-09 META** 上单独做单日反事实，不要用 07-06 当正例。

### 11.5 已做：07-09 META 强制入场（摘要）

完整表见 [`jul7_9_mover_vs_pick_analysis.md`](jul7_9_mover_vs_pick_analysis.md)「07-09 强制入场反事实」。

| 情景 | opt ret | 账户≈ |
|---|---:|---:|
| 实际 AMD | −18.4% | −3.7pp |
| 强制 META@12:22 | **−8.0%**（T+30；延长持仓仍负） | −1.6pp |
| 强制 TSLA@12:47 | **+11.1%** | +2.2pp |

→ 07-09 换 META 只是**少亏**；正股午后续涨但 1DTE 期权窗吃不到。另：TopK#2 GOOGL 被 regime 挡后 **不会递补** META。
