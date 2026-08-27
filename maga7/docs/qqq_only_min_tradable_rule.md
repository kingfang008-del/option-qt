# QQQ-only 最小可交易规则 — 验收规格

**日期：** 2026-07-22  
**角色：** `acceptance_spec` —— **独立于** Mag7 `peer3_v1` / `research_baseline`  
**一句话：** 用 QQQ 0DTE/1DTE 点差优势，先冻一条「能亏得起、能解释、能影子跑」的窄规则；不过门不上线。

相关：[`qqq_only_sleeve_research.md`](qqq_only_sleeve_research.md) · [`wave_confirm_spec.md`](wave_confirm_spec.md) · [`flow_state_event_architecture.md`](flow_state_event_architecture.md)

---

## 0. 边界（先锁死）

| 做 | 不做 |
|----|------|
| 标的 **仅 QQQ** | Mag7 / GOOGL 个股期权 |
| 合约 **ATM±1，DTE0 优先；DTE1 作对照** | OTM5 ladder 当主表达（sleeve L0 只作历史对照） |
| 方向来自 **因果价格路径规则**（A/B） | TFT / FCS / ExecutionEngineV8 |
| 持仓权来自 **成交后波段确认** | Rule-A 点火即拿满 T30 |
| 验收看 **双窗 + 毒性日 + 语义审计** | 单窗翻倍、事后调参升线 |

与现有两条线的关系：

| 线 | 状态 | 本规格中的位置 |
|----|------|----------------|
| Mag7 peer3 full_day | research_baseline，非 freeze | **对照外**；不并入 |
| `qqq_only_*` sleeve（Rule-A + OTM5） | 可行性 +66%/+92% | **旧壳**；信号源不沿用 |
| oracle / foresight / distill | 规则发现沙盒 | **入场候选源** |
| wave_confirm（Mag7） | shadow，弱窗尾损未过 | **语义借用**；参数在 QQQ 上重标定 |

---

## 1. 最小规则骨架（v0）

```
日态臂（禁做日）
      │
      ▼
候选点火（最多 2 笔/日：A 上午续 + B 拉伸回撤）
      │
      ▼
FILL 探针（ATM，标准仓；持仓权未授予）
      │
      ▼
WAVE_CONFIRM（股票路径为主）
  pass → ARMED → trail / toxic / 时钟安全阀
  fail → WAVE_ABORT → 立即平；禁止拖时钟
```

### 1.1 日态臂（禁做 / 缩仓）

任一触发 → **当日禁开新仓**（已持仓只允许减/平）：

| 条件 | 阈值（初值，后验可改一次） | 理由 |
|------|---------------------------|------|
| 事件日历硬风险 | 宏观/指数重磅窗口 | 与 Mag7 event 栈对齐语义 |
| RTH 开盘后 10 分钟内 | `09:30–09:40` 禁入 | 避开开盘拍卖噪声 |
| 极端静日 | `\|AM_ret→10:30\| < 25bp` 且 `range→10:30 < 40bp` | quiet_am 无波段可做 |
| 点差不可交易 | ATM mid>0 且 `(ask-bid)/mid > 3%` 持续 ≥30s | 点差优势失效日停手 |

**不定方向的新闻/叙事不作单笔确认。**

### 1.2 候选点火（最多日 2 笔）

来自 distill/foresight 的因果 A/B（工具已有）：

| 规则 | 窗 | 逻辑（初值对齐 distill 默认） |
|------|----|------------------------------|
| **A_am_continuation** | 09:40–10:15 | `\|from_open\| ≥ fo_min`（默认 80bp）且短窗同向 extend（默认 +10bp）→ 顺势 ATM 一笔 |
| **B_stretch_fade** | 10:30–15:00 | `\|from_open\| ≥ fo_fade`（默认 100bp）且短窗逆向 bounce（默认 10bp）→ 向开盘方向 fade 一笔 |

约束：

- A 与 B **非重叠**（B 仅在 A 已平或 A 未触发后）
- 每规则 **日最多 1 次**；总并发仓位 **≤1**
- 方向只由路径规则给出；**禁止**横截面 `|fp|` 帽当主门

### 1.3 WAVE_CONFIRM（成交后，生产语义）

借用 Mag7 knife-2b 结论，在 QQQ 上重标定：

| 项 | 生产默认 | 否决 |
|----|----------|------|
| 确认对象 | QQQ **股票** `signed` / `adverse` | 期权 MTM 单独 ARMED |
| 逆向废证 | `stock_min ≤ −A` → `WAVE_ABORT` | — |
| 可撤销 | ARMED 后 `revoke` 窗内再逆向 → ABORT | 只靠成交前 hard |
| 超时 | **`on_timeout=allow`**（未武装交给 tox/clock） | 硬 `timeout=abort`（Mag7 已证杀慢启动 TP） |
| 期权加速 | `opt_mtm ≤ −M` **且** 股票未同向 → 可加速 ABORT | 期权单独 ARMED |

未 ARMED **禁止**：hold_extend、加仓、Hunt。

### 1.4 ARMED 后退出（安全阀，不是收益引擎）

优先级固定：

1. `WAVE_ABORT` / revoke  
2. `trade_toxic`（期权阴跌、股票不跟）  
3. trail（activate/dd 由 distill 日网格给初值，不裸网格扫生产）  
4. 时钟上限（建议 A/B 默认 hold **600s**；仅安全阀）  
5. 日亏熔断 / 点差恶化强制平  

**禁止** always-on `delta_time_stop` / `roi_time_stop`（Mag7 L3 已 REJECT）。

---

## 2. 验收门（全部通过才谈 paper/shadow size-up）

评估协议：

- **Fill：** `FillSpec(0.75, 0.75)` ask/bid（与现有 QQQ 工具一致）  
- **强窗：** May–Jul（有 0DTE ATM 的交易日）  
- **弱窗：** Feb–Apr（需先补齐数据；补不齐则该窗记 **BLOCKED**，不得用单窗升线）  
- **毒性探针日：** 至少含 1 个「假突破后硬扛会深亏」的已知日（从 oracle 尸检点名，不事后挑）  
- **冻结：** 参数网格最多 **一轮**；通过后门禁再动旋钮  

### 门 A — 语义（硬）

| # | 要求 |
|---|------|
| A1 | 审计日志能区分：`PROBE` / `ARMED` / `WAVE_ABORT` / `TOXIC` / `TRAIL` / `CLOCK` |
| A2 | **禁止**大量「未 ARMED 却靠 CLOCK 离场」；目标 `clock_share_among_unarmed ≤ 5%` |
| A3 | 毒性探针日：相对「无确认硬扛」版本，最差笔亏损 **显著变浅**（至少浅 30% 相对，或不再 ≤−25%） |

### 门 B — 尾损（硬）

| # | 要求 |
|---|------|
| B1 | 强窗：`n(ret ≤ −25%)` ≤ 对照无确认版的 **50%**，且 `worst ≥ −20%`（期权收益） |
| B2 | 弱窗：`n(ret ≤ −25%)` **不恶化** vs 无确认版；`worst` 不差于无确认版 |
| B3 | 单日最大亏损（仓位归一后）有硬顶；初值建议 **日亏 ≤ −8%** 触发停手 |

### 门 C — 收益保留（软主门 / 硬否决带）

| # | 要求 |
|---|------|
| C1 | 相对「同 A/B 入场、无 WAVE、纯时钟」对照：强窗 total_ret **retain ≥ 0.70** |
| C2 | 若 retain `< 0.50` 且尾损未同时过 B → **规格失败**（照搬 Mag7 §6） |
| C3 | 弱窗 total_ret **≥ 0**（允许小正）；若弱窗为负且强窗靠少数暴利日 → **否决** |

### 门 D — 稳健与工程（硬）

| # | 要求 |
|---|------|
| D1 | 双窗可跑；任一窗数据缺口 → 状态 `BLOCKED`，不得 freeze |
| D2 | 离线 replay 与 stream 对拍：同信号同 fill 锚点，成交集合一致 |
| D3 | Live 镜像清单齐：scanner 有 WAVE 状态机；OMS 有 `WAVE_ABORT`；缺一不可 size-up |
| D4 | Paper/Shadow ≥ 10 个完整 RTH 日，无未解释的 ADVERSE_FILL / 状态机卡死 |

**升线一票否决：** 只靠收紧 SL、入场横截面「假确认」、单窗调参、或把 Mag7 peer3 数字拿来比绝对值。

---

## 3. 与现有工具的差距图

| 能力 | 已有 | 缺口 |
|------|------|------|
| 单日机会上界 | `run_qqq_oracle_day_opportunities.py` | 需固定「毒性探针日」清单写入本规格附录 |
| 单日 A/B vs oracle | `run_qqq_foresight_rules_day.py` | 未接 WAVE / toxic |
| 多日蒸馏 | `run_qqq_multi_day_rule_distill.py` | 输出阈值，**无**验收 scoreboard（门 A–C） |
| Morning sec 结构 | `run_qqq_0dte_morning_structures.py` 等 | 研究扫描，非生产状态机 |
| QQQ sleeve Rule-A | `run_qqq_only_sleeve_scoreboard.py` | 信号源不同；可作「旧壳对照」一行 |
| WAVE 状态机 | `wave_confirm.py` + Mag7 ablation | **未**接到 QQQ A/B 入场；QQQ 阈值未标定 |
| Live | Mag7 scanner/OMS | QQQ-only 路径 **未**产品化 |

---

## 4. 执行顺序（建议 2 周，不扩 scope）

1. **数据门：** 确认 Feb–Jun（或至少双窗）QQQ 股票 1s + DTE0 ATM 可发现日期表；缺口列表进附录。  
2. **对照 0：** 冻结 A/B 初值（distill 默认），跑「无 WAVE / 纯 hold600+trail」双窗 scoreboard → 记为 `CTRL0`。  
3. **对照 1：** 同入场 + WAVE（`timeout=allow` + revoke）→ `WAVE1`；只评门 A/B/C。  
4. **对照 2：** `WAVE1` + toxic → `WAVE1T`；毒性探针日必须过 A3/B1。  
5. **不过门则停：** 写失败原因（哪扇门），**禁止**立刻开大网格；只允许改 **一个** 语义旋钮再复跑。  
6. **过门后：** 写 `qqq_only_min_v1` profile（`role=shadow_candidate`，`frozen_at` 仍 null）+ live 镜像 checklist；Paper 10 日。

命令锚点（实现 scoreboard 后）：

```bash
# 发现 / 蒸馏（已有）
python -m maga7.tools.run_qqq_multi_day_rule_distill \
  --start-date 2026-02-01 --end-date 2026-06-30 \
  --out /mnt/s990/data/maga7/results/qqq_multi_day_rule_distill_v1

# 验收（待实现：同入场 CTRL0 vs WAVE1 vs WAVE1T）
python -m maga7.tools.run_qqq_only_min_rule_scoreboard \
  --out /mnt/s990/data/maga7/results/qqq_only_min_rule_accept_v1
```

---

## 5. 非目标（v0）

- 不并入 Mag7 peer3，不改 `wash_drop_min` 等 Mag7 阈值  
- 不追求接近 Mag7 研究基线的绝对收益倍数  
- 不做多因子选股、不做 peer 广度 Halt  
- 不把 Hunt / hold_extend 放进 v0  
- 不在本规格内训练新模型门

---

## 6. 冻结句

> **QQQ 最小可交易规则 = 日态臂 + 因果 A/B 各至多一笔 + 成交后可撤销波段确认 + 毒路径/trail/时钟安全阀。**  
> **双窗尾损与语义审计不过，就不上线；强窗 retain&lt;0.5 且尾损无改善，规格失败。**

---

## 附录 A — 第一次验收跑数（2026-07-22）

工具：`python -m maga7.tools.run_qqq_only_min_rule_scoreboard`  
产物：`/mnt/s990/data/maga7/results/qqq_only_min_rule_accept_v1/`

| 项 | 值 |
|----|----|
| 强窗 | May–Jun（dte0 止于 06-30；无 Jul）· 日历 41 日 · 成交日 11 |
| 弱窗 | Feb–Apr · 日历 55 日 · 成交日 13 |
| 数据缺口 | Jul 无 dte0 ATM → 强窗截断；D1 仍 PASS（双窗≥10 日） |
| 毒性探针日 | 2026-02-20 / 02-02 / 03-31（CTRL0 最差日自动） |
| 结论 | **`FAIL`** |

### Scoreboard（复利日收益）

| 窗 | 变体 | n | total_ret | ≤−25% | worst | WAVE_ABORT | TRADE_TOX |
|----|------|--:|----------:|------:|------:|-----------:|----------:|
| feb_apr | CTRL0 | 17 | **−76.7%** | 4 | −55% | 0 | 0 |
| feb_apr | WAVE1 | 17 | −67.2% | 3 | −45% | 7 | 0 |
| feb_apr | WAVE1T | 17 | −64.1% | 1 | −45% | 4 | 4 |
| may_jun | CTRL0 | 15 | **+41.2%** | 0 | −21% | 0 | 0 |
| may_jun | WAVE1 | 15 | +25.0% (retain **0.89**) | 0 | −25% | 2 | 0 |
| may_jun | WAVE1T | 15 | −2.3% (retain 0.69) | 0 | −25% | 2 | 2 |

### 门（WAVE1）

| 门 | 结果 | 读数 |
|----|------|------|
| A3 毒日 | FAIL | 02-20/03-31 变浅；**02-02 未变**（−45% 原样） |
| B1 强尾 | FAIL | ≤−25% 仍为 0，但 worst −21%→**−25%**（WAVE 略加深） |
| B2 弱尾 | PASS | 4→3，worst −55%→−45% |
| C1 retain≥0.70 | PASS | 0.89 |
| C3 弱窗≥0 | **FAIL** | CTRL0 已 −77%；WAVE 救不了入场 |
| D2–D4 | PENDING | 工程未做 |

### 读数（下一刀只改一个语义）

1. **入场规则本身在弱窗不成立**：CTRL0 Feb–Apr −77%，不是确认层能补的。  
2. WAVE1 在强窗 **retain 合格**，弱窗尾损略好，但毒日未全过、强窗 worst 略差。  
3. WAVE1T（tox20）**否决于强窗**（收益打穿、retain&lt;0.70）。  
4. **下一刀候选（择一）：**  
   - 提高日态臂：`quiet` / `|fo|` 更高才开 A；或弱窗只做 A、禁 B；  
   - 或先只验收 **A_am_continuation**（拿掉 fade），再叠 WAVE1。  
   禁止同时开大网格。
