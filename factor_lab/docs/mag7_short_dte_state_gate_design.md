# MAG7 Short-DTE State Gate 设计文档

> 状态：战略转向已确认（2026-07-10，对齐 `maga7.txt`）  
> 范围：QQQ/SPY 作锚点 + MAG7 并行扫描 × **0DTE / 1DTE / 2DTE**  
> 产品形态：**Nasdaq / MAG7 Intraday Option Scanner**（不是 QQQ-only 策略）  
> 目标：先按 **标的 × 日历日 × 到期日 / DTE** 分类数据；框架共用、参数独立；跨标的 TopK，而不是把 QQQ 0DTE 规则硬搬到个股。

## 已确认决策

| 项 | 决定 |
|---|---|
| 产品形态 | **跨标的 Opportunity Scanner**（并行扫描 + TopK），不是单标的 QQQ 策略 |
| QQQ / SPY 角色 | **市场状态锚点**（regime / beta 确认），主盈利不再押在 QQQ 上 |
| 主盈利标的 | 先 **NVDA + TSLA**（右尾弹性）；META 后置；AAPL/MSFT/AMZN/GOOGL 作稳定补充 |
| MAG7 Mon/Wed 起点 | 实测 **2026-02-02 (Mon) / 2026-02-04 (Wed)**；短 DTE 主窗从 **2026-02-02** 起算 |
| 并行原则 | **加大并行扫描，禁止无脑并行持仓**；日级最多 Top1–Top3 |
| 相关暴露 | 必须有 **Correlation / Beta Penalty**（同向 Nasdaq beta 不可叠满） |
| 规则迁移 | **框架共用，参数/阈值/执行独立**；禁止把 QQQ 失败规则直接复制到 MAG7 |
| DTE 定义 | **trading DTE**（按交易日计数，不用 calendar DTE） |
| DTE / Cross-symbol Router | **各 (symbol,dte) 门闸齐了再做跨标的排序**；此前不做全池混训 |
| 账户仓位 | 单笔按账户 **25%** 计权益（`position_frac=0.25`）；跨标的时总暴露另受 beta 上限约束 |
| QQQ-only 后续 | Jul OOS / 选时 / No-Trade **降级为 shadow**；QQQ 深度只做 **Regime / Tradeable / 选时质量**，不做端到端下单 |
| QQQ 深度数据 | Regime 用 **正股 1s（约 2022-03→今，~3 年）**；Tradeable/入场仍依赖 option micro（当前仅 2026） |
| Alpha 假设 | **预测机构主动流的起点（onset）**，禁止简单跟随已展开的流（跟流=高位接盘）；门控/选时条件化在 onset 前兆，不在 persistence 尾段 |

---

## 0. 一句话结论

> QQQ 应从「主交易标的」降级为「市场状态锚点」；真正收益通过 MAG7 并行扫描、TopK 排序、0/1/2DTE 动态表达来捕捉。

MAG7 从今年起常见 **周一 / 周三 / 周五** 到期后，每个交易日都可能同时存在：

```text
0DTE = 当天到期
1DTE = 下一近月到期
2DTE = 再下一近月到期
```

> **时间窗（重要）**：MAG7 **周一 / 周三** 到期经 Polygon 盘点确认：NVDA/TSLA **首个 Monday expiry = 2026-02-02，首个 Wednesday expiry = 2026-02-04**（与「约 2026-02 起」一致）；此前个股短期权仍以 **周五 weekly** 为主。  
> 因此 NVDA/TSLA 的 `dte∈{0,1,2}` 主研究窗应从 **2026-02-02** 起算；2026-01 仅作周五周权对照。短窗内首次「碰到 Mon/Wed expiry」的交易日可早到 **2026-01-29**（看向 2/2），但真正 Mon/Wed 到期日从 2 月才开始。

系统不能再按「固定周五 0DTE」或「永远只做 QQQ 0DTE」运行，而应变成：

```text
QQQ/SPY Regime（锚点）
        ↓
MAG7 相对强弱 / 独立驱动
        ↓
各 symbol×dte：Edge / RightTail / Liquidity（参数独立）
        ↓
跨标的 FinalScore → Top1–Top3（非全信号下单）
        ↓
Portfolio Risk（beta / 相关惩罚）+ Execution
```

数据层仍先保证：

```text
日历日 → 可用到期日集合 → DTE 标签 → 分桶数据宇宙
       → State Gate（按 symbol×dte 标定）
```
---

## 1. 背景与问题

### 1.1 旧假设（已失效）

| 旧假设 | 现状 |
|---|---|
| 个股主要周五 weekly，周一~周四没有真正 0DTE | MAG7 自约 **2026-02** 起常见 Mon/Wed/Fri 到期，周中也可出现 0/1/2DTE；此前仍以周五周权为主 |
| QQQ 日频 0DTE 足够代表短期权 | 1DTE/2DTE 是不同工具：容错与 IV/路径不同 |
| 一套 curated 规则可跨标的复用 | QQQ 0DTE 的 recovering/lunch 不能直接搬到 NVDA/TSLA |
| 数据按「一个 locked map」即可 | 必须按 **symbol × trade_date × expiry × dte** 分类 |

### 1.2 新约束

1. **同一交易日可并存多个 DTE**，必须先分类，再交易。  
2. **规则按 (symbol, dte) 标定**，不允许混池训练后直接共用阈值。  
3. **DTE 是工具选择，不是标签噪声**：0DTE 偏右尾/急动，1DTE 偏趋势容错，2DTE 偏方向+IV。  
4. **个股流动性显著差于 QQQ**：spread / min_ask / TopN 必须分 profile。  
5. **先扩数据覆盖，再谈更复杂 runner / RightTail 默认化**。

---

## 2. 核心设计：三层分类

所有原始与中间数据，统一打上三维标签：

```text
Symbol  ×  TradeDate  ×  Expiry/DTE
```

### 2.1 Symbol 层（三层角色，对齐 `maga7.txt`）

```text
第一层 · 市场锚点（主要不赚钱，管 regime）
  QQQ, SPY
  → 趋势 / 震荡 / 反转 / 扩波 / pin
  → 个股信号是否有指数确认；beta 行情 vs 个股独立行情

第二层 · 主盈利（优先建数据与门闸）
  NVDA, TSLA（先行）
  META（后置）
  → 日内波动大、期权弹性高、独立驱动多、0DTE 右尾空间大

第三层 · 稳定补充（后置）
  AAPL, MSFT, AMZN, GOOGL/GOOG
  → 更适合 1DTE/2DTE 趋势容错、相对强弱补充、广度确认
  → 不是天天主攻 0DTE
```
### 2.2 TradeDate 层（日历）

每个交易日先算「当天可交易到期日集合」：

```text
available_expiries(symbol, trade_date)
  = 该日存在且满足流动性门槛的到期日列表
```

对 MAG7 Mon/Wed/Fri 到期，典型映射（示意；**仅适用于约 2026-02 及以后**）：

| 交易日星期 | 常见可用 DTE | 说明 |
|---|---|---|
| Mon | 0 / 2 / 4… | 当天若有 Mon expiry → 0DTE；到 Wed → 2DTE |
| Tue | 1 / 3… | 到 Wed → 1DTE；到 Fri → 3DTE（本阶段可先不进主池） |
| Wed | 0 / 2… | 当天 Wed expiry → 0DTE；到 Fri → 2DTE |
| Thu | 1… | 到 Fri → 1DTE |
| Fri | 0 / 3…（若有下周 Mon） | 当天 Fri expiry → 0DTE |

> 注意：上表是**日历结构示意**，实盘必须以当日 option chain / locked map 为准，不能写死 weekday→dte。

### 2.3 DTE 层（研究主池）

本阶段只主攻：

```text
DTE ∈ {0, 1, 2}
```

| DTE | 角色 | 典型持仓时钟 | 适用场景 |
|---|---|---|---|
| 0 | 右尾 / gamma 急动工具 | 短（如 45s）或 state 时钟 | Edge 强、流动性好、预期快速移动 |
| 1 | 趋势容错工具 | 中（可长于 0DTE） | 方向对但路径噪，0DTE 易被震出 |
| 2 | 方向 + IV 工具 | 更长 | 需要更多时间演化 / 事件前后 |

**禁止**：把 0/1/2DTE 混成一个训练集，再用同一套 TopK/Confirm/Hold。

---

## 3. 数据分类规范（必须先做）

### 3.1 规范对象

每一行合约快照 / locked target / micro 日文件，至少具备：

```text
symbol
trade_date          # 交易日（美东日历）
expiry_date         # 到期日
dte                 # trading DTE（已确认；非 calendar DTE）
side                # CALL/PUT
bucket_id / moneyness
contract_symbol
liquidity fields    # spread, bid/ask size, volume 等
```

### 3.2 推荐目录 / 产物分层

```text
L0 Raw
  raw_1s/stocks/{SYM}/
  raw_1s/options/{SYM}/          # 个股 sniper
  raw_1s/dte{N}_options/QQQ/     # QQQ 分 DTE raw（历史遗留可保留）

L1 Locked Map（分类核心）
  locked_targets_map_{symbol}_{dte}.parquet
  或统一大表 + 强制列: symbol, date_str, expiry, dte, bucket_id, contract_symbol

L2 Micro（State Gate 输入）
  microstructure/{symbol}_{dte}/contract_1s/{SYM}/{SYM}_YYYY-MM-DD.parquet

L3 Score Cache
  results/state_gate_{symbol}_{dte}/cache/score_dataset_YYYY-MM.parquet

L4 Rules / Replay
  results/state_gate_{symbol}_{dte}/stability/
  results/state_gate_{symbol}_{dte}/curated_replay/
```

### 3.3 分类清单（先盘点，再下载）

对每个 `(symbol, dte)` 维护一张覆盖表：

| 字段 | 含义 |
|---|---|
| locked_map 是否存在 | 有无目标合约表 |
| raw_1s 覆盖起止 | sniper / 期权 1s |
| stock_1s 覆盖起止 | 正股 state |
| micro 覆盖起止 | State Gate 可跑区间 |
| 缺日列表 | 需要补下载 |
| 数据质量标记 | 有无 trade prints / quote events |

---

## 4. 当前数据现状（截至文档编写时）

### 4.1 QQQ

| 通道 | 状态 | 覆盖 | 备注 |
|---|---|---|---|
| 0DTE locked map | 有 | 2022-03 → 2026-06 | 仅 QQQ |
| 0DTE micro | 有 | **2026-01-02 → 2026-06-30**（123 日） | 已支撑 Apr–Jun curated；**更早月份可扩研究但未做完整 State Gate 标定** |
| 1DTE locked map | 有 | 2022-03 → **2026-03-18** | 仅 QQQ |
| 1DTE raw_1s | 有 | 2022-03 → **2026-03-18** | 与 0DTE Apr–Jun **不对齐** |
| 1DTE micro | 仅 smoke | 2026-01-02 → 01-15（10 日） | 骨架已通，未全量 |
| 2DTE locked / micro | 有 | micro 约 2026-01 → 2026-06 | short_dte micro 为 selected_dte=2 |
| QQQ stock 1s | 有 | 2022-03 → 2026-06 | 足够 |

**QQQ 已完成（研究）**

- 0DTE State Gate curated：confirm + state clock（及可选 RT v1）
- 通用 skeleton / profile / raw→micro 桥
- 1DTE / NVDA smoke

**QQQ 明确未完成（写进计划）**

1. **0DTE 更多月份的正式 State Gate 扩展**  
   - 现有 micro 已含 2026-01~03，但 curated 规则主要在 Apr–Jun 上冻结  
   - 需要：用 2026-01~03 做 **前推验证 / 再标定**，确认规则不是单季过拟合  
2. **1DTE 全量 micro + stability + curated**（至少 2026-01~03）  
3. **2DTE 独立 stability**（不要与 0DTE 混用规则）  
4. **1DTE/2DTE 与 0DTE 对齐到同一 OOS 窗口**（缺 2026-04+ 的 1DTE raw/micro 时先补数据）

### 4.2 NVDA / TSLA

| 通道 | NVDA | TSLA | 备注 |
|---|---|---|---|
| stock 1s | 有（→2026-03-18） | 有 | OK |
| option raw_1s | 有（→2026-03-18） | 有 | sniper 格式；缺完整 trade-flow micro |
| stock locked map | **缺失** | **缺失** | `locked_targets_map_stock_0dte.parquet` 不存在 |
| micro | smoke 仅 NVDA Jan | 无 | 非正式 |
| Mon/Wed/Fri 分 DTE map | **未建** | **未建** | 旧假设仍偏「周五 0DTE」 |

**关键缺口**：个股还没有按 `dte∈{0,1,2}` 分类的 locked map，因此无法正确做「动态应对三种 DTE」。

---

## 5. 动态应对 0/1/2DTE 的目标架构

```text
                    ┌─────────────────────────────┐
                    │  Daily Universe Builder      │
                    │  symbol, trade_date          │
                    │  → expiries → dte tags       │
                    └──────────────┬──────────────┘
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
   Panel 0DTE                 Panel 1DTE                 Panel 2DTE
   (独立 micro)               (独立 micro)               (独立 micro)
         │                         │                         │
         ▼                         ▼                         ▼
   StateGate_0                 StateGate_1               StateGate_2
   rules/confirm/hold          rules/confirm/hold        rules/confirm/hold
   在 (symbol,dte) 上标定       在 (symbol,dte) 上标定     在 (symbol,dte) 上标定
         │                         │                         │
         └─────────────┬───────────┴─────────────┘
                       ▼
              DTE Router (可选)
              选当天用哪个工具 /
              或是否多工具并行但分仓
```

### 5.1 每日动态流程（个股与 QQQ 共用）

```text
1. 读取当日 chain / locked candidates
2. 计算每个合约 dte
3. 过滤到 dte ∈ {0,1,2} 且流动性达标
4. 按 dte 分桶生成当日 panel
5. 对每个非空桶运行对应 StateGate_{dte}
6. Router:
   - 若只有一个桶有信号 → 用该桶
   - 若多桶有信号 → 用 DTE_Score 选主工具，或主仓+卫星仓
7. 执行与归因按 (symbol,dte,state,rule) 记账
```

### 5.2 DTE_Score（路由，不替代 State Gate）

沿用右尾文档思路，但第一版保持可解释：

```text
DTE_Score =
  ExpectedExecEdge
  / ExpectedPathNoise
  * LiquidityScore
  * StateConfidence
  * ThetaSafety
  * SpreadEfficiency
  * RightTailScore   # 可选，后置
```

经验规则（可先写死，后用数据校准）：

```text
优先 0DTE:
  高流动性 + 高 vol expansion + 短持仓 state 有效

优先 1DTE:
  方向 state 强，但 0DTE MFE 常被噪声打回 / spread 对 0DTE 不友好

优先 2DTE:
  需要更长演化，或 0/1 流动性不足但 2DTE 可交易
```

### 5.3 NVDA/TSLA 相对 QQQ 的差异化参数

| 维度 | QQQ | NVDA/TSLA |
|---|---|---|
| spread 门槛 | 更严（~5–6%） | 更松但仍截断（~10–12%） |
| min_ask | ~0.20 | ~0.25+ |
| TopN universe | 可更小 | 更依赖流动性排序 |
| 日历 | 几乎每日有短 DTE | 以 Mon/Wed/Fri expiry 为骨架，动态算 dte |
| state 命名 | 可用 underlying_* | **必须用 underlying_***，避免 qqq_* 语义 |
| 规则 | 每 (symbol,dte) 重做 stability | 同左；禁止直接 copy QQQ curated |
| RS 特征 | 可选 | **建议加入 vs QQQ/SPY**（后置增强） |

---

## 6. State Gate 在多 DTE 下的标定原则

### 6.1 什么可以共享

- 代码骨架：loader / score / state attach / TopK / path exec / confirm 框架  
- 状态定义模板：trend / vol / lunch / recovering（逻辑可共享）  
- 研究方法：Rule×State×Month stability → confirm → hold clock

### 6.2 什么必须分开

| 项目 | 分开粒度 |
|---|---|
| locked map / micro | `symbol × dte` |
| scorer 拟合 | `symbol × dte`（至少 dte 分开） |
| curated rules | `symbol × dte` |
| confirm 阈值 | `symbol × dte` |
| hold / state clock | `symbol × dte` |
| RT v1 模型 | 后置；默认先不做跨 dte 共用 |

### 6.3 研究窗口建议

```text
阶段 A（对齐现有能力）:
  QQQ 0DTE: 2026-01~06（先用已有 micro 扩月份验证）
  QQQ 1DTE: 2026-01~03（先全量，再补 04+）
  QQQ 2DTE: 2026-01~06（已有 micro，先 stability）

阶段 B（个股）:
  NVDA/TSLA: 先建 dte 分类 locked map
  研究窗优先 2026-01~03（与现有 raw 对齐）
  再补 2026-04+ 数据后做 OOS
```

---

## 7. 执行步骤（按依赖排序）

### Phase 0 — 分类与盘点（不改策略逻辑）

1. 定义统一 schema：`symbol, trade_date, expiry, dte, side, bucket, contract`  
2. 产出覆盖矩阵：QQQ/NVDA/TSLA × {0,1,2} × {locked, raw, stock, micro}  
3. 明确 MAG7 到期日历源（exchange calendar / chain 扫描）  
4. 冻结本阶段主池：仅 dte∈{0,1,2}

**交付物**：覆盖矩阵 CSV/表 + 本设计文档确认

### Phase 1 — QQQ 数据补齐与扩月份

1. **QQQ 0DTE 扩月份验证**  
   - 用已有 2026-01~03 micro，对当前 curated / state clock 做前推或再稳定性检验  
   - 目标：回答「Apr–Jun 规则是否只是单季有效」  
2. **QQQ 1DTE**  
   - raw→micro 或 Polygon micro 全量 2026-01~03  
   - 跑 stability → 生成 1DTE curated（禁止搬 0DTE 规则）  
3. **QQQ 2DTE**  
   - 基于现有 short_dte micro 跑独立 stability  
4. **对齐缺口**  
   - 若要与 0DTE 同用 Apr–Jun OOS：补 1DTE 2026-04+ locked/raw/micro

**交付物**：`state_gate_qqq_{0,1,2}dte` 各自的 stability / curated 结果

### Phase 2 — 个股 DTE 分类基础设施

1. 扫描 NVDA/TSLA 每日可用 expiry，生成：  
   `locked_targets_map_{sym}_dte{0,1,2}.parquet` 或统一表  
2. 按 dte 分桶下载/转换 micro  
3. 正股 1s 对齐（已有到 2026-03 可先用）  
4. profile：`nvda_0dte/1dte/2dte`, `tsla_0dte/1dte/2dte`  
5. 去掉「仅周五」硬编码，改为 **由当日 dte 可用性驱动**

**交付物**：个股分类 locked map + 至少一个月 micro smoke/全量

### Phase 3 — 个股 State Gate 标定

1. 每个 `(NVDA/TSLA, dte)` 独立跑 Rule×State×Month  
2. 选跨月稳定规则（宁缺毋滥）  
3. confirm / hold clock 各自拟合  
4. 与 QQQ 同 dte 做对照，但不当作同一规则

**交付物**：每标的每 DTE 的 curated 规则 JSON + replay 摘要

### Phase 4 — DTE Router（动态选择）

1. 定义可执行的 DTE_Score v0（规则版）  
2. 回测三种模式：  
   - only-0 / only-1 / only-2  
   - router 单选  
   - router 主仓+卫星（后置）  
3. 归因：哪些交易日 0 优于 1/2，与 state / 流动性关系

**交付物**：router 回测报告；默认策略建议

### Phase 5 — 增强（后置，不阻塞主线）

1. Polygon 级 micro（补 trade-flow，激活 flow states）  
2. RightTailScore 按 `(symbol,dte)` 监督训练  
3. vs QQQ 相对强度特征  
4. 扩展其余 MAG7

---

## 8. 明确不做 / 暂缓

1. 不把 QQQ 0DTE curated 规则直接套到 1DTE/2DTE 或 NVDA/TSLA  
2. 不把 0/1/2DTE 混训成一个模型再事后按 dte 切片解释  
3. 不在数据分类完成前上复杂分仓 runner 作为默认  
4. 不在缺 locked map 时用「全市场 raw 盲扫」充当正式宇宙  
5. 暂不以 3DTE+ 为主池（可留观察桶）

---

## 9. 成功标准

### 数据层

- [ ] QQQ/NVDA/TSLA 均有 `dte∈{0,1,2}` 的分类覆盖表  
- [ ] 每个非空 `(symbol,dte)` 有 locked map 与可加载 micro  
- [ ] 缺日与质量问题可追溯

### 策略层

- [ ] 每个 `(symbol,dte)` 有独立 stability 排名（即使结论是「无可交易规则」）  
- [ ] QQQ 0DTE 扩月份后，规则稳定性有明确结论  
- [ ] NVDA/TSLA 能按交易日动态暴露 0/1/2，而不是写死周五  

### 路由层

- [ ] Router 在多 DTE 并存日有可解释选择  
- [ ] 账户归因可拆到 symbol × dte × state

---

## 10. 冻结后的执行优先级

战略已按 `maga7.txt` 转向。**QQQ-only 深挖（选时 / No-Trade / Exit）降级**；主线改为「锚点 + 个股数据 + 跨标的 TopK」。

```text
P0  覆盖矩阵（已完成）
P1  QQQ 0DTE 扩月份 / Jul OOS / 选时（已完成；结论：不升默认，QQQ 改锚点）
P2★ NVDA/TSLA：trading-DTE 分类 locked map（动态 0/1/2）← 当前主阻塞
P3★ NVDA/TSLA micro + 正股 1s 对齐；按 (symbol,dte) 独立 State Gate
P4★ QQQ Deep Anchor：正股 3 年 Regime + micro 窗 Tradeable（因果滚动；非端到端）
P4b★ 机构指纹层：预测 onset（accel↑∩非高persist），禁止跟流接盘
P5  跨标的 FinalScore + TopK（日级 Top1–3）+ Correlation Penalty
P6  DTE Selector（0/1/2 动态表达）与 Portfolio Risk
P7  后置：META / 稳定补充标的；QQQ micro 向 2024–2025 回补（服务 Tradeable 扩窗）
```

### 立即下一步（三线）

- **P3（MAG7）**：dte0 TopK 已有结论；继续 dte1/2 stability + TSLA path 回放。  
- **P4★（QQQ Deep Anchor）**：`block_pred_1` shadow；1DTE ladder micro 用户侧续下。  
- **P4b★（机构指纹）**：先在 0DTE curated 亏单上归因三指纹，再改 Tradeable 标签/路由条件。

1. Micro MAG7：`/mnt/s990/data/microstructure/mag7_short_dte_api_ladder/`  
2. 工具：`qqq_btc/tools/train_0dte_qqq_deep_anchor.py`；指纹：`qqq_btc/tools/analyze_0dte_institutional_fingerprints.py`  
3. Stability：`stock_options/tools/run_mag7_short_dte_state_gate_stability.py`
### P0/P1 执行结果（2026-07-10）

**仓位**：账户汇总默认改为单笔 `position_frac=0.25`。

**P0 覆盖矩阵**：`qqq_btc/results/mag7_short_dte_coverage/`

| 标的 | dte0/1/2 gate_ready | 主要阻塞 |
|---|---|---|
| QQQ | 是 / 是（1DTE micro 仅 smoke）/ 是 | 1DTE micro 未全量 |
| NVDA/TSLA | 否 | **缺按 trading-DTE 分类的 locked map**；现有 sniper raw 仍偏周五周权，Jan 抽样未见同日 0DTE |

**P1 QQQ 0DTE 扩月份**（冻结规则：Apr+May confirm + rec45/lunch180，回放到 Jan–Jun）：

| 区间 | 单笔均收益 | 账户收益@25% | 回撤@25% | 结论 |
|---|---:|---:|---:|---|
| Jan–Mar | `-1.35%` | `-31.9%` | `-38.6%` | 反向时间可迁移失败 |
| Apr–Jun | `+8.15%` | `+293%` | `-10.1%` | 仍强 |
| 全年 H1 | `+2.83%` | `+168%` | `-38.6%` | 被 Q1 拖累 |

**含义**：当前 curated 不能当作 H1 默认；扩月份后必须先 **按滚动窗口重标定 / 加 regime 门控**，再谈 1DTE 与个股。

**已验证（见 §13.8）**：低内存因果滚动重标定相对 frozen curated 合并账户由 `-34.7%` → `+35.2%`，但仍有约 `-31%` 回撤，**不升默认**；下一步压回撤（No-Trade）+ 新增 OOS 月。

**下一优先（已转向 `maga7.txt`）**：

1. **主线**：NVDA/TSLA trading-DTE locked map + micro（P2★）  
2. **锚点**：把 QQQ curated / Jul 失败经验沉淀为 Regime 输入，而不是继续 QQQ-only 调参  
3. **暂缓**：QQQ No-Trade / 选时 / Exit 升默认；跨标的 Router 等个股门闸齐后再做
---

## 11. 与现有文档关系

| 文档 | 关系 |
|---|---|
| `maga7.txt` | **产品战略源**：QQQ 锚点 + MAG7 并行扫描 + TopK + beta 惩罚；本文件负责落地顺序 |
| `zero_dte_right_tail_capture_strategy.md` | 定义 0/1/2 工具角色与 RightTail/runner；本文件负责落地前的**数据分类与标定顺序** |
| `short_option_alpha_engine_profit_improvement.md` | 利润改进方向；本文件约束「先分类、再路由、再右尾」 |
| `qqq_btc/CONFIG/state_gate_profiles.json` | 现有 profile 雏形；后续应按本文件扩成 `symbol×dte` 完整矩阵 |
| `stock_options/` | 个股路径骨架（原 weekly-DTE）；需升级为 **short-DTE 0/1/2 动态**，与本文件对齐 |
---

## 12. 设计选择（已关闭）

1. **DTE 定义**：trading DTE ✅  
2. **个股主池**：NVDA + TSLA ✅  
3. **产品形态**：跨标的 Scanner（并行扫描 + TopK）✅（`maga7.txt`）  
4. **QQQ 角色**：市场锚点，非主盈利 ✅  
5. **Router 时机**：各 (symbol,dte) 门闸齐了再做跨标的排序 ✅  
6. **仓位**：单笔账户仓位 25% + 跨标的 beta 上限 ✅  
7. **QQQ-only 深挖**：Jul OOS / 选时后降级，不升默认 ✅  

下一步从 **P2 NVDA/TSLA trading-DTE locked map** 开始。
---

## 13. QQQ 0DTE Regime / Rule Selector 执行结果（2026-07-10）

### 13.1 实验边界

- Apr–Jun 发现的 curated 规则与 Apr 拟合的 base scorer 回放到 Jan–Mar，只能叫
  **反向时间可迁移诊断**，不能叫 forward OOS。
- Meta 层全部按月份做 expanding-window；不随机拆交易行。
- 标签使用已扣 ask→bid 与 commission 的 `path_exec_ret > 0`。
- 账户指标继续按单笔 `position_frac=0.25`。

### 13.2 失败归因与漂移

输出：`qqq_btc/results/0dte_state_gate_regime_drift_h1/`

```text
确认后交易：
  Jan–Mar: 93 笔，均笔 -1.35%，账户 -31.9%，从初始资金计最大回撤 -39.0%
  Apr–Jun: 73 笔，均笔 +8.15%，账户 +293.1%，从初始资金计最大回撤 -10.5%

26 个入场特征中：
  PSI >= 0.10: 12 个
  PSI >= 0.25: 2 个
  Spearman 方向翻转: 13 个
```

主要分布漂移集中在：

```text
vol_score
state_activity_q
stock_rv_60s
state_spread_q
spread_pct
```

主要收益关系翻转集中在：

```text
flow_toxicity_5s
state_spread_q
flow_imbalance_5s
tod_frac
spread_pct
```

结论：问题不仅是 State 命中频率变化，还包含明显的 **concept drift**；固定 confirm
阈值和固定特征方向不能直接跨季度使用。

### 13.3 未确认候选池

输出：

```text
qqq_btc/results/0dte_state_gate_curated_noconfirm_statehold_jan_jun/
qqq_btc/results/0dte_state_gate_h1_cache/
```

候选池共 230 笔、116 个交易日；月度候选为：

```text
Jan 40 / Feb 38 / Mar 43 / Apr 28 / May 40 / Jun 41
```

同时修复了 `run_0dte_state_gate_curated.py` 在使用新 `--cache-dir` 时不创建目录、
导致首月计算完成后保存失败的问题。

### 13.4 轻量 Rule Selector

输出：`qqq_btc/results/0dte_state_gate_rule_selector_h1/`

全可用 expanding-window 影子结果：

| 策略 | 交易数 | 均笔 | 账户收益@25% | 从初始资金计回撤 |
|---|---:|---:|---:|---:|
| 未确认候选全部执行 | 190 | `+2.81%` | `+205.5%` | `-37.0%` |
| 滚动经济阈值 Selector | 69 | `+0.35%` | `+0.5%` | `-33.4%` |
| 固定 `p>=0.60`（敏感性，不是预注册主策略） | 53 | `+4.70%` | `+73.1%` | `-13.4%` |

滚动主策略显著落后，不可升为默认。固定阈值显示“过滤可降回撤”的可能性，但存在
阈值后验选择和规则发现泄漏，只能作为待新增月份验证的候选。

### 13.5 因果慢 Regime

实现：`qqq_btc/tools/fit_0dte_causal_regime.py`

方法：

```text
过去月份拟合 diagonal GMM emission
→ 按低/中/高风险排序 component
→ 用训练期日内转移估计 transition
→ 测试日只做 forward Bayesian filtering
```

输出：`qqq_btc/results/0dte_causal_regime_h1/`

该模型平均 posterior confidence 达 `97.9%`，但这不是优点：在只有 230 个候选事件、
且只在候选时点更新的条件下，Gaussian emission 明显过度自信，状态持续性估计也不稳定。

### 13.6 Regime + Selector 同窗口消融

共同窗口为 Mar–Jun：

| 策略 | 交易数 | 均笔 | 账户收益@25% | 从初始资金计回撤 |
|---|---:|---:|---:|---:|
| 全候选基线 | 152 | `+2.36%` | `+103.8%` | `-36.6%` |
| Selector，不含 Regime | 54 | `+6.65%` | `+127.4%` | `-13.7%` |
| Selector，加入 Regime | 66 | `+4.71%` | `+98.9%` | `-17.3%` |

结论：

1. 去掉最早冷启动月后，轻量 Selector 有改善收益质量和回撤的迹象。
2. 当前 GMM/HMM-like Regime 特征 **降低** 了同窗口收益与回撤表现，不采用。
3. 结果仍受 curated 规则与 base scorer 的后见发现影响，不能称为生产 OOS。

### 13.7 深度模型与 Execution Model 门槛判断

**TCN / Transformer：暂不实现。**

原因：

```text
只有 6 个月
只有 116 个独立交易日
只有 230 个候选事件
```

秒级行数不能替代独立状态段数量；当前训练深度模型大概率只会记住月份与报价尺度。

**Fill / DeepLOB Execution Model：暂不实现。**

当前候选事件虽然都有 top-of-book quote，但：

```text
仅 55.7% 的候选时点 trade_count / trade_volume > 0
没有真实挂单排队位置
没有订单提交、成交、撤单与 fill-time 标签
没有完整多档 LOB
```

因此现在训练 fill probability 会制造不可验证的伪标签。现阶段继续使用 ask-entry /
bid-exit / commission 的严格执行回放。

### 13.8 因果滚动重标定（低内存 lean）

首版全量拼接 bar 面板在约 50GB+ 内存被杀；改为 lean 实现：

```text
实现：qqq_btc/tools/run_0dte_causal_rolling_recal_lean.py
输出：qqq_btc/results/0dte_causal_rolling_recal_lean_h1/
```

方法约束：

```text
expanding-window：用过去月拟合 scorer / 选规则，只在下一月回放
scorer 按月采样拟合（每折最多约 35 万行），不拼接全量 bar
规则评估走交易级 path PnL，不物化全候选 bar 矩阵
聚焦状态集 + tree_edge_score；训练期默认固定 45s hold
```

Feb–Jun 月度（账户@25%）：

| 测试月 | 训练窗 | rolling strict 均笔 / 账户 | frozen curated 均笔 / 账户 |
|---|---|---:|---:|
| Feb | Jan | `+3.64%` / `+39.5%` | `-1.31%` / `-12.2%` |
| Mar | Jan–Feb | `-0.76%` / `-10.6%` | `-1.53%` / `-16.5%` |
| Apr | Jan–Mar | `+0.17%` / `+0.1%` | `-3.08%` / `-20.1%` |
| May | Jan–Apr | `-0.26%` / `-4.1%` | `+0.41%` / `+3.2%` |
| Jun | Jan–May | `+1.53%` / `+13.0%` | `+0.85%` / `+8.0%` |

合并 Feb–Jun：

| 策略 | 交易数 | 均笔 | 账户收益@25% | 回撤@25% |
|---|---:|---:|---:|---:|
| rolling strict/loose | 192 | `+0.85%` | `+35.2%` | `-31.3%` |
| frozen curated | 192 | `-0.79%` | `-34.7%` | `-44.3%` |

结论：

1. **相对冻结 curated，因果滚动重标定明显更好**（合并账户由负转正）。
2. **仍不能升为生产默认**：回撤仍大（约 -31%）；Mar/May 单月为负；strict≈loose（池子几乎相同）。
3. 选中状态从 curated 的 `is_qqq_recovering` / lunch 组合，滚动到 `vol_expansion`、`put/call_trend`、`opening`、`breaking_down` 等，说明冻结状态集本身在漂移。
4. lean 局限（采样拟合、聚焦状态、固定 hold）可能低估完整搜索上限，但已证明「按月重标定」方向正确。

### 13.9 当前决策

```text
保留：
  漂移诊断
  未确认候选池
  轻量 Selector 研究管线
  因果滚动重标定 lean 管线（相对 frozen curated 的对照基线）
  固定 p>=0.60 作为新增月份 shadow candidate

不升默认：
  冻结 curated（全年）
  滚动经济阈值 Selector
  当前 GMM/HMM-like Regime
  当前 lean 滚动重标定（改善明显但回撤/稳定性不够）
  TCN / Transformer
  Fill / DeepLOB model

下一优先：
  1. No-Trade：day_risk_halt + regime/相似度（挡 Jul 坏日，保护 Apr–Jun）
  2. Exit：改前瞻标签（未来是否吐盈），再训 Exit Model；暂不推简单 TP
  3. 组合 shadow：No-Trade ∧ clock/ExitModel
  4. 并行：NVDA/TSLA trading-DTE locked map；补更多 OOS 月

重新开启深度/执行模型条件：
  至少新增 3–6 个完全未参与规则发现的月份
  或汇入 QQQ/NVDA/TSLA × 0/1/2DTE 后获得足够独立状态段
  Execution 模型另需真实 order/fill 或完整 LOB 标签
```

### 13.10 Forward OOS：冻结 Apr–Jun 规则 × 2026-07 第一周

决策调整：不再硬拟合 Jan–Mar（含少见冲击市）；以冻结 curated 做真正 forward 验证。

```text
窗口：2026-07-01/02/06/07/08/09（7/3 休市；7/10 当时尚未完整）
规则：冻结 recovering + lunch，confirm 阈值冻结自 Apr+May，
      state_hold rec45/lunch180，scorer fit 仍为 Apr 13–30
产物：qqq_btc/results/0dte_state_gate_curated_confirm_statehold_jul2026_w1_pos25/
```

| 指标 | 数值 |
|---|---:|
| 交易数 | 12 笔 / 6 日（每日 2 笔，无空仓日） |
| 单笔均收益 | **-6.65%** |
| 胜率 | 33.3% |
| 账户收益@25% | **-18.7%** |
| 最大回撤@25% | **-17.0%** |

日度（账户@25%）：

| 日期 | 笔 | 日均笔收益 | 日账户@25% | 各笔 |
|---|---:|---:|---:|---|
| 07-01 | 2 | -1.7% | -0.9% | +6.3% / -9.7% |
| 07-02 | 2 | +10.4% | +5.3% | +15.2% / +5.7% |
| 07-06 | 2 | -14.7% | -7.2% | -12.8% / -16.6% |
| 07-07 | 2 | -18.0% | -8.9% | -8.1% / -28.0% |
| 07-08 | 2 | -17.4% | -8.5% | -15.4% / -19.4% |
| 07-09 | 2 | +1.5% | +0.8% | +3.4% / -0.3% |

结论：

1. **这是真正的 forward OOS**；冻结 4–6 配方在 7 月初 **未通过**。
2. 7/6–7/8 连续三天双杀，说明问题不是单笔噪声，而是状态/边在新月份失效。
3. 同窗口 fixed `CALL+trend_down` 基线均笔 `+2.9%`、账户 `+4.3%`，进一步说明 curated 状态集本身可能已不适配。
4. **不升默认、不直接实盘**；下一步优先 No-Trade / 可交易性门控，而不是继续扩规则搜索。

### 13.11 7 月第一周失败归因

产物：`qqq_btc/results/0dte_state_gate_july_w1_failure_attribution/`

12 笔中 4 笔盈利、8 笔亏损，亏损并非单一原因：

| 桶 | 笔数 | 均笔 | 含义 |
|---|---:|---:|---|
| F_winner | 4 | +7.6% | 信号有时仍有效（集中 7/1–2、7/9 recovering） |
| A_direction_wrong | 3 | -20.0% | 正股逆向，State/Rule 失效（多为 lunch CALL） |
| C_mfe_but_exit_fail | 3 | -8.9% | 持仓内曾有 +3%~11% MFE，固定 clock 退出亏掉 |
| B_direction_ok_option_dead | 2 | -11.7% | 正股略顺 PUT，但期权不涨（偏执行/弹性） |

含义：

1. **不能整组删规则**：有 winner，也有「方向对但退出失败」的 C 类。
2. **A 与 C 各占亏损约 3/8**：需要并行处理 —— No-Trade 挡 A 类日，Exit/trailing 救 C 类。
3. **7/6–7/8 是 A/B 主导的坏区段**；若当时 No-Trade，整体 OOS 会完全不同。
4. 下一步：用 A/E 类构造 `tradeable=0`，用 F/C（有 MFE）构造可交易正例，训练轻量 No-Trade；同时对 C 类试 trailing / 提前止盈。

### 13.12 No-Trade / Tradeable 层（已落地 v1）

实现：`qqq_btc/tools/train_0dte_no_trade_gate.py`  
产物：`qqq_btc/results/0dte_state_gate_no_trade_gate_h1/`

架构位置：

```text
Curated 触发 → Confirm → No-Trade Gate → path 执行
                 ├─ tradeable logit（入场可交易概率）
                 ├─ day_risk_halt（严重亏损日后跳过）
                 └─ setup_veto（研究用；默认关闭）
```

协议：Apr+May（noconfirm）拟合 tradeable；Jun 选阈值；Jul confirm 做 forward OOS。

| 策略 | Apr–Jun 账户@25% | Jul 账户@25% | 说明 |
|---|---:|---:|---|
| 无门控基线 | **+293%** | **-18.7%** | 冻结 curated |
| tradeable logit | +293%（thr→0，等于无门控） | -18.7% | **未学到可迁移边界** |
| day_risk_halt（实盘一致） | +145% | **-11.4%** | Jul 有改善，但误伤 Apr–Jun winner |
| 静态 veto lunch CALL | +134% | **-1.7%** | Jul 最好，但严重破坏 Apr–Jun alpha |

结论：

1. **No-Trade 这一层方向正确，必须保留。**
2. **当前 logit 不够**：Jun 经济验证选出 `threshold=0`，对 Jul 无过滤力。
3. **day_risk_halt 是第一版可用控制**：7/6 严重日后跳过 7/7，再交易 7/8；Jul 回撤从 -23%→-15%，账户 -19%→-11%。
4. **不能用静态否决 lunch CALL**：它在 Jul 有效，是因为 lunch 边已失效；在 Apr–Jun 它是主要盈利来源之一。这再次证明 Alpha 规则必须 **regime-conditional**，不能永久 veto。
5. **默认姿态**：curated 仍不升实盘；No-Trade 以 `day_risk_halt` 做 shadow 默认，logit/veto 继续研究；下一迭代要做「像 Apr–Jun 才允许 lunch」的相似度/regime 条件，而不是删规则。

### 13.13 C 类 Exit / Trailing 消融

实现：`qqq_btc/tools/analyze_0dte_c_class_exit_trailing.py`  
产物：`qqq_btc/results/0dte_state_gate_c_class_exit_trailing/`

**C 类确认存在：**

| 集合 | C 笔数 | clock 均笔 | oracle MFE 均笔 |
|---|---:|---:|---:|
| Jul OOS | 3 | -8.9% | **+7.5%** |
| Apr–Jun | 4 | -4.1% | **+15.2%** |

说明入场后曾有可兑现浮盈，固定 state-clock 退出是真实失败模式。

**但简单提前止盈/trailing 不能升默认：**

| 策略 | Apr–Jun 账户@25% | Jul 账户@25% | Jul C 均笔 |
|---|---:|---:|---:|
| state clock（当前） | **+293%** | -18.7% | -8.9% |
| 全局 `tp_lock5` | +46% | -12.2% | +4.3% |
| lunch:`trail5_50` / recovering:clock | +124% | -11.9% | **+1.6%** |
| lunch:`tp_lock5` / recovering:clock | +115% | -10.0% | +4.3% |
| lunch:oracle_mfe / recovering:clock | +700%（上界） | +4.3% | +7.5% |

原因：Apr–Jun 的 lunch winner 平均 clock 收益约 **+15%**，`trail5`/`tp_lock5` 会把右尾砍到约 **+6%**。救 C 类的同一把刀，也砍掉主利润引擎。

**决策：**

```text
不升默认：全局/lunch 提前止盈、简单 trailing
保留：state clock 作为当前退出默认
Shadow 候选：lunch trail5_50（仅诊断，不生产）
下一迭代：特征化 Exit Model
  - 输入：edge decay、state 结束、spread 恶化、已实现 MFE/MAE
  - 目标：在保住右尾的前提下减少 C 类吐盈
  - 验证：Apr–Jun 账户不得显著劣于 clock；Jul C 转正
```

### 13.14 特征化 Exit Model（v1）

实现：`qqq_btc/tools/train_0dte_exit_model.py`  
产物：`qqq_btc/results/0dte_state_gate_exit_model_h1/`

方法：

```text
路径秒级样本（每 5s）
特征：未实现盈亏、MFE/MAE、giveback、edge decay、state_on、spread、持仓进度…
标签：y=1 若 ret[t] >= ret[clock] + 0.5%
训练：Apr+May；验证选阈值：Jun；OOS：Jul
策略：P(exit)>=thr 则提前出，否则拿到 state clock
```

| 策略 | Apr–Jun 账户@25% | Jul 账户@25% | Jul C 均笔 |
|---|---:|---:|---:|
| state clock | **+293%** | -18.7% | -8.9% |
| Exit Model thr=0.60 | **+283%** | -17.7% | -8.5% |
| 规则：peak≥5% 且回撤≥50% 且 state off | +293% | -18.7% | -8.9%（几乎未触发） |

补充：

- train AUC 0.67 / val AUC **0.57**（弱）
- Jul 仅 2/12 笔提前退出；对 C 类几乎没救起来
- Apr–Jun 损伤很小（账户约 -10pp），说明「至少没砍右尾」
- 可解释规则（giveback+state off）在当前参数下基本不触发

**决策：不升默认。** 作为 shadow 保留。标签「现在出优于 clock」噪声大，下一步应改成前瞻标签（未来 Δt 是否继续吐盈），或与 No-Trade 组合，而不是继续调简单 trailing。

### 13.15 入场时机消融（Entry Timing）

实现：`qqq_btc/tools/analyze_0dte_entry_timing_ablation.py`  
产物：`qqq_btc/results/0dte_state_gate_entry_timing_ablation/`

假设：交易次数本就很少，瓶颈是 state 内**何时开火**（追高/偏晚），不是再加厚 No-Trade。

方法：冻结 curated 规则 + confirm + state-hold clock；只改 state 内候选的 timing mask，再走同一 TopK/cooldown。

| 策略 | AJ 笔数 | AJ 账户@25% | Jul 笔数 | Jul 账户@25% | 备注 |
|---|---:|---:|---:|---:|---|
| baseline | 73 | **+293%** | 12 | **-18.7%** | age 中位 ~4s / ~3.5s |
| fresh_le_30 | 72 | +308% | 11 | -15.2% | 几乎已是 fresh；Jul 微改善 |
| edge_rising10 | 72 | +250% | 11 | -13.8% | Jul 最好的「不砍太多 AJ」候选 |
| onset_10_90 | 58 | +145% | 8 | -12.0% | 伤 AJ 过大 |
| delay / mid_30_180 | 24 | +14% | 4 | -3.2% | 靠少做把 Jul 亏缩小，不可取 |
| edge_not_extended | 15 | +28% | 1 | -2.8% | 几乎停机 |

关键发现：

1. **Baseline 已经很早入场**：AJ/Jul 成交 age 中位约 3.5–4s；`fresh_le_60/120` 与 baseline 完全重合 → 「追高偏晚」假设对当前 TopK **不成立**。
2. 温和选时（`fresh_le_30` / `edge_rising10`）最多把 Jul 从 -18.7% 提到约 -13.8%～-15%，**仍为负、未过 OOS**。
3. 更强 age/onset/not-extended 主要靠砍笔数「改善」Jul，同时严重破坏 Apr–Jun → 与「不能靠少做」一致。
4. 启发式推荐 `edge_rising10`（Jul lift +4.9pp，AJ 损伤约 -43pp），但 **不升默认、不实盘**。

**决策：** 简单 state-age / edge-slope 选时**不是** 7 月失败的主解。结合 `maga7.txt`，**不再把主迭代押在 QQQ-only 触发质量上**；QQQ 降级为锚点，主线改为 MAG7 并行扫描（见 §14）。

### 14. 产品转向：MAG7 Intraday Option Scanner（`maga7.txt`）

#### 14.1 为什么换思路

QQQ-only 路径已证明：

- Apr–Jun 有强 alpha，但跨 regime / Jul forward OOS 失败  
- 加厚 No-Trade → 稀交易系统停机  
- 入场选时 → baseline 已很早，救不了 Jul  

根因更像：**单标的机会频率低 + 规则易过拟合到一段 Nasdaq 状态**，而不是缺一层过滤器。

#### 14.2 正确 / 错误

| 对 | 错 |
|---|---|
| QQQ/SPY + MAG7 **同时扫描** | 所有标的有信号就一起下单 |
| 每天从池子里找最强 1–3 个机会 | QQQ 不行就全 MAG7 一起上 |
| 框架共用、参数独立 | 把 QQQ curated 直接复制到 NVDA/TSLA |
| 同向 beta 行情限制仓位数 | 把相关多头当成「分散」 |

#### 14.3 八层系统（落地顺序）

```text
1. QQQ/SPY Regime Engine          ← 复用现有正股状态 / curated 经验作锚点
2. MAG7 Relative Strength Engine  ← NVDA/TSLA 先行
3. Symbol Edge Engine             ← 每标的独立 Call/Put Edge
4. RightTailScore                 ← 是否值得用 0DTE 博倍增
5. DTE Selector                   ← 0/1/2 动态表达
6. Cross-symbol TopK              ← 只做最强 1–3
7. Portfolio Risk                 ← beta / 相关惩罚
8. Execution / Exit               ← spread、滑点、MFE
```

每分钟目标表（示意）：

```text
Rank  Symbol  DTE   Dir   Mode       Edge  RightTail  Liq  Final
1     NVDA    0     Call  Momentum   0.86  0.91       0.82 0.84
2     TSLA    0     Call  Reversal   0.81  0.76       0.78 0.75
3     QQQ     1     Put   Breakout   0.74  0.62       0.95 0.69
```

#### 14.4 Beta / Correlation 规则（初稿）

| QQQ regime | 同向 Call 上限 | 备注 |
|---|---|---|
| strong uptrend | 允许 2–3 个最强 | 仍按 FinalScore 排序，不是全开 |
| uncertain / chop | 只允许 1 个最强个股 | 禁止叠满 Nasdaq beta |
| downtrend | 个股 Call 需极强独立反转证据 | 否则只做 Put / 不做 |

#### 14.5 与代码目录

- `qqq_btc/`：继续承载 **QQQ 锚点研究** 与已有 0DTE 工具；不再塞 MAG7 主盈利逻辑  
- `stock_options/`：升级为 MAG7 short-DTE（0/1/2）数据与 per-symbol State Gate  
- 跨标的 TopK / Portfolio Risk：新建 `stock_options/tools/` 或后续 `mag7_scanner/`，等 P3 门闸齐后再写

#### 14.6 当前阻塞与下一动作

| 阻塞 | 动作 |
|---|---|
| NVDA/TSLA 无 trading-DTE∈{0,1,2} locked map | **P2：建 map + 下载** |
| Mon/Wed 约 **2026-02** 才上线 | 主研究窗从 2026-02（或实测首日）起；1 月仅周五周权对照 |
| 现有 sniper raw 偏周五周权 | 按 Mon/Wed/Fri 短到期重拉 |
| 跨标的排序尚无输入 | 等 NVDA/TSLA 各自有 Edge 面板后再做 TopK |

**立即执行项**：开始 P2——NVDA/TSLA 每日 expiry→trading DTE 盘点（标注 Mon/Wed 首现日），并设计/生成 short-DTE locked map。

#### 14.7 P2 盘点结果（2026-07-10）

工具：`stock_options/tools/inventory_mag7_short_dte_expiries.py`  
产物：`stock_options/results/mag7_short_dte_expiry_inventory/`

| 标的 | 首个 Mon expiry | 首个 Wed expiry | 首个 Fri expiry（对照） |
|---|---|---|---|
| NVDA | **2026-02-02** | **2026-02-04** | 2026-01-02 |
| TSLA | **2026-02-02** | **2026-02-04** | 2026-01-02 |

- 2026-01 无 Mon/Wed expiry（与用户判断一致）。  
- 2 月起 Mon/Wed/Fri 骨架齐全；post 窗短 DTE 的 dte1/dte2 可用日占比约 **60%**（非每个交易日都有完整 0/1/2）。  
- Locked map 构建器：`preprocess/download/build_mag7_short_dte_api_ladder_map.py`（默认 `--start-date 2026-02-02`）。  
- 已生成：`~/train_data/locked_targets_map_mag7_short_dte_api_ladder.parquet`（NVDA/TSLA × dte0/1/2，约 2026-02-02 → 2026-07-09；非每个交易日都有全部三个桶，符合 Mon/Wed/Fri 骨架）。  
- Micro 下载中：`/mnt/s990/data/microstructure/mag7_short_dte_api_ladder/`（`download_short_dte_microstructure.py`）。  
- 正股 1s 已回填：`/mnt/s990/data/raw_1s/stocks/{NVDA,TSLA}`（map 窗内空缺日补齐，见 `stock_options/results/mag7_stock_1s_backfill_report.json`）。

#### 14.8 P3 dte0 State Gate（2026-07-10）

工具：`stock_options/tools/run_mag7_short_dte_state_gate_stability.py`  
拟合窗：2026-02-02 → 2026-03-31；评估：2026-02 → 2026-06（按月）。  
**禁止**复制 QQQ curated；每标的独立拟合 scorer / 阈值。

| 标的 | 面板行数 | 稳定规则数 | 可读候选 | 初步结论 |
|---|---:|---:|---:|---|
| NVDA 0DTE | 2.14M | 3204 | 弱（仅 2 条过候选筛） | `time_score × recovering × CALL top1` 跨月最稳但均笔仅约 +1.1%；power_hour 右尾被 3 月单月拉高，不稳 |
| TSLA 0DTE | 3.18M | 3204 | 较强（24 条） | **`tree_edge_score × breaking_down × CALL top1`**：5 月均笔约 +3.5%，正月比 0.8，命中约 56%；recovering 次之 |

TSLA 最优月度明细（breaking_down CALL top1）：

| 月 | 笔数 | 均笔 | hit | PF |
|---|---:|---:|---:|---:|
| 02 | 11 | +8.6% | 64% | 4.9 |
| 03 | 13 | +5.3% | 69% | 2.1 |
| 04 | 12 | -3.2% | 42% | 0.6 |
| 05 | 13 | +1.8% | 62% | 1.5 |
| 06 | 13 | +5.0% | 46% | 2.7 |

**决策（dte0）**：TSLA 进入 curated 候选短名单；NVDA 0DTE 暂不升默认。下一步：dte1/dte2 同口径 stability，再做 path 回放 + confirm（仍独立于 QQQ）。

#### 14.9 TopK / 跨标的频率消融（dte0）

问题：日 Top1 ≈ 每天 1 笔，是否过少？

| 策略 | 有 0DTE 日均笔数 | 相对 NYSE 日 | 均笔（标签 30s） | 备注 |
|---|---:|---:|---:|---|
| TSLA breaking_down Top1 | **1.0** | 0.60 | **+3.5%**（stability 全量拟合） | 质量最好 |
| TSLA breaking_down Top2 | **2.0** | 1.18 | +1.2% | 频率翻倍，均笔腰斩 |
| TSLA breaking_down Top3 | **2.9** | 1.77 | +1.2% | 再加第 3 笔几乎不增质 |
| NVDA recovering Top1→3 | 1→3 | 0.59→1.78 | +1.1% → **-2.7%** | 加笔数明显伤质量 |
| 跨标的 Top1（分位对齐） | 1.0 | 0.61 | +1.8% | 多数选中 TSLA |
| 跨标的 Top2 | **2.0** | 1.19 | +1.4%，账户@25% **+42%** | NVDA/TSLA 约各半 |
| 跨标的 Top3 | **3.0** | 1.81 | -0.1% | 第 3 笔拖累 |

产物：`stock_options/results/mag7_topk_cross_symbol_ablation_dte0/`

**结论**：一天确实不止一次机会，但 **Top2 是频率/质量较平衡点**；Top3 开始稀释。跨标的扫描能把候选池做大，同时用日 Top2 控仓，比单标的硬加 Top3 更符合 `maga7.txt`。

---

## 15. QQQ + 深度：靠谱用法（已确认 2026-07-10）

### 15.1 产品定位

```text
QQQ 机会密度 > 个股（几乎每日 0DTE）→ 保留为研究与锚点主场
深度模型 ≠ 端到端下单
深度模型 = Regime / Tradeable / 选时质量（门控与排序）
主 alpha 仍来自 State Gate 规则（因果滚动重标定）
MAG7 = 并行扫描补频率与右尾，跨标的 TopK
```

### 15.2 数据现实（必须先认清）

| 层 | 覆盖 | 含义 |
|---|---|---|
| QQQ 正股 1s | **2022-03 → 2026-07**（~1086 日，约 3 年+） | **Regime 可先做 3 年** |
| QQQ option micro | **仅 2026-01 → 2026-07**（~129 日） | Tradeable / 入场标签暂不能吹成 3 年 |
| Locked / ladder map | 长历史有，但 micro 未全量回补 | 扩 Tradeable 窗 = 先补 micro，不是先上大模型 |

因此：**「3 年深度」先落在正股 Regime；option 侧深度随 micro 回补扩窗。**

### 15.3 三层深度（按优先级）

| 层 | 输入 | 标签 | 协议 | 输出用途 |
|---|---|---|---|---|
| A. Regime Engine | 正股 1s→日/分钟特征（3 年） | 无监督或弱监督（波动/趋势桶）；**不用未来收益定义状态** | expanding：只用过去训，预测下一月 | 给 QQQ/MAG7 降权、beta 上限、是否允许同向叠仓 |
| B. Tradeable / Meta | micro 窗内 curated 候选 | winner / MFE≥x vs direction-wrong / dead | train 过去月，val 选 thr，Jul+ 真 OOS | 过滤假阳性，不创造方向 |
| C. Entry Timing Quality | state 内序列（短） | 前瞻：未来 Δt 是否继续改善 | 仅 micro 窗 | 在 TopK 内重排，不替代 TopK |

**明确禁止**：全样本混训一个「买卖头」再回看 H1；用 3 年数据拟合后在同段宣称泛化。

### 15.4 与旧结论的关系

- §13.5–13.6 GMM Regime 同窗消融曾伤收益 → 新 Regime 必须 **因果 + 正股长窗**，且只做门控，不直接改 alpha 规则发现。  
- §13.7「6 个月不够训 TCN」→ 正股 3 年打开 Regime；option 序列模型仍等 micro 扩到 ≥2 个完整日历年再评估。  
- §13.12 No-Trade v1 → 升级为 B 层 Tradeable，标签改前瞻/归因桶，阈值因果滚动。

### 15.5 执行顺序（P4★）

```text
1. 正股日级 Regime 特征面板（2023–2026，可含 2022）
2. 因果滚动：MLP/浅层序列 预测次月 regime 概率（先浅后深）
3. 在 2026 micro 窗：用 regime 概率门控 curated / TSLA Top2，看 Jul OOS
4. 并行：按需回补 QQQ 2025（再 2024）micro，扩 Tradeable 训练窗
5. 达标后再考虑 option 侧短序列 Timing 模型
```

验收（Jul 或更新的 forward 月）：

- 相对无 Regime 门控：回撤下降，且 Apr–Jun 类强段不被砍到停机  
- Tradeable 在 OOS 有过滤力（不只是训/验 AUC）  
- 仍保持日 Top1–3，不靠「少做」伪装稳健

工具入口（骨架）：`qqq_btc/tools/train_0dte_qqq_deep_anchor.py`

### 15.6 骨架首跑（2026-07-10）

产物：`qqq_btc/results/0dte_qqq_deep_anchor_scaffold/`

| 项 | 结果 |
|---|---|
| 正股特征日 | 854（扩到 2026-07-09） |
| 因果月评 | 2024-03 → 2026-07（583 日预测） |
| 与 curated 重叠 | 103+6 交易日 |

**重要诊断（勿直接「禁做 stress」）**：Jan–Jun curated 上，`regime_pred=2` 日均 **+8.1%** / 账户 **+361%**，`regime_pred=1` **−2.2%** / **−42%**。  
0DTE alpha 落在高波动/趋势桶；Regime 更适合 **砍 mid 桶**，不是禁 stress。

### 15.7 Jul OOS 门控消融（2026-07-10）

工具：`qqq_btc/tools/analyze_0dte_qqq_deep_anchor_gate.py`  
产物：`qqq_btc/results/0dte_qqq_deep_anchor_gate_ablation/`

| 窗 | baseline | `block_pred_1` / `keep_pred_2` |
|---|---|---|
| Jan–Mar | 93 笔，账户 **−32%**，DD **−39%** | **12 笔，账户 +25%，DD −4.6%** |
| Apr–Jun | 73 笔，**+293%**，DD −10% | 69 笔，**+269%**，DD −10%（几乎不伤强段） |
| Jan–Jun | 166 笔，+168%，DD −39% | 81 笔，**+361%**，DD **−10%** |
| Jul W1 OOS | 12 笔，**−18.7%** | **无变化**（Jul 12 笔全是 pred=2） |

附加：

- Jul 内 `p_regime_2` 分位 **分不开** 盈亏（全在 0.94–0.99）。  
- `size_inv_stress` 能把 Jul 亏缩到约 −3%，但 Apr–Jun 从 +293% 砍到 +62% → **不升**。  
- 旧 Tradeable H1：Jun 选 thr=0（几乎不滤）；day_halt 仅把 Jul 亏收到 −11%。  
- **窥视 Jul 的 thr≥0.55 几乎打平，但禁止用 Jul 选 thr。**

**冻结结论**

1. **`block_pred_1` → shadow 升默认（Regime 门控）**：修弱月、保强月；不解决 Jul。  
2. **禁止「禁做 stress / block_pred_2」**：会清空 Jul 并伪装成改善，且毁掉主 alpha 桶。  
3. **Jul 失败在 stress 桶内部** → 下一刀是 Tradeable（含弱月训练）+ MAG7 分散，不是加深 Regime 禁做。

### 15.8 Tradeable 弱月扩训（同日）

`train_0dte_no_trade_gate.py --train-months Jan–May --val-month Jun`  
→ `qqq_btc/results/0dte_state_gate_tradeable_weakmonth_train/`

| 项 | 结果 |
|---|---|
| fit n / tradeable 率 | 189 / 0.68 |
| Jun 选 thr | **仍为 0.0**（强月目标函数不愿过滤） |
| Jul tradeable | pass=100%，无改善 |
| Jul + day_halt | 账户 −18.7% → **−11.4%**（与 H1 相同，仅 halt） |
| expanding 警示 | 用上月选 thr 时 Apr 曾被选成 abstain-all（pass=0）→ 局部 thr 协议不稳 |

**结论**：在「Jun 收益−0.5·DD」目标下，Tradeable **升不了默认**；要改目标（弱月过滤力 / 校准）或先补 2025 micro。Regime `block_pred_1` 与 Tradeable 解耦：前者 shadow 升，后者继续研究。

### 15.9 0DTE + 1DTE 组合（纠正后，2026-07-10）

**纠正**：`qqq_1dte` micro 仅 10 天是 smoke 残缺，**不是**日历事实。  
顺延口径下 locked map（2026 Jan–Jun）：

| DTE | 交易日 | 说明 |
|---|---:|---|
| 0 | **116** | 与 1 完全同集 |
| 1 | **116** | `0∩1=116`，不是「少很多」 |
| 2 | 85 | map/下载在 Apr–May 有缺口，暂不能当「只少 1 天」 |

数据阻塞：1DTE **raw 仅到 2026-03-18（52/116）**；micro 几乎未建。2DTE raw/micro=85 天可用，但是另一条线。

**为何值得合**：0DTE curated 在横盘（日振幅低三分位）日均 **−3.6%**，高振幅 **+6.4%**；idle 无成交日 13 + 低振幅亏损日 17 ≈ **30/116 日（26%）** 是 1DTE 路由候选。若仅「不做」这些低振幅亏日，0DTE 账户上界从 +168%→+463%（1DTE 需至少打平才兑现）。

**推荐路由（未验证，需独立规则）**

```text
同日并行扫描 0DTE + 1DTE（规则各自标定，禁止抄 0DTE curated）
优先 0DTE：高振幅 / regime_pred=2 / 趋势态
改走 1DTE：0DTE idle 或 低振幅横盘（少 theta 压榨）
跨 DTE 日 TopK=1~2，相关惩罚同向叠仓
```

产物：`qqq_btc/results/0dte_1dte_combo_coverage_reanalysis/`

**B 口径已冻结（2026-07-10）**：对齐 0DTE api_ladder，不用旧 6-bucket（含次月）。

| 项 | 值 |
|---|---|
| map | `~/train_data/locked_targets_map_1dte_api_ladder.parquet` |
| 构建 | `preprocess/download/build_qqq_1dte_api_ladder_map.py` |
| 合约 | **8/日**（PUT×4 + CALL×4），动态 ATM ladder |
| DTE | 全部 `selected_dte=1` / `trading_dte=1`（0 异常） |
| 日覆盖 | 2026-01-02→06-30 **123 日**，与 0DTE api_ladder **同集** |
| 旧 raw | 52 日 6-bucket **0 天全覆盖新 ladder** → 需按新 map **全量重下 123 日 / 984 约** |

**下一步**：用户侧续下 1DTE micro（跳过已有）→ stability → 0/1 路由消融。  
指纹层见 **§16**（与下载并行可先在 0DTE 上跑）。

---

## 16. 机构指纹层（P4b★）— 预测起点，不抓尾巴

> 修订（2026-07-10）：否决「跟随机构流 / 吃尾巴残差」。  
> 新口径：**只做 onset（起点）预测**；高 persistence 跟流 = 高位接盘。

### 16.1 为什么「跟随」不可行

短 DTE 主对手是做市与系统单。若等 `net_buy` / 同向流已经**高持续**再进：

- 价格与权利金已被扫过一截  
- 随后常见 inventory 回补 → 你变成接盘  
- 实证（Jan–Jun curated）：`flow_persist` 高三分位均笔 **−5.5%**；晚盘+高持续 **−1.8%**  

结论：F3「高持续」不是机会，是**已晚**的标记。产品目标改为：

```text
在机构主动流尚未展开（或刚加速的第一拍）时入场
一旦流进入高持续 / 晚盘追单 → 默认 No-Trade
```

### 16.2 起点 vs 尾巴（可操作定义）

| | Onset（要） | Tail / Chase（不要） |
|---|---|---|
| 流 | persist 低~中 + **accel>0**（刚加速） | persist 高分位（已同向很久） |
| 时间 | 偏早段；避免尾盘追 | 晚盘 + 高持续 |
| 报价 | 变薄/失衡作**前兆** | 变薄 + 高持续 = 已被扫后的薄书 |
| 决策 | 预测「下一小段会启动」 | 确认「已经在涨/跌」再跟 |

### 16.3 三个指纹（改写为 onset 前兆）

均在入场前因果窗；**禁止**用持仓期未来信息。

#### F1 — Pre-sweep inventory stress（扫单前的簿记压力）

```text
前兆（不是事后回补）：
  - quote_thinning 升、size_asymmetry 升，且此时 flow_persist 仍低
  - 含义：簿记变薄/偏斜，系统单尚未完全打穿
用法：
  - thin↑ + persist 低~中 → onset 候选加权
  - thin↑ + persist 高 → 接盘区，No-Trade
否决：把「sweep 后 replenish」当买入信号（那是尾巴）
```

#### F2 — Gamma-day ignition（振幅刚打开，不是已经走完）

```text
前兆：
  - rv_vs_prem 从低位抬升的过程（日级滚动），而非尾盘已经高振幅
  - 早段 range 仍有限但开始扩张 → 0DTE 可做
  - 全日仍极低 → 改 1DTE / 不做 0
用法：DTE 路由 + 是否允许 0DTE，不作「已大涨后追」
```

#### F3 — Flow ignition（加速起点，不是高持续）

```text
主信号：flow_accel > 0 且 persist ∈ 低~中
否决信号：persist 高分位（无论早晚）
辅助：session_minute 过大 + persist 高 → 强制降权
用法：选时 / Tradeable 核心；替代「跟流」
```

### 16.4 接入（仍挂在现架构上）

```text
State Gate           → 方向假设（state×rule）
F3 ignition          → 只允许 accel↑ 且非高 persist
F1 pre-stress        → 簿记前兆加权；薄书+高persist 一票否决
F2 ignition/route    → 0 vs 1 DTE
Regime block_pred_1  → 日级总开关（shadow）
禁止：FinalScore 里加「flow_persist 越高越好」
```

验收：

1. 相对 baseline：去掉「高 persist」后弱月回撤降，强月保留 ≥70% 笔数  
2. `mid_persist + accel+`（或等价 ignition）OOS 均笔优于 `high_persist`  
3. Jul 只报告：若仍偏晚盘高持续，说明入口时钟/onset 检测未达标，不调参粉饰

### 16.5 工具

`qqq_btc/tools/analyze_0dte_institutional_fingerprints.py`  
产物：`qqq_btc/results/0dte_institutional_fingerprints/`  
下一步：标签改为 **ignition mask**（accel↑ ∩ ¬高persist）；补 path 归因后验 F1 前兆。

### 16.6 实证快照（支持改写，2026-07-10）

Jan–Jun curated（n=166）：

| 代理 | n | 均笔 | 账户@25% | 解读 |
|---|---:|---:|---:|---|
| baseline | 166 | +2.8% | +168% | — |
| **TAIL** 高 persist | 55 | −0.2% | −8% | 跟流差 |
| **TAIL** 晚盘+高 persist | 21 | −1.8% | −12% | 典型接盘 |
| **ONSET** mid persist + accel+ | 19 | **+14.9%** | **+91%** | 起点代理（样本小，只作方向） |
| **ONSET** 早盘 + accel+ | 7 | +21.3% | +39% | 同向，n 更小 |
| 早盘但高 persist | 16 | −2.0% | — | 再早也已晚 |

交叉：`thin=high` 且 `persist=high` → 均笔 **−7.8%**（薄书+已展开流=接盘区）。

Jul W1（只报告）：12 笔均笔 −6.6%，**session 均约 205 分钟**（偏晚）→ 与「抓尾巴」病理一致，应用 onset 时钟重做入口，而不是加跟流权重。

**冻结**：方向可行，但必须是 **onset 预测**；简单跟随机构流 **不可行**。

### 16.7 完整验证（1DTE 齐套后，2026-07-10）

数据：1DTE ladder micro **123/123** 日齐；工具 `validate_0dte_onset_and_1dte_route.py`  
产物：`qqq_btc/results/0dte_onset_full_validation/`  
协议：阈值 fit 于 **Jan–Apr**；May–Jun shadow；Jul **只报告**。

| 门控 | May–Jun 账户 | Apr–Jun 账户 | Jan–Mar | Jul（报告） | 保留率 May–Jun | 结论 |
|---|---:|---:|---:|---:|---:|---|
| baseline | +206% | +293% | −32% | −19% | 100% | — |
| **block_chase_late** | **+223%** | +240% | −13% | **−6.6%** | 88% | **shadow 升** |
| block_thin_high_persist | +216% | **+305%** | −24% | −19% | 93% | shadow 候选 |
| block_high_persist | +174% | +191% | **+3.7%** | −8% | 74% | 弱月好，强月伤账户 |
| ignition_only | +47% | +69% | −3% | −4.5% | **25%** | 均笔高但过稀，不升默认 |

解读：

1. **禁止跟流**成立：砍「高 persist × 晚盘」接盘区，强段几乎不伤，Jul 亏幅收窄（报告项）。  
2. **纯 ignition_only** 样本太少，作研究档，不作默认。  
3. 1DTE micro **123/123 已齐**；stability + replay_v0 完成。  
4. 与 `block_pred_1` 可叠；Jul 全是 pred=2，onset 门控才是 Jul 的主要杠杆。

### 16.8 1DTE replay 与因果路由（同日）

| 项 | 结果 |
|---|---|
| 1DTE stability | 标签边弱（~0.5–0.8%/笔）；`tree_edge × ALL/CALL top1` 正月比 0.83 |
| 1DTE replay_v0 | 230 笔，均笔 **+0.70%**，账户@25% **+41%**，DD **−47%**（Jan 强、Feb–May 弱、Jun 回升） |
| 因果 low_f2 日（35 日） | 0DTE 均笔 ~0；1DTE 均笔 ~0 → **尚未证明 1DTE 能救横盘** |
| 因果 chase_risk 日（8 日） | 0DTE −5.7%；1DTE −1.8% → 1DTE 较不糟，但样本小 |
| 事后「0DTE 亏日→换 1DTE」 | **作废**（用了实现盈亏，前视） |

**冻结结论（完整验证）**

1. **`block_chase_late` → shadow 升默认**（0DTE onset 门控）。  
2. **1DTE 数据层完成**，但 v0 规则 **不升** 组合默认；需独立 curated（confirm/state-hold）后再做路由。  
3. 组合 alpha **未证实**；下一步是加强 1DTE 规则质量，不是把弱 1DTE 硬塞进路由器。  
4. 继续坚持 **onset 而非跟流**。

产物：
- `qqq_btc/results/0dte_onset_full_validation/`
- `qqq_btc/results/state_gate_qqq_1dte/stability_jan_jun/`
- `qqq_btc/results/state_gate_qqq_1dte/replay_v0_jan_jun/`
- `full_verdict_causal.json`

### 16.9 组合验证：QQQ onset + TSLA Top2（2026-07-10）

工具：`validate_portfolio_qqq_onset_tsla_top2.py`  
窗：2026-02-02→06-30；日仓：单名 25%，同日总上限 40%。  
TSLA：`tree_edge × breaking_down × CALL Top2`，**30s 标签**（非 path）。

| 组合（日复利） | 账户 | DD | 备注 |
|---|---:|---:|---|
| QQQ baseline | +122% | −28% | 对照 |
| QQQ `block_chase_late` | +147% | −27% | onset 单独已改善 |
| TSLA Top2 only | +16% | −10% | 边薄但回撤小 |
| QQQ onset + TSLA | **+151%** | −29% | 略优于 onset 单做 |
| QQQ onset+`block_pred_1`+TSLA | **+158%** | **−9%** | 回撤明显改善 |
| Jul QQQ onset（报告） | −6.6% | −13% | vs baseline −19% |

月度（onset+TSLA）：Feb +49%，**Mar −20%**，Apr −3%，May +63%，Jun +33% → Mar 仍痛。

**冻结**

1. **`block_chase_late` 继续 shadow 升**（QQQ）。  
2. **QQQ onset + TSLA Top2 → shadow 组合候选**（相对 QQQ 单做账户升、可叠 regime）。  
3. 不是圣杯：Mar 仍亏；TSLA 仍是标签回放，升默认前要 path 确认。  
4. **不是死路**：门控+分散在验证上优于「再赌大模型」。
