# Domain Schema

日期：`2026-04-23`

范围：

- `production/baseline/Domain/contracts.py`

目的：

- 为 `Domain` 层提供独立、稳定、可并线前审阅的 schema 文档
- 明确每个对象的职责边界、字段语义、时间语义和主要校验约束
- 为后续 adapter、旁路验证和主线接入提供统一参考

---

## 1. 设计原则

这套 `Domain` 模型遵循 4 条原则：

1. **时序语义优先**
   - 先定义分钟窗口、分钟冻结事实、秒级执行轨迹，再定义具体资产载荷。
2. **资产语义与时序语义分离**
   - 期权、股票、永续的差异由 `InstrumentTraits` 表达，不污染窗口契约。
3. **持仓语义与行情语义分离**
   - `PositionSnapshot` 只表达仓位本身，不混入 alpha 或 quote 载荷。
4. **关键字段显式化**
   - 关键字段进入正式 schema；`metadata` / `tags` 只能承载非关键扩展信息。

---

## 2. 对象关系

```text
InstrumentTraits
    ├─ AlphaFrameItem.instrument_traits
    └─ PositionSnapshot.instrument_traits

DecisionQuoteSnapshot
    └─ AlphaFrameItem.decision_quote

ExecutionQuote1s
    └─ ExecutionWindow.quotes_1s[*]

AlphaFrameItem
    └─ AlphaFrame.items[*]

AlphaFrame
    └─ ExecutionWindow.alpha_frame

PositionSnapshot
    └─ 与上述对象并列存在，用于持仓语义，不内嵌在 frame/window 里
```

---

## 3. 枚举

### `InstrumentKind`

| 值 | 含义 |
| --- | --- |
| `stock` | 现货股票 |
| `option` | 期权 |
| `perpetual` | 永续合约 |
| `future` | 期货 |
| `unknown` | 未知或未定资产类型 |

### `QuoteSourceKind`

| 值 | 含义 |
| --- | --- |
| `unknown` | 来源未知 |
| `current_batch` | 当前分钟/当前批次直接提供 |
| `latched_prev_second` | 上一分钟边界前锁存的上一秒快照 |
| `replay_feed` | replay 回放输入 |
| `live_feed` | 实时行情输入 |
| `execution_backfill` | 执行路径回填 |
| `synthetic` | 人工合成或兜底构造 |

### `PositionSide`

| 值 | 含义 |
| --- | --- |
| `flat` | 空仓 |
| `long` | 多头 |
| `short` | 空头 |

---

## 4. Schema 详情

## `InstrumentTraits`

职责：

- 表达资产类别本身的交易规则和语义能力
- 不带时间维度
- 不承载策略分数、行情或持仓状态

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `symbol` | `str` | 是 | 资产或合约标识 |
| `instrument_kind` | `InstrumentKind` | 是 | 资产类别 |
| `quote_currency` | `str` | 否 | 报价币种，默认 `USD` |
| `settlement_currency` | `str` | 否 | 结算币种，默认 `USD` |
| `contract_multiplier` | `float` | 否 | 合约乘数，必须 `> 0` |
| `min_price_increment` | `float` | 否 | 最小价格跳动，必须 `> 0` |
| `qty_step` | `float` | 否 | 最小数量步长，必须 `> 0` |
| `supports_long` | `bool` | 否 | 是否支持多头 |
| `supports_short` | `bool` | 否 | 是否支持空头 |
| `has_expiry` | `bool` | 否 | 是否具有到期日语义 |
| `has_strike` | `bool` | 否 | 是否具有行权价语义 |
| `has_iv` | `bool` | 否 | 是否具有隐波语义 |
| `mark_price_required` | `bool` | 否 | 是否要求 `mark_price` 作为主要估值锚点 |
| `funding_rate_supported` | `bool` | 否 | 是否支持 funding 语义 |
| `metadata` | `dict` | 否 | 非关键扩展字段 |

主要校验：

- `symbol` 非空
- `contract_multiplier / min_price_increment / qty_step > 0`
- 至少支持 `long` 或 `short` 之一

---

## `DecisionQuoteSnapshot`

职责：

- 表达策略做决定时真正看到的可交易快照
- 用于分钟冻结决策语义，不用于秒级执行轨迹

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `symbol` | `str` | 是 | 资产标识 |
| `instrument_kind` | `InstrumentKind` | 是 | 资产类别 |
| `quote_ts` | `float` | 是 | 决策所见快照时刻，必须 `> 0` |
| `last_price` | `float` | 否 | 最近成交价或参考价 |
| `best_bid` | `float` | 否 | 最优买价 |
| `best_ask` | `float` | 否 | 最优卖价 |
| `bid_size` | `float` | 否 | 最优买量 |
| `ask_size` | `float` | 否 | 最优卖量 |
| `mark_price` | `float?` | 否 | 衍生品估值锚点 |
| `index_price` | `float?` | 否 | 永续/指数锚点 |
| `contract_id` | `str` | 否 | 具体合约标识 |
| `venue` | `str` | 否 | 行情来源或交易 venue |
| `source_kind` | `QuoteSourceKind` | 否 | 快照来源类型 |
| `contract_multiplier` | `float` | 否 | 合约乘数 |
| `metadata` | `dict` | 否 | 非关键扩展字段 |

派生属性：

- `mid_price`
  - 优先 `(best_bid + best_ask) / 2`
  - 否则回退 `mark_price`
  - 再回退 `last_price`
- `has_book`
  - 当 `best_bid > 0` 且 `best_ask > 0` 时为真

主要校验：

- `quote_ts > 0`
- 价格与数量字段必须有限且 `>= 0`
- 若盘口存在，则 `best_ask >= best_bid`

---

## `ExecutionQuote1s`

职责：

- 表达秒级执行窗口中的行情轨迹
- 用于执行跟踪、报价回放和分秒级归因

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `symbol` | `str` | 是 | 资产标识 |
| `instrument_kind` | `InstrumentKind` | 是 | 资产类别 |
| `ts` | `float` | 是 | 秒级事件时刻，必须 `> 0` |
| `last_price` | `float` | 否 | 最近价 |
| `best_bid` | `float` | 否 | 最优买价 |
| `best_ask` | `float` | 否 | 最优卖价 |
| `bid_size` | `float` | 否 | 最优买量 |
| `ask_size` | `float` | 否 | 最优卖量 |
| `mark_price` | `float?` | 否 | 衍生品估值锚点 |
| `index_price` | `float?` | 否 | 指数锚点 |
| `contract_id` | `str` | 否 | 合约标识 |
| `venue` | `str` | 否 | 交易 venue / feed 源 |
| `source_kind` | `QuoteSourceKind` | 否 | 行情来源类型 |
| `sequence_no` | `int?` | 否 | 若上游提供严格顺序号，可用于重放排序 |
| `exchange_latency_ms` | `float?` | 否 | 行情链路观测延迟 |
| `metadata` | `dict` | 否 | 非关键扩展字段 |

辅助方法：

- `to_decision_snapshot()`
  - 将执行快照映射为 `DecisionQuoteSnapshot`
  - 用于分钟边界策略决策复用秒级快照

主要校验：

- `ts > 0`
- 价格/数量字段有限且 `>= 0`
- 若盘口存在，则 `best_ask >= best_bid`

---

## `AlphaFrameItem`

职责：

- 表达一个 symbol 在分钟边界冻结下来的单标的决策事实
- 是 `AlphaFrame` 的最小业务单元

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `symbol` | `str` | 是 | 标的标识 |
| `instrument_traits` | `InstrumentTraits` | 是 | 资产语义 |
| `alpha` | `float` | 是 | 核心 alpha 分数 |
| `alpha_label_ts` | `int` | 是 | 标签时刻，通常为 `minute_ts - 60` |
| `alpha_available_ts` | `int` | 是 | alpha 可交易时刻，通常等于 `minute_ts` |
| `batch_idx` | `int` | 否 | 在该分钟批次中的原始位置 |
| `frame_id` | `str` | 否 | 所属帧 id |
| `reference_price` | `float` | 否 | 冻结时的参考价格 |
| `cs_alpha_z` | `float` | 否 | 截面 alpha z-score |
| `vol_z` | `float` | 否 | 成交量或波动 z-score |
| `roc_5m` | `float` | 否 | 5 分钟动量 |
| `macd` | `float` | 否 | 分钟冻结 MACD 直方图或同义指标 |
| `macd_slope` | `float` | 否 | MACD 斜率 |
| `snap_roc` | `float` | 否 | 秒级快照动量 |
| `event_prob` | `float` | 否 | 事件概率，范围 `[0, 1]` |
| `is_ready` | `bool` | 否 | 是否通过 warmup / 可交易就绪 |
| `correction_mode` | `str` | 否 | 对 alpha 进行纠偏时的模式标记 |
| `decision_quote` | `DecisionQuoteSnapshot?` | 否 | 策略做决定时所见快照 |
| `tags` | `dict` | 否 | 非关键扩展标签 |

派生属性：

- `minute_ts`
  - 直接等于 `alpha_available_ts`

主要校验：

- `instrument_traits.symbol == symbol`
- `alpha_available_ts > alpha_label_ts`
- `alpha_available_ts - alpha_label_ts == 60`
- `event_prob ∈ [0, 1]`
- 若存在 `decision_quote`：
  - `decision_quote.symbol == symbol`
  - `decision_quote.instrument_kind == instrument_traits.instrument_kind`
  - `decision_quote.quote_ts <= alpha_available_ts`

---

## `AlphaFrame`

职责：

- 表达单个分钟边界上的横截面冻结事实
- 是策略分钟级输入的 canonical frame

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `frame_id` | `str` | 是 | 帧唯一标识，允许为空但建议全链路填写 |
| `minute_ts` | `int` | 是 | 当前分钟窗口左边界 |
| `alpha_label_ts` | `int` | 是 | 该帧消费的标签时刻 |
| `alpha_available_ts` | `int` | 是 | 该帧真正可决策时刻 |
| `items` | `list[AlphaFrameItem]` | 否 | 单标的冻结事实集合 |
| `index_trend` | `int` | 否 | 帧级市场方向 |
| `market_regime` | `str` | 否 | 帧级 regime 标签 |
| `is_zombie_market` | `bool` | 否 | 是否处于僵尸市场/应禁开仓状态 |
| `metadata` | `dict` | 否 | 非关键扩展字段 |

辅助构造：

- `AlphaFrame.from_items(minute_ts, items, frame_id=...)`
  - 自动推导：
    - `alpha_label_ts = minute_ts - 60`
    - `alpha_available_ts = minute_ts`

主要校验：

- `alpha_label_ts == minute_ts - 60`
- `alpha_available_ts == minute_ts`
- 帧内 `symbol` 不可重复
- 每个 `item` 的时间字段必须与 frame 一致
- 若 item 自带 `frame_id`，则必须与外层一致

---

## `ExecutionWindow`

职责：

- 绑定一个分钟帧和该分钟内的秒级执行行情
- 是回放、仿真、执行研究的窗口级契约

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `minute_ts` | `int` | 是 | 执行窗口左边界 |
| `alpha_label_ts` | `int` | 是 | 窗口消费的标签时刻 |
| `alpha_available_ts` | `int` | 是 | 窗口可执行时刻 |
| `alpha_frame` | `AlphaFrame` | 是 | 分钟冻结事实 |
| `quotes_1s` | `list[ExecutionQuote1s]` | 否 | 秒级执行行情序列 |

辅助构造：

- `ExecutionWindow.from_frame(minute_ts, alpha_frame, quotes_1s)`
  - 自动推导：
    - `alpha_label_ts = minute_ts - 60`
    - `alpha_available_ts = minute_ts`

主要校验：

- `alpha_label_ts == minute_ts - 60`
- `alpha_available_ts == minute_ts`
- `alpha_frame.minute_ts == minute_ts`
- `quotes_1s[*].ts ∈ [minute_ts, minute_ts + 60)`
- `quotes_1s[*].ts` 非递减

---

## `PositionSnapshot`

职责：

- 只表达仓位状态，不表达交易信号或行情快照
- 可以单独与 `frame_id` / `quote_ts` 关联，方便审计

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `symbol` | `str` | 是 | 标的标识 |
| `instrument_traits` | `InstrumentTraits` | 是 | 资产语义 |
| `side` | `PositionSide` | 否 | 仓位方向，默认 `flat` |
| `quantity` | `float` | 否 | 绝对仓位量 |
| `avg_entry_price` | `float` | 否 | 平均开仓价格 |
| `entry_ts` | `float` | 否 | 开仓时刻 |
| `contract_id` | `str` | 否 | 具体合约标识 |
| `entry_frame_id` | `str` | 否 | 来源分钟帧 |
| `entry_quote_ts` | `float?` | 否 | 决策所见 quote 的时刻 |
| `realized_pnl` | `float` | 否 | 已实现盈亏 |
| `unrealized_pnl` | `float` | 否 | 未实现盈亏 |
| `max_favorable_excursion` | `float` | 否 | 最大有利偏移 |
| `max_adverse_excursion` | `float` | 否 | 最大不利偏移 |
| `metadata` | `dict` | 否 | 非关键扩展字段 |

派生属性：

- `is_open`
  - `side != flat and quantity > 0`
- `signed_quantity`
  - `long -> +quantity`
  - `short -> -quantity`
  - `flat -> 0`

主要校验：

- `instrument_traits.symbol == symbol`
- `quantity >= 0`
- `avg_entry_price >= 0`
- `entry_ts >= 0`
- `flat` 必须 `quantity == 0`
- 非 `flat` 必须：
  - `quantity > 0`
  - `avg_entry_price > 0`
  - `entry_ts > 0`
- `short` 方向要求 `instrument_traits.supports_short == True`

---

## 5. 时间语义总表

| 字段 | 所在对象 | 语义 |
| --- | --- | --- |
| `alpha_label_ts` | `AlphaFrameItem` / `AlphaFrame` / `ExecutionWindow` | 该窗口消费的分钟标签时刻 |
| `alpha_available_ts` | `AlphaFrameItem` / `AlphaFrame` / `ExecutionWindow` | alpha 真正允许下单/决策的时刻 |
| `minute_ts` | `AlphaFrame` / `ExecutionWindow` | 当前分钟窗口左边界 |
| `quote_ts` | `DecisionQuoteSnapshot` | 策略实际看到决策快照的时刻 |
| `ts` | `ExecutionQuote1s` | 秒级执行行情的事件时刻 |
| `entry_ts` | `PositionSnapshot` | 持仓建仓时刻 |
| `entry_quote_ts` | `PositionSnapshot` | 建仓时看到的 quote 时刻 |

关系约束：

- `minute_ts == alpha_available_ts`
- `alpha_label_ts == minute_ts - 60`
- `decision_quote.quote_ts <= alpha_available_ts`
- `quotes_1s[*].ts` 必须落在当前分钟窗口中

---

## 6. 多资产扩展约定

当前 schema 已经尽量做成资产无关，但扩展时遵循以下规则：

1. 资产差异优先进 `InstrumentTraits`
2. 决策快照和执行快照优先使用通用字段：
   - `best_bid`
   - `best_ask`
   - `bid_size`
   - `ask_size`
   - `mark_price`
   - `index_price`
3. 期权特有字段、永续特有字段如需加入：
   - 优先先经过 schema 评审
   - 不直接塞入 `metadata` 替代正式字段

示例：

- 期权可新增：
  - `strike`
  - `expiry`
  - `iv`
  - `delta/gamma/vega/theta`
- 永续可新增：
  - `funding_rate`
  - `open_interest`
  - `basis`
  - `liquidation_buffer`

---

## 7. 当前文档的使用方式

推荐把这份文档当成 3 种用途：

1. **代码评审清单**
   - 新字段是否应该进正式 schema
2. **adapter 开发说明**
   - 旧 dict payload 要如何映射为 `Domain` 对象
3. **并线前验收标准**
   - 新结构是否真的满足时间和一致性约束
