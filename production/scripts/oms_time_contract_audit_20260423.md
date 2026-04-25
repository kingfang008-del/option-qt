# OMS 时序契约审计与执行计划

日期：`2026-04-23`  
范围：

- `production/baseline/execution_engine_v8.py`
- `production/baseline/signal_engine_v8.py`
- `production/baseline/DAO/realtime_feature_engine.py`

目的：

- 审计 OMS 当前用于关联 `alpha`、秒级行情、分钟标签的结构是否足够稳固
- 判断这套结构是否真的有助于防止 replay / realtime / 实盘之间的语义漂移
- 给出后续从“弱字典载荷”升级到“强契约对象”的执行计划
- 评估该结构未来扩展到 `BTC` 永续合约时的可复用边界与必要改造

---

## 1. 总结结论

当前设计方向是对的，而且已经明显优于“散字段临时拼 payload”的旧方式。

核心优点有三点：

1. 已经显式定义了分钟窗口 `ExecutionWindow`，把 `minute_ts / alpha_label_ts / alpha_available_ts / alpha_frame / quotes_1s` 固化成统一时序契约。
2. OMS 已经把策略评估集中到 `ALPHA_FRAME` 上执行，避免策略在秒级路径里被行情噪声反复改写。
3. 下单与成交日志已经开始透传 `alpha_label_ts / alpha_available_ts / order_submit_ts / fill_ts`，后续具备做链路归因的基础。

但当前仍然不是“彻底防漂移系统”，而是“半结构化防漂移框架”。

主要原因：

1. 外层有 `dataclass`，内层关键载荷仍然大量使用 `Dict[str, Any]`
2. realtime 主链路并没有完全围绕 typed window / typed frame 运行
3. 校验器当前主要校验时间边界，没有系统校验 frame、symbol、quote 来源一致性
4. 当前分钟边界上的 `decision_opt_data` 仍然保留旧语义，容易成为后续解释和扩展时的歧义源

结论判断：

- **对于当前美股分钟 alpha + 秒级期权执行链路，这套结构已经有明显正面价值**
- **对于未来 BTC 永续接入，这套结构的“时序壳”可复用，但“交易载荷层”必须抽象，否则会被 call/put/iv/strike 等期权字段卡死**

---

## 2. 当前结构里已经正确的部分

### 2.1 分钟窗口契约已经被显式化

`ExecutionWindow` 已经把“分钟事实”和“秒级执行”分成两个层次：

- `minute_ts`
- `alpha_label_ts`
- `alpha_available_ts`
- `alpha_frame`
- `quotes_1s`

这解决了过去最常见的一个问题：同一轮决策里到底用了哪一分钟的 alpha、从哪个时刻开始允许下单、秒级执行行情属于哪个窗口。

现有定义说明，这个方向是正确的。

### 2.2 Signal Engine 到 OMS 的分钟边界已经相对清晰

`signal_engine_v8.py` 会构造 canonical `ALPHA_FRAME`，并把分钟级 `alpha_items` 一次性发给 OMS。

这意味着：

- alpha 事实在分钟边界冻结
- OMS 不再自己重建 alpha 历史
- 策略不需要在秒级路径里重新推导分钟标签

这是防漂移的关键一步。

### 2.3 Timing 字段已经具备链路归因雏形

执行侧已经能记录：

- `alpha_label_ts`
- `alpha_available_ts`
- `order_submit_ts`
- `fill_ts`
- `alpha_to_submit_ms`
- `submit_to_fill_ms`
- `alpha_to_fill_ms`

这对后续诊断“漂移来自模型标签、quote、下单延迟还是成交延迟”非常重要。

---

## 3. 重点审计清单

下面按优先级列出审计项。

### P0. `ExecutionWindow` 是否已经是唯一窗口契约

目标：

- 确认系统里是否真的只有一套“分钟决策窗口”定义

现状：

- `ExecutionWindow` 已经存在
- `execute_window()` 也把分钟决策和秒级执行分开
- 但 realtime 主链路并没有彻底围绕它运行

风险：

- 文档里有唯一窗口，运行时却存在第二套隐式窗口语义

验收标准：

- 决策窗口定义只有一套
- 任意一笔交易都可追溯到唯一 `frame_id + minute_ts`

### P0. `alpha_item` 仍然是弱字典，不是强契约

目标：

- 确认分钟 alpha、分钟标签、决策用秒级快照之间是否被稳定绑定

现状：

- `alpha_items.append({...})` 仍然使用 dict
- 关键字段包括：
  - `symbol`
  - `stock_price`
  - `alpha`
  - `cs_alpha_z`
  - `vol_z`
  - `roc_5m`
  - `macd`
  - `macd_slope`
  - `snap_roc`
  - `event_prob`
  - `alpha_label_ts`
  - `alpha_available_ts`
  - `opt_data`

风险：

- 字段名漂移无法静态发现
- 缺字段时默认值可能吞错
- `opt_data` schema 变动时 OMS 侧难以及时发现

验收标准：

- `AlphaFrameItem` 需要成为显式对象
- 关键字段缺失时必须显式告警或拒绝

### P0. `alpha_label_ts / alpha_available_ts / frame.ts` 是否全链路一致

目标：

- 避免“标签时刻”和“可交易时刻”被混用

现状：

- `ExecutionWindow` 明确约束：
  - `alpha_label_ts = minute_ts - 60`
  - `alpha_available_ts = minute_ts`
- `signal_engine_v8.py` 也在透传这两个字段

风险：

- 一旦某段代码把 `curr_ts`、`frame.ts`、`alpha_available_ts` 当成同义词使用，就会出现隐藏的一分钟错位

验收标准：

- 任意订单都能还原完整时间链：
  - `label minute -> available minute -> submit -> fill`

### P0. 分钟边界上的 `decision_opt_data` 语义是否明确

目标：

- 确认策略真正看到的是哪一拍秒级 quote

现状：

- 在整分钟边界，策略决策优先使用 `st.last_tick_opt_data`
- 同时 alpha log 仍然可能使用当前批次的 `opt_data`

这是当前最容易造成解释分裂的点。

风险：

- 决策 quote 与记录 quote 不是同一拍
- 后续 debug 时，日志和行为不完全对得上
- 扩展到其他资产时会把这种歧义继续复制过去

验收标准：

- 明确定义并命名 `decision_quote`
- 必须显式记录 `decision_quote_ts`
- 同一笔单能够看出 quote 来源类型

### P1. `ctx` 是否混合了分钟冻结事实与运行时副作用

目标：

- 防止策略读取到不同时间层的字段，却误以为来自同一时刻

现状：

- `ctx` 同时包含：
  - 分钟 alpha 指标
  - 当前 / 锁存 quote
  - 持仓状态
  - runtime state

风险：

- 策略未来迭代时容易误用动态字段替代冻结字段
- replay / realtime 行为差异会被藏进 `ctx`

验收标准：

- 至少在结构和命名上区分：
  - `alpha_snapshot`
  - `decision_quote`
  - `position_snapshot`
  - `runtime_state`

### P1. `frame_id` 是否真正贯穿到交易闭环

目标：

- 确保不是只有 frame 有 id，而是订单、成交、状态都可挂回同一 frame

现状：

- `ALPHA_FRAME` 有 `frame_id`
- 提交订单时也在透传
- 但 `alpha_item`、quote 审计、状态持久化未必都完全绑定

风险：

- 只能按时间模糊反查，而不能精确归因

验收标准：

- 任意 OPEN / CLOSE / GHOST / BLOCKED 记录都能反查到唯一 `frame_id`

### P1. `validate()` 当前只做边界校验，没有做一致性校验

目标：

- 从“窗口时间合法”升级到“窗口内容一致”

当前建议新增校验：

1. `alpha_frame.ts == minute_ts`
2. `alpha_label_ts == minute_ts - 60`
3. `alpha_available_ts == minute_ts`
4. `quotes_1s` 时间有序
5. `items[*].symbol` 唯一
6. `items[*].frame_id` 与外层一致
7. `decision_quote.quote_ts <= alpha_available_ts`
8. `items[*]` 与 `quotes_1s` 的 symbol universe 一致或可解释缺失

验收标准：

- 提供 soft validation 与 strict validation 两层

### P2. 日志是否足够支持漂移归因

目标：

- 后续能回答“到底漂在哪一层”

建议确保日志具备：

- `frame_id`
- `minute_ts`
- `alpha_label_ts`
- `alpha_available_ts`
- `decision_quote_ts`
- `decision_bid`
- `decision_ask`
- `decision_mid`
- `order_submit_ts`
- `fill_ts`
- `decision_quote_source_kind`

---

## 4. 详细执行计划

### 阶段 0：字段盘点与冻结

目标：

- 先冻结当前事实定义，避免后续改造时语义继续滑动

任务：

1. 列出对象级字段表：
   - `ExecutionWindow`
   - `ALPHA_FRAME`
   - `alpha_item`
   - `opt_data`
   - `ctx`
   - `trade log timing meta`
2. 为每个字段标记：
   - 所属时间层：分钟 / 秒级 / 持仓 / 运行时
   - 生产者模块
   - 消费者模块
   - 是否关键字段
   - 是否存在默认值吞错风险

交付：

- 一张字段级审计表

### 阶段 1：引入正式对象模型

目标：

- 把关键载荷从弱字典升级成强契约对象

建议新增对象：

1. `AlphaFrame`
2. `AlphaFrameItem`
3. `DecisionQuoteSnapshot`
4. `ExecutionQuote1s`

建议原则：

- 外层对象表达时序
- 内层对象表达交易载荷
- Optional 字段只用于真实可缺省字段，不再用于掩盖 schema 漏洞

交付：

- typed dataclass / model 定义
- 从 dict 到对象的构造函数与校验函数

### 阶段 2：升级校验器

目标：

- 从“边界没越界”升级到“结构彼此一致”

任务：

1. 为 `ExecutionWindow` 增加强一致性校验
2. 为 `AlphaFrame / AlphaFrameItem / DecisionQuoteSnapshot` 增加 `validate()`
3. 把 silent fallback 改成显式 warning 或 hard fail

交付：

- `soft_validate()`
- `strict_validate()`

### 阶段 3：统一 realtime / replay 的对象入口

目标：

- 保证运行时真正共享一套对象模型，而不是只共享部分函数

任务：

1. realtime 收到 `ALPHA_FRAME` 后先反序列化为 typed `AlphaFrame`
2. OMS 策略入口优先消费 typed `AlphaFrameItem`
3. replay 和 realtime 统一通过同一层对象转换再进入策略

交付：

- typed ingestion boundary
- 统一策略入口

### 阶段 4：定版分钟边界 quote 语义

目标：

- 彻底消除 `decision_opt_data` 的语义歧义

任务：

1. 明确策略使用：
   - 上一秒锁存 quote
   - 当前 frame quote
   - 最近 `<= minute_ts` 的 quote
   三者中的哪一种
2. 新增 `decision_quote_source_kind`
3. 新增 `decision_quote_ts`

交付：

- 文档化的 quote 选择规则
- 可审计的 quote 来源字段

### 阶段 5：补齐日志与观测

目标：

- 为后续 dashboard、归因分析、故障排查留证据

任务：

1. 补充 frame 级 trace
2. 补充 quote 来源 trace
3. 把决策到成交链路完整打到日志

交付：

- 可直接用于 dashboard 的时序链路字段

### 阶段 6：测试与回归保护

目标：

- 把结构正确性固化成自动回归保障

建议测试：

1. 契约测试
2. quote 边界语义测试
3. frame 追踪一致性测试
4. replay / realtime 对齐测试
5. 构造错位字段时的拒绝 / 告警测试

交付：

- 结构契约测试
- 漂移回归测试

---

## 5. 字段设计上最值得优先重构的地方

如果只做最小但高收益的改造，我建议按下面顺序推进：

1. `alpha_item` 从 dict 变 typed object
2. `opt_data` 从“期权快照混合结构”拆成明确 `decision_quote`
3. `frame_id` 贯穿到日志与状态
4. `ctx` 拆层，避免把分钟冻结事实和动态状态混用
5. `validate()` 从边界校验升级到一致性校验

这样做的收益最大，而且不会一开始就大面积改动策略逻辑。

---

## 6. 这套结构是否适合扩展到 BTC 永续合约

结论：

**可以扩展，但必须先把“时序契约”与“期权载荷语义”拆开。**

换句话说：

- `ExecutionWindow` 这一层，天然适合继续复用到 `BTC` 永续
- 但当前 `alpha_item.opt_data` 和 OMS 内部若干字段明显写死在期权语义上，不能原样平移

### 6.1 可以直接复用的部分

以下部分对 `BTC` 永续依然成立：

1. `minute_ts / alpha_label_ts / alpha_available_ts`
2. `frame_id`
3. 分钟冻结决策、秒级执行跟踪的两层结构
4. `label -> available -> submit -> fill` 的 timing 链路
5. `ExecutionWindow`
6. `AlphaFrame`
7. `AlphaFrameItem` 中与 alpha 强度、市场 regime、事件概率相关的字段

这些都是**资产无关的时序基础设施**。

### 6.2 不能直接复用、必须抽象的部分

以下部分当前明显带有“美股期权”绑定：

1. `call_price / put_price`
2. `call_bid / call_ask / put_bid / put_ask`
3. `call_id / put_id`
4. `call_iv / put_iv`
5. `call_k / put_k`
6. `strike_price / expiry_date`
7. `entry_iv / last_valid_iv`
8. 持仓方向目前隐含了 `call=1 / put=-1` 的期权语义

对于 `BTC` 永续，这些概念要么不存在，要么应该替换成：

- `instrument_kind = perpetual`
- `side = long / short`
- `mark_price`
- `index_price`
- `best_bid / best_ask`
- `bid_size / ask_size`
- `funding_rate`
- `open_interest`
- `venue`
- `symbol`
- `contract_multiplier`

因此，**不能把当前 `opt_data` 原封不动当成 BTC 行情对象使用**。

### 6.3 推荐的抽象方式

推荐把当前的期权快照层抽象成“可交易标的快照”：

#### 统一的交易快照基类

- `instrument_kind`
- `symbol`
- `quote_ts`
- `last_price`
- `best_bid`
- `best_ask`
- `bid_size`
- `ask_size`
- `mark_price`
- `venue`
- `contract_id`
- `multiplier`

#### 针对期权的扩展字段

- `option_right`
- `strike`
- `expiry`
- `iv`
- `delta / gamma / vega / theta`

#### 针对 BTC 永续的扩展字段

- `funding_rate`
- `index_price`
- `open_interest`
- `basis`
- `liquidation_buffer`

这样做之后：

- OMS 的时序契约不需要因资产类型而重写
- 只需要更换交易载荷与执行适配层

### 6.4 当前结构扩展到 BTC 永续的最大障碍

最大的障碍不是 `ExecutionWindow`，而是**OMS 和 SE 内部大量默认“交易对象就是一对 call/put 期权”**。

目前明显存在这些硬编码：

1. 根据 `dir == 1 / -1` 决定 `call` 还是 `put`
2. 根据 `call_* / put_*` 取 bid/ask/price/id/iv
3. 持仓状态中内置了 `strike_price / expiry_date / entry_iv`
4. 部分 fair price / ROI / quote backfill 逻辑默认使用期权价格语义

这意味着：

- **时序框架可复用**
- **执行载荷与状态层必须做资产类型抽象**

### 6.5 如果后续一定要接 BTC 永续，建议怎么演进

建议采用两层拆法：

#### 第一层：先把结构做资产无关化

目标：

- 把“分钟决策窗口”和“交易标的快照”分开

任务：

1. `AlphaFrameItem.opt_data` 改名或替换为 `decision_quote`
2. `decision_quote` 不再内含 `call_* / put_*`
3. 把 `side`、`instrument_kind`、`venue` 纳入统一字段

#### 第二层：再加资产特定扩展

目标：

- 保持统一主流程，同时给期权和 BTC 永续各自保留扩展字段

任务：

1. 定义 `OptionQuoteSnapshot`
2. 定义 `PerpQuoteSnapshot`
3. OMS 只依赖公共接口
4. 资产特定逻辑放到执行适配层和风控适配层

#### 第三层：再接策略与风险规则

目标：

- 避免把期权世界的持仓、风险、成交假设直接挪到永续合约

需新增：

1. funding / mark / index 相关风控
2. 杠杆、保证金、强平缓冲语义
3. 多空持仓与仓位价值计算
4. 交易所级别的撮合与订单状态差异

---

## 7. 推荐的最终目标形态

一个更稳的最终结构，应该是：

1. `ExecutionWindow`
   - 只负责分钟窗口与秒级执行窗口语义
2. `AlphaFrame`
   - 只负责分钟冻结事实
3. `AlphaFrameItem`
   - 只负责单标的分钟级决策事实
4. `DecisionQuoteSnapshot`
   - 只负责策略做决定时所见的可交易快照
5. `ExecutionQuote1s`
   - 只负责秒级执行跟踪
6. `PositionSnapshot`
   - 只负责持仓语义
7. `InstrumentTraits`
   - 只负责期权 / 永续 / 股票等资产类型差异

如果做到这一步：

- 美股期权与 BTC 永续可以共用同一套时序主骨架
- 资产差异只留在快照扩展字段与执行适配层
- replay / realtime / dry / live 更容易长期保持语义一致

---

## 8. 下一步建议

最推荐的下一步不是直接大改代码，而是先做两件事：

1. 产出一张**字段级审计表**
   - 标记每个字段是强约束、弱约束、建议新增、建议删除
2. 设计一版**资产无关的 `DecisionQuoteSnapshot` 草案**
   - 同时兼容期权和 BTC 永续

这样后面无论先做“防漂移收敛”，还是先做“多资产扩展”，都不会返工。
