# Domain Adapter 真实 Payload 采样演练方案

日期：`2026-04-23`

范围：

- `production/scripts/domain_adapter_dryrun.py`
- `production/baseline/signal_engine_v8.py`
- `production/baseline/execution_engine_v8.py`
- `production/baseline/orchestrator_state_manager.py`
- `production/scripts/oms_state_recovery.py`

目的：

- 定义如何从**当前主线真实运行链路**中采样 `ALPHA_FRAME / quote / state`
- 明确这些样本分别从哪里抓、抓成什么 JSON 形状
- 说明如何批量送入 `domain_adapter_dryrun.py` 做旁路验证
- 在**完全不接入主线**的前提下，提前暴露 schema 漏洞、字段漂移和时间语义问题

---

## 1. 总原则

这次演练遵循 3 个原则：

1. **只读采样**
   - 不修改 Redis 流、不修改 PG 状态、不影响 OMS / SE 正常运行
2. **旁路转换**
   - 只把旧 payload 导出成 JSON，再交给 `domain_adapter_dryrun.py`
3. **先采样，后批跑**
   - 先确认样本来源正确，再做批量 dryrun 和错误统计

---

## 2. 当前主线里 3 类目标样本

### 2.1 `ALPHA_FRAME`

用途：

- 构造 `AlphaFrame`
- 同时覆盖 `AlphaFrameItem` 和 `DecisionQuoteSnapshot`

主来源：

- Redis Stream: `orch_trade_signals`

代码来源：

- `production/baseline/signal_engine_v8.py`
- `_publish_alpha_frame(...)` 中通过：
  - `self.r.xadd('orch_trade_signals', {'data': ser.pack(payload)}, maxlen=5000)`

实际 payload 关键字段：

- `action = ALPHA_FRAME`
- `ts`
- `frame_id`
- `items`
- `index_trend`
- `global_regime_band`
- `is_zombie_market`

结论：

- **这是最核心、最优先采样的对象**

---

### 2.2 秒级 execution quote

用途：

- 构造 `ExecutionQuote1s`
- 进一步构造 `ExecutionWindow`

主来源有两种：

#### 路径 A：直接采 fused market 输入

来源：

- `fused_market_stream`
- 或 replay 里构造的 fused batch payload

优点：

- 更贴近上游真实输入
- 适合回放 / 批量复盘

缺点：

- 需要额外做一次“按 symbol 拆 quote”的转换

#### 路径 B：采 OMS 进程内缓存 `latest_execution_quote_by_symbol`

来源：

- `production/baseline/execution_engine_v8.py`
- `_cache_execution_market_packet(...)`

缓存字典形状：

- `self.latest_execution_quote_by_symbol[sym] = { ... }`
- 关键字段：
  - `ts`
  - `wall_ts`
  - `stock_price`
  - `call_price`
  - `put_price`
  - `call_bid`
  - `call_ask`
  - `put_bid`
  - `put_ask`

优点：

- 与 OMS 实际消费后的 execution quote 更接近
- 可直接喂给当前 adapter

缺点：

- 这是进程内缓存，不是天然持久化对象
- 需要增加一个只读导出脚本或调试钩子才能方便抓取

结论：

- **短期演练建议优先用 replay/fused payload 导出**
- **中期如果要看 OMS 真正消费后的 execution 视图，再补进程内只读导出**

---

### 2.3 state / position snapshot

用途：

- 构造 `PositionSnapshot`

主来源：

- PostgreSQL 表：`symbol_state`

代码来源：

- `production/baseline/orchestrator_state_manager.py`
- `_load_state_from_db()`
- `save_state()`

当前权威读取口径：

- `namespace = OMS_STATE_NAMESPACE`
- 表：`symbol_state`
- 关键字段位于 `data` JSON 中

关键持仓字段：

- `symbol`
- `position`
- `qty`
- `entry_price`
- `entry_ts`
- `contract_id`
- `strike_price`
- `expiry_date`
- `opt_type`
- `entry_iv`
- `last_valid_iv`
- `open_fill_confirmed`

结论：

- **state 样本最适合直接从 PG 导出**
- 因为它本来就是 OMS 恢复口径的权威快照

---

## 3. 建议的采样顺序

### 阶段 A：先只采 `ALPHA_FRAME`

目标：

- 先验证 `AlphaFrame / AlphaFrameItem / DecisionQuoteSnapshot` 是否能稳定吃下真实主线 payload

原因：

- 这是 Domain 结构里最核心、最复杂的一层
- 也是后面 strategy / minute 漂移问题的主轴

建议样本量：

- 先抓 `20 ~ 50` 条真实 `ALPHA_FRAME`

关注点：

- `alpha_label_ts`
- `alpha_available_ts`
- `items[*].opt_data`
- `frame_id`
- `event_prob`
- `correction_mode`

---

### 阶段 B：再补秒级 quote 样本

目标：

- 验证 `ExecutionWindow` 的秒级轨迹约束

建议样本量：

- 每个 `ALPHA_FRAME` 配 `5 ~ 20` 条同分钟 quote

关注点：

- `ts` 是否落在 `[minute_ts, minute_ts + 60)`
- `best_ask >= best_bid`
- `call/put` quote 是否能正确映射

---

### 阶段 C：最后补 state 样本

目标：

- 验证 `PositionSnapshot` 在主线旧语义下的转换是否正确

重点关注：

- `position = -1` 的期权仓位不能被错映射成 `SHORT`
- `put` 仓位应保留为 long option position + `option_right=put`

---

## 4. 真实样本从哪里抓

## 4.1 抓 `ALPHA_FRAME`

### 推荐来源

- Redis Stream: `orch_trade_signals`

### 识别规则

只保留：

- `action == "ALPHA_FRAME"`

### 数据格式

流字段不是纯 JSON，而是：

- `data = ser.pack(payload)`

因此导出时要先：

1. 读 stream 条目
2. 取 `data`
3. 调 `production/baseline/utils/serialization_utils.py` 里的 `ser.unpack(...)`
4. 再把解出的 dict 写成 JSON

### 建议抓法

优先抓：

- 最新 `N` 条 `ALPHA_FRAME`
- 再按时间分桶补几个特殊时段：
  - 开盘前后
  - 整点 / 半点附近
  - 高频波动时段
  - 收盘前

---

## 4.2 抓 quote 样本

### 方案 1：从 fused market/replay batch 导出

适合：

- replay
- 离线批量对比
- 不想碰 OMS 进程内缓存时

建议导出后的 JSON 形状：

```json
{
  "NVDA": {
    "ts": 1710000061.0,
    "call_price": 10.1,
    "call_bid": 10.0,
    "call_ask": 10.2
  }
}
```

这是当前 `execution_window_from_legacy(...)` 最容易直接消费的形式。

### 方案 2：从 OMS 内部 `latest_execution_quote_by_symbol` 导出

适合：

- 想验证 OMS 实际已经缓存好的 execution quote 口径

位置：

- `production/baseline/execution_engine_v8.py`
- `self.latest_execution_quote_by_symbol`

建议：

- 后续可单独增加一个只读调试导出脚本
- 当前先不改主线

---

## 4.3 抓 state 样本

### 推荐来源

- PostgreSQL `symbol_state`

### 过滤规则

- `namespace = OMS_STATE_NAMESPACE`
- 先排除 `_GLOBAL_STATE_`
- 再优先采：
  - `position != 0`
  - `open_fill_confirmed = true/false` 两类都留样

### 推荐抓法

先导出两组：

1. 当前有持仓的 state
2. 当日空仓但带历史字段的 state

这样既能测正常映射，也能测边界恢复语义。

---

## 5. 建议的采样文件组织

建议在工作区外或临时目录组织成这样：

```text
samples/
  20260423/
    alpha/
      alpha_frame_001.json
      alpha_frame_002.json
    quotes/
      frame_001_quotes.json
      frame_002_quotes.json
    state/
      state_open_001.json
      state_flat_001.json
```

命名建议：

- `alpha_frame_<frame_id>.json`
- `quotes_<frame_id>.json`
- `state_<symbol>.json`

这样最利于后续批跑和错误追踪。

---

## 6. 如何单条演练

当前已有脚本：

- `production/scripts/domain_adapter_dryrun.py`

单条运行方式：

```bash
python3 production/scripts/domain_adapter_dryrun.py \
  --alpha-frame /path/to/alpha_frame.json \
  --quotes /path/to/quotes.json \
  --state /path/to/state.json \
  --strict
```

输出重点看：

1. `frame validation errors`
2. `window validation errors`
3. `position validation errors`
4. `Sample Item`

判读建议：

- 先追 `frame` 错误
- 再追 `window`
- 最后追 `position`

因为大多数结构漂移问题，最早会在 `AlphaFrameItem` 层暴露。

---

## 7. 如何批量跑

建议分三轮：

### 第一轮：只跑 `ALPHA_FRAME`

命令模式：

```bash
for f in samples/20260423/alpha/*.json; do
  python3 production/scripts/domain_adapter_dryrun.py \
    --alpha-frame "$f" \
    --strict || break
done
```

目标：

- 先确认 frame 结构本身稳定

### 第二轮：补 quotes

命令模式：

```bash
python3 production/scripts/domain_adapter_dryrun.py \
  --alpha-frame samples/20260423/alpha/alpha_frame_xxx.json \
  --quotes samples/20260423/quotes/quotes_xxx.json \
  --strict
```

目标：

- 看 `ExecutionWindow` 是否出现时间错位、book crossed、symbol 对不上

### 第三轮：补 state

命令模式：

```bash
python3 production/scripts/domain_adapter_dryrun.py \
  --alpha-frame samples/20260423/alpha/alpha_frame_xxx.json \
  --state samples/20260423/state/state_xxx.json \
  --strict
```

目标：

- 看 `PositionSnapshot` 是否稳定保留旧期权语义

---

## 8. 每轮演练的验收标准

### A. `ALPHA_FRAME` 验收

要求：

- `frame_id` 可读
- `minute_ts > 0`
- `alpha_label_ts == minute_ts - 60`
- `alpha_available_ts == minute_ts`
- `items[*].symbol` 无重复
- `decision_quote.quote_ts <= alpha_available_ts`

### B. quote 验收

要求：

- `quotes_1s[*].ts` 落在当前分钟窗口
- `best_ask >= best_bid`
- `call/put` 侧别映射正确

### C. state 验收

要求：

- 期权 `position=-1` 不被映射成 `SHORT`
- `opt_type=put` 保留为 `option_right=put`
- `quantity / entry_price / entry_ts` 恢复正常

---

## 9. 推荐先采哪些真实样本

优先级从高到低：

1. **开盘后前 10 分钟**
   - 最容易出现 warmup / label 边界问题
2. **有真实持仓的 frame**
   - 最容易看出 `decision_quote` 与 state 追踪是否一致
3. **波动放大时段**
   - 最容易暴露 quote 侧别和时间错位
4. **收盘前 30 分钟**
   - 最容易暴露 stale quote / EOD 风险边界

---

## 10. 当前最推荐的执行路线

按投入产出比，建议按这个顺序：

1. 先做 **`ALPHA_FRAME` 导出**
2. 用 `domain_adapter_dryrun.py` 单条跑通
3. 再做一批 `20~50` 条批跑
4. 统计最常见错误类型
5. 再决定是否值得补 quote/state 批量导出脚本

原因：

- `ALPHA_FRAME` 是 schema 风险最高的层
- 也是最容易直接从 Redis 流采的层
- 一旦这层不稳，后面 quote/state 演练价值会下降

---

## 11. 下一步最自然的实现

如果继续推进，下一步最值得做的是新增一个**只读样本导出脚本**，功能大致是：

1. 从 `orch_trade_signals` 拉最新 `N` 条
2. 自动筛出 `ALPHA_FRAME`
3. 调 `ser.unpack(...)`
4. 落成 JSON 文件
5. 可选从 PG 导出 `symbol_state`
6. 可选直接调用 `domain_adapter_dryrun.py`

这样就能把：

- 采样
- 转换
- 校验

串成一条真正可重复执行的旁路链路。
