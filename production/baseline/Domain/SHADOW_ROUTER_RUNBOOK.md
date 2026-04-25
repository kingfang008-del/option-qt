# Domain Shadow Router Runbook

日期：`2026-04-23`

范围：

- `production/baseline/Domain/shadow_router.py`
- `production/baseline/Domain/adapters.py`
- `production/baseline/Domain/contracts.py`

目的：

- 说明如何在**不影响主线逻辑**的前提下启用 `Domain Shadow Router`
- 说明启用后会校验哪些对象
- 说明如何看日志、如何落样本、如何安全关闭

---

## 1. 这是什么

`Domain Shadow Router` 是一条**旁路校验链路**。

它的工作方式是：

1. 主线继续按原来的逻辑运行
2. 在主线边缘把旧 payload 复制一份
3. 送进 `Domain` adapter 做转换与 `validate`
4. 只记录日志和可选样本，不回写任何主线状态

所以它的定位不是替代主线，而是：

- 提前发现 schema 漂移
- 提前发现旧 payload 和新 Domain 模型的语义不一致
- 为后续正式并线提供真实运行数据基础

---

## 2. 当前已经挂在哪些位置

目前影子路由已经挂在 3 条主线边缘：

### 2.1 `ALPHA_FRAME`

位置：

- `production/baseline/signal_engine_v8.py`
- `_publish_alpha_frame(...)`

旁路动作：

- 旧 `ALPHA_FRAME` payload -> `AlphaFrame`
- 调 `validate()`

### 2.2 秒级 execution quote

位置：

- `production/baseline/execution_engine_v8.py`
- `_cache_execution_market_packet(...)`

旁路动作：

- execution quote dict -> `ExecutionQuote1s`
- 调 `validate()`

### 2.3 position/state snapshot

位置：

- `production/baseline/orchestrator_state_manager.py`
- `save_state()`

旁路动作：

- 旧 state row -> `PositionSnapshot`
- 调 `validate()`

---

## 3. 安全性保证

这套影子路由遵循以下安全约束：

1. **默认关闭**
2. **不改主线逻辑**
3. **不阻断主链路**
4. **所有异常都吞掉，只记日志**
5. **只读样本落盘**

也就是说：

- 不会影响下单
- 不会影响 alpha 发布
- 不会影响 save_state
- 不会影响 Redis / PG 的主线写入

---

## 4. 如何开启

最小开启方式：

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=1
```

然后按原方式启动你的主程序即可。

例如：

```bash
DOMAIN_SHADOW_ROUTER_ENABLE=1 python production/baseline/run_live_signal.py
```

或者：

```bash
DOMAIN_SHADOW_ROUTER_ENABLE=1 python production/baseline/run_live_exec.py
```

如果你用的是自己已有的启动脚本，也只需要把这个环境变量带进去。

---

## 5. 如何开启样本落盘

如果只想看日志，不需要任何额外配置。

如果还想把经过 shadow router 的原始 payload 和转换后对象都落盘：

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=1
export DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS=1
export DOMAIN_SHADOW_ROUTER_DUMP_DIR=/tmp/domain-shadow
```

说明：

- `DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS=1`
  - 开启样本落盘
- `DOMAIN_SHADOW_ROUTER_DUMP_DIR`
  - 指定样本输出目录

落盘目录结构大致是：

```text
/tmp/domain-shadow/
  alpha_frame/
  execution_quote/
  position_state/
```

每条样本会保存成一个 JSON，里面包含：

- `raw_payload`
- `converted_payload`
- `errors`
- `identity`

---

## 6. 如何控制成功日志频率

默认情况下，成功样本不是每条都打日志，而是做节流。

默认值：

```bash
DOMAIN_SHADOW_ROUTER_LOG_OK_EVERY=50
```

意思是：

- 第 1 条成功样本会打日志
- 之后每 50 条成功样本打一次

如果想更频繁一点，比如每 10 条成功样本打一条：

```bash
export DOMAIN_SHADOW_ROUTER_LOG_OK_EVERY=10
```

---

## 7. 日志里看什么

日志统一会带：

```text
[DomainShadow]
```

主要有两类：

### 7.1 成功日志

示例形态：

```text
[DomainShadow] alpha_frame ok identity=frame-123 ok=50 error=0 detail=minute_ts=1710000060 items=38
```

看点：

- `kind`
  - `alpha_frame`
  - `execution_quote`
  - `position_state`
- `identity`
  - 哪条样本
- `ok / error`
  - 当前累计统计
- `detail`
  - 关键摘要

### 7.2 失败日志

示例形态：

```text
[DomainShadow] execution_quote invalid identity=NVDA_1710000061 errors=1 detail=ts=1710000061.000 bid=10.3000 ask=10.1000 source=unknown first=best_ask must be >= best_bid
```

看点：

- `invalid`
  - 表示 Domain validate 未通过
- `errors`
  - 错误数量
- `first`
  - 第一条错误信息

### 7.3 路由异常日志

示例形态：

```text
[DomainShadow] alpha_frame route failed: ...
```

这表示：

- 旁路自己出错了
- 但主线仍会继续跑

如果看到这类日志，优先修 shadow router 或 adapter，本身不会说明主线出问题。

---

## 8. 首次启用建议步骤

建议按下面顺序启用：

### 第一步：只开日志，不落盘

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=1
unset DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS
unset DOMAIN_SHADOW_ROUTER_DUMP_DIR
```

观察目标：

- 主程序是否正常运行
- 日志里是否出现 `[DomainShadow]`
- 是否有大量 `route failed`

### 第二步：再开样本落盘

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=1
export DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS=1
export DOMAIN_SHADOW_ROUTER_DUMP_DIR=/tmp/domain-shadow
```

观察目标：

- 目录是否成功生成
- 三类样本是否都有输出
- 错误样本是否能对照 `raw_payload` / `converted_payload`

### 第三步：在盘中只跑小规模观察

建议先观察：

1. 开盘后前 10 分钟
2. 有持仓时段
3. 波动较大时段

这样最容易发现时间语义和旧 payload 偏差。

---

## 9. 关闭方式

最简单的关闭方式：

```bash
unset DOMAIN_SHADOW_ROUTER_ENABLE
unset DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS
unset DOMAIN_SHADOW_ROUTER_DUMP_DIR
unset DOMAIN_SHADOW_ROUTER_LOG_OK_EVERY
```

或者显式关闭：

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=0
```

关闭后：

- 主线回到完全无 shadow router 参与的状态
- 不再做旁路转换和 validate

---

## 10. 建议重点关注的错误

启用后，优先看这几类错误：

### A. `alpha_frame`

重点错误：

- `alpha_label_ts must equal minute_ts - 60`
- `alpha_available_ts must equal minute_ts`
- `decision_quote.quote_ts must be <= alpha_available_ts`
- `symbol duplicated`

说明：

- 这些通常是分钟时序契约问题

### B. `execution_quote`

重点错误：

- `best_ask must be >= best_bid`
- `ts must be > 0`
- 价格字段非数值或负数

说明：

- 这些通常是盘口数据或 quote 映射问题

### C. `position_state`

重点错误：

- `instrument does not support short positions`
- `open positions must have quantity > 0`
- `open positions must have avg_entry_price > 0`
- `open positions must have entry_ts > 0`

说明：

- 这些通常是 state 恢复边界或旧语义映射问题

---

## 11. 推荐的试运行姿势

如果你现在只是想“提前测试”，最推荐这样开：

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=1
export DOMAIN_SHADOW_ROUTER_LOG_OK_EVERY=20
```

先不要落盘，只看日志。

如果日志显示：

- `ok` 正常增长
- `invalid` 数量很少
- 没有 `route failed`

再切到：

```bash
export DOMAIN_SHADOW_ROUTER_ENABLE=1
export DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS=1
export DOMAIN_SHADOW_ROUTER_DUMP_DIR=/tmp/domain-shadow
export DOMAIN_SHADOW_ROUTER_LOG_OK_EVERY=20
```

这样就能把问题样本留下来，方便后面精修 schema 和 adapter。

---

## 12. 后续自然下一步

在这份 runbook 基础上，后续最自然的动作有两个：

1. 加一个小的 **stats / summary 导出接口**
   - 定时把 `ok / error` 计数打印或落盘
2. 加一个 **样本回放脚本**
   - 直接对 `DOMAIN_SHADOW_ROUTER_DUMP_DIR` 里的错误样本批量复跑 `domain_adapter_dryrun.py`

这样就能把：

- 运行时影子接入
- 样本沉淀
- 离线复盘

串成闭环。
