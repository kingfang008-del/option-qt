# “步进同步”高压仿真测试计划 (Step-Sync Simulation Test Plan)

本测试计划旨在利用 **Redis 帧同步锁 (`sync:orch_done`)**，在 `LIVEREPLAY` 模式下运行完整的生产环境异步逻辑（而非简化的同步逻辑），从而精准捕捉由于执性能、状态竞争或逻辑漏洞导致的“少开仓、不平仓”等异常。

---

## 🎯 测试目标
1.  **定位少开仓原因**：检查是否由于 `is_pending` 锁冲突、Alpha 传输延迟（导致动量守卫拦截）或盘口数据缺失引发。
2.  **定位不平仓原因**：检查是否由于平仓信号未触发、止损门槛偏移或异步发单超时导致。
3.  **性能基准评估**：测量生产环境异步逻辑处理每一帧数据（包含所有 `asyncio` 协程）的真实耗时。

---

## 🛠️ 测试环境配置

### 1. 配置 `config.py`
```python
ONLY_LOG_ALPHA = False
SYNC_EXECUTION = False   # 💡 关键：关闭同步模式，启用完整生产异步逻辑
IS_LIVEREPLAY = True     # 开启重放模式，使用 Redis 帧锁进行流量控制
DISABLE_ICEBERG = False  # 开启冰山拆单仿真
```

### 2. 启动组件顺序
1.  **数据驱动 (Driver)**：启动 `s2_run_realtime_replay_sqlite.py`。
    - 它会发出一帧数据到 `raw_stream`，然后立刻在 Redis 上阻塞等待 `sync:orch_done`。
2.  **特征服务 (Feature Service)**：启动 `feature_compute_service_v8.py`。
3.  **策略引擎 (Orchestrator)**：启动 `system_orchestrator_v8.py`。
    - **原理**：它处理完一帧并执行完所有协程后，才会 ACK 释放帧锁。这保证了无论处理耗时多久，下一帧都不会“溢出”或丢失，从而实现 100% 逻辑回放。

---

## 🔍 核心观察指标与定位方法

### 情况 A：回测开仓了，仿真/实盘没开仓
- **排查工具**：`StrategyCore` 诊断日志。
- **定位**：
  - 如果日志显示 `[Entry Reject] ... reason: SNAP_ROC` -> **延迟杀手**：虽然我们用了帧锁保证不丢数据，但在这一帧中，Alpha 对应的价格与当前模拟价格已产生偏差。
  - 如果日志显示 `[并发锁防线] ... is_pending=True` -> **逻辑锁死**：说明上一个任务（可能是撤单）卡住了，没有在 `finally` 中释放 `is_pending`。
  - 如果日志显示 `[定价非法] ... price=NaN` -> **数据断流**：期权订阅数据没有及时同步。

### 情况 B：回测平仓了，仿真/实盘没平仓
- **排查点**：
  - `SymbolState.update_indicators`：检查 `is_new_minute` 标志是否正确触发，导致移动止损（Trailing Stop）逻辑未被触达。
  - `_execute_exit` 的 Pricing 流程：检查 `bid/ask` 是否为 0。

### 性能监控 (Performance Profiling)
在 `V8Orchestrator.process_batch` 中手动注入计时器：
```python
start_t = time.perf_counter()
# ... 执行逻辑 ...
duration = (time.perf_counter() - start_t) * 1000
if duration > 100:
    logger.warning(f"⚠️ [性能告警] 处理 {sym} 帧耗时过长: {duration:.1f}ms")
```
- **阈值**：如果单帧（12 个标的）在 `LIVEREPLAY` 下总耗时超过 **800ms**，则在实盘 1s-Tick 下极度危险，必然会引发“协程饥饿”。

---

## 📅 测试阶段 (Phases)

1. **Phase 1: 完美回放** (禁用随机滑点，最小化干扰)
2. **Phase 2: 压力回放** (使用 1s 高频数据，观察处理链长度)
3. **Phase 3: 状态注入** (故意通过对账器人为修改内存状态，测试自愈逻辑)

---

> [!TIP]
> **结论**：如果在此模式下开仓次数恢复到回测水平（~100次），说明问题纯粹是**传输延迟**导致的；如果在此模式下依然只有 1-5 次，说明问题出在 **`LIVEREPLAY` 与 `backtest` 的代码路径分支差异**上。
