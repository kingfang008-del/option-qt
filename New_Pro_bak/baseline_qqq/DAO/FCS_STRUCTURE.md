# FCS/SE 结构说明（可扩展版）

本文档记录当前 `FeatureComputeService`（FCS）与 `SignalEngine`（SE）的分层结构，以及未来扩展到 BTC 期权时的接入方式。

## 1. 分层总览

- `feature_compute_service_v8.py`
  - 框架编排层（时间对齐、分钟冻结、归一化、payload 组装、发布）
  - 不再直接绑定单一市场实现，转为依赖 adapter/profile
- `fcs_realtime_pipeline.py`
  - 秒级数据接入与分钟边界推进
  - 时段规则由 `market_profile` 提供
- `fcs_persistence_handler.py`
  - 分钟 K 线与期权快照落地、原子提交、同步 ACK
- `fcs_support_handler.py`
  - 门控状态机、快照完整性、审计与状态持久化
- `fcs_engine_adapter.py`
  - 特征引擎适配器工厂（目前支持 `equity_options_v1` / `btc_options_v1`）
- `fcs_market_profile.py`
  - 市场规则抽象（时段、RTH 口径、warmup 门槛、非交易标的）

## 2. 关键扩展点

### 2.1 Feature Engine Adapter

- 环境变量：`FEATURE_ENGINE_ADAPTER`
- 当前值：
  - `equity_options_v1`：美股实现
  - `btc_options_v1`：BTC 骨架（当前回退到 equity 引擎，便于框架先跑通）
- 入口：`build_feature_engine_adapter(...)`
- 统一接口：`compute_all_inputs(**kwargs) -> Dict[str, Dict]`

### 2.2 Market Profile

- 环境变量：`MARKET_PROFILE`
- 当前值：
  - `equity_us`（默认）
  - `crypto_247`
- 核心能力：
  - `accept_realtime_tick(dt_ny)`
  - `should_flush_premarket(dt_ny, last_flush_date)`
  - `history_keep_mask(idx, dt_ny)`
  - `is_rth_minute(dt_ny)`
  - `count_effective_history(idx, label_floor)`
  - `calc_effective_minutes(start_dt, end_dt)`

## 3. Option Gate 配置入口

- 配置聚合函数：`config.get_option_gate_profile()`
- FCS 初始化只读取 profile，不再在服务内散落硬编码
- 支持环境变量覆盖：
  - `OPTION_GATE_MIN_PASS`
  - `OPTION_GATE_MAX_FAIL`
  - `OPTION_GATE_GRACE_MINUTES`
  - `OPTION_GATE_MIN_IV`
  - `OPTION_GATE_REQUIRE_FRAME_CONSISTENCY`

## 4. 当前数据流（简版）

1. 秒级行情进入 `FCSRealtimePipeline.process_market_data`
2. 分钟边界冻结快照并触发 `finalization_barrier`
3. FCS 调用 `engine_adapter.compute_all_inputs`
4. 组装分钟 payload（含 `live_options`、`vol_z_dict`、warmup 指标）
5. `atomic_commit_minute_payload` 发布至 `STREAM_INFERENCE`
6. `SignalEngine.process_batch` 消费并产出 alpha/信号

## 5. 接入 BTC 的最小步骤

1. 设置环境变量：
   - `MARKET_PROFILE=crypto_247`
   - `FEATURE_ENGINE_ADAPTER=btc_options_v1`
2. 在 `realtime_feature_engine_btc.py` 中逐步替换 fallback 逻辑：
   - BTC 合约解析
   - 24/7 时间特征
   - BTC 盘口/波动特征
3. 根据市场特性调整 gate/warmup 参数

## 6. 回归测试入口

分钟聚合与 gate 稳定性测试：

```bash
python3 production/scripts/run_minute_regression_tests.py
```

当前覆盖：
- 60 秒期权快照 -> 1 分钟聚合
- 60 秒 stock -> 1 分钟 OHLCV 聚合
- 3 分钟秒级 gate/valid 稳定性

