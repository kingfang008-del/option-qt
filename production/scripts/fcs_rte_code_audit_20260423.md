# FCS / RTE Code Audit

日期：`2026-04-23`  
范围：

- `production/baseline/DAO/feature_compute_service_v8.py`
- `production/baseline/DAO/realtime_feature_engine.py`
- `production/baseline/DAO/fcs_realtime_pipeline.py`
- `production/baseline/DAO/fcs_persistence_handler.py`
- `production/baseline/DAO/fcs_warmup_handler.py`
- `production/baseline/DAO/fcs_support_handler.py`

目的：

- 回头系统梳理本轮发现的问题与修复，防止遗漏
- 区分“逻辑纠错”和“实验/过渡代码”
- 为后续 replay / realtime_dry 对比提供统一检查清单

---

## 1. 总结结论

目前已经确认并修复的核心问题，不是单纯的“`5min` 和 `10min` 的正常差异”，而是几处会直接改变输入语义或破坏时间连续性的逻辑问题：

1. `5min` 股票慢特征与 `5min` 期权慢特征之前不在同一时间相位
2. `options_*` 历史序列会被“当前 snapshot”整段覆盖
3. `hour / day_of_week / minute` 曾出现错位或被错误缩放
4. 跨日 / 开盘前缓存没有完整清理，可能把上一交易日状态带入新会话
5. 导数特征 `options_iv_momentum / options_gamma_accel / options_iv_divergence` 之前存在写回顺序错误，并且重启后 lookback 队列不会恢复
6. 非 RTH 分钟仍可能进入 commit 链路
7. `option_gate_metrics` PG 分区缺失时会报错

这些问题里，前 5 条都足以让 replay 和 realtime_dry 出现“收益异常好/异常差/不稳定”的现象。

---

## 2. 已确认并修复的问题

### 2.1 `5min` 慢特征只使用已完成 bar

问题：

- `5min` 慢特征之前可能会吃到正在形成中的半根 `5min` bar
- 这会让 slow branch 的语义不稳定，实际更像“半实时慢特征”

修复：

- 在 [realtime_feature_engine.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/realtime_feature_engine.py#L528) 新增 `_map_1m_targets_to_5m_buckets`
- 默认映射为“上一根已完成 `5min` bar”
- `10:11-10:14` 使用 `10:05`
- `10:15` 才切到 `10:10`

结论：

- 这是逻辑纠错，不是临时优化

---

### 2.2 `5min option snapshot` 上游已维护，但 RTE 之前没有真正使用

问题：

- FCS 上游已经维护：
  - `option_snapshot_5m`
  - `frozen_option_snapshot_5m`
- 但 RTE 在 `5min` 分支里直接把 `opts_bh_5m = opts_bh`
- 等于 `5min option` 慢特征实际仍在吃当前 `1min` snapshot

原问题位置：

- [realtime_feature_engine.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/realtime_feature_engine.py#L694)

修复：

- 新增 `_pad_option_snapshot(...)`
- `5min` 分支优先使用 `option_snapshot_5m`
- 没有 `option_snapshot_5m` 才回退到 `option_snapshots`

结论：

- 这是逻辑纠错
- 修复前，`5min stock` 和 `5min option` 属于两套不同相位

---

### 2.3 `options_*` 历史序列会被当前 snapshot 覆盖

问题：

- `compute_batch_features()` 先从 `prices_bh` 读历史特征
- 之后又把当前 `option_snapshot` 算出的 `options_*` 扩展成整段 `[B, L]`
- 结果原本已经写进历史序列的 `options_*`，会被当前值整段覆盖

问题位置：

- 历史读入：[realtime_feature_engine.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/realtime_feature_engine.py#L770)
- 覆盖发生：[realtime_feature_engine.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/realtime_feature_engine.py#L894)

修复：

- 改为 `_merge_with_fallback(...)`
- 只有历史缺失时才用当前 snapshot 补值

结论：

- 这是非常关键的逻辑修复
- 修复前会导致期权相关慢特征“看起来有历史，其实是一条水平线”

---

### 2.4 Calendar 特征修复

问题：

- `hour / day_of_week / minute` 曾出现错位
- 还可能被 normalizer 当连续实数缩放

修复：

- 在 [feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py#L89) 新增 `inject_calendar_features_into_raw_vec(...)`
- 在计算后重新强制注入 label time 对应的 NY calendar 值
- 在 [realtime_feature_engine.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/realtime_feature_engine.py#L120) 增加 `NO_STATS_SCALE_FEATURES`
- 在 [feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py#L151) 增加 `_enforce_categorical_identity_stats()`

结论：

- 这是逻辑纠错，不应回退

---

### 2.5 跨日 / 开盘前 stale cache 清理不完整

问题：

- 之前开盘前清理只处理了 `history_1min`
- 但正式计算在 `is_new_minute=True` 时实际读取：
  - `committed_history_1min`
  - `committed_history_5min`
  - `frozen_option_snapshot_5m`
  - `committed_latest_opt_buckets`
- 这样上一交易日数据仍可能污染开盘后分钟推理

修复：

- [fcs_realtime_pipeline.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_realtime_pipeline.py#L140)
- 开盘前统一清理：
  - `history_1min / committed_history_1min`
  - `history_5min / committed_history_5min`
  - `option_snapshot / option_snapshot_5m`
  - `frozen_* / committed_*`
  - `current_bars_5s`
  - `option_minute_agg`
  - `global_last_minute / committed_last_minute`

结论：

- 这是逻辑纠错

---

### 2.6 非 RTH 分钟进入 commit 链路

问题：

- 分钟 commit 之前没有统一拦 RTH
- replay 时会逐分钟扫过非交易时段

修复：

- [fcs_persistence_handler.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_persistence_handler.py#L55)
- `commit_ready_minutes()` 里直接跳过 `not profile.is_rth_minute(commit_dt)`

结论：

- 这是逻辑纠错

---

### 2.7 `option_gate_metrics` PG 分区缺失

问题：

- replay 带 `--mirror-pg` 时，`option_gate_metrics` 缺日分区会报错

修复：

- [fcs_support_handler.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_support_handler.py#L315)
- 新增 `ensure_option_gate_metrics_partition(...)`
- 插入前自动补齐日分区

结论：

- 这是基础设施修复，不影响模型语义，但会影响 replay 稳定性

---

### 2.8 导数特征三件套写回顺序错误

涉及特征：

- `options_iv_momentum`
- `options_gamma_accel`
- `options_iv_divergence`

问题：

- 这些特征是在 FCS 里由 `_inject_temporal_derivatives(...)` 计算
- 旧代码先把 `raw_vec` 写回 `history_1min`
- 然后才计算导数
- 结果历史列会落后一拍

修复：

- 现在顺序改为：
  1. `_fill_raw_vec_from_result(...)`
  2. `_inject_temporal_derivatives(...)`
  3. 再写回 `history_1min`
- 位置：[feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py#L1190)

结论：

- 这是逻辑纠错

---

### 2.9 导数特征三件套在 warmup / PG 恢复后不连续

问题：

- `deriv_history` 是内存队列
- 旧代码重启后不会从历史重建
- 前几分钟导数特征会重新从 0 附近起步

修复：

- 在 [feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py#L945) 新增 `_rebuild_deriv_history_from_history(...)`
- 在 warmup 后调用：
  - [fcs_warmup_handler.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_warmup_handler.py#L187)
  - [fcs_warmup_handler.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_warmup_handler.py#L267)
- 在 PG 状态恢复后调用：
  - [fcs_support_handler.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_support_handler.py#L481)
  - [fcs_support_handler.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/fcs_support_handler.py#L493)

结论：

- 这是逻辑纠错

---

### 2.10 导数特征在无效 IV 下不应污染 lookback 队列

问题：

- 如果当前 `options_vw_iv` 无效或价格无效，旧逻辑仍可能 append 进 `deriv_history`

修复：

- 在 [feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py#L996) 增加守卫：
  - `iv/gamma/price` 必须 finite
  - `price > 0`
  - `iv >= option_gate_min_iv`

结论：

- 这是逻辑纠错

---

## 3. 当前默认口径

### 3.1 Normalizer 刷新间隔

当前默认值：

- [feature_compute_service_v8.py](/Users/fangshuai/Documents/GitHub/option-qt/production/baseline/DAO/feature_compute_service_v8.py#L125)
- `FCS_NORMALIZER_STATS_UPDATE_INTERVAL = 5`

说明：

- `5` 和 `10` 都属于低频台阶更新
- 当前默认改回 `5`
- 仍保留环境变量覆盖能力

建议：

- 主线默认使用 `5`
- 对比实验再显式切 `10`

---

## 4. 已新增测试

### 4.1 `5min option` 相位与历史覆盖测试

文件：

- [test_realtime_feature_engine_option_phasing.py](/Users/fangshuai/Documents/GitHub/option-qt/production/tests/test_realtime_feature_engine_option_phasing.py)

覆盖：

1. `5min option_snapshot_5m` 是否真正生效
2. `options_*` 历史序列是否还会被当前 snapshot 覆盖

说明：

- 由于本机 `python3` / `python3.10` 环境依赖不完整，实际是用函数级方式验证
- 逻辑验证结果已通过

### 4.2 导数特征连续性测试

文件：

- [test_fcs_temporal_derivatives.py](/Users/fangshuai/Documents/GitHub/option-qt/production/tests/test_fcs_temporal_derivatives.py)

覆盖：

1. 从 `history_1min` 重建 `deriv_history`
2. 无效 IV 不污染队列
3. 当前导数值计算正确

结果：

- `3 passed`

---

## 5. 哪些代码属于“应保留的逻辑修复”

建议保留：

1. `5min` completed bucket 映射
2. `5min option_snapshot_5m` 真接入
3. `options_*` 历史序列只补缺、不覆盖
4. calendar repair
5. categorical/time no-scale
6. 跨日 stale cache 完整清理
7. 非 RTH commit 跳过
8. 导数特征写回顺序修复
9. warmup / PG restore 后重建 `deriv_history`

---

## 6. 哪些代码更像“实验 / 过渡代码”

这些代码建议后续统一清理或明确标注实验用途：

1. `runtime_payload_audit_*`
2. `minute_write_audit_*`
3. `feature_parity_*`
4. `TRACE-NVDA`
5. `DebugSlow-Guard`
6. `realtime_feature_engine.py` 里的 `_postprocess_5m_feature_tensor(...)`
7. `realtime_feature_engine_slowfilter.py`
8. `fcs_engine_adapter.py` 里的 `equity_options_slowfilter_v1`

说明：

- 这些不一定错误
- 但它们属于排障 / A/B / 实验基础设施
- 不应混淆为主线语义的一部分

---

## 7. 仍需继续观察的风险点

### 7.1 `options_gamma_accel` 配置语义仍需复核

现状：

- 在 `slow_feature.json` 中，`options_gamma_accel` 被标成 `5min`
- 但在 `fast_feature.json` / `feature_all.json` 中也存在
- 它本质上是 FCS 基于分钟序列导出来的特征，不是原生 `5min option snapshot` 特征

风险：

- 配置层面存在“快慢双重归属”的歧义

建议：

- 后续明确它到底应该只属于 `1min`，还是在 slow branch 中做广播使用

### 7.2 warmup 过程不会逐分钟重放导数链

现状：

- 当前 warmup 采用“从历史列重建 `deriv_history`”
- 这比冷启动从 0 开始更好
- 但它不是“逐分钟逐帧真实重放导数计算”

风险：

- 极端情况下，warmup 后最开始几分钟仍可能与连续运行有微小差异

建议：

- 若后续还发现开盘前几分钟异常，可再考虑做“导数链专用 warmup replay”

### 7.3 `stats_update_interval=5` 与 `10` 仍需用同一时间窗复测

现状：

- `5` 与 `10` 语义接近
- 但收益差异还需要同样数据窗口下复核

建议：

- 用同一天、同一批 alpha、同一套 OMS/FCS 代码分别跑：
  - `FCS_NORMALIZER_STATS_UPDATE_INTERVAL=5`
  - `FCS_NORMALIZER_STATS_UPDATE_INTERVAL=10`

---

## 8. 建议的下一步检查顺序

1. 先用当前代码重新跑一小段 replay
2. 检查 `debug_slow` / `history_1min` 中：
   - `options_iv_momentum`
   - `options_gamma_accel`
   - `options_iv_divergence`
   是否还会落后一拍
3. 对比 `FCS_NORMALIZER_STATS_UPDATE_INTERVAL=5` 与 `10`
4. 若收益仍与旧超级收益差异很大，再单独审查：
   - `options_gamma_accel` 的快慢分支归属
   - warmup 是否需要更像真实 replay

---

## 9. 本次文档对应的主要验证

- `python3 -m py_compile ...` 通过
- `git diff --check` 通过
- `pytest -q production/tests/test_fcs_temporal_derivatives.py` 通过
- `test_realtime_feature_engine_option_phasing.py` 已做函数级逻辑验证

---

## 10. 最终判断

当前主线最值得保留的是“逻辑正确性修复”，而不是去回退到旧的偶然高收益状态。

如果后续收益仍和旧版本有明显差距，优先怀疑的应该是：

1. 旧 alpha 本身建立在错误输入定义上
2. `options_gamma_accel` 等导数特征的 branch 归属还有配置歧义
3. warmup / replay 与连续运行之间仍有少量连续性差异

不建议为了追求旧收益，回退已经确认正确的语义修复。
