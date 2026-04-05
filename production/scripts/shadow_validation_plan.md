# 影子系统校验执行计划 (Shadow System Validation Plan)

本计划提供了执行“影子系统”校验的标准步骤，以确保生产引擎 (`V8Orchestrator`) 的运行逻辑与离线研究结果完全匹配。

---

## 🟢 第一阶段：Alpha 信号一致性对比 (Alpha Consistency)
**目的**：确保实时流式计算出的 Alpha 值与离线回测中使用的 Alpha 值完全一致。

1. **配置影子模式**:
   在 `production/baseline/config.py` 中设置：
   ```python
   ONLY_LOG_ALPHA = True
   ```

2. **执行实时回放**:
   ```bash
   RUN_MODE=LIVEREPLAY python production/preprocess/backtest/s2_run_realtime_replay_sqlite.py --symbol NVDA --date 2026-03-02
   ```

3. **运行对比工具**:
   使用 `verify_alpha_consistency.py` 检查生成的 `market_*.db` 与离线特征库的差异：
   ```bash
   python production/scripts/verify_alpha_consistency.py \
       --ref data/history_sqlite/db_all_features_20260302.sqlite \
       --shadow data/history_sqlite/market_20260302.db
   ```
   *   **成功标准**：相关性 (Correlation) > 0.99, 均方根误差 (RMSE) < 0.05。

---

## 🟢 第二阶段：策略逻辑一致性校验 (Playback Verification)
**目的**：排除数据差异，单纯验证 **生产环境代码** 与 **离线代码** 的交易决策逻辑是否一致。

1. **配置同步执行模式**:
   在 `production/baseline/config.py` 中设置：
   ```python
   ONLY_LOG_ALPHA = False
   DISABLE_ICEBERG = True
   SYNC_EXECUTION = True  # 💡 关键：开启同步执行以消除异步随机性
   ```

2. **启动回放驱动脚本**:
   ```bash
   python production/scripts/verify_strategy_playback_sqlite.py --date 20260302
   ```

3. **启动 Orchestrator 核心**:
   （在另一个终端运行）
   ```bash
   RUN_MODE=LIVEREPLAY python production/baseline/system_orchestrator_v8.py --mode backtest --stream verification_stream
   ```

4. **分析报告**:
   检查 `logs/playback_equity_*.csv`。其交易时间点、入场价和盈亏应与离线 S4 回测结果 **完全重合**。

---

## 🟢 第三阶段：全系统实战模拟 (Full System Trade Validation)
**目的**：在不开启同步模式、开启流动性模拟的情况下，验证系统整体在该交易日的最终表现。

1. **配置模拟环境**:
   在 `production/baseline/config.py` 中设置：
   ```python
   ONLY_LOG_ALPHA = False
   DISABLE_ICEBERG = False # 恢复为真实模拟设置
   SYNC_EXECUTION = False  # 恢复为异步实时逻辑
   ```

2. **运行全量回放**:
   ```bash
   RUN_MODE=LIVEREPLAY python production/preprocess/backtest/s2_run_realtime_replay_sqlite.py --symbol NVDA --date 2026-03-02
   ```

3. **对比最终交易单**:
   ```bash
   python production/scripts/verify_trade_consistency.py \
       --ref /path/to/offline/trades_ref.csv \
       --sha /path/to/current/replay_trades.db
   ```

---

## 🔴 后续计划任务 (Next Steps)
- [ ] 自动化脚本：编写一个 Shell 脚本一键执行上述三个阶段。
- [ ] PnL 监控：在 Dashboard 中增加离线 vs 在线 PnL 实时对比曲线。
- [ ] 告警集成：当 Alpha 相关性低于 0.98 时，通过飞书/钉钉发送自动告警。

---

> [!IMPORTANT]
> 每次执行完校验任务后，请务必还原 `config.py` 中的 `SYNC_EXECUTION` 和 `ONLY_LOG_ALPHA` 设置，以免影响真实实盘交易。
