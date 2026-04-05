# 1s vs 1m Backtest Parity Audit & Fix Plan

## 1. 核心目标
解决秒级 (1s) 与分钟级 (1m) 回测结果的巨大差异，确保 Alpha 信号与成交逻辑在分钟边界达到 99% 以上的一致性。

## 2. 已发现的问题 (Local Audit 结论)
- **价格中断 (Price Gaps)**: 部分标的（如 META）在 1s 模式下记录的 Alpha 日志价格为 0.0，导致推理失效。
- **波动率异常 (Vol_Z Corruption)**: `vol_z` 出现 360 亿级别的离谱数值，直接摧毁 Alpha 稳定性。
- **状态频率不匹配 (Frequency Drift)**: `SymbolState.prices` 在秒级模式下累积了过多 Tick 数据，导致基于 `prices[-2]` 的 `pct_change` 计算逻辑与分钟级（上分钟收盘）完全脱节。

---

## 3. 测试与修复阶段 (Server Execution)

### 阶段 1：数据源完整性自检
**目标**: 确认服务器上的 1s Parquet 数据是否完整。
- **检查项**: 对比 `NVDA` 与 `META` 的文件大小。
- **命令**: 
  ```bash
  ls -lh /mnt/s990/data/raw_1s/stocks/META | head -n 5
  ls -lh /mnt/s990/data/raw_1s/stocks/NVDA | head -n 5
  ```

### 阶段 2：带有调试手段的 Alpha 生成 (1s)
**目标**: 捕捉 `vol_z` 崩溃的瞬间。
- **操作**: 已经在 `signal_engine_v8.py` 中注入了 `🚨 [VOL_DEBUG]` 打印。
- **命令**:
  ```bash
  PYTHONPATH=./baseline:./model:./DB:./history_replay \
  python3 production/preprocess/backtest/s2_run_realtime_replay_sqlite_1s.py \
  --start-date 20260102 --end-date 20260102 --turbo --skip-warmup
  ```
- **观察**: 如果屏幕刷出大量 `VOL_DEBUG`，说明 `FeatureComputeService` 或驱动传来的 `fast_vol` 数组索引错位或存在未处理的 0 值。

### 阶段 3：代码逻辑修复 (即将实施)
- **修复 A: 指标步长对齐**
  在 `SymbolState.update_tick_state` 中，不再无限制追加 `self.prices`，而是确保其保持与分钟级一致的“上一分钟收盘”引用。
- **修复 B: Vol_Z 防护**
  在 `_update_volatility_metrics` 中增强对 inputs 的校验，如果 `r_v` (fast_vol) 始终为 0，则退回到基于 `pct_change` 的本地计算，而非依赖已断流的特征输入。
- **修复 C: 1s 驱动补全**
  修改 `s2_run_realtime_replay_sqlite_1s.py`，确保即使某秒没有成交，也向特征引擎发送“Last Known State”，防止 `latest_prices` 掉入 0 值的坑。

### 阶段 4：自动化对齐验证
**目标**: 使用 `audit_parity_v8.py` 验证修复效果。
- **操作**:
  1. 重新运行 1s 生成。
  2. 运行 1m 生成。
  3. 执行对齐脚本：
     ```bash
     python3 production/tests/audit_parity_v8.py
     ```
- **预期**: Alpha MAE < 0.01, Price MAE < 0.001, Vol_Z MAE < 0.1。

---

## 4. 注意事项
> [!IMPORTANT]
> 执行前请确保服务器已安装 `zstandard`。
> 
> [!WARNING]
> 大盘趋势 (`index_trend`) 在 1s 模式下受开盘前几分钟的波动影响可能与 1m 存在相位差，计划在 RTH 开始后的第 5 分钟再进行严格比对。
