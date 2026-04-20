# Event-Driven 压测蓝图（回测贴近实盘）

目标：在不依赖“连续跑几天 paper”的情况下，快速评估系统健壮性与回测-实盘一致性。

## 1) 三层压测模型

- `L0 Deterministic`：同数据重复回放，检查结果稳定性（逻辑一致性）。
- `L1 Realistic Execution`：加入执行扰动（延迟、追价收紧），检查执行口径偏差。
- `L2 Chaos`：组合扰动（延迟+追价+潜在队列抖动），检查鲁棒性。

建议顺序：先 L0，再 L1，最后 L2。

---

## 2) 指标体系（可量化验收）

每个场景至少重复 `N=3~10` 次，关注：

- `alpha_rows`：alpha 产出规模（是否漏算）。
- `trade_close`：平仓笔数（行为一致性）。
- `pnl_close_sum`：已平仓 PnL（结果一致性）。
- `success_ratio`：脚本成功率（可用性）。
- `CV`（变异系数）：
  - `trade_close_cv`
  - `pnl_cv`

建议阈值（可按策略再细化）：

- `success_ratio >= 0.99`
- `trade_close_cv <= 0.01`
- `pnl_cv <= 0.05`（L0 建议更严 <= 0.02）
- 相对 baseline 的漂移：
  - `|close_count_drift_vs_baseline_pct| <= 2%`
  - `|pnl_drift_vs_baseline_pct| <= 10%`（L1/L2 可放宽）

---

## 3) 已落地脚手架

脚本：`production/scripts/run_event_stress_harness.py`

能力：

- 多场景、多轮 replay 自动执行。
- 每轮结束自动采集 SQLite 指标（`alpha_logs/trade_logs`）。
- 自动输出：
  - `runs.csv`（逐轮明细）
  - `aggregate.csv`（场景聚合）
  - `report.json`（完整报告）

默认场景：

- `baseline`
- `delayed_exec`
- `requote_tight`
- `stress_combo`

---

## 4) 推荐运行命令

```bash
python production/scripts/run_event_stress_harness.py \
  --start-date 20260310 \
  --end-date 20260312 \
  --runs 5 \
  --skip-warmup \
  --enable-oms \
  --scenario baseline \
  --scenario delayed_exec \
  --scenario stress_combo
```

---

## 5) 下一步增强（建议）

- 引入统一事件信封（`event_id/frame_id/source/event_ts/recv_ts/payload_hash`）。
- 在 replay driver 中加入可控故障注入：
  - 乱序概率
  - 丢包概率
  - 重复包概率
  - 抖动延迟分布
- 增加 PG 侧指标校验（`alpha_logs_* / trade_logs_*` 分区对比）。
- 将阈值校验接入 CI（不达标直接 fail）。

