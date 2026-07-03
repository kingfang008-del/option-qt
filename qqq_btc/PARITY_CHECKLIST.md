# G3 Shadow Parity 核对表

> **用途**: REALTIME_DRY 影子模式期间,逐日/逐周核对 live 与 strict replay 是否同口径。  
> **前置**: **G2 已通过** — 必须先有 `run_replay` 报告 + `export_replay_trades.py` 基准,再跑 G3。  
> **时序图**: [ARCHITECTURE.md §2.6](./ARCHITECTURE.md#26-端到端时序bar-close--fill)  
> **命令索引**: [EXECUTION_PLAN.md §阶段 3](./EXECUTION_PLAN.md)

---

## 1. 启动前一次性检查

| # | 项 | 命令/动作 | 通过标准 |
|---|-----|-----------|----------|
| 1 | G2 基准已存档 | `run_replay.py` + 报告 JSON | fill PnL>0; 去 best-2-day 仍为正 |
| 2 | replay trades 导出 | 见下方「基准文件」 | 含 `exit_reason` 列 |
| 3 | 环境变量 | 三进程均设 `QQQ_BTC_LIVE=1` | bootstrap patch 生效 |
| 4 | tick exit 模式 | **不要**设 `QQQ_BTC_TICK_EXITS=legacy` | disaster_only |
| 5 | checkpoint | Signal 与 G2 同 ckpt | fail-fast 无 silent fallback |
| 6 | shadow 目录 | `~/quant_project/shadow/` 可写 | fill_audit.csv 持续追加 |

### 基准文件( G2 出口生成,shadow 全程只读 )

```bash
# strict replay 基准 trades(含 exit_reason)
python qqq_btc/tools/export_replay_trades.py \
  --parquet /path/to/qqq_infer.parquet \
  --output ~/quant_project/shadow/replay_trades_baseline.csv

# 可选:存档 G2 summary
cp /path/to/replay_summary.json ~/quant_project/shadow/g2_baseline.json
```

### 双引擎启动

```bash
# 终端 1 — FCS(沿用原启动)
# 终端 2 — Signal
cd New_Pro/baseline_qqq
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_signal_qqq.py \
  --checkpoint ~/quant_project/checkpoints_qqq_net_edge_v2/best.pth

# 终端 3 — OMS
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_exec_qqq.py
```

---

## 2. 每日核对(每个 RTH 交易日)

日期: `__________`  操作人: `__________`

### 2.1 进程健康(开盘前 / 盘中)

| # | 检查项 | 方法 | ✓ | 备注 |
|---|--------|------|---|------|
| D1 | FCS 有 bar 输出 | Dashboard / Redis `unified_inference_stream` | ☐ | |
| D2 | SE 有 ALPHA_FRAME | Redis `orch_trade_signals` 或 SE 日志 | ☐ | 连续 N bar 无 frame → 告警 |
| D3 | OMS 消费 frame | OMS 日志无长期 stall | ☐ | |
| D4 | 无 CRITICAL 手工干预 | 日志 / Dashboard | ☐ | slow_feature 一致; ckpt fail-fast |
| D5 | REALTIME_DRY | 无真实下单或 mock 模式确认 | ☐ | |

### 2.2 收盘后 parity 三条

```bash
SHADOW=~/quant_project/shadow
DATE=2026-07-03   # 替换为当日

# 1) 特征 parity(需当日 offline parquet + live FCS dump)
python qqq_btc/tools/parity_audit.py feature \
  --offline /path/to/offline/QQQ_${DATE}.parquet \
  --live ${SHADOW}/fcs_bars_${DATE}.parquet \
  --output ${SHADOW}/parity/feature_${DATE}.json

# 2) fill 点差位
python qqq_btc/tools/parity_audit.py fill \
  --audit-log ${SHADOW}/fill_audit.csv \
  --output ${SHADOW}/parity/fill_cumulative.json

# 3) exit reason 分布 vs G2 基准
python qqq_btc/tools/parity_audit.py exits \
  --audit-log ${SHADOW}/fill_audit.csv \
  --replay-trades ${SHADOW}/replay_trades_baseline.csv \
  --output ${SHADOW}/parity/exits_cumulative.json
```

| 指标 | 阈值 | 当日值 | ✓ |
|------|------|--------|---|
| feature `pass_rate` | > **0.95** | | ☐ |
| fill `median` spread_frac | **0.75 – 0.80**(目标 0.775) | | ☐ |
| exit `distribution_l1` | ≤ **0.35** | | ☐ |

### 2.3 异常记录(当日)

| 时间 | 现象 | 可能原因 | 处理 |
|------|------|----------|------|
| | | | |

---

## 3. 每周汇总(第 1/2 周 shadow)

周次: `第 __ 周`  区间: `____` – `____`

| 指标 | 阈值 | 本周累计/中位 | ✓ |
|------|------|---------------|---|
| 有效交易日数 | ≥ 5 | | ☐ |
| feature pass_rate | 每日 >0.95 | | ☐ |
| fill median | 0.75–0.80 | | ☐ |
| exit L1 | ≤ 0.35 | | ☐ |
| ghost position | 0 | | ☐ |
| state 漂移 | 无未解释偏差 | | ☐ |
| disaster_stop 触发 | 记录次数与 replay 量级可比 | | ☐ |

**G3 出口定义**(Phase 0): 上述指标 **连续 5 个交易日** 达标,且 replay→live 无未解释偏差。

---

## 4. 挂掉排查速查

| 症状 | 最可能原因 | 动作 |
|------|------------|------|
| feature pass_rate 低 | FCS 与 offline norm 统计量不一致; time/trend 未 enrich | 核对 rolling_norm artifact; 确认 `fcs_adapter` 生效 |
| fill median 偏离 >0.03 | OMS 未 patch 0.775; 仍走 legacy 0.20–0.45 | 确认 `QQQ_BTC_LIVE=1`; 查 `oms_integration` 日志 |
| exit L1 超标 | legacy 秒级阶梯/FLASH 污染 max_roi | 禁用 `QQQ_BTC_TICK_EXITS=legacy`; 仅 disaster |
| live 多入场 / replay 少 | entry_delay 不对齐; SESSION 窗口不一致 | 对拍 gate_trace; 核对 `ReplayConfig.entry_delay_bars` |
| 连续无 ALPHA_FRAME | SE crash / checkpoint 路径错 / GPU OOM | SE 日志; fail-fast 重启 |
| fill_audit 无 CLOSE 行 | shadow 未成交或 audit writer 未 patch | 确认 `fill_audit_writer` patch |

---

## 5. 归档清单(G3 通过后)

- [ ] `g2_baseline.json` — G2 replay summary
- [ ] `replay_trades_baseline.csv` — exit reason 基准
- [ ] `shadow/parity/feature_*.json` — 每日特征对拍
- [ ] `shadow/parity/fill_cumulative.json` — fill 累计
- [ ] `shadow/parity/exits_cumulative.json` — exit 累计
- [ ] `fill_audit.csv` — 完整 shadow 成交审计
- [ ] 5 日连续达标日期列表

---

## 6. 相关文档

- [ARCHITECTURE.md §2.6](./ARCHITECTURE.md#26-端到端时序bar-close--fill) — bar close → fill 时序
- [EXECUTION_PLAN.md](./EXECUTION_PLAN.md) — G0–G3 阶段命令
- [CLUSTER_ROADMAP.md](./CLUSTER_ROADMAP.md) — G3 通过后 → Phase 2 NVDA Pilot
