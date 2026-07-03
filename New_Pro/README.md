# New_Pro — QQQ 0DTE 实盘栈（精简版）

自 `New_Pro_bak` 裁剪，仅保留 **qqq_btc 双引擎** 实际运行的代码。

**代码分层说明见** [`baseline_qqq/docs/LAYOUT.md`](baseline_qqq/docs/LAYOUT.md)。

## 进程拓扑

```
IBKR (DAO/ibkr_connector_v8.py)
  → fused_market_stream
FCS (DAO/feature_compute_service_v8.py)
  → unified_inference_stream
Signal (run_live_signal.py / qqq_btc run_live_signal_qqq.py)
  → orch_trade_signals (ALPHA_FRAME)
OMS (run_live_exec.py / qqq_btc run_live_exec_qqq.py)
  → IBKR 下单
Dashboard (DAO/dashboard_monitor_ultimate.py) — 只读监控
```

## 推荐启动（qqq_btc 路径）

```bash
cd New_Pro/baseline_qqq
set -a && source config/minimal_stack.env && set +a

# 1. IBKR + 行情
python DAO/ibkr_connector_v8.py

# 2. FCS（需 PYTHONPATH 含 repo 根以便 qqq_btc enrich）
PYTHONPATH=../../:$PYTHONPATH python DAO/feature_compute_service_v8.py

# 3. Signal
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_signal_qqq.py --checkpoint <best.pth>

# 4. OMS
QQQ_BTC_LIVE=1 python ../../qqq_btc/tools/run_live_exec_qqq.py

# 5. Dashboard（可选）
streamlit run DAO/dashboard_monitor_ultimate.py
```

## 目录分层

| 路径 | 职责 |
|------|------|
| `baseline_qqq/strategy/` | V0 策略、门控、exec profile |
| `baseline_qqq/signal_engine/` | SignalEngineV8、alpha 归一化 |
| `baseline_qqq/oms/` | ExecutionEngineV8、编排器 |
| `baseline_qqq/infra/` | Mock IBKR、启动清理 |
| `baseline_qqq/DAO/` | FCS、IBKR、Dashboard |
| `baseline_qqq/tools/` | gate 统计、路径分析 |
| `baseline_qqq/compat/` | legacy 扁平 import shim（21 个，勿在新代码使用） |
| `baseline_qqq/docs/` | LAYOUT 等文档 |
| `baseline_qqq/archive/` | verify/TREND/Domain（非实盘） |
| `CONFIG/` | 特征 JSON、锚点配置 |
| `model/` | legacy TFT（SE import；qqq_btc 会覆盖 checkpoint） |
| `qqq_btc/` | 标签、replay、live patch（主开发） |
| `qqq_btc/CLUSTER_ROADMAP.md` | Phase 0–5 里程碑与模型集群设计 |

legacy 扁平 import 在 `compat/`（由 `baseline_paths` 加载）；新代码请按 `docs/LAYOUT.md` 从分层包 import。

完整备份与审计脚本见 `New_Pro_bak/`。
