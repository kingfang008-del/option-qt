# baseline_qqq 代码分层

根目录只保留 **配置、路径引导、启动脚本**；实现代码在分层包；旧扁平 import 在 `compat/`。

```
baseline_qqq/
├── README.md
├── config.py              # 全局运行参数（env / Redis / 模式）
├── config/minimal_stack.env
├── baseline_paths.py      # 启动时 sys.path 引导
├── run_live_exec.py       # OMS 入口
├── run_live_signal.py     # SE 入口
│
├── strategy/              # 策略决策（V0-only）
├── signal_engine/         # SignalEngineV8、alpha 归一化
├── oms/                   # ExecutionEngineV8、编排器
├── infra/                 # Mock IBKR、启动清理
├── compat/                # legacy 扁平 import shim（21 个）
├── DAO/                   # FCS、IBKR、Dashboard
├── tools/                 # gate_trace_stats 等运维脚本
├── utils/                 # 序列化 / Greeks
├── docs/                  # 本文件
└── archive/               # verify_*、TREND、Domain、back（非实盘）
```

### strategy/

| 文件 | 原路径 |
|------|--------|
| `selector.py` | strategy_selector |
| `core_v0.py` | strategy_core_v0 |
| `config0.py` | strategy_config0 |
| `regime.py` | bidirectional_regime |
| `entry_risk_rules.py` | entry_risk_rules |
| `exit_rails.py` | strategy_exit_rails |
| `exec_profile.py` | exec_profile |
| `liquidity_rules.py` | liquidity_rules |

### compat/

`execution_engine_v8.py`、`strategy_selector.py` 等 **仅 re-export**，由 `baseline_paths` 加入 `sys.path`。
新代码勿在此目录增文件。

## 与 qqq_btc 的分工

| 层 | baseline_qqq | qqq_btc |
|----|--------------|---------|
| 标签 / 回放 | — | `common/`, `tools/run_replay.py` |
| 模型 | legacy TFT（SE fallback） | `model/backbone` + checkpoint |
| 入场 / 出场语义 | V0 门控 + patch 宿主 | `entry_decision`, `exit_rails` |
| 实盘 patch | OMS 宿主 | `live/*_bridge`, `oms_integration` |

`QQQ_BTC_LIVE=1` 时 OMS bootstrap patch：

- `decide_entry` → `choose_entry`（`strategy_entry_bridge`）
- `check_exit` → `exit_rails`（`strategy_exit_bridge`）

## import 约定

**新代码（推荐）：**

```python
from strategy import StrategyCore, StrategyConfig
from signal_engine import SignalEngineV8
from oms import ExecutionEngineV8
```

**Legacy（经 compat/，仍支持）：**

```python
import baseline_paths  # 启动脚本首行
from execution_engine_v8 import ExecutionEngineV8
from strategy_selector import StrategyCore
```

## 策略版本

仅 **V0**（`strategy/core_v0.py`）。TREND / V1 在 `archive/`。
