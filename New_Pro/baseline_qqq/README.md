# baseline_qqq

QQQ 0DTE 实盘栈（V0 + qqq_btc 双引擎）。

| 入口 | 说明 |
|------|------|
| `run_live_exec.py` / `run_live_signal.py` | legacy 启动 |
| `qqq_btc/tools/run_live_*_qqq.py` | 推荐（replay 对齐 patch） |
| `config/minimal_stack.env` | 环境变量 |
| `docs/LAYOUT.md` | **目录分层与 import 约定** |

分层包：`strategy/` · `signal_engine/` · `oms/` · `infra/` · `DAO/`  
兼容层：`compat/`（旧扁平 import，勿在新代码中直接使用）
