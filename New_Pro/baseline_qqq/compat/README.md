# compat — legacy 扁平 import 兼容层

旧代码与 `qqq_btc` 集成使用 `from execution_engine_v8 import ...` 等扁平名。
这些文件仅做 re-export，**实现均在** `strategy/` · `signal_engine/` · `oms/` · `infra/`。

`baseline_paths.py` 会把本目录加入 `sys.path`；新代码请直接从分层包 import。
