# -*- coding: utf-8 -*-
"""执行层：OMS、编排器、IBKR 路由。"""

__all__ = ["ExecutionEngineV8"]


def __getattr__(name: str):
    if name == "ExecutionEngineV8":
        from oms.execution_engine import ExecutionEngineV8
        return ExecutionEngineV8
    raise AttributeError(name)
