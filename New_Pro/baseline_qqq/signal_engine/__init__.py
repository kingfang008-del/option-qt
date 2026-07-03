# -*- coding: utf-8 -*-
"""信号层：FCS bar → TFT 推理 → ALPHA_FRAME。"""

__all__ = ["SignalEngineV8", "SymbolState"]


def __getattr__(name: str):
    if name in ("SignalEngineV8", "SymbolState"):
        from signal_engine.engine_v8 import SignalEngineV8, SymbolState
        return SignalEngineV8 if name == "SignalEngineV8" else SymbolState
    raise AttributeError(name)
