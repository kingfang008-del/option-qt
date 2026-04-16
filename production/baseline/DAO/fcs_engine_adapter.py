from __future__ import annotations

from typing import Any, Dict, Optional, Protocol


class FeatureEngineAdapter(Protocol):
    """
    特征引擎适配器协议：
    - 对 FCS 暴露统一入口，屏蔽不同市场/标的引擎的差异。
    """

    engine: Any

    def compute_all_inputs(self, **kwargs) -> Dict[str, Dict]:
        ...


class EquityOptionsEngineAdapter:
    """
    默认适配器：美股期权实时特征引擎。
    """

    def __init__(self, device: str):
        try:
            from realtime_feature_engine import RealTimeFeatureEngine
        except ImportError as e:
            raise ImportError("Missing realtime_feature_engine.py") from e
        self.engine = RealTimeFeatureEngine(stats_path=None, device=device)

    def compute_all_inputs(self, **kwargs) -> Dict[str, Dict]:
        return self.engine.compute_all_inputs(**kwargs)


class BTCOptionsEngineAdapter:
    """
    BTC 期权适配器（骨架）。
    当前先复用 fallback 逻辑，接口已稳定，可逐步替换底层实现。
    """

    def __init__(self, device: str):
        try:
            from realtime_feature_engine_btc import BTCRealtimeFeatureEngine
        except ImportError as e:
            raise ImportError("Missing realtime_feature_engine_btc.py") from e
        self.engine = BTCRealtimeFeatureEngine(device=device)

    def compute_all_inputs(self, **kwargs) -> Dict[str, Dict]:
        return self.engine.compute_all_inputs(**kwargs)


def build_feature_engine_adapter(adapter_name: Optional[str], *, device: str) -> FeatureEngineAdapter:
    """
    根据配置构建适配器。
    目前仅内置 equity_options_v1，后续可扩展 btc_options_v1 等实现。
    """
    name = (adapter_name or "equity_options_v1").strip().lower()
    if name in {"equity_options_v1", "equity_options", "default"}:
        return EquityOptionsEngineAdapter(device=device)
    if name in {"btc_options_v1", "btc_options", "btc"}:
        return BTCOptionsEngineAdapter(device=device)
    raise ValueError(f"Unsupported feature engine adapter: {adapter_name}")

