"""qqq_btc 实盘薄层 —— Signal/OMS 集成,不修改 New_Pro 源码。"""

from qqq_btc.live.bootstrap import bootstrap_qqq_btc_live, is_qqq_btc_live, tick_exits_mode

__all__ = [
    "bootstrap_qqq_btc_live",
    "is_qqq_btc_live",
    "tick_exits_mode",
]
