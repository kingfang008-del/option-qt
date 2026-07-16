"""Option fill helpers — wrap qqq_btc fill_model (single source of truth)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from qqq_btc.common.fill_model import OptionSpreadFillModel, spread_interpolate


@dataclass(frozen=True)
class FillSpec:
    entry_frac: float = 0.8
    exit_frac: float = 0.8

    def model(self) -> OptionSpreadFillModel:
        return OptionSpreadFillModel(entry_frac=self.entry_frac, exit_frac=self.exit_frac)

    def buy(self, bid: float, ask: float) -> float:
        return float(spread_interpolate(bid, ask, self.entry_frac, "BUY"))

    def sell(self, bid: float, ask: float) -> float:
        return float(spread_interpolate(bid, ask, self.exit_frac, "SELL"))

    def sell_series(self, bid, ask):
        return spread_interpolate(bid, ask, self.exit_frac, "SELL")
