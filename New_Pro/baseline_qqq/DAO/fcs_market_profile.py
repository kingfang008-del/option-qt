from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from typing import Iterable, Tuple

import pandas as pd


@dataclass
class BaseMarketProfile:
    """
    FCS 市场规则抽象：
    - 接收时段
    - RTH 口径
    - warmup 门槛
    - 非交易标的列表
    """

    name: str
    ny_tz: object
    warmup_required_len: int = 31
    non_tradable_symbols: Tuple[str, ...] = field(default_factory=tuple)

    def accept_realtime_tick(self, dt_ny: pd.Timestamp) -> bool:
        return True

    def should_flush_premarket(self, dt_ny: pd.Timestamp, last_flush_date) -> bool:
        return False

    def history_keep_mask(self, idx: pd.DatetimeIndex, dt_ny: pd.Timestamp):
        return pd.Series([True] * len(idx), index=idx)

    def is_rth_minute(self, dt_ny: pd.Timestamp) -> bool:
        return True

    def rth_window(self, label_floor: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        start = label_floor.replace(hour=0, minute=0, second=0, microsecond=0)
        end = label_floor.replace(hour=23, minute=59, second=59, microsecond=0)
        return start, end

    def count_effective_history(self, idx: pd.DatetimeIndex, label_floor: pd.Timestamp) -> int:
        start, end = self.rth_window(label_floor)
        return int(((idx >= start) & (idx <= min(label_floor, end))).sum())

    def calc_effective_minutes(self, start_dt: datetime, end_dt: datetime) -> float:
        if end_dt <= start_dt:
            return 0.0
        return max(0.0, (end_dt - start_dt).total_seconds() / 60.0)

    def get_non_tradable_set(self) -> set[str]:
        return set(self.non_tradable_symbols or ())


@dataclass
class EquityUSProfile(BaseMarketProfile):
    market_close_guard: dt_time = dt_time(16, 15)
    rth_start_time: dt_time = dt_time(9, 30)
    rth_end_time: dt_time = dt_time(16, 0)

    def accept_realtime_tick(self, dt_ny: pd.Timestamp) -> bool:
        return dt_ny.time() <= self.market_close_guard

    def should_flush_premarket(self, dt_ny: pd.Timestamp, last_flush_date) -> bool:
        return dt_ny.time() >= self.rth_start_time and last_flush_date != dt_ny.date()

    def history_keep_mask(self, idx: pd.DatetimeIndex, dt_ny: pd.Timestamp):
        today_start = dt_ny.replace(hour=0, minute=0, second=0, microsecond=0)
        rth_start = dt_ny.replace(
            hour=self.rth_start_time.hour, minute=self.rth_start_time.minute, second=0, microsecond=0
        )
        rth_end = dt_ny.replace(
            hour=self.rth_end_time.hour, minute=self.rth_end_time.minute, second=0, microsecond=0
        )
        return (idx < today_start) | ((idx >= rth_start) & (idx <= rth_end))

    def is_rth_minute(self, dt_ny: pd.Timestamp) -> bool:
        t = dt_ny.time()
        return self.rth_start_time <= t < self.rth_end_time

    def rth_window(self, label_floor: pd.Timestamp):
        start = label_floor.replace(
            hour=self.rth_start_time.hour, minute=self.rth_start_time.minute, second=0, microsecond=0
        )
        end = label_floor.replace(
            hour=self.rth_end_time.hour, minute=self.rth_end_time.minute, second=0, microsecond=0
        )
        return start, end

    def calc_effective_minutes(self, start_dt: datetime, end_dt: datetime) -> float:
        if end_dt <= start_dt:
            return 0.0

        minutes = 0.0
        curr = start_dt
        while curr.date() <= end_dt.date():
            if curr.weekday() >= 5:
                curr = (curr + timedelta(days=1)).replace(
                    hour=self.rth_start_time.hour, minute=self.rth_start_time.minute, second=0, microsecond=0
                )
                continue

            market_open = curr.tzinfo.localize(datetime.combine(curr.date(), self.rth_start_time))
            market_close = curr.tzinfo.localize(datetime.combine(curr.date(), self.rth_end_time))
            daily_start = max(curr, market_open)
            daily_end = min(end_dt, market_close)
            if daily_start < daily_end:
                minutes += (daily_end - daily_start).total_seconds() / 60.0

            curr = (curr + timedelta(days=1)).replace(
                hour=self.rth_start_time.hour, minute=self.rth_start_time.minute, second=0, microsecond=0
            )
        return max(0.0, minutes)


@dataclass
class Crypto247Profile(BaseMarketProfile):
    """
    24/7 市场规则（BTC 等）。
    """

    def accept_realtime_tick(self, dt_ny: pd.Timestamp) -> bool:
        return True

    def should_flush_premarket(self, dt_ny: pd.Timestamp, last_flush_date) -> bool:
        return False

    def is_rth_minute(self, dt_ny: pd.Timestamp) -> bool:
        return True

    def rth_window(self, label_floor: pd.Timestamp):
        start = label_floor.replace(hour=0, minute=0, second=0, microsecond=0)
        end = label_floor.replace(hour=23, minute=59, second=59, microsecond=0)
        return start, end


def build_market_profile(
    name: str,
    *,
    ny_tz,
    warmup_required_len: int = 31,
    non_tradable_symbols: Iterable[str] = (),
) -> BaseMarketProfile:
    profile = (name or "equity_us").strip().lower()
    kwargs = dict(ny_tz=ny_tz, warmup_required_len=int(warmup_required_len), non_tradable_symbols=tuple(non_tradable_symbols))
    if profile in {"equity_us", "equity", "default"}:
        return EquityUSProfile(name="equity_us", **kwargs)
    if profile in {"crypto_247", "btc", "crypto"}:
        return Crypto247Profile(name="crypto_247", **kwargs)
    raise ValueError(f"Unsupported market profile: {name}")

