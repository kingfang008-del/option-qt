"""Causal entry reinforce gates for session H120 sleeves.

Base signal remains 60s stock momentum. Optional AND-gates (all causal on 1m):
  - mf10 same-sign
  - streak_up/dn >= min
  - peer mf10 align count
  - vol_z >= min
  - from_open chase block (same-sign extension too large)
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd

from maga7.common.from_open_gate import session_from_open
from maga7.common.replay import to_ny
from maga7.common.signals import count_peer_align

NY = "America/New_York"


@dataclass(frozen=True)
class SessionReinforceConfig:
    require_mf: bool = False
    streak_min: int = 0
    peer_min: int = 0
    peer_mode: str = "mf10"
    vol_z_min: float = 0.0
    from_open_max: float = 0.0  # 0 = off; e.g. 0.035 blocks same-sign chase
    fail_open_missing: bool = True  # missing feature → allow (except peer/mf if required)


def parse_reinforce(raw: Any) -> SessionReinforceConfig:
    if not isinstance(raw, dict):
        return SessionReinforceConfig()
    return SessionReinforceConfig(
        require_mf=bool(raw.get("require_mf", False)),
        streak_min=int(raw.get("streak_min", 0) or 0),
        peer_min=int(raw.get("peer_min", 0) or 0),
        peer_mode=str(raw.get("peer_mode") or "mf10"),
        vol_z_min=float(raw.get("vol_z_min", 0.0) or 0.0),
        from_open_max=float(raw.get("from_open_max", 0.0) or 0.0),
        fail_open_missing=bool(raw.get("fail_open_missing", True)),
    )


def _bar_at(sdf: pd.DataFrame | None, date: str, ts: pd.Timestamp) -> pd.Series | None:
    if sdf is None or sdf.empty:
        return None
    day = sdf[sdf["date"].astype(str) == str(date)]
    if day.empty:
        return None
    asof = to_ny(ts)
    upto = day[day["timestamp"] <= asof]
    if upto.empty:
        return None
    return upto.iloc[-1]


def evaluate_reinforce(
    *,
    stock_by: dict[str, pd.DataFrame],
    symbol: str,
    date: str,
    entry_ts: pd.Timestamp,
    direction: str,
    cfg: SessionReinforceConfig,
    peer_symbols: list[str] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Return ``(ok, meta)``. Base 60s momentum is assumed already passed."""
    d = str(direction or "").strip().upper()
    meta: dict[str, Any] = {"dir": d}
    if d not in ("UP", "DN"):
        return False, {**meta, "reason": "bad_dir"}

    bar = _bar_at(stock_by.get(symbol), date, entry_ts)
    if bar is None:
        return bool(cfg.fail_open_missing), {**meta, "reason": "missing_bar"}

    mf = float(bar["mf10"]) if "mf10" in bar.index and np.isfinite(bar["mf10"]) else float("nan")
    su = int(bar["streak_up"]) if "streak_up" in bar.index else 0
    sd = int(bar["streak_dn"]) if "streak_dn" in bar.index else 0
    vz = float(bar["vol_z"]) if "vol_z" in bar.index and np.isfinite(bar["vol_z"]) else float("nan")
    meta.update({"mf10": mf, "streak_up": su, "streak_dn": sd, "vol_z": vz})

    if cfg.require_mf:
        if not np.isfinite(mf):
            if not cfg.fail_open_missing:
                return False, {**meta, "reason": "mf_missing"}
        else:
            ok_mf = (d == "UP" and mf > 0) or (d == "DN" and mf < 0)
            if not ok_mf:
                return False, {**meta, "reason": "mf_against"}

    if int(cfg.streak_min) > 0:
        st = su if d == "UP" else sd
        meta["streak"] = st
        if st < int(cfg.streak_min):
            return False, {**meta, "reason": "streak_low"}

    if float(cfg.vol_z_min) > 0:
        if not np.isfinite(vz):
            if not cfg.fail_open_missing:
                return False, {**meta, "reason": "vol_z_missing"}
        elif vz < float(cfg.vol_z_min):
            return False, {**meta, "reason": "vol_z_low"}

    if int(cfg.peer_min) > 0:
        peers = [p for p in (peer_symbols or []) if p != symbol]
        n_peer = count_peer_align(
            stock_by,
            date=str(date),
            asof_ts=entry_ts,
            direction=d,
            peer_symbols=peers,
            mode=str(cfg.peer_mode or "mf10"),
        )
        meta["peer_n"] = int(n_peer)
        if n_peer < int(cfg.peer_min):
            return False, {**meta, "reason": "peer_low"}

    if float(cfg.from_open_max) > 0:
        fo = session_from_open(stock_by.get(symbol), date=str(date), asof_ts=entry_ts)
        meta["from_open"] = fo
        if fo is not None:
            # block same-sign chase past threshold
            if d == "UP" and fo > float(cfg.from_open_max):
                return False, {**meta, "reason": "from_open_chase"}
            if d == "DN" and fo < -float(cfg.from_open_max):
                return False, {**meta, "reason": "from_open_chase"}

    return True, {**meta, "reason": "pass"}


def cfg_to_dict(cfg: SessionReinforceConfig) -> dict[str, Any]:
    return {f.name: getattr(cfg, f.name) for f in fields(cfg)}
