"""LightGBM Smart Bouncer — Rule-A candidate veto via tabular features.

Default **off**. Complements ``tcn_gate``: same hook shape (block|scale), but
features are session-structure + MF context, labels prefer option-path ternary.

Profile::

    "lgbm_bouncer": {
      "enabled": false,
      "action": "scale",          # off|block|scale
      "p_min": 0.55,              # P(allow) = 1 - P(toxic)
      "scale_when_low": 0.5,
      "model_path": "maga7/results/lgbm_bouncer/model.txt",
      "block_on_missing": false
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NY = "America/New_York"


def _to_ny(ts: pd.Timestamp) -> pd.Timestamp:
    """Local helper to avoid circular import with replay."""
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)

FEATURE_COLS = (
    "dir_sign",
    "mf10",
    "mf_fast",
    "streak",
    "from_prev",
    "vol_z",
    "tod_min",
    "gap_open",
    "bounce_lod",
    "above_open",
    "above_vwap",
    "qqq_from_prev",
    "qqq_gap_open",
    "qqq_above_open",
    "qqq_above_vwap",
)


@dataclass(frozen=True)
class LgbmBouncerConfig:
    enabled: bool = False
    action: str = "scale"
    p_min: float = 0.55
    scale_when_low: float = 0.5
    model_path: str | None = None
    block_on_missing: bool = False
    feature_cols: tuple[str, ...] = FEATURE_COLS
    # If non-empty, only score these directions; others passthrough.
    only_directions: tuple[str, ...] = ()

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "LgbmBouncerConfig":
        prof = profile or {}
        raw = prof.get("lgbm_bouncer")
        if raw is None:
            raw = (prof.get("signal") or {}).get("lgbm_bouncer")
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", False))
        action = str(raw.get("action") or ("scale" if enabled else "off")).strip().lower()
        if not enabled:
            action = "off"
        cols = raw.get("feature_cols") or list(FEATURE_COLS)
        if isinstance(cols, str):
            cols = [x.strip() for x in cols.split(",") if x.strip()]
        mp = raw.get("model_path")
        only = raw.get("only_directions") or raw.get("only_dirs") or ()
        if isinstance(only, str):
            only = [x.strip() for x in only.split(",") if x.strip()]
        only_t = tuple(str(x).upper() for x in only)
        return cls(
            enabled=enabled and action not in {"", "off", "none", "false", "0"},
            action=action if action in {"off", "block", "scale"} else "scale",
            p_min=float(raw.get("p_min", 0.55) or 0.55),
            scale_when_low=float(raw.get("scale_when_low", raw.get("scale", 0.5)) or 0.5),
            model_path=str(mp) if mp else None,
            block_on_missing=bool(raw.get("block_on_missing", False)),
            feature_cols=tuple(str(c) for c in cols),
            only_directions=only_t,
        )


@dataclass
class LgbmBouncerDecision:
    allow: bool
    size_scale: float = 1.0
    p: float | None = None
    reason: str = "off"
    meta: dict[str, Any] = field(default_factory=dict)


def _bar_at(sdf: pd.DataFrame | None, asof_ts: pd.Timestamp) -> pd.Series | None:
    if sdf is None or sdf.empty or "timestamp" not in sdf.columns:
        return None
    asof = _to_ny(asof_ts)
    ts = pd.to_datetime(sdf["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    upto = sdf.loc[ts <= asof]
    if upto.empty:
        return None
    return upto.iloc[-1]


def extract_bouncer_features(
    *,
    symbol: str,
    direction: str,
    asof_ts: pd.Timestamp,
    stock_df: pd.DataFrame | None,
    qqq_df: pd.DataFrame | None,
) -> dict[str, float] | None:
    """Causal tabular features at Rule-A fire."""
    # Lazy import: replay imports this module.
    from maga7.common.replay import _session_structure_at

    asof = _to_ny(asof_ts)
    date = asof.strftime("%Y-%m-%d")
    bar = _bar_at(stock_df, asof)
    if bar is None:
        return None
    st = _session_structure_at(stock_df, date=date, asof_ts=asof)
    if st is None:
        return None
    qst = _session_structure_at(qqq_df, date=date, asof_ts=asof) if qqq_df is not None else None
    qbar = _bar_at(qqq_df, asof) if qqq_df is not None else None

    d = str(direction).upper()
    streak = float(bar["streak_up"] if d == "UP" else bar["streak_dn"]) if "streak_up" in bar.index else 0.0
    mf_fast = bar["mf_fast"] if "mf_fast" in bar.index else (bar["mf_short"] if "mf_short" in bar.index else np.nan)
    qfp = float(qbar["from_prev"]) if qbar is not None and "from_prev" in qbar.index and pd.notna(qbar["from_prev"]) else np.nan

    feat = {
        "dir_sign": 1.0 if d == "UP" else -1.0,
        "mf10": float(bar["mf10"]) if "mf10" in bar.index and pd.notna(bar["mf10"]) else 0.0,
        "mf_fast": float(mf_fast) if pd.notna(mf_fast) else 0.0,
        "streak": float(streak) if pd.notna(streak) else 0.0,
        "from_prev": float(bar["from_prev"]) if "from_prev" in bar.index and pd.notna(bar["from_prev"]) else 0.0,
        "vol_z": float(bar["vol_z"]) if "vol_z" in bar.index and pd.notna(bar["vol_z"]) else 0.0,
        "tod_min": float(asof.hour * 60 + asof.minute),
        "gap_open": float(st["px"] / st["open"] - 1.0),
        "bounce_lod": float(st["bounce_from_lod"] or 0.0),
        "above_open": 1.0 if st["above_open"] else 0.0,
        "above_vwap": 1.0 if st["above_vwap"] else 0.0,
        "qqq_from_prev": float(qfp) if np.isfinite(qfp) else 0.0,
        "qqq_gap_open": float(qst["px"] / qst["open"] - 1.0) if qst else 0.0,
        "qqq_above_open": 1.0 if (qst and qst["above_open"]) else 0.0,
        "qqq_above_vwap": 1.0 if (qst and qst["above_vwap"]) else 0.0,
    }
    return feat


def label_option_ternary(
    *,
    mfe: float,
    mae: float,
    good_mfe: float = 0.40,
    good_mae_max: float = 0.15,
    toxic_mae: float = 0.30,
) -> int:
    """+1 quality, 0 chop, -1 toxic reversal (option return path)."""
    if mae >= float(toxic_mae):
        return -1
    if mfe >= float(good_mfe) and mae <= float(good_mae_max):
        return 1
    return 0


def label_underlying_ternary(
    stock_df: pd.DataFrame | None,
    *,
    asof_ts: pd.Timestamp,
    direction: str,
    horizon_minutes: int = 30,
    good_mfe: float = 0.004,
    good_mae_max: float = 0.002,
    toxic_mae: float = 0.006,
) -> int | None:
    """Fallback ternary on underlying path (same spirit, smaller thresholds)."""
    if stock_df is None or stock_df.empty:
        return None
    asof = _to_ny(asof_ts)
    end = asof + pd.Timedelta(minutes=int(horizon_minutes))
    df = stock_df.copy()
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    df["_ts"] = ts
    entry = df[df["_ts"] <= asof].tail(1)
    if entry.empty:
        return None
    px0 = float(entry.iloc[0]["close"])
    if not np.isfinite(px0) or px0 <= 0:
        return None
    fut = df[(df["_ts"] > asof) & (df["_ts"] <= end)]
    if fut.empty or len(fut) < 5:
        return None
    hi = float(pd.to_numeric(fut["high"], errors="coerce").max())
    lo = float(pd.to_numeric(fut["low"], errors="coerce").min())
    if str(direction).upper() == "UP":
        mfe, mae = hi / px0 - 1.0, 1.0 - lo / px0
    else:
        mfe, mae = 1.0 - lo / px0, hi / px0 - 1.0
    return label_option_ternary(
        mfe=mfe, mae=mae, good_mfe=good_mfe, good_mae_max=good_mae_max, toxic_mae=toxic_mae
    )


def option_path_mfe_mae(
    path: pd.DataFrame | None,
    entry_ts: pd.Timestamp,
    *,
    entry_frac: float = 0.8,
    exit_frac: float = 0.8,
    hold_minutes: int = 30,
) -> tuple[float, float, float] | None:
    """Return (mfe, mae, end_ret) on option sell marks over hold window."""
    from maga7.common.fills import FillSpec

    if path is None or path.empty:
        return None
    entry_ts = _to_ny(entry_ts)
    after = path[path["timestamp"] >= entry_ts]
    if after.empty:
        return None
    fill = FillSpec(entry_frac=entry_frac, exit_frac=exit_frac)
    bid0 = float(after.iloc[0]["bid"])
    ask0 = float(after.iloc[0]["ask"])
    entry = fill.buy(bid0, ask0)
    if not np.isfinite(entry) or entry <= 0:
        return None
    end_ts = entry_ts + pd.Timedelta(minutes=int(hold_minutes))
    win = after[after["timestamp"] <= end_ts]
    if win.empty:
        win = after
    sell = np.asarray(
        fill.sell_series(win["bid"].astype(float), win["ask"].astype(float)),
        dtype=float,
    )
    rets = sell / entry - 1.0
    if rets.size == 0:
        return None
    mfe = float(np.nanmax(rets))
    mae = float(-np.nanmin(rets))  # adverse = positive magnitude
    return mfe, mae, float(rets[-1])


@dataclass
class NullLgbmBouncer:
    cfg: LgbmBouncerConfig = field(default_factory=LgbmBouncerConfig)

    def decide(
        self,
        *,
        symbol: str,
        direction: str,
        asof_ts: pd.Timestamp,
        stock_df: pd.DataFrame | None,
        qqq_df: pd.DataFrame | None = None,
    ) -> LgbmBouncerDecision:
        return LgbmBouncerDecision(allow=True, size_scale=1.0, reason="off")


@dataclass
class LgbmBouncer:
    cfg: LgbmBouncerConfig
    model: Any
    feature_cols: tuple[str, ...]
    model_outputs: str = "p_allow"  # or p_toxic

    def predict_p_allow(self, feat: dict[str, float]) -> float:
        x = np.array([[float(feat.get(c, 0.0)) for c in self.feature_cols]], dtype=np.float32)
        p = float(np.asarray(self.model.predict(x)).reshape(-1)[0])
        if self.model_outputs == "p_toxic":
            return 1.0 - p
        return p

    def decide(
        self,
        *,
        symbol: str,
        direction: str,
        asof_ts: pd.Timestamp,
        stock_df: pd.DataFrame | None,
        qqq_df: pd.DataFrame | None = None,
    ) -> LgbmBouncerDecision:
        if self.cfg.action == "off" or not self.cfg.enabled:
            return LgbmBouncerDecision(allow=True, size_scale=1.0, reason="off")
        if self.cfg.only_directions and str(direction).upper() not in self.cfg.only_directions:
            return LgbmBouncerDecision(allow=True, size_scale=1.0, reason="lgbm_skip_dir")
        feat = extract_bouncer_features(
            symbol=symbol,
            direction=direction,
            asof_ts=asof_ts,
            stock_df=stock_df,
            qqq_df=qqq_df,
        )
        if feat is None:
            if self.cfg.block_on_missing:
                return LgbmBouncerDecision(allow=False, size_scale=0.0, reason="missing_feat")
            return LgbmBouncerDecision(allow=True, size_scale=1.0, reason="missing_passthrough")
        p = self.predict_p_allow(feat)
        if self.cfg.action == "block":
            ok = p >= float(self.cfg.p_min)
            return LgbmBouncerDecision(
                allow=ok,
                size_scale=1.0 if ok else 0.0,
                p=p,
                reason="ok" if ok else "lgbm_block",
                meta=feat,
            )
        # scale
        if p >= float(self.cfg.p_min):
            return LgbmBouncerDecision(allow=True, size_scale=1.0, p=p, reason="ok", meta=feat)
        return LgbmBouncerDecision(
            allow=True,
            size_scale=float(self.cfg.scale_when_low),
            p=p,
            reason="lgbm_scale",
            meta=feat,
        )


def save_lgbm_model(model: Any, path: Path, *, meta: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    path.with_suffix(path.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )


def load_lgbm_bouncer(profile: dict[str, Any] | None) -> NullLgbmBouncer | LgbmBouncer:
    cfg = LgbmBouncerConfig.from_profile(profile)
    if not cfg.enabled or cfg.action == "off":
        return NullLgbmBouncer(cfg=cfg)
    if not cfg.model_path:
        return NullLgbmBouncer(cfg=cfg)
    path = Path(cfg.model_path).expanduser()
    if not path.is_file():
        return NullLgbmBouncer(cfg=cfg)
    import lightgbm as lgb

    booster = lgb.Booster(model_file=str(path))
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    cols = cfg.feature_cols
    model_outputs = "p_allow"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("feature_cols"):
            cols = tuple(meta["feature_cols"])
        model_outputs = str(meta.get("model_outputs") or meta.get("target") or "p_allow")
        if model_outputs == "toxic":
            model_outputs = "p_toxic"
        if model_outputs == "allow":
            model_outputs = "p_allow"
    return LgbmBouncer(cfg=cfg, model=booster, feature_cols=cols, model_outputs=model_outputs)
