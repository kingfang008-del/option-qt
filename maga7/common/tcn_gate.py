"""Pluggable TCN probability gate for Mag7 Rule-A entries.

Default is **off**. When enabled, scores a short causal minute-bar tensor and either
blocks or scales size. Missing torch / model / features never crash the baseline
path unless ``block_on_missing`` is set.

Profile (under ``signal.tcn_gate`` or top-level ``tcn_gate``)::

    {
      "enabled": false,
      "action": "scale",          # off|block|scale
      "p_min": 0.75,
      "scale_when_low": 0.5,      # used when action=scale and p < p_min
      "scale_mode": "floor",      # floor: size*=scale_when_low; linear: size*=clamp(p, floor, 1)
      "window": 15,
      "model_path": null,
      "block_on_missing": false,
      "channels": ["net$", "ret1", "range_pct", "qqq_ret1"]
    }
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

NY = "America/New_York"

DEFAULT_CHANNELS = ("net$", "ret1", "range_pct", "qqq_ret1")


def _opt_float(raw: dict[str, Any], *keys: str) -> float | None:
    for k in keys:
        if k in raw and raw[k] is not None and str(raw[k]).strip() != "":
            return float(raw[k])
    only = raw.get("only_when")
    if isinstance(only, dict):
        for k in keys:
            # allow nested keys without only_when_ prefix
            short = k.replace("only_when_", "") if k.startswith("only_when_") else k
            if short in only and only[short] is not None and str(only[short]).strip() != "":
                return float(only[short])
            if k in only and only[k] is not None and str(only[k]).strip() != "":
                return float(only[k])
    return None


@dataclass(frozen=True)
class TcnGateConfig:
    enabled: bool = False
    action: str = "scale"  # off|block|scale
    p_min: float = 0.75
    scale_when_low: float = 0.5
    scale_mode: str = "floor"  # floor|linear
    window: int = 15
    model_path: str | None = None
    block_on_missing: bool = False
    channels: tuple[str, ...] = DEFAULT_CHANNELS
    # Apply gate only in "weak" regime; otherwise passthrough (size unchanged).
    only_when_vixy_z_max: float | None = None  # e.g. 0.5 → gate when vixy_z <= 0.5
    only_when_vixy_z_min: float | None = None
    only_when_abs_qqq_fp_max: float | None = None  # gate when |qqq_from_prev| <= max
    only_when_abs_qqq_fp_min: float | None = None  # gate when |qqq_from_prev| >= min

    @classmethod
    def from_profile(cls, profile: dict[str, Any] | None) -> "TcnGateConfig":
        prof = profile or {}
        raw = prof.get("tcn_gate")
        if raw is None:
            raw = (prof.get("signal") or {}).get("tcn_gate")
        if not isinstance(raw, dict):
            raw = {}
        enabled = bool(raw.get("enabled", False))
        action = str(raw.get("action") or ("scale" if enabled else "off")).strip().lower()
        if action in {"size", "half"}:
            action = "scale"
        if not enabled:
            action = "off"
        ch = raw.get("channels") or list(DEFAULT_CHANNELS)
        if isinstance(ch, str):
            ch = [x.strip() for x in ch.split(",") if x.strip()]
        mp = raw.get("model_path")
        return cls(
            enabled=enabled and action not in {"", "off", "none", "false", "0"},
            action=action if action in {"off", "block", "scale"} else "scale",
            p_min=float(raw.get("p_min", 0.75) or 0.75),
            scale_when_low=float(raw.get("scale_when_low", raw.get("scale", 0.5)) or 0.5),
            scale_mode=str(raw.get("scale_mode") or "floor").strip().lower(),
            window=max(5, int(raw.get("window", 15) or 15)),
            model_path=str(mp) if mp else None,
            block_on_missing=bool(raw.get("block_on_missing", False)),
            channels=tuple(str(c) for c in ch),
            only_when_vixy_z_max=_opt_float(raw, "only_when_vixy_z_max", "vixy_z_max"),
            only_when_vixy_z_min=_opt_float(raw, "only_when_vixy_z_min", "vixy_z_min"),
            only_when_abs_qqq_fp_max=_opt_float(
                raw, "only_when_abs_qqq_fp_max", "abs_qqq_fp_max", "abs_qqq_from_prev_max"
            ),
            only_when_abs_qqq_fp_min=_opt_float(
                raw, "only_when_abs_qqq_fp_min", "abs_qqq_fp_min", "abs_qqq_from_prev_min"
            ),
        )

    def regime_applies(
        self,
        *,
        regime_vixy_z: float | None = None,
        regime_qqq_fp: float | None = None,
    ) -> bool:
        """True if only_when constraints are inactive or satisfied."""
        need = (
            self.only_when_vixy_z_max is not None
            or self.only_when_vixy_z_min is not None
            or self.only_when_abs_qqq_fp_max is not None
            or self.only_when_abs_qqq_fp_min is not None
        )
        if not need:
            return True
        if self.only_when_vixy_z_max is not None:
            if regime_vixy_z is None or not np.isfinite(regime_vixy_z):
                return False
            if float(regime_vixy_z) > float(self.only_when_vixy_z_max):
                return False
        if self.only_when_vixy_z_min is not None:
            if regime_vixy_z is None or not np.isfinite(regime_vixy_z):
                return False
            if float(regime_vixy_z) < float(self.only_when_vixy_z_min):
                return False
        if self.only_when_abs_qqq_fp_max is not None:
            if regime_qqq_fp is None or not np.isfinite(regime_qqq_fp):
                return False
            if abs(float(regime_qqq_fp)) > float(self.only_when_abs_qqq_fp_max):
                return False
        if self.only_when_abs_qqq_fp_min is not None:
            if regime_qqq_fp is None or not np.isfinite(regime_qqq_fp):
                return False
            if abs(float(regime_qqq_fp)) < float(self.only_when_abs_qqq_fp_min):
                return False
        return True


@dataclass
class TcnGateDecision:
    allow: bool
    size_scale: float = 1.0
    p: float | None = None
    reason: str = "off"
    meta: dict[str, Any] = field(default_factory=dict)


class TcnGate(Protocol):
    cfg: TcnGateConfig

    def decide(
        self,
        *,
        symbol: str,
        direction: str,
        asof_ts: pd.Timestamp,
        stock_df: pd.DataFrame | None,
        qqq_df: pd.DataFrame | None = None,
        regime_vixy_z: float | None = None,
        regime_qqq_fp: float | None = None,
    ) -> TcnGateDecision: ...


@dataclass
class NullTcnGate:
    """No-op gate (baseline path)."""

    cfg: TcnGateConfig = field(default_factory=TcnGateConfig)

    def decide(
        self,
        *,
        symbol: str,
        direction: str,
        asof_ts: pd.Timestamp,
        stock_df: pd.DataFrame | None,
        qqq_df: pd.DataFrame | None = None,
        regime_vixy_z: float | None = None,
        regime_qqq_fp: float | None = None,
    ) -> TcnGateDecision:
        return TcnGateDecision(allow=True, size_scale=1.0, p=None, reason="off")


def _to_ny_ts(ts: pd.Timestamp) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        return t.tz_localize(NY)
    return t.tz_convert(NY)


def build_feature_tensor(
    stock_df: pd.DataFrame | None,
    qqq_df: pd.DataFrame | None,
    *,
    asof_ts: pd.Timestamp,
    window: int,
    channels: tuple[str, ...] = DEFAULT_CHANNELS,
    direction: str = "UP",
) -> np.ndarray | None:
    """Causal (window, n_channels) float32 tensor ending at/before ``asof_ts``.

    Channels (minimal set; missing optional channels zero-filled):
      - net$: minute money-flow proxy (requires attach_mf_features)
      - ret1: close-to-close return
      - range_pct: (high-low)/close
      - qqq_ret1: aligned QQQ 1m return (0 if unavailable)
    Direction flips signed flow/returns for DN so the net always sees "favorable" as +.
    """
    if stock_df is None or stock_df.empty:
        return None
    asof = _to_ny_ts(asof_ts)
    df = stock_df
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_datetime(df["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert(NY)
    else:
        ts = ts.dt.tz_convert(NY)
    day = df.copy()
    day["_ts"] = ts
    day = day[day["_ts"] <= asof].sort_values("_ts")
    if len(day) < int(window):
        return None
    win = day.iloc[-int(window) :].copy()
    close = pd.to_numeric(win["close"], errors="coerce")
    high = pd.to_numeric(win["high"], errors="coerce") if "high" in win.columns else close
    low = pd.to_numeric(win["low"], errors="coerce") if "low" in win.columns else close
    if "net$" in win.columns:
        net = pd.to_numeric(win["net$"], errors="coerce")
    else:
        # fallback proxy if features not attached
        hl = (high - low).replace(0, np.nan)
        buy = ((close - low) / hl).fillna(0.5) * pd.to_numeric(win.get("volume", 0), errors="coerce")
        sell = ((high - close) / hl).fillna(0.5) * pd.to_numeric(win.get("volume", 0), errors="coerce")
        net = (buy - sell) * close
    ret1 = close.pct_change().fillna(0.0)
    range_pct = ((high - low) / close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    qqq_ret = pd.Series(0.0, index=win.index)
    if qqq_df is not None and not qqq_df.empty and "timestamp" in qqq_df.columns:
        q = qqq_df.copy()
        qts = pd.to_datetime(q["timestamp"])
        if getattr(qts.dt, "tz", None) is None:
            qts = qts.dt.tz_localize("UTC").dt.tz_convert(NY)
        else:
            qts = qts.dt.tz_convert(NY)
        q["_ts"] = qts
        q = q[q["_ts"] <= asof].sort_values("_ts")
        if len(q) >= 2 and "close" in q.columns:
            q["ret1"] = pd.to_numeric(q["close"], errors="coerce").pct_change()
            # asof align: last qqq bar at/before each stock bar
            q_idx = q.set_index("_ts")["ret1"]
            aligned = []
            for t in win["_ts"]:
                pos = q_idx.index.searchsorted(t, side="right") - 1
                aligned.append(float(q_idx.iloc[pos]) if pos >= 0 else 0.0)
            qqq_ret = pd.Series(aligned, index=win.index).fillna(0.0)

    sign = 1.0 if str(direction).upper() == "UP" else -1.0
    last_row = win.iloc[-1]
    vol_z = 0.0
    if "vol_z" in win.columns:
        try:
            vz = float(pd.to_numeric(last_row["vol_z"], errors="coerce"))
            if np.isfinite(vz):
                vol_z = vz
        except Exception:
            vol_z = 0.0
    # minutes from 09:30 / 390
    tod = ((asof.hour * 60 + asof.minute) - (9 * 60 + 30)) / 390.0
    tod = float(np.clip(tod, 0.0, 1.0))
    ones = np.ones(len(win), dtype=np.float64)
    chan_map = {
        "net$": (net.fillna(0.0) * sign).to_numpy(dtype=np.float64),
        "ret1": (ret1.fillna(0.0) * sign).to_numpy(dtype=np.float64),
        "range_pct": range_pct.fillna(0.0).to_numpy(dtype=np.float64),
        "qqq_ret1": (qqq_ret.fillna(0.0) * sign).to_numpy(dtype=np.float64),
        "vol_z": ones * vol_z,
        "tod": ones * tod,
    }
    cols = []
    for name in channels:
        arr = chan_map.get(name)
        if arr is None:
            arr = np.zeros(len(win), dtype=np.float64)
        # per-window z-ish scale for net$
        if name == "net$":
            s = float(np.std(arr))
            if s > 1e-12:
                arr = arr / s
        cols.append(arr.astype(np.float32))
    x = np.stack(cols, axis=-1)  # (T, C)
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def label_smooth_trend(
    stock_df: pd.DataFrame,
    *,
    asof_ts: pd.Timestamp,
    direction: str,
    horizon_minutes: int = 30,
    breakout_pct: float = 0.004,
    max_adverse_pct: float = 0.002,
    label_mode: str = "smooth",
) -> int | None:
    """Binary label on underlying path over next H minutes.

    Modes:
      - ``smooth``: MFE >= breakout_pct AND MAE <= max_adverse_pct
      - ``mfe``: MFE >= breakout_pct only (more learnable; default research path)
      - ``soft``: signed close-to-close return over horizon > 0
    """
    if stock_df is None or stock_df.empty or "timestamp" not in stock_df.columns:
        return None
    mode = str(label_mode or "smooth").strip().lower()
    asof = _to_ny_ts(asof_ts)
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
    if fut.empty or len(fut) < max(5, horizon_minutes // 3):
        return None
    hi = float(pd.to_numeric(fut["high"], errors="coerce").max())
    lo = float(pd.to_numeric(fut["low"], errors="coerce").min())
    last = float(pd.to_numeric(fut["close"], errors="coerce").iloc[-1])
    if str(direction).upper() == "UP":
        mfe = hi / px0 - 1.0
        mae = 1.0 - lo / px0
        signed_ret = last / px0 - 1.0
    else:
        mfe = 1.0 - lo / px0
        mae = hi / px0 - 1.0
        signed_ret = 1.0 - last / px0
    if mode in {"soft", "ret", "sign"}:
        return int(signed_ret > 0.0)
    if mode in {"mfe", "breakout"}:
        return int(mfe >= float(breakout_pct))
    # smooth (default)
    return int(mfe >= float(breakout_pct) and mae <= float(max_adverse_pct))


class TinyTCN:
    """Small causal TCN classifier; imported only when torch is available."""

    def __init__(self, n_channels: int, hidden: int = 32, n_layers: int = 3, kernel: int = 3):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        self._torch = torch
        self._F = F
        self.x_mean: np.ndarray | None = None  # (C,)
        self.x_std: np.ndarray | None = None  # (C,)

        class _Block(nn.Module):
            def __init__(self, c_in, c_out, k, dilation):
                super().__init__()
                pad = (k - 1) * dilation
                self.conv = nn.Conv1d(c_in, c_out, k, padding=pad, dilation=dilation)
                self.down = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else None
                self.drop = nn.Dropout(0.1)

            def forward(self, x):
                y = self.conv(x)
                # causal: trim future pad
                y = y[..., : x.size(-1)]
                y = F.relu(y)
                y = self.drop(y)
                res = x if self.down is None else self.down(x)
                return F.relu(y + res)

        layers = []
        c_in = n_channels
        for i in range(n_layers):
            layers.append(_Block(c_in, hidden, kernel, dilation=2**i))
            c_in = hidden
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, 1)

    def set_norm(self, mean: np.ndarray | None, std: np.ndarray | None) -> None:
        if mean is None or std is None:
            self.x_mean = None
            self.x_std = None
            return
        self.x_mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        self.x_std = np.asarray(std, dtype=np.float32).reshape(-1)

    def _normalize(self, x):
        torch = self._torch
        if self.x_mean is None or self.x_std is None:
            return x
        mean = torch.as_tensor(self.x_mean, dtype=x.dtype, device=x.device).view(1, 1, -1)
        std = torch.as_tensor(self.x_std, dtype=x.dtype, device=x.device).view(1, 1, -1)
        return (x - mean) / std.clamp_min(1e-6)

    def forward(self, x):
        # x: (B, T, C) -> (B, C, T)
        torch = self._torch
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float32)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self._normalize(x)
        h = self.net(x.transpose(1, 2))
        h = h.mean(dim=-1)
        return self.head(h).squeeze(-1)

    def predict_proba(self, x: np.ndarray) -> float:
        torch = self._torch
        self.net.eval()
        self.head.eval()
        with torch.no_grad():
            logit = self.forward(x)
            p = torch.sigmoid(logit).item()
        return float(p)


@dataclass
class TorchTcnGate:
    cfg: TcnGateConfig
    model: Any = None
    device: str = "cpu"

    def decide(
        self,
        *,
        symbol: str,
        direction: str,
        asof_ts: pd.Timestamp,
        stock_df: pd.DataFrame | None,
        qqq_df: pd.DataFrame | None = None,
        regime_vixy_z: float | None = None,
        regime_qqq_fp: float | None = None,
    ) -> TcnGateDecision:
        if not self.cfg.regime_applies(
            regime_vixy_z=regime_vixy_z, regime_qqq_fp=regime_qqq_fp
        ):
            return TcnGateDecision(
                True,
                1.0,
                None,
                "tcn_skip_regime",
                {
                    "vixy_z": regime_vixy_z,
                    "qqq_fp": regime_qqq_fp,
                },
            )
        if self.model is None:
            if self.cfg.block_on_missing:
                return TcnGateDecision(False, 1.0, None, "tcn_no_model")
            return TcnGateDecision(True, 1.0, None, "tcn_no_model_passthrough")
        x = build_feature_tensor(
            stock_df,
            qqq_df,
            asof_ts=asof_ts,
            window=self.cfg.window,
            channels=self.cfg.channels,
            direction=direction,
        )
        if x is None:
            if self.cfg.block_on_missing:
                return TcnGateDecision(False, 1.0, None, "tcn_feat_missing")
            return TcnGateDecision(True, 1.0, None, "tcn_feat_missing_passthrough")
        try:
            p = float(self.model.predict_proba(x))
        except Exception as exc:  # noqa: BLE001 — gate must not kill replay
            if self.cfg.block_on_missing:
                return TcnGateDecision(False, 1.0, None, "tcn_infer_error", {"err": str(exc)})
            return TcnGateDecision(True, 1.0, None, "tcn_infer_error_passthrough", {"err": str(exc)})

        if self.cfg.action == "block":
            ok = p >= self.cfg.p_min
            return TcnGateDecision(ok, 1.0 if ok else 0.0, p, "ok" if ok else "tcn_block")

        # scale
        if self.cfg.scale_mode == "linear":
            floor = max(0.0, min(1.0, float(self.cfg.scale_when_low)))
            scale = float(max(floor, min(1.0, p)))
        else:
            scale = 1.0 if p >= self.cfg.p_min else max(0.0, min(1.0, float(self.cfg.scale_when_low)))
        return TcnGateDecision(True, scale, p, "ok" if scale >= 1.0 - 1e-12 else "tcn_scale")


def load_tcn_gate(profile: dict[str, Any] | None) -> TcnGate:
    """Factory: returns NullTcnGate unless enabled + loadable checkpoint."""
    cfg = TcnGateConfig.from_profile(profile)
    if not cfg.enabled or cfg.action == "off":
        return NullTcnGate(cfg=cfg)
    if not cfg.model_path:
        return TorchTcnGate(cfg=cfg, model=None)
    path = Path(cfg.model_path).expanduser()
    if not path.is_file():
        return TorchTcnGate(cfg=cfg, model=None)
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        n_ch = int(ckpt.get("n_channels") or len(cfg.channels))
        hidden = int(ckpt.get("hidden", 32))
        n_layers = int(ckpt.get("n_layers", 3))
        model = TinyTCN(n_channels=n_ch, hidden=hidden, n_layers=n_layers)
        model.net.load_state_dict(ckpt["net"])
        model.head.load_state_dict(ckpt["head"])
        model.set_norm(ckpt.get("x_mean"), ckpt.get("x_std"))
        model.net.eval()
        model.head.eval()
        # allow checkpoint to override channel list / window
        ch = ckpt.get("channels")
        window = int(ckpt.get("window") or cfg.window)
        if ch:
            cfg = TcnGateConfig(
                enabled=cfg.enabled,
                action=cfg.action,
                p_min=cfg.p_min,
                scale_when_low=cfg.scale_when_low,
                scale_mode=cfg.scale_mode,
                window=window,
                model_path=cfg.model_path,
                block_on_missing=cfg.block_on_missing,
                channels=tuple(ch),
                only_when_vixy_z_max=cfg.only_when_vixy_z_max,
                only_when_vixy_z_min=cfg.only_when_vixy_z_min,
                only_when_abs_qqq_fp_max=cfg.only_when_abs_qqq_fp_max,
                only_when_abs_qqq_fp_min=cfg.only_when_abs_qqq_fp_min,
            )
        return TorchTcnGate(cfg=cfg, model=model)
    except Exception:
        return TorchTcnGate(cfg=cfg, model=None)


def save_tcn_checkpoint(
    model: TinyTCN,
    path: str | Path,
    *,
    channels: tuple[str, ...] | list[str],
    window: int,
    meta: dict[str, Any] | None = None,
) -> Path:
    import torch

    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "net": model.net.state_dict(),
        "head": model.head.state_dict(),
        "n_channels": len(channels),
        "channels": list(channels),
        "window": int(window),
        "hidden": int(model.head.in_features),
        "n_layers": len(list(model.net.children())),
        "x_mean": None if model.x_mean is None else np.asarray(model.x_mean, dtype=np.float32),
        "x_std": None if model.x_std is None else np.asarray(model.x_std, dtype=np.float32),
        "meta": meta or {},
    }
    torch.save(payload, out)
    return out
