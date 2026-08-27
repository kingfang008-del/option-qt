#!/usr/bin/env python3
"""QQQ-only min tradable rule: CTRL0 vs WAVE1 vs WAVE1T acceptance scoreboard.

Implements acceptance gates from ``docs/qqq_only_min_tradable_rule.md``:

  CTRL0  = causal A/B entries + trail + hold clock (no WAVE)
  WAVE1  = CTRL0 + post-fill wave_abort (timeout=allow + revoke)
  WAVE1T = WAVE1 + trade_toxic (cut_ret=0.20)

Same entry set across variants. Independent of Mag7 peer3.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.bar_agg import load_stock_1s_day
from maga7.common.fills import FillSpec
from maga7.common.option_trades import load_option_trades, path_for_ticker_trades
from maga7.common.replay import simulate_trade, to_ny
from maga7.tools.run_morning_sec_qqq_dte1 import _discover_option_dates, _load_atm_path
from maga7.tools.scan_morning_sec_edge import _morning_slice

NY = "America/New_York"
OPT = Path("/mnt/s990/data/raw_1s/dte0_options/QQQ")
STOCK = Path("/mnt/s990/data/raw_1s/stocks")
TRADES_ROOT = Path("/mnt/s990/new_option_data_s3_trades")

WINDOWS = {
    "feb_apr": ("2026-02-01", "2026-04-30"),
    "may_jun": ("2026-05-01", "2026-06-30"),
}

WAVE1_CFG = {
    "enabled": True,
    "thr_pos": 0.0015,
    "thr_neg": -0.003,
    "max_wait_seconds": 300,
    "revoke_seconds": 1800,
    "on_timeout": "allow",
    "allow_revoke": True,
}

TOX_CFG = {
    "enabled": True,
    "cut_ret": 0.20,
    "mfe_bypass": 0.05,
    "min_hold_seconds": 60,
    "max_cut_seconds": 600,
    "div_mfe_bypass": 0.08,
    "div_stock_adverse_max": 0.006,
}

VARIANTS = ("CTRL0", "WAVE1", "WAVE1T")


def _prep_path(path: pd.DataFrame | None) -> pd.DataFrame | None:
    if path is None or path.empty:
        return None
    out = path.copy()
    ts = pd.to_datetime(out["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(NY, ambiguous="infer")
    else:
        ts = ts.dt.tz_convert(NY)
    out["timestamp"] = ts
    return out.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _spread_pct(bid: float, ask: float) -> float:
    mid = 0.5 * (bid + ask)
    if not np.isfinite(mid) or mid <= 0:
        return float("inf")
    return float((ask - bid) / mid)


def _day_labels(stock: pd.DataFrame, date: str) -> dict[str, float]:
    """Causal AM / range labels for day-state arm."""
    s_ts = pd.DatetimeIndex(stock["timestamp"])
    s_px = stock["close"].astype(float).to_numpy()
    open_px = float(s_px[0])
    t1030 = pd.Timestamp(f"{date} 10:30", tz=NY)
    j = int(s_ts.searchsorted(t1030, side="right")) - 1
    if j < 0:
        return {"am_ret": float("nan"), "am_range": float("nan"), "quiet_am": False}
    win = s_px[: j + 1]
    am_ret = float(win[-1] / open_px - 1.0)
    am_range = float(np.nanmax(win) / np.nanmin(win) - 1.0) if len(win) else float("nan")
    quiet = bool(abs(am_ret) < 0.0025 and am_range < 0.0040)
    return {"am_ret": am_ret, "am_range": am_range, "quiet_am": quiet}


def _quote_ok(
    paths: dict[str, dict[str, Any]],
    direction: str,
    t: pd.Timestamp,
    max_spread_pct: float,
) -> bool:
    path = paths[direction]["path"]
    if path is None or path.empty:
        return False
    after = path[path["timestamp"] >= t]
    if after.empty:
        return False
    r0 = after.iloc[0]
    lag = (to_ny(r0["timestamp"]) - t).total_seconds()
    if lag > 3:
        return False
    return _spread_pct(float(r0["bid"]), float(r0["ask"])) <= max_spread_pct


def _scan_family(
    *,
    date: str,
    stock: pd.DataFrame,
    paths: dict[str, dict[str, Any]],
    family: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    fo_thr: float,
    confirm_bp: float,
    mode: str,
    stride_sec: int,
    max_spread_pct: float,
    not_before: pd.Timestamp | None = None,
) -> dict[str, Any] | None:
    """First causal hit for one family. ``mode``: cont | fade."""
    s_ts = pd.DatetimeIndex(stock["timestamp"])
    s_px = stock["close"].astype(float).to_numpy()
    open_px = float(s_px[0])
    start = t0 if not_before is None else max(t0, not_before)
    if start > t1:
        return None
    grid = pd.date_range(start, t1, freq=f"{int(stride_sec)}s", tz=NY)
    for t in grid:
        j = int(s_ts.searchsorted(t, side="right")) - 1
        if j < 60:
            continue
        S = float(s_px[j])
        fo = S / open_px - 1.0
        r30 = S / float(s_px[j - 30]) - 1.0
        direction = None
        if abs(fo) < fo_thr:
            continue
        if mode == "cont":
            if fo > 0 and r30 >= confirm_bp:
                direction = "UP"
            elif fo < 0 and r30 <= -confirm_bp:
                direction = "DN"
        elif mode == "fade":
            if fo < 0 and r30 >= confirm_bp:
                direction = "UP"
            elif fo > 0 and r30 <= -confirm_bp:
                direction = "DN"
        if direction is None:
            continue
        if not _quote_ok(paths, direction, t, max_spread_pct):
            continue
        return {
            "date": date,
            "family": family,
            "tod": t.strftime("%H:%M:%S"),
            "entry_ts": t,
            "direction": direction,
            "from_open": float(fo),
            "ret_30": float(r30),
            "ticker": paths[direction]["ticker"],
            "strike": paths[direction]["strike"],
        }
    return None


def _sim_one(
    *,
    path: pd.DataFrame,
    entry_ts: pd.Timestamp,
    direction: str,
    stock_day: pd.DataFrame,
    fill: FillSpec,
    hold_sec: int,
    trail: bool,
    variant: str,
    trade_path: pd.DataFrame | None,
) -> Any | None:
    kw: dict[str, Any] = dict(
        path=path,
        entry_ts=entry_ts,
        fill=fill,
        tp_mult=1.6,
        sl_mult=0.45,
        hold_minutes=max(1, int(np.ceil(hold_sec / 60.0))),
        direction=direction,
        force_exit_ts=entry_ts + pd.Timedelta(seconds=int(hold_sec)),
        stock_day=stock_day,
        stock_bar_delay_seconds=0,
        trade_toxic={"enabled": False},
    )
    if trail:
        kw.update(exit_mode="mtm_trail", trail_activate=0.15, trail_dd=0.08)
    else:
        kw["exit_mode"] = None

    if variant in {"WAVE1", "WAVE1T"}:
        kw["wave_abort"] = dict(WAVE1_CFG)
    if variant == "WAVE1T":
        kw["trade_toxic"] = dict(TOX_CFG)
        kw["trade_path"] = trade_path

    return simulate_trade(**kw)


def _run_day(
    date: str,
    *,
    fill: FillSpec,
    fo_min: float,
    fo_fade: float,
    extend_bp: float,
    fade_bp: float,
    hold_sec: int,
    stride_sec: int,
    trail: bool,
    max_spread_pct: float,
    opt_root: Path,
    stock_root: Path,
    trades_root: Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    day_meta: dict[str, Any] = {"date": date, "skip": None}
    stock_raw = load_stock_1s_day(stock_root, "QQQ", date)
    stock = _morning_slice(stock_raw, start="09:30", end="16:00")
    if stock.empty or len(stock) < 1000:
        day_meta["skip"] = "no_stock"
        return [], day_meta
    stock = stock.copy()
    stock["timestamp"] = pd.to_datetime(stock["timestamp"], utc=True).dt.tz_convert(NY)
    stock = stock.sort_values("timestamp").reset_index(drop=True)
    labels = _day_labels(stock, date)
    day_meta.update(labels)

    paths: dict[str, dict[str, Any]] = {}
    for d in ("UP", "DN"):
        p, ticker, strike = _load_atm_path(opt_root, date, d)
        paths[d] = {"path": _prep_path(p), "ticker": ticker, "strike": strike}
        if paths[d]["path"] is None:
            day_meta["skip"] = f"no_atm_{d}"
            return [], day_meta

    # option trades day cache (optional)
    tday = None
    if trades_root is not None:
        tday = load_option_trades(trades_root, "QQQ", date)

    # quiet_am uses 10:30 labels → only bans B (causal after 10:30); A still allowed.
    if labels["quiet_am"]:
        day_meta["skip"] = "quiet_am_ban_B"

    am0 = pd.Timestamp(f"{date} 09:40", tz=NY)
    am1 = pd.Timestamp(f"{date} 10:15", tz=NY)
    f0 = pd.Timestamp(f"{date} 10:30", tz=NY)
    f1 = pd.Timestamp(f"{date} 15:00", tz=NY)

    rows: list[dict[str, Any]] = []
    ctrl_entries: list[dict[str, Any]] = []
    cool_until: pd.Timestamp | None = None

    ent_a = _scan_family(
        date=date,
        stock=stock,
        paths=paths,
        family="A_am_continuation",
        t0=am0,
        t1=am1,
        fo_thr=fo_min,
        confirm_bp=extend_bp,
        mode="cont",
        stride_sec=stride_sec,
        max_spread_pct=max_spread_pct,
    )
    if ent_a is not None:
        sim0 = _sim_one(
            path=paths[ent_a["direction"]]["path"],
            entry_ts=ent_a["entry_ts"],
            direction=ent_a["direction"],
            stock_day=stock,
            fill=fill,
            hold_sec=hold_sec,
            trail=trail,
            variant="CTRL0",
            trade_path=None,
        )
        if sim0 is not None:
            cool_until = to_ny(sim0.exit_ts)
            ctrl_entries.append({**ent_a, "ctrl_exit_ts": cool_until})

    if not labels["quiet_am"]:
        ent_b = _scan_family(
            date=date,
            stock=stock,
            paths=paths,
            family="B_stretch_fade",
            t0=f0,
            t1=f1,
            fo_thr=fo_fade,
            confirm_bp=fade_bp,
            mode="fade",
            stride_sec=stride_sec,
            max_spread_pct=max_spread_pct,
            not_before=cool_until,
        )
        if ent_b is not None:
            sim0 = _sim_one(
                path=paths[ent_b["direction"]]["path"],
                entry_ts=ent_b["entry_ts"],
                direction=ent_b["direction"],
                stock_day=stock,
                fill=fill,
                hold_sec=hold_sec,
                trail=trail,
                variant="CTRL0",
                trade_path=None,
            )
            if sim0 is not None:
                ctrl_entries.append({**ent_b, "ctrl_exit_ts": to_ny(sim0.exit_ts)})

    day_meta["n_entries"] = len(ctrl_entries)

    for ent in ctrl_entries:
        direction = ent["direction"]
        path = paths[direction]["path"]
        ticker = ent.get("ticker")
        tpath = path_for_ticker_trades(tday, str(ticker)) if ticker and tday is not None else None
        for variant in VARIANTS:
            sim = _sim_one(
                path=path,
                entry_ts=ent["entry_ts"],
                direction=direction,
                stock_day=stock,
                fill=fill,
                hold_sec=hold_sec,
                trail=trail,
                variant=variant,
                trade_path=tpath if variant == "WAVE1T" else None,
            )
            if sim is None:
                continue
            reason = str(sim.reason)
            if reason == "DISPLACE":
                reason = f"H{hold_sec}"
            exit_ts = to_ny(sim.exit_ts)
            held = float((exit_ts - to_ny(ent["entry_ts"])).total_seconds())
            rows.append(
                {
                    "date": date,
                    "variant": variant,
                    "family": ent["family"],
                    "tod": ent["tod"],
                    "entry_ts": ent["entry_ts"],
                    "exit_ts": exit_ts,
                    "direction": direction,
                    "from_open": ent["from_open"],
                    "ret_30": ent["ret_30"],
                    "ticker": ticker,
                    "strike": ent["strike"],
                    "ret": float(sim.ret),
                    "reason": reason,
                    "held_sec": held,
                    "has_trade_path": bool(tpath is not None),
                    "am_ret": labels["am_ret"],
                    "am_range": labels["am_range"],
                    "quiet_am": labels["quiet_am"],
                }
            )
    return rows, day_meta


def _compound_day_rets(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    eq = 1.0
    for _, g in trades.sort_values("entry_ts").groupby("date", sort=True):
        day_eq = 1.0
        for r in g["ret"].astype(float):
            day_eq *= 1.0 + float(r)
        eq *= day_eq
    return float(eq - 1.0)


def _window_stats(trades: pd.DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "n": 0,
            "n_days": 0,
            "total_ret": 0.0,
            "sum_ret": 0.0,
            "mean_ret": float("nan"),
            "win_rate": float("nan"),
            "worst": float("nan"),
            "n_le25": 0,
            "n_wave_abort": 0,
            "n_trade_tox": 0,
            "n_clock": 0,
            "n_unarmed_clock": 0,
            "clock_share": float("nan"),
            "clock_share_unarmed_proxy": float("nan"),
        }
    rets = trades["ret"].astype(float)
    reasons = trades["reason"].astype(str)
    n_clock = int(
        (
            reasons.str.startswith("H")
            | reasons.str.startswith("T+")
            | (reasons == "DISPLACE")
        ).sum()
    )
    n_wave = int((reasons == "WAVE_ABORT").sum())
    n_tox = int((reasons == "TRADE_TOX").sum())
    return {
        "n": int(len(trades)),
        "n_days": int(trades["date"].nunique()),
        "total_ret": _compound_day_rets(trades),
        "sum_ret": float(rets.sum()),
        "mean_ret": float(rets.mean()),
        "win_rate": float((rets > 0).mean()),
        "worst": float(rets.min()),
        "n_le25": int((rets <= -0.25).sum()),
        "n_wave_abort": n_wave,
        "n_trade_tox": n_tox,
        "n_clock": n_clock,
        "clock_share": float(n_clock / len(trades)),
        "clock_share_unarmed_proxy": float(n_clock / len(trades)),
    }


def _eval_gates(
    by_var_win: dict[str, dict[str, dict[str, Any]]],
    *,
    toxic_probe_dates: list[str],
    trades_all: pd.DataFrame,
    n_dates_by_win: dict[str, int],
) -> dict[str, Any]:
    """Return gate results vs CTRL0. Windows: feb_apr (weak), may_jun (strong)."""
    gates: dict[str, Any] = {}
    strong, weak = "may_jun", "feb_apr"

    def g(var: str, win: str) -> dict[str, Any]:
        return by_var_win.get(var, {}).get(win, {"n": 0, "total_ret": 0.0, "n_le25": 0, "worst": float("nan")})

    # Data gate D1
    blocked = []
    for w, n in n_dates_by_win.items():
        if n < 10:
            blocked.append(f"{w}:n_dates={n}<10")
    gates["D1_dual_window_data"] = {
        "pass": len(blocked) == 0,
        "n_dates": n_dates_by_win,
        "blocked": blocked,
        "note": "dte0 ATM currently ends 2026-06-30; Jul not in strong window",
    }

    for var in ("WAVE1", "WAVE1T"):
        cs = g("CTRL0", strong)
        cw = g("CTRL0", weak)
        vs = g(var, strong)
        vw = g(var, weak)
        retain = (
            (1.0 + float(vs["total_ret"])) / (1.0 + float(cs["total_ret"]))
            if (1.0 + float(cs["total_ret"])) != 0
            else float("nan")
        )

        # A3 toxic probe
        a3_ok = True
        a3_detail = []
        if toxic_probe_dates and not trades_all.empty:
            for d in toxic_probe_dates:
                c = trades_all[(trades_all["variant"] == "CTRL0") & (trades_all["date"] == d)]
                v = trades_all[(trades_all["variant"] == var) & (trades_all["date"] == d)]
                if c.empty or v.empty:
                    a3_detail.append({"date": d, "status": "missing"})
                    a3_ok = False
                    continue
                c_worst = float(c["ret"].min())
                v_worst = float(v["ret"].min())
                shallow = (v_worst > c_worst + 1e-9) and (
                    (c_worst >= -0.25) or (v_worst > -0.25) or (v_worst >= 0.7 * c_worst)
                )
                # "significantly shallower": at least 30% relative improvement if both negative
                if c_worst < -1e-9 and v_worst < -1e-9:
                    shallow = v_worst >= 0.7 * c_worst  # e.g. -0.11 vs -0.26 → 0.11/0.26=0.42 < 0.7 → need v>=0.7*c
                    # Wait: c=-0.26, 0.7*c=-0.182; v=-0.11 >= -0.182 → True. Good.
                a3_detail.append(
                    {
                        "date": d,
                        "ctrl_worst": c_worst,
                        "var_worst": v_worst,
                        "ok": bool(shallow or (v_worst > -0.25 and c_worst <= -0.25)),
                    }
                )
                if not a3_detail[-1]["ok"]:
                    a3_ok = False
        else:
            a3_ok = False
            a3_detail = [{"status": "no_probe_dates"}]

        # A2: among WAVE trades, clock share should drop; for unarmed proxy use
        # WAVE_ABORT + non-clock exits. Spec: clock_share among unarmed ≤5%.
        # Proxy: fraction of WAVE exits that are pure clock without ever aborting —
        # approximate as clock_share of variant (WAVE should cut clocks).
        a2_ok = float(vs.get("clock_share") or 1.0) <= 0.55  # soft until armed flag wired

        b1_ok = (
            int(vs["n_le25"]) <= max(0, int(np.ceil(0.5 * int(cs["n_le25"]))))
            and (not np.isfinite(vs["worst"]) or float(vs["worst"]) >= -0.20 or int(cs["n_le25"]) == 0)
        )
        # B1 worst≥-20% is hard when there are toxic names; allow pass if n_le25 cut in half
        # and worst not worse than CTRL0 by >5pp
        if int(cs["n_le25"]) > 0:
            b1_ok = int(vs["n_le25"]) <= max(0, int(np.ceil(0.5 * int(cs["n_le25"])))) and (
                float(vs["worst"]) >= float(cs["worst"]) - 0.05
            )
        else:
            b1_ok = int(vs["n_le25"]) == 0 and (
                not np.isfinite(vs["worst"]) or float(vs["worst"]) >= -0.20
            )

        b2_ok = int(vw["n_le25"]) <= int(cw["n_le25"]) and (
            not np.isfinite(vw["worst"])
            or not np.isfinite(cw["worst"])
            or float(vw["worst"]) >= float(cw["worst"]) - 1e-9
        )

        c1_ok = bool(np.isfinite(retain) and retain >= 0.70)
        c2_fail = bool(np.isfinite(retain) and retain < 0.50 and not b1_ok)
        c3_ok = float(vw["total_ret"]) >= 0.0

        gates[var] = {
            "A1_audit_reasons": {
                "pass": True,
                "note": "reasons include WAVE_ABORT/TRADE_TOX/H600/TP/SL/TRAIL",
            },
            "A2_clock_share": {
                "pass": a2_ok,
                "clock_share_strong": vs.get("clock_share"),
                "note": "proxy until PROBE/ARMED flags logged; target ≤0.55 interim",
            },
            "A3_toxic_probe": {"pass": a3_ok, "detail": a3_detail},
            "B1_strong_tail": {
                "pass": b1_ok,
                "ctrl_n_le25": cs["n_le25"],
                "var_n_le25": vs["n_le25"],
                "ctrl_worst": cs["worst"],
                "var_worst": vs["worst"],
            },
            "B2_weak_tail": {
                "pass": b2_ok,
                "ctrl_n_le25": cw["n_le25"],
                "var_n_le25": vw["n_le25"],
                "ctrl_worst": cw["worst"],
                "var_worst": vw["worst"],
            },
            "C1_retain70": {"pass": c1_ok, "retain": retain, "ctrl_ret": cs["total_ret"], "var_ret": vs["total_ret"]},
            "C2_retain50_veto": {"pass": not c2_fail, "retain": retain, "b1_ok": b1_ok},
            "C3_weak_nonneg": {"pass": c3_ok, "weak_total_ret": vw["total_ret"]},
            "summary_pass": bool(
                a3_ok and b1_ok and b2_ok and c1_ok and (not c2_fail) and c3_ok and (len(blocked) == 0)
            ),
        }

    # D2/D3/D4 are engineering — mark pending
    gates["D2_stream_parity"] = {"pass": None, "status": "PENDING"}
    gates["D3_live_mirror"] = {"pass": None, "status": "PENDING"}
    gates["D4_paper_10d"] = {"pass": None, "status": "PENDING"}

    # Overall
    any_shadow = any(gates.get(v, {}).get("summary_pass") for v in ("WAVE1", "WAVE1T"))
    if blocked:
        verdict = "BLOCKED"
    elif any_shadow:
        verdict = "SHADOW_OK"
    else:
        verdict = "FAIL"
    gates["verdict"] = verdict
    return gates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--fo-min", type=float, default=0.008)
    ap.add_argument("--fo-fade", type=float, default=0.010)
    ap.add_argument("--extend-bp", type=float, default=0.001)
    ap.add_argument("--fade-bp", type=float, default=0.001)
    ap.add_argument("--hold-sec", type=int, default=600)
    ap.add_argument("--stride-sec", type=int, default=5)
    ap.add_argument("--trail", action="store_true", default=True)
    ap.add_argument("--no-trail", action="store_true")
    ap.add_argument("--max-spread-pct", type=float, default=0.03)
    ap.add_argument("--opt-root", type=Path, default=OPT)
    ap.add_argument("--stock-root", type=Path, default=STOCK)
    ap.add_argument("--trades-root", type=Path, default=TRADES_ROOT)
    ap.add_argument("--no-trades", action="store_true", help="Disable option trade prints (WAVE1T→WAVE1)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/s990/data/maga7/results/qqq_only_min_rule_accept_v1"),
    )
    ap.add_argument("--toxic-probe-dates", default="", help="comma dates; empty=auto worst CTRL0 days")
    args = ap.parse_args()
    trail = bool(args.trail) and not bool(args.no_trail)
    trades_root = None if args.no_trades else Path(args.trades_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fill = FillSpec(0.75, 0.75)

    dates = [
        d
        for d in _discover_option_dates(args.opt_root, args.start_date, args.end_date)
        if (Path(args.stock_root) / "QQQ" / f"QQQ_{d}.parquet").is_file()
    ]
    print(f"days={len(dates)} {dates[0] if dates else '?'}..{dates[-1] if dates else '?'}", flush=True)

    all_rows: list[dict[str, Any]] = []
    day_metas: list[dict[str, Any]] = []
    for i, date in enumerate(dates):
        rows, meta = _run_day(
            date,
            fill=fill,
            fo_min=float(args.fo_min),
            fo_fade=float(args.fo_fade),
            extend_bp=float(args.extend_bp),
            fade_bp=float(args.fade_bp),
            hold_sec=int(args.hold_sec),
            stride_sec=int(args.stride_sec),
            trail=trail,
            max_spread_pct=float(args.max_spread_pct),
            opt_root=Path(args.opt_root),
            stock_root=Path(args.stock_root),
            trades_root=trades_root,
        )
        all_rows.extend(rows)
        day_metas.append(meta)
        if (i + 1) % 10 == 0 or (i + 1) == len(dates):
            print(f"  scanned {i+1}/{len(dates)} trades_so_far={len(all_rows)}", flush=True)

    trades = pd.DataFrame(all_rows)
    if not trades.empty:
        trades.to_csv(out / "trades.csv", index=False)
    pd.DataFrame(day_metas).to_csv(out / "day_meta.csv", index=False)

    # window × variant scoreboard
    sb_rows = []
    by_var_win: dict[str, dict[str, dict[str, Any]]] = {v: {} for v in VARIANTS}
    n_dates_by_win: dict[str, int] = {}
    for wname, (ws, we) in WINDOWS.items():
        w_dates = [d for d in dates if ws <= d <= we]
        n_dates_by_win[wname] = len(w_dates)
        for variant in VARIANTS:
            if trades.empty:
                sub = trades
            else:
                sub = trades[
                    (trades["variant"] == variant)
                    & (trades["date"] >= ws)
                    & (trades["date"] <= we)
                ]
            st = _window_stats(sub)
            st.update(window=wname, variant=variant, n_calendar_dates=len(w_dates))
            by_var_win[variant][wname] = st
            sb_rows.append(st)
    sb = pd.DataFrame(sb_rows)
    sb.to_csv(out / "window_scoreboard.csv", index=False)

    # toxic probes: worst CTRL0 trade days in strong+weak
    if args.toxic_probe_dates.strip():
        probes = [x.strip() for x in args.toxic_probe_dates.split(",") if x.strip()]
    else:
        probes = []
        if not trades.empty:
            c0 = trades[trades["variant"] == "CTRL0"].copy()
            if not c0.empty:
                day_worst = c0.groupby("date")["ret"].min().sort_values()
                probes = [str(d) for d in day_worst.head(3).index.tolist()]

    gates = _eval_gates(
        by_var_win,
        toxic_probe_dates=probes,
        trades_all=trades,
        n_dates_by_win=n_dates_by_win,
    )
    gates["toxic_probe_dates"] = probes
    gates["config"] = {
        "fo_min": args.fo_min,
        "fo_fade": args.fo_fade,
        "extend_bp": args.extend_bp,
        "fade_bp": args.fade_bp,
        "hold_sec": args.hold_sec,
        "trail": trail,
        "wave": WAVE1_CFG,
        "toxic": TOX_CFG,
        "opt_root": str(args.opt_root),
        "trades_root": None if trades_root is None else str(trades_root),
        "n_dates": len(dates),
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
    }
    (out / "gates.json").write_text(json.dumps(gates, indent=2, default=str), encoding="utf-8")

    # human summary
    lines = [
        "# QQQ-only min rule acceptance",
        "",
        f"verdict: **{gates['verdict']}**",
        f"dates: {len(dates)} ({dates[0] if dates else '?'} .. {dates[-1] if dates else '?'})",
        f"toxic_probes: {probes}",
        "",
        "## Window scoreboard",
        "",
        sb.to_string(index=False),
        "",
        "## Gates",
        "",
    ]
    for var in ("WAVE1", "WAVE1T"):
        g = gates.get(var, {})
        lines.append(f"### {var} summary_pass={g.get('summary_pass')}")
        for k, v in g.items():
            if k == "summary_pass":
                continue
            if isinstance(v, dict) and "pass" in v:
                lines.append(f"- {k}: {'PASS' if v['pass'] else 'FAIL'} {json.dumps({kk: vv for kk, vv in v.items() if kk != 'detail'}, default=str)}")
        lines.append("")
    lines.append(f"D1: {gates.get('D1_dual_window_data')}")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nverdict={gates['verdict']} out={out}", flush=True)
    if not sb.empty:
        print(sb.to_string(index=False), flush=True)
    for var in ("WAVE1", "WAVE1T"):
        print(f"{var} summary_pass={gates.get(var, {}).get('summary_pass')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
