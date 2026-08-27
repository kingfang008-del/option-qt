#!/usr/bin/env python3
"""Hold-time **feature** exits on AM pocket (trade-mark only).

Not fixed hold. At each option print, read causal stock features and decide
stay/exit. Target: lift mean_capture vs TP8 toward ≥20% with econ dual PASS.

Entry default: ``vd_soft`` on ``no_b_up`` (same book as stock_up work).

Feature families:
  - micro_mom: stock signed ret over last L seconds; exit when flips after arm
  - stall: option no new high for T sec + stock micro_mom≤0 after arm
  - mf_flip: mf100 sign against dir after min_hold
  - soft_floor: after +arm, stop to BE/floor; stay for higher TP while stock mom>0
  - baselines: TP8, widen_tp2+stock_up

Example:
  PYTHONPATH=. python -m maga7.tools.scan_am_pocket_hold_feature_exit \\
    --tag research_am_pocket_hold_feature_exit
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
from maga7.common.config import load_profile
from maga7.common.option_trade_tpsl import simulate_trade_tpsl
from maga7.common.option_trades import load_option_trades
from maga7.common.replay import to_ny
from maga7.common.session_1s_features import prepare_day_arrays
from maga7.common.stock_path_whipsaw import signed_stock_ret
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _stats
from maga7.tools.scan_am_pocket_exit_design import _path_window
from maga7.tools.scan_am_pocket_multi_gate import build_gates
from maga7.tools.scan_am_pocket_path_exit import _stock_series
from maga7.tools.scan_am_pocket_regime_ladder_v2 import (
    LADDERS_V2,
    _apply_ladder,
    _stock_upgrade,
    _window_of,
    classify_regime_v2,
)
from maga7.tools.scan_am_pocket_risk_optimize import POCKET_SETS, _equity_stats
from maga7.tools.scan_session_horizon_foresight import _paths_by_ticker

DEFAULT_ENRICHED = Path(
    "/mnt/s990/data/maga7/results/research_am_pocket_multi_gate/enriched_probes.csv"
)
DEFAULT_TRADES = Path("/mnt/s990/new_option_data_s3_trades")
DEFAULT_STOCK = Path("/mnt/s990/data/raw_1s/stocks")
PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
WINDOWS = (
    ("may_jul09", "2026-05-01", "2026-07-09"),
    ("jul10_23", "2026-07-10", "2026-07-23"),
)


def _stock_idx_at(ts_ns: np.ndarray, entry_ns: int, hold_sec: float) -> int:
    t = entry_ns + int(hold_sec * 1e9)
    j = int(np.searchsorted(ts_ns, t, side="right") - 1)
    return max(0, min(j, len(ts_ns) - 1))


def _micro_mom(
    close: np.ndarray,
    ts_ns: np.ndarray,
    j: int,
    look_sec: float,
    direction: str,
) -> float:
    """Signed stock ret over last look_sec ending at index j."""
    t_end = int(ts_ns[j])
    t0 = t_end - int(look_sec * 1e9)
    j0 = int(np.searchsorted(ts_ns, t0, side="left"))
    j0 = max(0, min(j0, j))
    a = float(close[j0])
    b = float(close[j])
    if not np.isfinite(a) or not np.isfinite(b) or a <= 0 or b <= 0:
        return 0.0
    return float(signed_stock_ret(b, a, direction))


def simulate_hold_features(
    opt_rets: np.ndarray,
    opt_holds: np.ndarray,
    *,
    stock_arr: dict[str, np.ndarray],
    entry_ts: pd.Timestamp,
    direction: str,
    mode: str,
    params: dict[str, Any],
) -> dict[str, Any]:
    tp = float(params.get("tp", 0.08))
    sl = float(params.get("sl", 0.15))
    max_h = float(params.get("max_hold", 300))
    min_hold = float(params.get("min_hold", 15))
    arm = float(params.get("arm", 0.08))
    look = float(params.get("look_sec", 15))
    stall_sec = float(params.get("stall_sec", 20))
    mom_cut = float(params.get("mom_cut", 0.0))
    floor = float(params.get("floor", 0.0))
    tp2 = float(params.get("tp2", 0.25))

    ts_ns = stock_arr["ts_ns"]
    close = stock_arr["close"]
    mf100 = stock_arr["mf100"]
    entry_ns = int(to_ny(entry_ts).value)
    # stock entry px
    j_e = int(np.searchsorted(ts_ns, entry_ns, side="left"))
    if j_e >= len(ts_ns):
        return {"ret": float(opt_rets[-1]), "reason": "no_stock", "hold_sec": float(opt_holds[-1])}
    entry_px = float(close[j_e])
    if entry_px <= 0:
        return {"ret": float(opt_rets[-1]), "reason": "bad_stock", "hold_sec": float(opt_holds[-1])}

    peak = -1.0
    peak_t = 0.0
    armed = False
    stop = -sl

    for i in range(1, len(opt_rets)):
        r = float(opt_rets[i])
        h = float(opt_holds[i])
        if h > max_h:
            return {"ret": float(opt_rets[i - 1]), "reason": "max_hold", "hold_sec": float(opt_holds[i - 1])}

        if r > peak:
            peak = r
            peak_t = h
        if r >= arm:
            armed = True
            if mode == "soft_floor":
                stop = max(stop, floor)

        # hard SL / soft floor
        if r <= stop:
            return {"ret": r, "reason": "sl" if stop < -1e-9 else "floor", "hold_sec": h}

        # hard TP (primary or secondary)
        if mode in {"tpsl", "micro_mom", "stall", "mf_flip"} and r >= tp:
            return {"ret": r, "reason": "tp", "hold_sec": h}
        if mode == "soft_floor" and r >= tp2:
            return {"ret": r, "reason": "tp2", "hold_sec": h}

        if h < min_hold:
            continue

        j = _stock_idx_at(ts_ns, entry_ns, h)
        mom = _micro_mom(close, ts_ns, j, look, direction)
        s_from_entry = float(signed_stock_ret(float(close[j]), entry_px, direction))
        mf = float(mf100[j]) if np.isfinite(mf100[j]) else 0.0
        mf_ok = (mf > 0) if direction == "UP" else (mf < 0)

        if mode == "micro_mom" and armed:
            # stock short-window turns against → cut (even if below hard TP)
            if mom <= mom_cut:
                return {"ret": r, "reason": "micro_mom_flip", "hold_sec": h}

        if mode == "stall" and armed:
            if (h - peak_t) >= stall_sec and mom <= mom_cut:
                return {"ret": r, "reason": "stall", "hold_sec": h}

        if mode == "mf_flip" and armed:
            if not mf_ok and s_from_entry < 0.001:
                return {"ret": r, "reason": "mf_flip", "hold_sec": h}

        if mode == "soft_floor" and armed:
            # stay while micro mom supportive; cut when mom flips after peak
            if mom <= mom_cut and (h - peak_t) >= float(params.get("give_sec", 10)):
                return {"ret": r, "reason": "mom_give", "hold_sec": h}

        if mode == "combo" and armed:
            # stall OR (mom flip and not making HH)
            if (h - peak_t) >= stall_sec and mom <= mom_cut:
                return {"ret": r, "reason": "combo_stall", "hold_sec": h}
            if mom <= mom_cut and r < peak - float(params.get("opt_give", 0.03)):
                return {"ret": r, "reason": "combo_mom", "hold_sec": h}
            if r >= tp2:
                return {"ret": r, "reason": "tp2", "hold_sec": h}
            # soft floor after arm
            if r <= floor:
                return {"ret": r, "reason": "floor", "hold_sec": h}

    return {"ret": float(opt_rets[-1]), "reason": "max_hold", "hold_sec": float(opt_holds[-1])}


def _exit_cfgs() -> list[dict[str, Any]]:
    cfgs: list[dict[str, Any]] = [
        {"name": "baseline_tp8", "kind": "tpsl", "tp": 0.08, "sl": 0.15, "max_hold": 240},
        {"name": "widen_stock_up", "kind": "widen"},
    ]
    # micro_mom: wider TP so features can act before hard TP
    for look in (10, 15, 30):
        for mom_cut in (0.0, -0.0005, -0.001):
            for tp in (0.15, 0.25):
                cfgs.append(
                    {
                        "name": f"mom_L{look}_c{mom_cut:g}_tp{tp:g}",
                        "kind": "feat",
                        "mode": "micro_mom",
                        "look_sec": look,
                        "mom_cut": mom_cut,
                        "tp": tp,
                        "sl": 0.15,
                        "arm": 0.05,
                        "min_hold": 20,
                        "max_hold": 480,
                    }
                )
    # stall
    for stall in (15, 25, 40):
        for tp in (0.15, 0.25):
            cfgs.append(
                {
                    "name": f"stall{stall}_tp{tp:g}",
                    "kind": "feat",
                    "mode": "stall",
                    "stall_sec": stall,
                    "look_sec": 15,
                    "mom_cut": 0.0,
                    "tp": tp,
                    "sl": 0.15,
                    "arm": 0.06,
                    "min_hold": 20,
                    "max_hold": 480,
                }
            )
    # mf_flip
    for tp in (0.12, 0.20):
        cfgs.append(
            {
                "name": f"mf_flip_tp{tp:g}",
                "kind": "feat",
                "mode": "mf_flip",
                "tp": tp,
                "sl": 0.15,
                "arm": 0.05,
                "min_hold": 30,
                "max_hold": 480,
            }
        )
    # soft floor + mom
    for floor in (0.0, 0.03):
        for tp2 in (0.20, 0.30):
            cfgs.append(
                {
                    "name": f"soft_f{floor:g}_tp2{tp2:g}",
                    "kind": "feat",
                    "mode": "soft_floor",
                    "arm": 0.08,
                    "floor": floor,
                    "tp2": tp2,
                    "sl": 0.15,
                    "look_sec": 15,
                    "mom_cut": 0.0,
                    "give_sec": 12,
                    "min_hold": 20,
                    "max_hold": 600,
                    "tp": 9.0,
                }
            )
    # combo
    for stall in (20, 30):
        for tp2 in (0.20, 0.30):
            cfgs.append(
                {
                    "name": f"combo_st{stall}_tp2{tp2:g}",
                    "kind": "feat",
                    "mode": "combo",
                    "stall_sec": stall,
                    "look_sec": 15,
                    "mom_cut": 0.0,
                    "opt_give": 0.04,
                    "arm": 0.06,
                    "floor": 0.0,
                    "tp2": tp2,
                    "sl": 0.15,
                    "min_hold": 20,
                    "max_hold": 600,
                    "tp": 9.0,
                }
            )
    return cfgs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--enriched", default=str(DEFAULT_ENRICHED))
    ap.add_argument("--trades-root", default=str(DEFAULT_TRADES))
    ap.add_argument("--stock-1s", default=str(DEFAULT_STOCK))
    ap.add_argument("--tag", default="research_am_pocket_hold_feature_exit")
    ap.add_argument("--entry", default="vd_soft")
    ap.add_argument("--slip", type=float, default=0.01)
    ap.add_argument("--position-frac", type=float, default=0.20)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--cooldown-minutes", type=float, default=10.0)
    ap.add_argument("--capture-target", type=float, default=0.20)
    args = ap.parse_args(argv)

    prof = load_profile(args.profile)
    out = Path(prof["_paths"]["results_dir"]) / args.tag
    out.mkdir(parents=True, exist_ok=True)

    probes = pd.read_csv(args.enriched)
    probes["entry_ts"] = pd.to_datetime(probes["entry_ts"])
    probes = probes[probes["enrich_ok"] == True].copy()  # noqa: E712
    pdf = pd.DataFrame(sorted(POCKET_SETS["no_b_up"]), columns=["session", "tod_bucket", "dir"])
    probes = probes.merge(pdf, on=["session", "tod_bucket", "dir"], how="inner")
    gfn = dict(build_gates())[str(args.entry)]
    probes = probes[probes.apply(gfn, axis=1)].copy()
    print(f"entry={args.entry} probes={len(probes)}", flush=True)

    trades_root = Path(args.trades_root)
    stock_root = Path(args.stock_1s)
    tcache: dict = {}
    scache: dict = {}

    def paths_for(date, sym):
        key = (date, sym)
        if key not in tcache:
            tday = load_option_trades(trades_root, sym, date)
            tcache[key] = _paths_by_ticker(tday) if tday is not None and not tday.empty else {}
        return tcache[key]

    def stock_for(date, sym):
        key = (date, sym)
        if key not in scache:
            raw = load_stock_1s_day(stock_root, sym, date)
            scache[key] = None if raw is None or raw.empty else prepare_day_arrays(raw)
        return scache[key]

    bundles = []
    for _, r in probes.iterrows():
        date, sym = str(r["date"]), str(r["symbol"])
        w = _window_of(date)
        if w is None:
            continue
        ticker = str(r["ticker"]).replace("O:", "")
        arr = paths_for(date, sym).get(ticker)
        if arr is None:
            continue
        et = to_ny(pd.Timestamp(r["entry_ts"]))
        pw = _path_window(arr[0], arr[1], et, max_hold_sec=600, slip=float(args.slip))
        if pw is None:
            continue
        rets, holds, _, _ = pw
        sarr = stock_for(date, sym)
        if sarr is None:
            continue
        ss = _stock_series(sarr, et, 600)
        if ss is None:
            continue
        sh, sp = ss
        bundles.append(
            {
                "date": date,
                "symbol": sym,
                "dir": str(r["dir"]),
                "session": str(r["session"]),
                "ticker": ticker,
                "window": w,
                "et": et,
                "rets": rets,
                "holds": holds,
                "pts": arr[0],
                "plast": arr[1],
                "sarr": sarr,
                "sh": sh,
                "sp": sp,
                "oracle": float(r["oracle_ret"]),
                "row": r,
                "reg0": classify_regime_v2(r),
            }
        )
    print(f"bundles={len(bundles)}", flush=True)

    score_rows = []
    cfgs = _exit_cfgs()
    print(f"cfgs={len(cfgs)}", flush=True)

    for ci, cfg in enumerate(cfgs):
        win_raw: dict[str, list] = {w[0]: [] for w in WINDOWS}
        for b in bundles:
            kind = cfg["kind"]
            if kind == "tpsl":
                sim = simulate_trade_tpsl(
                    b["pts"], b["plast"], b["et"],
                    tp=float(cfg["tp"]), sl=float(cfg["sl"]),
                    max_hold_sec=int(cfg["max_hold"]), slip=float(args.slip),
                )
                if sim is None:
                    continue
                ret, hold, reason = float(sim["ret"]), float(sim["hold_sec"]), str(sim["reason"])
            elif kind == "widen":
                reg, tag = _stock_upgrade(
                    rets=b["rets"], holds=b["holds"],
                    stock_holds=b["sh"], stock_px=b["sp"],
                    direction=b["dir"], regime0=b["reg0"],
                    confirm_sec=30.0, stock_min=0.002, opt_min=0.01,
                )
                sim = _apply_ladder(b["rets"], b["holds"], reg, pack=LADDERS_V2)
                if sim is None or not np.isfinite(sim.get("ret", np.nan)):
                    continue
                ret, hold = float(sim["ret"]), float(sim["hold_sec"])
                reason = str(sim["reason"]) + (f"+{tag}" if tag else "")
            else:
                sim = simulate_hold_features(
                    b["rets"], b["holds"],
                    stock_arr=b["sarr"], entry_ts=b["et"], direction=b["dir"],
                    mode=str(cfg["mode"]), params=cfg,
                )
                ret, hold, reason = float(sim["ret"]), float(sim["hold_sec"]), str(sim["reason"])
                if not np.isfinite(ret):
                    continue

            win_raw[b["window"]].append(
                {
                    "date": b["date"],
                    "symbol": b["symbol"],
                    "dir": b["dir"],
                    "session": b["session"],
                    "entry_ts": b["et"],
                    "exit_ts": b["et"] + pd.Timedelta(seconds=hold),
                    "ticker": b["ticker"],
                    "ret": ret,
                    "exit_reason": reason,
                    "hold_sec": hold,
                    "oracle_ret": b["oracle"],
                    "window": b["window"],
                    "cell": cfg["name"],
                }
            )

        sized_all = []
        win_stats = {}
        for wname, _, _ in WINDOWS:
            raw = win_raw[wname]
            by_d: dict[str, list] = {}
            for tr in raw:
                by_d.setdefault(str(tr["date"]), []).append(tr)
            sized = []
            for _, rs in sorted(by_d.items()):
                sized.extend(
                    _portfolio_day(
                        sorted(rs, key=lambda x: (x["entry_ts"], x["symbol"])),
                        position_frac=float(args.position_frac),
                        max_concurrent=int(args.max_concurrent),
                        cooldown_minutes=float(args.cooldown_minutes),
                    )
                )
            st = _stats(sized) if sized else {"n": 0}
            ste = _equity_stats(pd.DataFrame(sized)) if sized else {}
            for k, v in ste.items():
                st[f"eq_{k}"] = v
            # reason mix
            if sized:
                vc = pd.Series([t["exit_reason"].split("+")[0] for t in sized]).value_counts(normalize=True)
                st["top_reason"] = str(vc.index[0])
                st["top_reason_frac"] = float(vc.iloc[0])
            win_stats[wname] = st
            sized_all.extend(sized)

        disc = float(win_stats["may_jul09"].get("eq_compound") or 0)
        blind = float(win_stats["jul10_23"].get("eq_compound") or 0)
        n_d = int(win_stats["may_jul09"].get("n") or 0)
        n_b = int(win_stats["jul10_23"].get("n") or 0)
        if sized_all:
            o = np.array([t["oracle_ret"] for t in sized_all], dtype=float)
            rr = np.array([t["ret"] for t in sized_all], dtype=float)
            cap = float(rr.mean() / o.mean()) if np.nanmean(o) > 0 else float("nan")
            mean_ret = float(rr.mean())
            tw = float((rr > 0).mean())
        else:
            cap = mean_ret = tw = float("nan")
        econ = bool(n_d >= 8 and n_b >= 3 and disc > 0 and blind > 0)
        row = {
            "name": cfg["name"],
            "kind": cfg["kind"],
            "mode": cfg.get("mode"),
            "n": len(sized_all),
            "disc_compound": disc,
            "blind_compound": blind,
            "disc_n": n_d,
            "blind_n": n_b,
            "mean_capture": cap,
            "mean_ret": mean_ret,
            "trade_win": tw,
            "econ_dual": econ,
            "hit_capture_target": bool(cap >= float(args.capture_target)),
            "disc_top_reason": win_stats["may_jul09"].get("top_reason"),
            "disc_top_reason_frac": win_stats["may_jul09"].get("top_reason_frac"),
        }
        score_rows.append(row)
        if (ci + 1) % 15 == 0 or ci == 0:
            print(
                f"[{ci+1}/{len(cfgs)}] {cfg['name']}: cap={cap:.3f} disc={disc:+.3f} "
                f"blind={blind:+.3f} econ={econ}",
                flush=True,
            )

    sb = pd.DataFrame(score_rows)
    sb.to_csv(out / "scoreboard.csv", index=False)
    base = sb[sb.name == "baseline_tp8"].iloc[0]
    widen = sb[sb.name == "widen_stock_up"].iloc[0]

    # rank: econ, capture, disc
    ok = sb[sb.econ_dual == True].copy()  # noqa: E712
    ok["cap_lift"] = ok["mean_capture"] - float(base["mean_capture"])
    ok["disc_lift"] = ok["disc_compound"] - float(base["disc_compound"])
    ok = ok.sort_values(
        ["hit_capture_target", "mean_capture", "disc_compound"],
        ascending=[False, False, False],
    )
    ok.to_csv(out / "ranked.csv", index=False)

    hit = ok[ok.hit_capture_target == True]  # noqa: E712
    best = ok.iloc[0].to_dict() if len(ok) else None
    best_feat = ok[ok.kind == "feat"].iloc[0].to_dict() if len(ok[ok.kind == "feat"]) else None

    promote = "NONE"
    if len(hit):
        promote = f"CAPTURE20_{hit.iloc[0]['name']}"
    elif best_feat and float(best_feat["mean_capture"]) > float(base["mean_capture"]) + 0.01:
        promote = f"CAPTURE_LIFT_{best_feat['name']}"

    summary = {
        "protocol": "am_pocket_hold_feature_exit",
        "entry": args.entry,
        "mark": "option_trade_last_slip",
        "capture_target": float(args.capture_target),
        "baseline_tp8": base.to_dict(),
        "widen_stock_up": widen.to_dict(),
        "n_cfgs": len(cfgs),
        "n_hit_capture20": int(len(hit)),
        "best_overall": best,
        "best_feature": best_feat,
        "promote": promote,
        "top10": ok.head(10).to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== TOP feature / all ===", flush=True)
    cols = ["name", "kind", "mean_capture", "disc_compound", "blind_compound", "mean_ret", "trade_win", "econ_dual", "disc_top_reason"]
    print(ok[cols].head(12).to_string(index=False), flush=True)
    print(json.dumps({"promote": promote, "n_hit_20": int(len(hit)), "best_feature": best_feat}, indent=2, default=str))
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
