#!/usr/bin/env python3
"""Pre-close inventory/imbalance proxy → 1DTE option accept.

No auction/MOC feed in-repo. Proxy inventory with causal 1s features at
``asof ∈ {15:50, 15:52}``:

  INV_MF   — sign(mf300) + |mf300| cross-section strong + volume_ratio_60≥vr_min
  INV_IDIO — INV_MF AND same-sign idio vs QQQ over last 5m
  CTRL_MOM — pure ret(last 60s) thr (no inventory)
  CTRL_VR  — volume surge only, direction from ret60

Entry: first quote after asof. Exit: quote TP15/SL20 or max_hold (default 300s),
never past 15:58. Contract: ATM 1DTE-only.

Example:
  PYTHONPATH=. python -m maga7.tools.run_preclose_inv_1dte_accept \\
    --tag research_preclose_inv_1dte_20260728
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
from maga7.common.fills import FillSpec
from maga7.common.open_lock import (
    load_multidte_lock_index,
    resolve_open_lock_contract,
    resolve_otm_rungs,
)
from maga7.common.option_quote_tpsl import simulate_quote_tpsl
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.session_1s_features import features_at, prepare_day_arrays
from maga7.common.stock_1s import session_dates
from maga7.tools.run_morning_sec_option_fill import _portfolio_day
from maga7.tools.scan_am_delayed_confirm_quote_dual import _prep_path

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
NY = "America/New_York"
TP, SL = 0.15, 0.20
POS, CMAX, CD = 0.20, 2, 5
FLATTEN = "15:58"


def _stats(tr: pd.DataFrame) -> dict[str, Any]:
    if tr is None or tr.empty:
        return {
            "n": 0,
            "win": None,
            "mean": None,
            "add": 0.0,
            "mult": 1.0,
            "maxdd": 0.0,
            "day_win": None,
            "n_days": 0,
            "n_up": 0,
            "n_dn": 0,
        }
    d = tr.groupby("date")["pnl_frac"].sum().sort_index()
    eq = (1.0 + d).cumprod()
    dd = eq / eq.cummax() - 1.0
    return {
        "n": int(len(tr)),
        "win": float((tr["ret"] > 0).mean()),
        "mean": float(tr["ret"].mean()),
        "add": float(d.sum()),
        "mult": float(eq.iloc[-1]),
        "maxdd": float(dd.min()) if len(dd) else 0.0,
        "day_win": float((d > 0).mean()),
        "n_days": int(len(d)),
        "n_up": int((tr["dir"] == "UP").sum()),
        "n_dn": int((tr["dir"] == "DN").sum()),
    }


def _verdict(may: dict[str, Any], feb: dict[str, Any]) -> str:
    if not may["n"] or may["mean"] is None or may["mean"] <= 0:
        return "FAIL"
    if may["n"] < 15:
        return "THIN"
    if may["day_win"] is None or may["day_win"] < 0.55:
        return "FAIL"
    if may["maxdd"] < -0.25:
        return "FAIL"
    if feb["n"] >= 8 and feb["mean"] is not None and feb["mean"] <= 0:
        return "FAIL"
    if may["mean"] >= 0.08 and (feb["mean"] is None or feb["mean"] >= 0):
        return "PASS"
    return "WEAK"


def _hold_cap(date: str, asof: pd.Timestamp, max_hold: int) -> int:
    flat = pd.Timestamp(f"{date} {FLATTEN}", tz=NY)
    sec = max(30, int((flat - asof).total_seconds()))
    return min(int(max_hold), sec)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tag", default="research_preclose_inv_1dte_20260728")
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-23")
    ap.add_argument("--asofs", default="15:50,15:52")
    ap.add_argument("--max-hold-sec", type=int, default=300)
    ap.add_argument("--vr-min", type=float, default=1.2)
    ap.add_argument("--mom-min", type=float, default=0.0005)
    ap.add_argument("--mf-q", type=float, default=0.70, help="cross-section |mf300| quantile")
    args = ap.parse_args(argv)

    prof = load_profile(PROFILE)
    paths = prof["_paths"]
    out = Path(paths["results_dir"]) / str(args.tag)
    out.mkdir(parents=True, exist_ok=True)
    symbols = [str(s).upper() for s in (prof.get("symbols") or [])]
    stock_1s = Path(paths["stock_1s_root"])
    quote_root = Path(paths["quote_1s_root"])
    lock = load_multidte_lock_index(Path(paths["open_locked_map"]))
    otm = resolve_otm_rungs(prof)
    fill = FillSpec(entry_frac=0.75, exit_frac=0.75)
    dates = session_dates(args.start_date, args.end_date)
    asofs = [x.strip() for x in str(args.asofs).split(",") if x.strip()]
    print(f"[init] dates={len(dates)} symbols={len(symbols)} asofs={asofs}", flush=True)

    # arms keyed by (asof, arm_name)
    arm_names = ["INV_MF", "INV_IDIO", "CTRL_MOM", "CTRL_VR"]
    books: dict[tuple[str, str], list[dict[str, Any]]] = {
        (a, n): [] for a in asofs for n in arm_names
    }

    for di, date in enumerate(dates):
        if di % 15 == 0:
            print(f"[day] {date} ({di+1}/{len(dates)})", flush=True)
        # load QQQ + names
        arrs: dict[str, dict[str, np.ndarray]] = {}
        for sym in ["QQQ", *symbols]:
            day = load_stock_1s_day(stock_1s, sym, date)
            if day is None or day.empty:
                continue
            arrs[sym] = prepare_day_arrays(day)
        if "QQQ" not in arrs:
            continue
        qday_cache: dict[str, pd.DataFrame | None] = {}

        for asof_hm in asofs:
            asof = pd.Timestamp(f"{date} {asof_hm}", tz=NY)
            # cross-section snapshots
            snaps: list[dict[str, Any]] = []
            for sym in symbols:
                if sym not in arrs:
                    continue
                f = features_at(arrs[sym], asof)
                if f is None:
                    continue
                fq = features_at(arrs["QQQ"], asof)
                idio = np.nan
                rs = f.get("ret_120", f.get("ret_60", np.nan))
                rq = fq.get("ret_120", fq.get("ret_60", np.nan)) if fq is not None else np.nan
                if np.isfinite(rs) and np.isfinite(rq):
                    idio = float(rs) - float(rq)
                snaps.append(
                    {
                        "symbol": sym,
                        "px": float(f["px"]),
                        "mf300": float(f["mf300"]) if np.isfinite(f["mf300"]) else np.nan,
                        "mf100": float(f["mf100"]) if np.isfinite(f["mf100"]) else np.nan,
                        "vr": float(f["volume_ratio_60"])
                        if np.isfinite(f["volume_ratio_60"])
                        else np.nan,
                        "ret60": float(f.get("ret_60", np.nan))
                        if np.isfinite(f.get("ret_60", np.nan))
                        else np.nan,
                        "ret120": float(f.get("ret_120", np.nan))
                        if np.isfinite(f.get("ret_120", np.nan))
                        else np.nan,
                        "idio": idio,
                    }
                )
            if len(snaps) < 3:
                continue
            sdf = pd.DataFrame(snaps)
            abs_mf = sdf["mf300"].abs()
            thr_mf = float(abs_mf.quantile(float(args.mf_q))) if abs_mf.notna().any() else np.nan

            candidates: dict[str, list[dict[str, Any]]] = {n: [] for n in arm_names}
            for _, r in sdf.iterrows():
                sym = str(r["symbol"])
                mf = float(r["mf300"]) if np.isfinite(r["mf300"]) else np.nan
                vr = float(r["vr"]) if np.isfinite(r["vr"]) else np.nan
                ret60 = float(r["ret60"]) if np.isfinite(r["ret60"]) else np.nan
                idio = float(r["idio"]) if np.isfinite(r["idio"]) else np.nan
                base = {
                    "date": date,
                    "symbol": sym,
                    "asof": asof_hm,
                    "px": float(r["px"]),
                    "mf300": mf,
                    "vr": vr,
                    "ret60": ret60,
                    "idio": idio,
                }
                # INV_MF
                if (
                    np.isfinite(mf)
                    and np.isfinite(thr_mf)
                    and abs(mf) + 1e-12 >= thr_mf
                    and np.isfinite(vr)
                    and vr + 1e-12 >= float(args.vr_min)
                    and np.isfinite(ret60)
                    and abs(ret60) + 1e-12 >= float(args.mom_min)
                    and np.sign(mf) == np.sign(ret60)
                ):
                    candidates["INV_MF"].append(
                        {**base, "dir": "UP" if mf > 0 else "DN", "arm": "INV_MF"}
                    )
                # INV_IDIO
                if (
                    np.isfinite(mf)
                    and np.isfinite(thr_mf)
                    and abs(mf) + 1e-12 >= thr_mf
                    and np.isfinite(vr)
                    and vr + 1e-12 >= float(args.vr_min)
                    and np.isfinite(idio)
                    and abs(idio) + 1e-12 >= float(args.mom_min)
                    and np.sign(mf) == np.sign(idio)
                ):
                    candidates["INV_IDIO"].append(
                        {**base, "dir": "UP" if mf > 0 else "DN", "arm": "INV_IDIO"}
                    )
                # CTRL_MOM
                if np.isfinite(ret60) and abs(ret60) + 1e-12 >= max(0.001, float(args.mom_min) * 2):
                    candidates["CTRL_MOM"].append(
                        {**base, "dir": "UP" if ret60 > 0 else "DN", "arm": "CTRL_MOM"}
                    )
                # CTRL_VR
                if (
                    np.isfinite(vr)
                    and vr + 1e-12 >= float(args.vr_min)
                    and np.isfinite(ret60)
                    and abs(ret60) + 1e-12 >= float(args.mom_min)
                ):
                    candidates["CTRL_VR"].append(
                        {**base, "dir": "UP" if ret60 > 0 else "DN", "arm": "CTRL_VR"}
                    )

            # rank: strongest |mf| / |ret| keep top1–2 later via portfolio
            for aname, rows in candidates.items():
                if not rows:
                    continue
                # prefer strongest signal
                if aname.startswith("INV"):
                    rows = sorted(rows, key=lambda x: abs(float(x["mf300"])), reverse=True)
                else:
                    rows = sorted(rows, key=lambda x: abs(float(x["ret60"])), reverse=True)
                hold = _hold_cap(date, asof, int(args.max_hold_sec))
                for row in rows[:4]:  # cap raw emits; portfolio will thin
                    sym = row["symbol"]
                    by_dte = lock.get((sym, date))
                    if not by_dte:
                        continue
                    if sym not in qday_cache:
                        qday_cache[sym] = _prep_path(load_quotes(quote_root, sym, date))
                    qday = qday_cache[sym]
                    if qday is None or qday.empty:
                        continue
                    ticker, dte, _ = resolve_open_lock_contract(
                        by_dte,
                        direction=row["dir"],
                        moneyness="ATM",
                        spot=float(row["px"]),
                        prefer_dte=1,
                        allowed_dte=[1],
                        clear_otm_thresh=0.01,
                        ladder=True,
                        otm_rungs=otm,
                    )
                    if not ticker or int(dte) != 1:
                        continue
                    path = _prep_path(path_for_ticker(qday, ticker))
                    if path is None or path.empty:
                        continue
                    sim = simulate_quote_tpsl(
                        path,
                        asof,
                        tp=TP,
                        sl=SL,
                        max_hold_sec=hold,
                        fill=fill,
                        max_lag_sec=5.0,
                        max_spread_pct=0.15,
                        min_mid=0.05,
                    )
                    if sim is None:
                        continue
                    books[(asof_hm, aname)].append(
                        {
                            **row,
                            "entry_ts": str(sim.get("entry_ts") or asof),
                            "exit_ts": str(sim.get("exit_ts") or ""),
                            "ticker": ticker,
                            "dte": 1,
                            "ret": float(sim["ret"]),
                            "exit_reason": str(sim.get("reason") or ""),
                            "hold_sec": float(sim.get("hold_sec") or 0.0),
                            "entry_mid": float(sim.get("entry_mid") or np.nan),
                        }
                    )

    score_rows: list[dict[str, Any]] = []
    for (asof_hm, aname), rows in books.items():
        tag = f"{asof_hm.replace(':','')}__{aname}"
        raw = pd.DataFrame(rows)
        if raw.empty:
            score_rows.append(
                {"name": tag, "asof": asof_hm, "arm": aname, "verdict": "EMPTY", "may_n": 0}
            )
            continue
        raw = raw.sort_values(["date", "entry_ts", "symbol"]).reset_index(drop=True)
        parts: list[pd.DataFrame] = []
        for _, g in raw.groupby("date", sort=True):
            sized = _portfolio_day(
                g.to_dict(orient="records"),
                position_frac=POS,
                max_concurrent=CMAX,
                cooldown_minutes=CD,
            )
            if sized:
                parts.append(pd.DataFrame(sized))
        book = (
            pd.concat(parts, ignore_index=True)
            if parts
            else raw.assign(size=POS, pnl_frac=raw["ret"].astype(float) * POS)
        )
        raw.to_csv(out / f"raw_{tag}.csv", index=False)
        book.to_csv(out / f"book_{tag}.csv", index=False)
        may = _stats(book[(book.date >= "2026-05-01") & (book.date <= "2026-07-23")])
        feb = _stats(book[(book.date >= "2026-02-01") & (book.date <= "2026-03-31")])
        verd = _verdict(may, feb)
        score_rows.append(
            {
                "name": tag,
                "asof": asof_hm,
                "arm": aname,
                "verdict": verd,
                **{f"may_{k}": v for k, v in may.items()},
                **{f"feb_{k}": v for k, v in feb.items()},
            }
        )
        print(
            f"[{tag}] may n={may['n']} win={may['win']} mean={may['mean']} "
            f"mult={may['mult']:.3f} maxdd={may['maxdd']:.2%} → {verd}",
            flush=True,
        )

    sb = pd.DataFrame(score_rows)
    order = {"PASS": 0, "WEAK": 1, "THIN": 2, "FAIL": 3, "EMPTY": 4}
    sb["_o"] = sb["verdict"].map(order)
    sb = sb.sort_values(["_o", "may_mult", "may_mean"], ascending=[True, False, False]).drop(
        columns=["_o"]
    )
    sb.to_csv(out / "scoreboard.csv", index=False)
    summary = {
        "tag": args.tag,
        "note": (
            "Pre-close inventory PROXY only (no auction/MOC feed). "
            "1s mf300 + volume_ratio + optional QQQ idio at 15:50/15:52 → 1DTE ATM."
        ),
        "asofs": asofs,
        "vr_min": args.vr_min,
        "mom_min": args.mom_min,
        "mf_q": args.mf_q,
        "max_hold_sec": args.max_hold_sec,
        "flatten": FLATTEN,
        "promote": sb.loc[sb.verdict == "PASS", "name"].tolist(),
        "weak": sb.loc[sb.verdict == "WEAK", "name"].tolist(),
        "scoreboard": sb.to_dict(orient="records"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"out": str(out), "promote": summary["promote"], "weak": summary["weak"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
