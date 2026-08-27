#!/usr/bin/env python3
"""Freeze-day OOS stress: ask/bid fills + portfolio risk + rolling monthly folds.

Does **not** retune thresholds. Answers whether the peer3 extend stack remains
viable once:

1. fills move to opponent prices (entry_frac=exit_frac=1.0);
2. 2026-07-20 is included as a frozen holdout day (Redis fused replay);
3. strong / weak / monthly folds are scored with the same locked profile.

Verdict rules (collapse → demote flow rules to research feature):
- weak_ret / strong_ret < 0.50  (or weak ≤ 0 while strong > 0)
- askbid_strong / fill075_strong < 0.50
- freeze day total_ret ≤ −0.25 with same-direction concentration ≥ 0.8
- ≥2 of 3 monthly folds have total_ret ≤ 0 while the combined strong window > 0
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.tools.run_live_fused_replay import run_fused_replay

DEFAULT_PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_SESSION = (
    "/mnt/s990/data/maga7/live_sessions/2026-07-20/"
    "live_20260720_083539_29843e"
)


def _summary_row(tag: str, s: dict[str, Any], *, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "tag": tag,
        "total_ret": float(s.get("total_ret") or 0.0),
        "maxdd": float(s.get("maxdd") or 0.0),
        "n_trades": int(s.get("n_trades") or 0),
        "trade_win": s.get("trade_win"),
        "avg_trade_ret": s.get("avg_trade_ret") or s.get("mean_ret"),
        "n_skip_max_concurrent": s.get("n_skip_max_concurrent"),
        "peer_align_min": s.get("peer_align_min"),
        "max_concurrent_positions": s.get("max_concurrent_positions"),
        "fill_frac": s.get("fill_frac"),
    }
    if extra:
        row.update(extra)
    return row


def _portfolio_concentration(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "n_trades": 0,
            "up_frac": None,
            "same_dir_frac": None,
            "n_symbols": 0,
            "pseudo_div_score": None,
        }
    t = trades.copy()
    if "direction" in t.columns:
        dirs = t["direction"].astype(str).str.upper()
    elif "dir" in t.columns:
        dirs = t["dir"].astype(str).str.upper()
    else:
        dirs = pd.Series(["?"] * len(t))
    up_frac = float((dirs == "UP").mean()) if len(dirs) else None
    # same-direction concentration: max(UP, DN) share
    same = float(max((dirs == "UP").mean(), (dirs == "DN").mean())) if len(dirs) else None
    n_sym = int(t["symbol"].astype(str).str.upper().nunique()) if "symbol" in t.columns else 0
    # 1.0 = all same direction (pseudo-diversification); ~0.5 = balanced
    return {
        "n_trades": int(len(t)),
        "up_frac": up_frac,
        "same_dir_frac": same,
        "n_symbols": n_sym,
        "pseudo_div_score": same,
    }


def _live_freeze_row(session_dir: Path) -> dict[str, Any]:
    """Ground-truth frozen day from live POSITION_CLOSE events."""
    path = session_dir / "order_events.jsonl"
    closes: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("kind") or "").upper() != "POSITION_CLOSE":
                continue
            closes.append(row)
    if not closes:
        return {
            "tag": "freeze_20260720_live_book",
            "total_ret": 0.0,
            "maxdd": 0.0,
            "n_trades": 0,
            "start": "2026-07-20",
            "end": "2026-07-20",
            "mode": "LIVE_BOOK",
            "port_same_dir_frac": None,
            "port_up_frac": None,
            "port_n_symbols": 0,
            "note": "no POSITION_CLOSE events",
        }
    t = pd.DataFrame(closes)
    rets = pd.to_numeric(t.get("ret"), errors="coerce").fillna(0.0)
    # compound approx via equity path of unit stakes
    eq = 1.0
    peak = 1.0
    maxdd = 0.0
    for r in rets:
        eq *= 1.0 + float(r)
        peak = max(peak, eq)
        maxdd = min(maxdd, eq / peak - 1.0)
    conc = _portfolio_concentration(t)
    return {
        "tag": "freeze_20260720_live_book",
        "total_ret": float(rets.sum()),
        "equity_compound": float(eq - 1.0),
        "maxdd": float(maxdd),
        "n_trades": int(len(t)),
        "trade_win": float((rets > 0).mean()),
        "avg_trade_ret": float(rets.mean()),
        "start": "2026-07-20",
        "end": "2026-07-20",
        "mode": "LIVE_BOOK",
        "entry_frac": None,
        "exit_frac": None,
        **{f"port_{k}": v for k, v in conc.items()},
    }


def _run_offline(
    base: dict[str, Any],
    *,
    start: str,
    end: str,
    entry_frac: float,
    exit_frac: float,
    tag: str,
    out: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    p = copy.deepcopy(base)
    p["date_range"] = {"start": start, "end": end}
    p.setdefault("fill", {})["entry_frac"] = float(entry_frac)
    p.setdefault("fill", {})["exit_frac"] = float(exit_frac)
    # keep router off — freeze stack evaluation, not router search
    rr = dict(p.get("regime_router") or {})
    rr["enabled"] = False
    p["regime_router"] = rr
    res = run_offline_replay(p, scheme="single")
    s = res["summary"]
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    res["daily"].to_csv(sub / "daily.csv", index=False)
    res["trades"].to_csv(sub / "trades.csv", index=False)
    conc = _portfolio_concentration(res["trades"])
    row = _summary_row(
        tag,
        s,
        extra={
            "start": start,
            "end": end,
            "entry_frac": entry_frac,
            "exit_frac": exit_frac,
            **{f"port_{k}": v for k, v in conc.items()},
        },
    )
    (sub / "row.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row, res["daily"], res["trades"]


def _run_freeze_day(
    *,
    session_dir: Path,
    profile_path: str,
    entry_frac: float,
    exit_frac: float,
    tag: str,
    out: Path,
    redis_db: int,
) -> dict[str, Any]:
    summary = run_fused_replay(
        session_dir,
        scheme="single",
        redis_db=redis_db,
        disable_prevention=True,
        tag=tag,
        fill_overrides={"entry_frac": entry_frac, "exit_frac": exit_frac},
        profile_path_override=profile_path,
    )
    # copy artifacts into eval out
    sess_out = session_dir / f"fused_replay_{tag}"
    sub = out / tag
    sub.mkdir(parents=True, exist_ok=True)
    (sub / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    trades_path = sess_out / "trades.csv"
    trades = pd.read_csv(trades_path) if trades_path.is_file() else pd.DataFrame()
    if not trades.empty:
        trades.to_csv(sub / "trades.csv", index=False)
    conc = _portfolio_concentration(trades)
    row = _summary_row(
        tag,
        summary,
        extra={
            "start": "2026-07-20",
            "end": "2026-07-20",
            "entry_frac": entry_frac,
            "exit_frac": exit_frac,
            "mode": "LIVE_FUSED_REPLAY",
            **{f"port_{k}": v for k, v in conc.items()},
        },
    )
    (sub / "row.json").write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")
    return row


def _verdict(rows: pd.DataFrame) -> dict[str, Any]:
    def _get(tag: str) -> dict[str, Any] | None:
        hit = rows[rows["tag"] == tag]
        if hit.empty:
            return None
        return hit.iloc[0].to_dict()

    strong075 = _get("strong_may_jul_fill075")
    strong_ask = _get("strong_may_jul_askbid")
    weak_ask = _get("weak_feb_apr_askbid")
    freeze_fused = _get("freeze_20260720_askbid")
    freeze_live = _get("freeze_20260720_live_book")
    monthly = [
        _get("roll_may_askbid"),
        _get("roll_jun_askbid"),
        _get("roll_jul_askbid"),
    ]
    monthly = [m for m in monthly if m is not None]

    flags: list[str] = []
    ratios: dict[str, Any] = {}

    if strong_ask and weak_ask:
        sr = float(strong_ask["total_ret"])
        wr = float(weak_ask["total_ret"])
        ratios["weak_over_strong"] = (wr / sr) if sr > 1e-9 else None
        if sr > 0 and wr <= 0:
            flags.append("weak_nonpositive_vs_strong_positive")
        elif ratios["weak_over_strong"] is not None and ratios["weak_over_strong"] < 0.50:
            flags.append("weak_below_50pct_of_strong")
        # harsh: weak retains <15% of strong under opponent fills → structural regime fragility
        if ratios["weak_over_strong"] is not None and ratios["weak_over_strong"] < 0.15:
            flags.append("weak_regime_collapse")

    if strong075 and strong_ask:
        a = float(strong_ask["total_ret"])
        b = float(strong075["total_ret"])
        ratios["askbid_over_fill075_strong"] = (a / b) if b > 1e-9 else None
        if b > 0 and a <= 0:
            flags.append("askbid_kills_strong_edge")
        elif ratios["askbid_over_fill075_strong"] is not None and ratios["askbid_over_fill075_strong"] < 0.50:
            flags.append("askbid_strong_below_50pct_of_fill075")

    # Prefer live book as freeze ground truth; fused may under-count vs live bugs/path.
    freeze = freeze_live or freeze_fused
    if freeze:
        fr = float(freeze["total_ret"])
        same = freeze.get("port_same_dir_frac")
        ratios["freeze_ret"] = fr
        ratios["freeze_same_dir_frac"] = same
        ratios["freeze_source"] = freeze.get("tag")
        if freeze_fused is not None:
            ratios["freeze_fused_ret"] = float(freeze_fused["total_ret"])
            ratios["freeze_fused_n"] = int(freeze_fused.get("n_trades") or 0)
        if fr <= -0.25 and (same is None or float(same) >= 0.8):
            flags.append("freeze_day_large_loss_concentrated")
        elif fr <= -0.15:
            flags.append("freeze_day_material_loss")

    if monthly and strong_ask and float(strong_ask["total_ret"]) > 0:
        n_bad = sum(1 for m in monthly if float(m["total_ret"]) <= 0)
        ratios["monthly_nonpositive"] = n_bad
        if n_bad >= 2:
            flags.append("rolling_months_unstable")

    demote_flags = {
        "askbid_kills_strong_edge",
        "weak_nonpositive_vs_strong_positive",
        "weak_regime_collapse",
        "freeze_day_large_loss_concentrated",
        "rolling_months_unstable",
    }
    if not flags:
        decision = "HOLD_RESEARCH_EDGE"
        note = "Ask/bid + freeze day did not meet collapse thresholds; keep as research, not auto-promote."
    elif any(f in demote_flags for f in flags):
        decision = "DEMOTE_TO_RESEARCH_FEATURE"
        note = (
            "OOS collapsed under ask/bid and/or freeze-day stress; "
            "demote pure flow+T30 to a research feature — do not keep tuning thresholds for live."
        )
    else:
        decision = "WARN_UNSTABLE"
        note = "Partial stress failure; pause live, require state-gate + event exits before any size-up."

    return {"decision": decision, "flags": flags, "ratios": ratios, "note": note}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=DEFAULT_PROFILE)
    ap.add_argument("--session-dir", default=DEFAULT_SESSION)
    ap.add_argument("--out", default="/mnt/s990/data/maga7/results/freeze_oos_askbid_20260720")
    ap.add_argument("--redis-db", type=int, default=0)
    ap.add_argument("--skip-offline", action="store_true")
    ap.add_argument("--skip-freeze", action="store_true")
    ap.add_argument("--skip-fill075", action="store_true", help="skip fill=0.75 strong control")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = load_profile(args.profile)
    profile_path = str(Path(args.profile) if Path(args.profile).is_file() else ROOT / args.profile)

    rows: list[dict[str, Any]] = []

    if not args.skip_offline:
        jobs = []
        if not args.skip_fill075:
            jobs.append(("strong_may_jul_fill075", "2026-05-01", "2026-07-17", 0.75, 0.75))
        jobs += [
            ("strong_may_jul_askbid", "2026-05-01", "2026-07-17", 1.0, 1.0),
            ("weak_feb_apr_askbid", "2026-02-01", "2026-04-30", 1.0, 1.0),
            ("roll_may_askbid", "2026-05-01", "2026-05-30", 1.0, 1.0),
            ("roll_jun_askbid", "2026-06-01", "2026-06-30", 1.0, 1.0),
            ("roll_jul_askbid", "2026-07-01", "2026-07-17", 1.0, 1.0),
        ]
        for tag, start, end, ef, xf in jobs:
            print(f"[offline] {tag} {start}→{end} fill={ef}/{xf}", flush=True)
            row, _, _ = _run_offline(
                base, start=start, end=end, entry_frac=ef, exit_frac=xf, tag=tag, out=out
            )
            rows.append(row)
            print(
                f"  ret={row['total_ret']:+.3f} maxdd={row['maxdd']:+.3f} "
                f"n={row['n_trades']} same_dir={row.get('port_same_dir_frac')}",
                flush=True,
            )

    if not args.skip_freeze:
        print("[freeze] 2026-07-20 live book (ground truth)", flush=True)
        live_row = _live_freeze_row(Path(args.session_dir))
        rows.append(live_row)
        (out / "freeze_20260720_live_book").mkdir(parents=True, exist_ok=True)
        (out / "freeze_20260720_live_book" / "row.json").write_text(
            json.dumps(live_row, indent=2, default=str), encoding="utf-8"
        )
        print(
            f"  sum_ret={live_row['total_ret']:+.3f} compound={live_row.get('equity_compound')} "
            f"n={live_row['n_trades']} same_dir={live_row.get('port_same_dir_frac')}",
            flush=True,
        )
        print("[freeze] 2026-07-20 fused ask/bid", flush=True)
        row = _run_freeze_day(
            session_dir=Path(args.session_dir),
            profile_path=profile_path,
            entry_frac=1.0,
            exit_frac=1.0,
            tag="freeze_20260720_askbid",
            out=out,
            redis_db=int(args.redis_db),
        )
        rows.append(row)
        print(
            f"  ret={row['total_ret']:+.3f} maxdd={row['maxdd']:+.3f} "
            f"n={row['n_trades']} same_dir={row.get('port_same_dir_frac')}",
            flush=True,
        )

    rdf = pd.DataFrame(rows)
    rdf.to_csv(out / "scoreboard.csv", index=False)
    verdict = _verdict(rdf)
    report = {
        "profile": profile_path,
        "session_dir": args.session_dir,
        "portfolio_controls": {
            "peer_align_min": (base.get("signal") or {}).get("peer_align_min"),
            "max_concurrent_positions": (base.get("trade") or {}).get("max_concurrent_positions"),
            "position_frac": (base.get("trade") or {}).get("position_frac"),
            "day_loss_streak_halt": (base.get("regime") or {}).get("day_loss_streak_halt"),
        },
        "rows": rows,
        "verdict": verdict,
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Freeze OOS Ask/Bid Eval",
        "",
        f"**Decision: `{verdict['decision']}`**",
        "",
        verdict["note"],
        "",
        "## Flags",
        "",
    ]
    if verdict["flags"]:
        lines += [f"- `{f}`" for f in verdict["flags"]]
    else:
        lines.append("- (none)")
    lines += ["", "## Ratios", "", "```json", json.dumps(verdict["ratios"], indent=2), "```", ""]
    lines += ["## Scoreboard", "", rdf.to_markdown(index=False), ""]
    lines += [
        "## Interpretation",
        "",
        "- Ask/bid = opponent fill (`entry_frac=exit_frac=1.0`).",
        "- Portfolio risk already in profile: peer3 + max_concurrent=2 + streak halt.",
        "- Freeze ground truth = live POSITION_CLOSE book; fused ask/bid is secondary (may under-count).",
        "- DEMOTE ⇒ stop threshold tuning; rebuild as flow→response→state→event-exit.",
        "",
    ]
    (out / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(verdict, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
