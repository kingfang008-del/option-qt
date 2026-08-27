#!/usr/bin/env python3
"""P3.4：日终 Hunt 对齐 — live session 日志 vs offline replay。

对某一交易日：
  1) 读 session ``order_events.jsonl`` / ``scanner_state.json`` 里 ``event_source=hunt``
  2) 跑 offline research_baseline，筛 ``event_source=hunt`` 成交
  3) 按 (symbol, direction) 集合对拍；可选时刻容差

Example:
  PYTHONPATH=. python -m maga7.tools.run_hunt_session_eod_align \\
    --date 2026-07-02 \\
    --session-dir /mnt/s990/data/maga7/live_sessions/<sid> \\
    --tag hunt_eod_align_20260702
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay

PROFILE = (
    "maga7/CONFIG/strategy_profiles/"
    "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def _load_events(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "order_events.jsonl"
    if not path.is_file():
        # alternate name used by some sessions
        alt = session_dir / "oms_events.jsonl"
        path = alt if alt.is_file() else path
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _session_hunts(session_dir: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for ev in _load_events(session_dir):
        src = str(ev.get("event_source") or "")
        meta = ev.get("meta") if isinstance(ev.get("meta"), dict) else {}
        sig = ev.get("signal") if isinstance(ev.get("signal"), dict) else {}
        sig_meta = sig.get("meta") if isinstance(sig.get("meta"), dict) else {}
        if src != "hunt" and meta.get("event_source") != "hunt" and sig_meta.get("event_source") != "hunt":
            # POSITION_OPEN / ENTRY_INTENT with nested signal
            if str(ev.get("kind") or "") not in {"POSITION_OPEN", "ENTRY_INTENT", "ORDER_SUBMITTED"}:
                continue
            if sig_meta.get("event_source") != "hunt" and meta.get("event_source") != "hunt":
                continue
        kind = str(ev.get("kind") or "")
        if kind not in {"POSITION_OPEN", "ENTRY_INTENT", "ORDER_SUBMITTED", "HUNT"} and src != "hunt":
            if sig_meta.get("event_source") != "hunt":
                continue
        sym = str(ev.get("symbol") or sig.get("symbol") or "").upper()
        direction = str(ev.get("direction") or sig.get("direction") or "").upper()
        if not sym or direction not in {"UP", "DN"}:
            continue
        out.append(
            {
                "symbol": sym,
                "direction": direction,
                "kind": kind,
                "ts": ev.get("ts"),
                "contract": ev.get("contract") or sig.get("contract"),
                "source": "session_events",
            }
        )
    # Also pull scanner_state signals
    snap_path = session_dir / "scanner_state.json"
    if snap_path.is_file():
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except Exception:
            snap = {}
        for item in list(snap.get("signals") or []) + list(snap.get("day_fires") or []):
            if not isinstance(item, dict):
                continue
            meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
            if str(meta.get("event_source") or "") != "hunt":
                continue
            out.append(
                {
                    "symbol": str(item.get("symbol") or "").upper(),
                    "direction": str(item.get("direction") or "").upper(),
                    "kind": "SCANNER_SIGNAL",
                    "ts": item.get("sig_ts"),
                    "contract": item.get("contract"),
                    "source": "scanner_state",
                }
            )
    # de-dupe by symbol+dir preferring POSITION_OPEN
    best: dict[tuple[str, str], dict[str, Any]] = {}
    rank = {"POSITION_OPEN": 3, "ORDER_SUBMITTED": 2, "ENTRY_INTENT": 1, "SCANNER_SIGNAL": 0}
    for r in out:
        key = (r["symbol"], r["direction"])
        prev = best.get(key)
        if prev is None or rank.get(r["kind"], -1) > rank.get(prev["kind"], -1):
            best[key] = r
    return list(best.values())


def _offline_hunts(profile: dict, date: str) -> list[dict[str, Any]]:
    p = copy.deepcopy(profile)
    p["date_range"] = {"start": str(date), "end": str(date)}
    res = run_offline_replay(p, scheme="single")
    trades = res.get("trades")
    if trades is None or (isinstance(trades, pd.DataFrame) and trades.empty):
        return []
    if not isinstance(trades, pd.DataFrame):
        trades = pd.DataFrame(trades)
    if "event_source" not in trades.columns:
        return []
    sub = trades[trades["event_source"].astype(str) == "hunt"]
    rows = []
    for _, r in sub.iterrows():
        rows.append(
            {
                "symbol": str(r.get("symbol") or "").upper(),
                "direction": str(r.get("direction") or r.get("dir") or "").upper(),
                "entry_ts": str(r.get("entry_ts") or ""),
                "ret": r.get("ret"),
                "source": "offline",
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", required=True)
    ap.add_argument("--session-dir", required=True, type=Path)
    ap.add_argument("--profile", default=PROFILE)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--skip-offline", action="store_true")
    args = ap.parse_args(argv)

    session_dir = Path(args.session_dir).expanduser().resolve()
    if not session_dir.is_dir():
        print(f"session-dir missing: {session_dir}", file=sys.stderr)
        return 2

    prof = load_profile(args.profile)
    tag = args.tag or f"hunt_eod_align_{args.date.replace('-', '')}"
    out = Path(prof["_paths"]["results_dir"]) / "watchdog" / tag
    out.mkdir(parents=True, exist_ok=True)

    sess = _session_hunts(session_dir)
    off: list[dict[str, Any]] = []
    if not args.skip_offline:
        print(f"offline hunt replay {args.date} …", flush=True)
        off = _offline_hunts(prof, str(args.date))

    sess_keys = {(r["symbol"], r["direction"]) for r in sess}
    off_keys = {(r["symbol"], r["direction"]) for r in off}
    both = sess_keys & off_keys
    only_sess = sess_keys - off_keys
    only_off = off_keys - sess_keys

    summary = {
        "date": str(args.date),
        "session_dir": str(session_dir),
        "n_session_hunt": len(sess),
        "n_offline_hunt": len(off),
        "n_both": len(both),
        "only_session": sorted([f"{a}:{b}" for a, b in only_sess]),
        "only_offline": sorted([f"{a}:{b}" for a, b in only_off]),
        "both": sorted([f"{a}:{b}" for a, b in both]),
        "ok_set_equal": bool(sess_keys == off_keys) if not args.skip_offline else None,
        "verdict": (
            "PASS_SET"
            if (not args.skip_offline and sess_keys == off_keys)
            else (
                "SESSION_ONLY"
                if args.skip_offline
                else ("PARTIAL" if both else "MISS")
            )
        ),
    }
    pd.DataFrame(sess).to_csv(out / "session_hunts.csv", index=False)
    pd.DataFrame(off).to_csv(out / "offline_hunts.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {out}", flush=True)
    return 0 if summary["verdict"] in {"PASS_SET", "SESSION_ONLY"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
