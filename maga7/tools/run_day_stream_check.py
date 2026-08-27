#!/usr/bin/env python3
"""One-day stream check (production-style): pitch day → trade_log → diff offline.

This is the simple Mag7 equivalent of production's historical pitcher + trade_log:
  1) stream one day's 1s bars into scanner→OMS (Redis S5 if available, else in-process)
  2) write ``trade_log.csv`` (OPEN/CLOSE rows)
  3) compare entry/exit vs offline replay

Usage:
  python -m maga7.tools.run_day_stream_check --date 2026-05-28
  python -m maga7.tools.run_day_stream_check --date 2026-05-28 --force-local
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.replay import run_offline_replay
from maga7.common.trade_log import offline_trades_to_trade_log, write_trade_log
from maga7.live.oms_dry import Mag7OmsDryRunner
from maga7.live.scanner import write_signal_audit
from maga7.tools.run_oms_dry_run import _drive_interleaved

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
DEFAULT_DATE = "2026-05-28"


def _redis_ok(host: str, port: int, db: int) -> bool:
    try:
        import redis  # type: ignore

        return bool(redis.Redis(host=host, port=port, db=db, socket_connect_timeout=1.0).ping())
    except Exception:
        return False


def _norm_ts(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace("T", " ", regex=False)


def _compare_trades(stream_df: pd.DataFrame, offline_df: pd.DataFrame) -> dict[str, Any]:
    a = stream_df.copy() if stream_df is not None else pd.DataFrame()
    b = offline_df.copy() if offline_df is not None else pd.DataFrame()
    # Both flat → PASS (no-trade days).
    if a.empty and b.empty:
        empty = pd.DataFrame(columns=["date", "symbol", "_merge"])
        return {
            "stream_n": 0,
            "offline_n": 0,
            "matched": 0,
            "only_stream": 0,
            "only_offline": 0,
            "max_abs_ret_diff": 0.0,
            "reason_mismatch": 0,
            "ok": True,
            "merge": empty,
        }
    if "direction" in a.columns and "dir" not in a.columns:
        a["dir"] = a["direction"]
    if "dir" not in b.columns and "direction" in b.columns:
        b["dir"] = b["direction"]
    keys = ["date", "symbol"]
    if "entry_ts" in a.columns and "entry_ts" in b.columns:
        keys = ["date", "symbol", "entry_ts"]
    for col in keys:
        if col not in a.columns:
            a[col] = pd.Series(dtype=object)
        if col not in b.columns:
            b[col] = pd.Series(dtype=object)
    a["date"] = a["date"].astype(str)
    b["date"] = b["date"].astype(str)
    if "symbol" in keys:
        a["symbol"] = a["symbol"].astype(str)
        b["symbol"] = b["symbol"].astype(str)
    if "entry_ts" in keys:
        a["entry_ts"] = _norm_ts(a["entry_ts"])
        b["entry_ts"] = _norm_ts(b["entry_ts"])
    m = a.merge(b, on=keys, how="outer", suffixes=("_stream", "_off"), indicator=True)
    both = m[m["_merge"] == "both"]
    delta = 0.0
    ret_s = "ret_stream" if "ret_stream" in both.columns else None
    ret_o = "ret_off" if "ret_off" in both.columns else None
    if ret_s and ret_o and len(both):
        delta = float((both[ret_s] - both[ret_o]).abs().max())
    reason_mismatch = 0
    if "reason_stream" in both.columns and "reason_off" in both.columns and len(both):
        reason_mismatch = int((both["reason_stream"].astype(str) != both["reason_off"].astype(str)).sum())
    only_stream = int((m["_merge"] == "left_only").sum())
    only_offline = int((m["_merge"] == "right_only").sum())
    ok = bool(only_stream == 0 and only_offline == 0 and reason_mismatch == 0 and delta < 1e-9)
    return {
        "stream_n": int(len(a)),
        "offline_n": int(len(b)),
        "matched": int(len(both)),
        "only_stream": only_stream,
        "only_offline": only_offline,
        "max_abs_ret_diff": delta,
        "reason_mismatch": reason_mismatch,
        "ok": ok,
        "merge": m,
    }


def _run_local(profile: dict, start: str, end: str, scheme: str, out_dir: Path) -> dict[str, Any]:
    runner = Mag7OmsDryRunner(profile)
    # 1m research bars + lookback (same stock frames as offline). Fills still use 1s quotes.
    scanner = _drive_interleaved(
        profile, start, end, ingest="1m", scheme=scheme, runner=runner
    )
    write_signal_audit(scanner.signals, out_dir / "signals.jsonl")
    summary = runner.finalize_summary()
    summary.update(
        {
            "mode": "DAY_STREAM_LOCAL_1M",
            "ingest": "stock_1m_lookback",
            "scheme": scheme,
            "start": start,
            "end": end,
            "n_signals": len(scanner.signals),
            "profile": profile.get("profile_id") or profile.get("profile"),
        }
    )
    runner.summary = summary
    runner.write(out_dir)
    write_trade_log(runner.trades, out_dir)
    stream_df = pd.DataFrame([t.__dict__ for t in runner.trades])
    return {"summary": summary, "trades": stream_df, "transport": "local_1m"}


def _run_s5(
    profile_path: str,
    start: str,
    end: str,
    scheme: str,
    tag: str,
    redis_host: str,
    redis_db: int,
) -> dict[str, Any]:
    py = sys.executable
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT)
        + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    cmd = [
        py,
        "-m",
        "maga7.tools.run_maga7_redis_sim",
        "--profile",
        profile_path,
        "--start-date",
        start,
        "--end-date",
        end,
        "--scheme",
        scheme,
        "--options",
        "--compare-offline",
        "--sync",
        "--redis-host",
        redis_host,
        "--redis-db",
        str(redis_db),
        "--tag",
        tag,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    # find newest run dir under tag
    results = Path(load_profile(profile_path)["_paths"]["results_dir"]) / tag
    run_dirs = sorted([p for p in results.glob("*") if p.is_dir()], key=lambda p: p.stat().st_mtime)
    if not run_dirs:
        raise SystemExit(
            f"S5 produced no out dir (rc={proc.returncode})\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    out_dir = run_dirs[-1]
    trades_path = out_dir / "trades.csv"
    stream_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
    if not stream_df.empty:
        write_trade_log(stream_df.to_dict("records"), out_dir)
    cmp = {}
    cmp_path = out_dir / "compare_summary.json"
    if cmp_path.exists():
        cmp = json.loads(cmp_path.read_text(encoding="utf-8"))
    summary = {}
    if (out_dir / "summary.json").exists():
        summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    if proc.returncode != 0 and not cmp:
        raise SystemExit(f"S5 failed rc={proc.returncode}\n{proc.stderr[-2000:]}")
    return {
        "summary": summary,
        "trades": stream_df,
        "transport": "redis_s5",
        "out_dir": out_dir,
        "compare_from_s5": cmp,
        "rc": proc.returncode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default=str(FREEZE))
    ap.add_argument("--date", default=DEFAULT_DATE)
    ap.add_argument("--end-date", default=None)
    ap.add_argument("--scheme", default=None)
    ap.add_argument("--force-local", action="store_true", help="skip Redis; in-process 1s stream")
    ap.add_argument("--redis-host", default=os.environ.get("MAG7_REDIS_HOST", "127.0.0.1"))
    ap.add_argument("--redis-db", type=int, default=int(os.environ.get("MAG7_REDIS_DB", "1")))
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    profile = load_profile(args.profile)
    scheme = args.scheme or profile.get("recommended_scheme") or "single"
    start = args.date
    end = args.end_date or args.date
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or f"day_stream_check_{start}_{end}"

    use_s5 = (not args.force_local) and _redis_ok(args.redis_host, 6379, args.redis_db)
    print(
        f"==> day stream check date={start}..{end} scheme={scheme} "
        f"transport={'redis_s5' if use_s5 else 'local_1m'}",
        flush=True,
    )

    if use_s5:
        packed = _run_s5(
            str(args.profile),
            start,
            end,
            scheme,
            tag=f"{tag}__s5",
            redis_host=args.redis_host,
            redis_db=args.redis_db,
        )
        out_dir = Path(packed["out_dir"])
        stream_df = packed["trades"]
        # Prefer S5's own compare if present; still emit offline trade_log for eyeball.
        off = run_offline_replay(
            {**profile, "date_range": {"start": start, "end": end}}, scheme=scheme
        )
        offline_trades_to_trade_log(off["trades"]).to_csv(out_dir / "trade_log_offline.csv", index=False)
        if packed.get("compare_from_s5"):
            cmp = {
                **packed["compare_from_s5"],
                "transport": "redis_s5",
                "out_dir": str(out_dir),
            }
            # normalize ok key
            ok = bool(cmp.get("ok"))
        else:
            c = _compare_trades(stream_df, off["trades"])
            c["merge"].to_csv(out_dir / "compare_offline.csv", index=False)
            cmp = {k: v for k, v in c.items() if k != "merge"}
            cmp["transport"] = "redis_s5"
            cmp["out_dir"] = str(out_dir)
            ok = bool(cmp["ok"])
    else:
        out_dir = Path(profile["_paths"]["results_dir"]) / tag / stamp
        out_dir.mkdir(parents=True, exist_ok=True)
        profile = load_profile(args.profile)
        profile["date_range"] = {"start": start, "end": end}
        packed = _run_local(profile, start, end, scheme, out_dir)
        stream_df = packed["trades"]
        off = run_offline_replay(profile, scheme=scheme)
        offline_trades_to_trade_log(off["trades"]).to_csv(out_dir / "trade_log_offline.csv", index=False)
        write_trade_log(stream_df.to_dict("records") if len(stream_df) else [], out_dir)
        c = _compare_trades(stream_df, off["trades"])
        c["merge"].to_csv(out_dir / "compare_offline.csv", index=False)
        cmp = {k: v for k, v in c.items() if k != "merge"}
        cmp["transport"] = "local_1m"
        cmp["out_dir"] = str(out_dir)
        ok = bool(cmp["ok"])

    # Always ensure stream trade_log exists
    if (out_dir / "trade_log.csv").exists() is False and len(stream_df):
        write_trade_log(stream_df.to_dict("records"), out_dir)

    report = {
        "ok": ok,
        "date": start,
        "end": end,
        "scheme": scheme,
        "profile": profile.get("profile_id") or profile.get("profile"),
        "out_dir": str(out_dir),
        "trade_log": str(out_dir / "trade_log.csv"),
        "trade_log_offline": str(out_dir / "trade_log_offline.csv"),
        "compare": cmp,
        "note": "PASS = same opens/closes/rets as offline. Fail → inspect trade_log vs trade_log_offline.",
    }
    (out_dir / "day_stream_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if ok:
        print("PASS: stream trade_log matches offline opens/closes", flush=True)
        return 0
    print("FAIL: inspect trade_log.csv vs trade_log_offline.csv", flush=True)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
