#!/usr/bin/env python3
"""Premarket hardening for freeze profile: fault tests + dry + optional S5/parity.

Catches live-path regressions without burning a trading day. Exit code 0 only if
all selected stages pass.

Examples:
  python -m maga7.tools.run_premarket_hardening
  python -m maga7.tools.run_premarket_hardening --date 2026-05-28 --with-s5 --with-parity
  python -m maga7.tools.run_premarket_hardening --faults-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FREEZE = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)
# Golden smoke day: freeze profile OMS dry ↔ offline exact match (as of 2026-07-18).
DEFAULT_SMOKE_DATE = "2026-05-28"


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
) -> dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "ok": proc.returncode == 0,
            "returncode": proc.returncode,
            "elapsed_sec": round(time.time() - t0, 2),
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
        }
    except subprocess.TimeoutExpired as e:
        return {
            "cmd": cmd,
            "ok": False,
            "returncode": -9,
            "elapsed_sec": round(time.time() - t0, 2),
            "stdout_tail": (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else "",
            "stderr_tail": f"TIMEOUT after {timeout}s",
        }


def _redis_ok(host: str, port: int, db: int) -> bool:
    try:
        import redis  # type: ignore

        r = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=1.5)
        return bool(r.ping())
    except Exception:
        return False


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=str(FREEZE))
    p.add_argument(
        "--date",
        default=DEFAULT_SMOKE_DATE,
        help=f"single-day smoke date (NYSE); default {DEFAULT_SMOKE_DATE} (dry↔offline golden)",
    )
    p.add_argument("--end-date", default=None, help="optional end (default=--date)")
    p.add_argument("--scheme", default=None, help="default: profile recommended_scheme or single")
    p.add_argument("--tag", default=None)
    p.add_argument("--faults-only", action="store_true")
    p.add_argument("--skip-faults", action="store_true")
    p.add_argument("--skip-dry", action="store_true")
    p.add_argument(
        "--allow-dry-mismatch",
        action="store_true",
        help="dry process success is enough (do not require trade-level offline match)",
    )
    p.add_argument("--with-s5", action="store_true", help="run Redis S5 sim (needs Redis)")
    p.add_argument("--with-parity", action="store_true", help="run G2 stream parity on the window")
    p.add_argument("--redis-host", default=os.environ.get("MAG7_REDIS_HOST", "127.0.0.1"))
    p.add_argument("--redis-port", type=int, default=int(os.environ.get("MAG7_REDIS_PORT", "6379")))
    p.add_argument("--redis-db", type=int, default=int(os.environ.get("MAG7_REDIS_DB", "1")))
    p.add_argument("--dry-timeout", type=int, default=900)
    p.add_argument("--s5-timeout", type=int, default=1800)
    p.add_argument("--parity-timeout", type=int, default=1200)
    args = p.parse_args()

    from maga7.common.config import load_profile

    profile = load_profile(args.profile)
    scheme = args.scheme or profile.get("recommended_scheme") or "single"
    start = args.date
    end = args.end_date or args.date
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = args.tag or f"premarket_hardening_{start}_{end}_{stamp}"
    out_dir = Path(profile["_paths"]["results_dir"]) / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    env = {**os.environ, "PYTHONPATH": str(ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")}

    stages: list[dict[str, Any]] = []
    overall_ok = True

    # --- 1) fault injection unit tests ---
    if not args.skip_faults:
        print("==> [1/4] fault injection pytest", flush=True)
        r = _run(
            [
                py,
                "-m",
                "pytest",
                "-q",
                "maga7/tests/test_live_fault_injection.py",
                "maga7/tests/test_risk_guards.py",
            ],
            cwd=ROOT,
            env=env,
            timeout=180,
        )
        r["stage"] = "fault_tests"
        stages.append(r)
        overall_ok = overall_ok and r["ok"]
        print(f"    ok={r['ok']} elapsed={r['elapsed_sec']}s", flush=True)
        if not r["ok"]:
            print(r.get("stdout_tail") or r.get("stderr_tail"), flush=True)

    if args.faults_only:
        summary = {
            "tag": tag,
            "profile": str(args.profile),
            "scheme": scheme,
            "start": start,
            "end": end,
            "overall_ok": overall_ok,
            "stages": stages,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps({"overall_ok": overall_ok, "out": str(out_dir)}, indent=2))
        return 0 if overall_ok else 2

    # --- 2) OMS dry + compare offline ---
    if not args.skip_dry:
        print("==> [2/4] OMS dry-run --compare-offline", flush=True)
        dry_tag = f"{tag}__oms_dry"
        r = _run(
            [
                py,
                "-m",
                "maga7.tools.run_oms_dry_run",
                "--profile",
                str(args.profile),
                "--start-date",
                start,
                "--end-date",
                end,
                "--scheme",
                scheme,
                "--ingest",
                "1s",
                "--tag",
                dry_tag,
                "--compare-offline",
            ],
            cwd=ROOT,
            env=env,
            timeout=args.dry_timeout,
        )
        cmp_path = Path(profile["_paths"]["results_dir"]) / dry_tag / "compare_summary.json"
        cmp = _load_json(cmp_path) or {}
        process_ok = bool(r["ok"]) and bool(cmp)
        match_ok = bool(cmp.get("ok") is True)
        dry_ok = process_ok if args.allow_dry_mismatch else (process_ok and match_ok)
        r["stage"] = "oms_dry"
        r["compare"] = cmp
        r["ok"] = dry_ok
        r["process_ok"] = process_ok
        r["match_ok"] = match_ok
        stages.append(r)
        overall_ok = overall_ok and dry_ok
        print(
            f"    ok={dry_ok} process={process_ok} match={match_ok} "
            f"elapsed={r['elapsed_sec']}s matched={cmp.get('matched')} "
            f"max_ret_diff={cmp.get('max_abs_ret_diff')}",
            flush=True,
        )
        if not dry_ok:
            print(r.get("stdout_tail") or r.get("stderr_tail"), flush=True)
            if process_ok and not match_ok and not args.allow_dry_mismatch:
                print(
                    "    hint: dry↔offline mismatch on this date; "
                    f"try --date {DEFAULT_SMOKE_DATE} or pass --allow-dry-mismatch",
                    flush=True,
                )

    # --- 3) optional G2 parity ---
    if args.with_parity:
        print("==> [3/4] stream parity (G2)", flush=True)
        parity_tag = f"{tag}__parity"
        r = _run(
            [
                py,
                "-m",
                "maga7.tools.run_stream_parity",
                "--profile",
                str(args.profile),
                "--scheme",
                scheme,
                "--start-date",
                start,
                "--end-date",
                end,
                "--stock-source",
                "stock_1s",
                "--tag",
                parity_tag,
            ],
            cwd=ROOT,
            env=env,
            timeout=args.parity_timeout,
        )
        r["stage"] = "stream_parity"
        stages.append(r)
        overall_ok = overall_ok and r["ok"]
        print(f"    ok={r['ok']} elapsed={r['elapsed_sec']}s", flush=True)
        if not r["ok"]:
            print(r.get("stdout_tail") or r.get("stderr_tail"), flush=True)

    # --- 4) optional S5 Redis ---
    if args.with_s5:
        print("==> [4/4] Redis S5 sim", flush=True)
        if not _redis_ok(args.redis_host, args.redis_port, args.redis_db):
            r = {
                "stage": "s5_redis",
                "ok": False,
                "returncode": -1,
                "elapsed_sec": 0,
                "stderr_tail": f"Redis unreachable {args.redis_host}:{args.redis_port}/{args.redis_db}",
                "cmd": [],
            }
            stages.append(r)
            overall_ok = False
            print(f"    SKIP/FAIL: {r['stderr_tail']}", flush=True)
        else:
            s5_tag = f"{tag}__s5"
            r = _run(
                [
                    py,
                    "-m",
                    "maga7.tools.run_maga7_redis_sim",
                    "--profile",
                    str(args.profile),
                    "--start-date",
                    start,
                    "--end-date",
                    end,
                    "--scheme",
                    scheme,
                    "--options",
                    "--compare-offline",
                    "--redis-host",
                    args.redis_host,
                    "--redis-db",
                    str(args.redis_db),
                    "--tag",
                    s5_tag,
                    "--sync",
                ],
                cwd=ROOT,
                env=env,
                timeout=args.s5_timeout,
            )
            r["stage"] = "s5_redis"
            stages.append(r)
            overall_ok = overall_ok and r["ok"]
            print(f"    ok={r['ok']} elapsed={r['elapsed_sec']}s", flush=True)
            if not r["ok"]:
                print(r.get("stdout_tail") or r.get("stderr_tail"), flush=True)

    summary = {
        "tag": tag,
        "profile": str(args.profile),
        "profile_id": profile.get("profile_id") or profile.get("profile"),
        "scheme": scheme,
        "start": start,
        "end": end,
        "overall_ok": overall_ok,
        "stages": [
            {
                "stage": s.get("stage"),
                "ok": s.get("ok"),
                "returncode": s.get("returncode"),
                "elapsed_sec": s.get("elapsed_sec"),
                "compare": s.get("compare"),
                "stderr_tail": s.get("stderr_tail"),
            }
            for s in stages
        ],
        "out_dir": str(out_dir),
        "note": (
            "Passing premarket hardening ≠ G4/G5/G6. "
            "Still require real Shadow/Paper session evidence before live."
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # also keep full stage dumps
    (out_dir / "stages_full.json").write_text(json.dumps(stages, indent=2), encoding="utf-8")
    print(json.dumps({"overall_ok": overall_ok, "out": str(out_dir), "stages": [s["stage"] for s in stages]}, indent=2))
    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
