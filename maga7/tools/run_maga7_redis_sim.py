#!/usr/bin/env python3
"""S5 Mag7 Redis sim — 1s market data → fused_market_stream → scanner → OMS.

Reuses New_Pro / qqq_btc Redis bus (``fused_market_stream`` + msgpack batch),
**not** FCS/TFT/QQQ OMS. Mag7 strategy stays Mag7Scanner + Mag7OmsStub.

Default: ``--options`` on (publish option_contracts) and OMS prefers Redis
quote book. Fills are causal: no full-day quote warmup/lookahead.

Usage:
    export PYTHONPATH=$PWD
    python -m maga7.tools.run_maga7_redis_sim \\
      --start-date 2026-05-01 --end-date 2026-05-01 --options --compare-offline

    # stock-only bus + disk fills
    python -m maga7.tools.run_maga7_redis_sim \\
      --start-date 2026-05-01 --no-options --disk-quotes
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.provenance import code_fingerprint
from maga7.common.replay import run_offline_replay
from maga7.live.oms_stub import Mag7OmsStub
from maga7.live.redis_consumer import Mag7RedisScannerLoop
from maga7.live.redis_fused import redis_client
from maga7.live.redis_pitcher import Mag7FusedPitcher
from maga7.live.scanner import write_signal_audit

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger("maga7.s5")

PROD_TEMP = (
    ROOT
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "m5c_qqq_onlywin_open_ladder_atm5otm_mf_flip_p20_v1.json"
)


def _dates(start: str, end: str) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]


def _parse_speed(raw: str) -> float:
    s = str(raw).strip().lower()
    if s in {"inf", "infinite", "max", "0"}:
        return float("inf")
    return float(s)


def main() -> None:
    p = argparse.ArgumentParser(description="Mag7 S5 Redis fused_market sim")
    p.add_argument("--profile", default=str(PROD_TEMP))
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", default=None)
    p.add_argument("--scheme", default="m5_circuit")
    p.add_argument("--speed", default="inf", help="inf | wall-clock factor (1=realtime)")
    p.add_argument("--sync", action="store_true", help="pitcher waits Mag7 consumer ACK each second")
    p.add_argument(
        "--no-sync",
        action="store_true",
        help="disable sync barrier (faster; risk dropping if consumer lags — prefer --sync for parity)",
    )
    opt = p.add_mutually_exclusive_group()
    opt.add_argument(
        "--options",
        dest="options",
        action="store_true",
        default=True,
        help="publish option_contracts on fused_market_stream (default: on)",
    )
    opt.add_argument(
        "--no-options",
        dest="options",
        action="store_false",
        help="stock-only fused batches",
    )
    p.add_argument(
        "--prefer-redis-quotes",
        action="store_true",
        default=False,
        help="force OMS Redis quote book (default on whenever options are published)",
    )
    p.add_argument(
        "--disk-quotes",
        action="store_true",
        help="force OMS disk 1s quotes even if options are on the bus",
    )
    p.add_argument("--redis-db", type=int, default=1)
    p.add_argument("--redis-host", default="127.0.0.1")
    p.add_argument("--run-id", default=None, help="optional explicit isolated run id")
    p.add_argument("--tag", default=None)
    p.add_argument("--max-seconds", type=int, default=None, help="cap ticks on last day (smoke)")
    p.add_argument("--compare-offline", action="store_true")
    args = p.parse_args()

    profile = load_profile(args.profile)
    fingerprint = code_fingerprint(profile["_profile_path"])
    end = args.end_date or args.start_date
    profile["date_range"]["start"] = args.start_date
    profile["date_range"]["end"] = end
    dates = _dates(args.start_date, end)
    speed = _parse_speed(args.speed)
    if args.no_sync:
        sync = False
    elif args.sync:
        sync = True
    else:
        sync = speed == float("inf")

    publish_options = bool(args.options)
    if args.disk_quotes:
        prefer_redis = False
    else:
        prefer_redis = publish_options or bool(args.prefer_redis_quotes)

    tag = args.tag or f"s5_redis_{args.scheme}_{args.start_date}_{end}"
    if publish_options and prefer_redis:
        tag = args.tag or f"s5_redis_opt_{args.scheme}_{args.start_date}_{end}"

    r = redis_client(host=args.redis_host, port=6379, db=args.redis_db)
    pitcher = Mag7FusedPitcher(
        symbols=list(profile["symbols"]),
        stock_1s_root=profile["_paths"]["stock_1s_root"],
        quote_1s_root=profile["_paths"].get("quote_1s_root"),
        host=args.redis_host,
        db=args.redis_db,
        run_id=args.run_id,
        publish_options=publish_options,
    )
    run_id = pitcher.init_redis(reset=True)
    # Never let concurrent runs overwrite each other's artifacts.
    out_dir = Path(profile["_paths"]["results_dir"]) / tag / run_id
    out_dir.mkdir(parents=True, exist_ok=False)

    stub = Mag7OmsStub.from_profile(
        profile,
        fill_audit_path=out_dir / "fill_audit_live.csv",
        redis_publish=False,
        prefer_redis_quotes=prefer_redis,
    )
    loop = Mag7RedisScannerLoop.from_profile(
        profile,
        r,
        run_id=run_id,
        scheme=args.scheme,
        stub=stub,
        consumer_name=f"maga7_s5_{run_id}",
    )

    consumer_err: list[BaseException] = []

    def _consume() -> None:
        try:
            # generous wall for multi-day; idle exit when pitcher marks DONE
            loop.run_until_done(idle_sec=3.0, max_wall_sec=None)
        except BaseException as exc:  # noqa: BLE001
            consumer_err.append(exc)
            logger.exception("Mag7 Redis consumer failed")

    th = threading.Thread(target=_consume, name="maga7-s5-consumer", daemon=True)
    th.start()
    time.sleep(0.3)  # let group subscribe

    t0 = time.time()
    n_ticks = pitcher.run(
        dates,
        speed=speed,
        sync=sync,
        progress_every=1800,
        max_seconds=args.max_seconds,
    )
    # Wait consumer drain
    th.join(timeout=600)
    if th.is_alive():
        loop.stop = True
        th.join(timeout=30)

    if consumer_err:
        raise SystemExit(f"consumer error: {consumer_err[0]}")

    stub.flush_pending()
    stream_len = int(r.xlen(loop.keys["stream"]))
    frame_integrity_ok = bool(
        loop.n_batches == n_ticks == len(loop.seen_frame_ids) == stream_len
        and loop.n_duplicate_frames == 0
        and loop.n_foreign_frames == 0
        and loop.n_rejected_frames == 0
    )
    summary = stub.finalize_summary(n_signals=len(loop.scanner.signals))
    summary.update(
        {
            "mode": "MAG7_S5_REDIS",
            "run_id": run_id,
            "ingest": "redis_fused_1s",
            "scheme": args.scheme,
            "speed": speed if speed != float("inf") else "inf",
            "sync": sync,
            "publish_options": publish_options,
            "prefer_redis_quotes": prefer_redis,
            "causal_redis_fills": prefer_redis,
            "stream": loop.keys["stream"],
            "consumer_group": loop.keys["group"],
            "pitcher_ticks": n_ticks,
            "consumer_batches": loop.n_batches,
            "consumer_unique_frames": len(loop.seen_frame_ids),
            "stream_len": stream_len,
            "duplicate_frames": loop.n_duplicate_frames,
            "foreign_frames": loop.n_foreign_frames,
            "rejected_frames": loop.n_rejected_frames,
            "frame_integrity_ok": frame_integrity_ok,
            "consumer_ticks": loop.n_ticks,
            "consumer_option_prints": loop.n_option_prints,
            "start": args.start_date,
            "end": end,
            "profile": profile.get("profile_id") or profile.get("profile"),
            "elapsed_sec": time.time() - t0,
            "redis_db": args.redis_db,
            "strategy_fingerprint": fingerprint,
        }
    )
    stub.summary = summary
    stub.write(out_dir)
    write_signal_audit(loop.scanner.signals, out_dir / "signals.jsonl")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"→ {out_dir}")
    for t in stub.trades[:15]:
        print(
            f"  {t.date} {t.symbol} {t.direction} ret={t.ret:+.1%} "
            f"size={t.qty_frac:.3f} {t.reason}"
        )

    if not frame_integrity_ok:
        raise SystemExit(
            "frame integrity failed: "
            f"pitched={n_ticks} consumed={loop.n_batches} "
            f"unique={len(loop.seen_frame_ids)} stream={stream_len}"
        )

    if args.compare_offline:
        result = run_offline_replay(profile, scheme=args.scheme)
        off_sum = result["summary"]
        off_sum["strategy_fingerprint"] = fingerprint
        (out_dir / "offline_summary.json").write_text(json.dumps(off_sum, indent=2), encoding="utf-8")
        ot = result["trades"].copy()
        ot["date"] = ot["date"].astype(str)
        dry = pd.DataFrame([t.__dict__ for t in stub.trades])
        if dry.empty:
            cmp = {
                "s5_total_ret": summary.get("total_ret"),
                "offline_total_ret": off_sum.get("total_ret"),
                "s5_n": 0,
                "offline_n": int(len(ot)),
                "matched": 0,
                "only_s5": 0,
                "only_offline": int(len(ot)),
                "max_abs_ret_diff": None,
                "ok": False,
                "note": "s5 produced zero trades",
                "strategy_fingerprint": fingerprint,
            }
            (out_dir / "compare_summary.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
            print(json.dumps(cmp, indent=2))
            return

        def _norm_ts(s: pd.Series) -> pd.Series:
            return s.astype(str).str.replace("T", " ", regex=False)

        if len(dry) and "entry_ts" in ot.columns and "entry_ts" in dry.columns:
            ot = ot.copy()
            dry = dry.copy()
            ot["entry_ts"] = _norm_ts(ot["entry_ts"])
            dry["entry_ts"] = _norm_ts(dry["entry_ts"])
            m = dry.merge(
                ot,
                on=["date", "symbol", "entry_ts"],
                how="outer",
                suffixes=("_s5", "_off"),
                indicator=True,
            )
        else:
            m = dry.merge(ot, on=["date", "symbol"], how="outer", suffixes=("_s5", "_off"), indicator=True)
        m.to_csv(out_dir / "compare_offline.csv", index=False)
        both = m[m["_merge"] == "both"]
        delta = None
        if len(both) and "ret_s5" in both.columns and "ret_off" in both.columns:
            delta = (both["ret_s5"] - both["ret_off"]).abs().max()
        cmp = {
            "s5_total_ret": summary.get("total_ret"),
            "offline_total_ret": off_sum.get("total_ret"),
            "s5_n": int(len(dry)),
            "offline_n": int(len(ot)),
            "matched": int(len(both)),
            "only_s5": int((m["_merge"] == "left_only").sum()),
            "only_offline": int((m["_merge"] == "right_only").sum()),
            "max_abs_ret_diff": float(delta) if delta is not None and pd.notna(delta) else None,
            "ok": bool(
                len(both) == len(dry) == len(ot) and delta is not None and float(delta) < 1e-9
            ),
            "strategy_fingerprint": fingerprint,
        }
        (out_dir / "compare_summary.json").write_text(json.dumps(cmp, indent=2), encoding="utf-8")
        print(json.dumps(cmp, indent=2))


if __name__ == "__main__":
    main()
