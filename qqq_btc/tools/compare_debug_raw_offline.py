#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate-1 特征门控：PG debug_raw（FCS 归一化前）vs 离线 quote_features_raw。

必须先过本门，再比 rolling/frozen norm（Gate-2 / compare_debug_slow_offline.py），
最后才做交易对拍（Gate-3）。

用法:
  # 流式需 FCS_DEBUG_RAW=1
  python qqq_btc/tools/compare_debug_raw_offline.py \\
      --dates 2026-07-01 \\
      --offline ~/train_data/july_w1_v4_honest_openwin/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet \\
      --out qqq_btc/results/.../feat_parity_gate1_raw.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# 复用 Gate-2 的对齐/判据逻辑，仅换表名与默认离线路径
from qqq_btc.tools.compare_debug_slow_offline import (  # noqa: E402
    SKIP_FEATURES,
    _common_feat_cols,
    _load_offline,
    compare_day,
)

NY = "America/New_York"
DEFAULT_PG = "dbname=quant_trade user=postgres password=postgres host=localhost port=5432"


def _load_debug_raw(dates: list[str], symbol: str, pg_url: str) -> pd.DataFrame:
    import psycopg2

    frames = []
    conn = psycopg2.connect(pg_url)
    try:
        for d in dates:
            ymd = d.replace("-", "")
            part = f"debug_raw_{ymd}"
            with conn.cursor() as c:
                c.execute("SELECT to_regclass(%s)", (f"public.{part}",))
                if c.fetchone()[0] is None:
                    print(f"[warn] missing partition {part} (need FCS_DEBUG_RAW=1 during stream)")
                    continue
                c.execute(
                    f"SELECT * FROM {part} WHERE symbol=%s ORDER BY ts",
                    (symbol,),
                )
                cols = [d[0] for d in c.description]
                rows = c.fetchall()
            if not rows:
                print(f"[warn] empty {part} symbol={symbol}")
                continue
            frames.append(pd.DataFrame(rows, columns=cols))
    finally:
        conn.close()
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-1: debug_raw vs quote_features_raw")
    ap.add_argument("--dates", required=True, help="comma YYYY-MM-DD")
    ap.add_argument(
        "--offline",
        default=str(
            Path.home()
            / "train_data/july_w1_v4_honest_openwin/quote_features_raw"
            / "QQQ/regular/09:30-16:00/1min/2026-07.parquet"
        ),
        help="离线 quote_features_raw parquet（未归一化）",
    )
    ap.add_argument("--symbol", default="QQQ")
    ap.add_argument("--pg-url", default=os.environ.get("PG_DB_URL", DEFAULT_PG))
    # raw 量纲与 z-score 不同：默认略宽中位误差，仍靠 corr 抓系统性偏差
    ap.add_argument("--med-tol", type=float, default=1e-3, help="median |err| hard gate (raw scale)")
    ap.add_argument("--corr-min", type=float, default=0.95)
    ap.add_argument(
        "--ts-shift-sec",
        type=int,
        default=60,
        help="live.ts + shift 再与 offline timestamp 对齐",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    offline = _load_offline(Path(args.offline).expanduser(), dates)
    live = _load_debug_raw(dates, args.symbol, args.pg_url)
    if offline.empty:
        print("ERROR: offline quote_features_raw empty")
        return 2
    if live.empty:
        print("ERROR: debug_raw empty — run FCS with FCS_DEBUG_RAW=1 first")
        return 2

    feats = _common_feat_cols(offline, live)
    # raw 门禁仍跳过 SKIP_FEATURES（日历/SE 补算/已知脏列）
    _ = SKIP_FEATURES
    print(
        f"[Gate-1 RAW] offline rows={len(offline)} live rows={len(live)} "
        f"common_feats={len(feats)} ts_shift_sec={args.ts_shift_sec}"
    )

    by_day = []
    for d in dates:
        rep = compare_day(
            offline,
            live,
            date=d,
            feats=feats,
            med_tol=args.med_tol,
            corr_min=args.corr_min,
            ts_shift_sec=args.ts_shift_sec,
        )
        by_day.append(rep)
        status = "PASS" if rep.get("pass") else "FAIL"
        print(
            f"\n=== {d} [{status}] matched={rep.get('n_matched')} "
            f"pass_rate={rep.get('pass_rate', 0):.1%} shift={rep.get('ts_shift_sec')} ==="
        )
        if rep.get("reason"):
            print(f"  reason: {rep['reason']}")
        for w in rep.get("worst") or []:
            if w.get("pass"):
                continue
            print(
                f"  FAIL {w['feature']:28s} med={w.get('med_abs_err')} "
                f"max={w.get('max_abs_err')} corr={w.get('corr')}"
            )
        fails = rep.get("failed_features") or []
        if fails:
            print(f"  failed({len(fails)}): {fails[:12]}")

    overall = all(r.get("pass") for r in by_day)
    summary = {
        "gate": 1,
        "name": "debug_raw_vs_quote_features_raw",
        "offline": str(Path(args.offline).expanduser()),
        "symbol": args.symbol,
        "med_tol": args.med_tol,
        "corr_min": args.corr_min,
        "ts_shift_sec": args.ts_shift_sec,
        "n_feats": len(feats),
        "overall_pass": overall,
        "by_day": [{k: v for k, v in r.items() if k != "columns"} for r in by_day],
        "next": "Gate-2 compare_debug_slow_offline.py only if overall_pass",
    }
    print(f"\n=== GATE-1 RAW OVERALL: {'PASS' if overall else 'FAIL'} ===")
    if args.out:
        out = Path(args.out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        full = dict(summary)
        full["by_day_full"] = by_day
        out.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str))
        print(f"wrote {out}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
