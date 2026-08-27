#!/usr/bin/env python3
"""Ablate winner-safe L2 path Greeks exits on Jul-20 fused books.

Compares presets on:
  - clock (T+30/SL) exit horizon
  - wash_and_opt exit horizon (stack: L2 can only exit *earlier*)

Uses live Redis db=0 option ``localSymbol`` mids.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.path_greeks_exit import PRESETS, PathGreeksState, cfg_from_preset
from maga7.live.redis_fused import redis_client, run_keys, unpack_batch

NY = "America/New_York"
SESSION = Path(
    "/mnt/s990/data/maga7/live_sessions/2026-07-20/live_20260720_083539_29843e"
)
CLOCK = SESSION / "fused_replay_lit_m5_v1_clock/trades.csv"
WASH = SESSION / "fused_replay_wash_and_opt_v1/trades.csv"
OUT = Path("/mnt/s990/data/maga7/results/path_greeks_exit_ablation_v1")


def _parse_occ(contract: str):
    s = str(contract).replace(" ", "")
    m = re.match(r"^([A-Z]+)(\d{6})([CP])(\d{8})$", s)
    if not m:
        raise ValueError(contract)
    sym, ymd, cp, strike_s = m.groups()
    return sym, ymd, float(strike_s) / 1000.0, cp.lower()


def extract_paths(trades: pd.DataFrame, session_id: str) -> dict[str, pd.DataFrame]:
    wants = {}
    for _, tr in trades.iterrows():
        key = str(tr["contract"]).replace(" ", "")
        sym, _, _, _ = _parse_occ(tr["contract"])
        wants[key] = (
            sym,
            pd.Timestamp(tr["entry_ts"]).tz_convert(NY),
            pd.Timestamp(tr["exit_ts"]).tz_convert(NY),
        )
    # extend horizon to max of both books later; for now use each trade exit
    r = redis_client(db=0)
    stream = run_keys(session_id)["stream"]
    t_min = min(v[1] for v in wants.values()) - pd.Timedelta(seconds=2)
    t_max = max(v[2] for v in wants.values()) + pd.Timedelta(minutes=35)
    buckets = {k: [] for k in wants}
    suffixes = {k: k[-15:] for k in wants}
    cursor = "-"
    while True:
        chunk = r.xrange(stream, min=cursor, max="+", count=500)
        if not chunk:
            break
        for mid, fields in chunk:
            raw = fields.get(b"batch")
            if raw is None:
                continue
            for payload in unpack_batch(raw):
                if not isinstance(payload, dict):
                    continue
                sym = str(payload.get("symbol") or "").upper()
                ts = float(payload.get("ts") or 0.0)
                t = pd.Timestamp(ts, unit="s", tz="UTC").tz_convert(NY)
                if t < t_min or t > t_max:
                    continue
                S = float((payload.get("stock") or {}).get("close") or 0.0)
                for oc in list(payload.get("option_contracts") or []):
                    if not isinstance(oc, dict):
                        continue
                    c = str(oc.get("localSymbol") or "").replace(" ", "")
                    hit = None
                    for key, suf in suffixes.items():
                        if wants[key][0].upper() != sym:
                            continue
                        if c == key or (len(c) >= 15 and c[-15:] == suf):
                            hit = key
                            break
                    if hit is None:
                        continue
                    mid_px = float(oc.get("mid") or 0.0)
                    bid = float(oc.get("bid") or 0.0)
                    ask = float(oc.get("ask") or 0.0)
                    if mid_px <= 0:
                        mid_px = (
                            0.5 * (bid + ask)
                            if bid > 0 and ask > 0
                            else (bid or ask)
                        )
                    if mid_px <= 0:
                        continue
                    buckets[hit].append({"ts": t, "S": S, "mid": mid_px, "unix": ts})
            cursor = "(" + (mid.decode() if isinstance(mid, bytes) else str(mid))
        if len(chunk) < 500:
            break
    out = {}
    for key, rows in buckets.items():
        if not rows:
            out[key] = pd.DataFrame()
            continue
        df = pd.DataFrame(rows).drop_duplicates("ts").sort_values("ts")
        out[key] = df.reset_index(drop=True)
    return out


def simulate_book(
    trades: pd.DataFrame,
    paths: dict[str, pd.DataFrame],
    preset: str,
    *,
    book: str,
) -> pd.DataFrame:
    cfg, naive = cfg_from_preset(preset)
    rows = []
    for _, tr in trades.iterrows():
        key = str(tr["contract"]).replace(" ", "")
        sym, ymd, K, cp = _parse_occ(tr["contract"])
        entry_ts = pd.Timestamp(tr["entry_ts"]).tz_convert(NY)
        exit_ts = pd.Timestamp(tr["exit_ts"]).tz_convert(NY)
        path = paths.get(key, pd.DataFrame())
        clock_ret = float(tr["ret"])
        if path.empty:
            rows.append(
                {
                    "book": book,
                    "preset": preset,
                    "symbol": sym,
                    "ok": False,
                    "final_ret": clock_ret,
                    "base_ret": clock_ret,
                    "reason": str(tr["reason"]),
                    "l2_reason": None,
                    "lift": 0.0,
                }
            )
            continue
        # path up to base exit (L2 can only fire earlier when stacking)
        path = path.loc[(path["ts"] >= entry_ts) & (path["ts"] <= exit_ts)].copy()
        expiry = pd.Timestamp(
            f"20{ymd[:2]}-{ymd[2:4]}-{ymd[4:6]}", tz=NY
        ) + pd.Timedelta(hours=16)
        st = PathGreeksState(
            entry_px=float(tr["entry"]),
            K=K,
            cp=cp,
            expiry_ts=float(expiry.timestamp()),
            cfg=cfg,
            naive_half_peak=naive,
            entry_ts=float(entry_ts.timestamp()),
        )
        l2_reason = None
        l2_ret = None
        l2_hold = None
        for _, row in path.iterrows():
            reason, met = st.on_tick(
                ts=float(row["unix"]), mid=float(row["mid"]), S=float(row["S"])
            )
            if reason:
                l2_reason = reason
                l2_ret = float(met["opt_ret"])
                l2_hold = float((row["ts"] - entry_ts).total_seconds())
                break
        final = l2_ret if l2_ret is not None else clock_ret
        rows.append(
            {
                "book": book,
                "preset": preset,
                "symbol": sym,
                "contract": str(tr["contract"]),
                "ok": True,
                "base_ret": clock_ret,
                "base_reason": str(tr["reason"]),
                "final_ret": final,
                "l2_reason": l2_reason,
                "l2_hold_sec": l2_hold,
                "lift": float(final - clock_ret),
                "mfe": float(st.peak_ret),
            }
        )
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument(
        "--presets",
        default="off,naive,winner_safe,toxic_only",
    )
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    clock = pd.read_csv(CLOCK)
    wash = pd.read_csv(WASH)
    # union horizon for path extract
    both = pd.concat([clock, wash], ignore_index=True)
    # for each contract keep max exit
    both["contract_key"] = both["contract"].astype(str).str.replace(" ", "", regex=False)
    both["exit_ts"] = pd.to_datetime(both["exit_ts"], utc=True)
    both["entry_ts"] = pd.to_datetime(both["entry_ts"], utc=True)
    agg = (
        both.sort_values("exit_ts")
        .groupby("contract_key", as_index=False)
        .agg({"contract": "last", "entry_ts": "min", "exit_ts": "max", "symbol": "last"})
    )
    # fake trades frame for extract
    extract_df = agg.rename(columns={})
    print("extracting paths...", flush=True)
    paths = extract_paths(extract_df, SESSION.name)
    for k, df in paths.items():
        print(f"  {k}: n={len(df)}", flush=True)

    presets = [x.strip() for x in args.presets.split(",") if x.strip()]
    parts = []
    score = []
    for preset in presets:
        for book, tr in [("clock", clock), ("wash_and_opt", wash)]:
            df = simulate_book(tr, paths, preset, book=book)
            parts.append(df)
            ok = df[df["ok"]]
            score.append(
                {
                    "book": book,
                    "preset": preset,
                    "sum_base": float(ok["base_ret"].sum()),
                    "sum_final": float(ok["final_ret"].sum()),
                    "sum_lift": float(ok["lift"].sum()),
                    "n_l2": int(ok["l2_reason"].notna().sum()),
                    "n": int(len(ok)),
                }
            )
            print(
                f"{book:14s} {preset:12s} sum {ok['final_ret'].sum():+.3f} "
                f"(base {ok['base_ret'].sum():+.3f}) l2={ok['l2_reason'].notna().sum()}/{len(ok)}",
                flush=True,
            )

    detail = pd.concat(parts, ignore_index=True)
    detail.to_csv(out / "trades_detail.csv", index=False)
    sb = pd.DataFrame(score)
    sb.to_csv(out / "scoreboard.csv", index=False)
    (out / "scoreboard.json").write_text(sb.to_json(orient="records", indent=2))
    (out / "presets.json").write_text(json.dumps(PRESETS, indent=2))
    print(sb.to_string(index=False))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
