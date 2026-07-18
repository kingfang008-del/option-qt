#!/usr/bin/env python3
"""Build Rule-A tabular dataset for LightGBM Smart Bouncer.

Samples = Mag7 first Rule-A fires. Labels prefer option-path ternary (MFE/MAE
on 1s quotes); fallback to underlying ternary when quotes/lock missing.

Output parquet + meta for ``train_lgbm_bouncer.py``. Does not mutate freeze.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.config import load_profile
from maga7.common.lgbm_bouncer import (
    FEATURE_COLS,
    extract_bouncer_features,
    label_option_ternary,
    label_underlying_ternary,
    option_path_mfe_mae,
)
from maga7.common.open_lock import load_multidte_lock_index, resolve_open_lock_contract, resolve_otm_rungs
from maga7.common.replay import load_quotes, month_list, path_for_ticker, to_ny
from maga7.common.signals import (
    attach_mf_features,
    build_all_first_rule_a_signals,
    load_stock_month_files,
)


def _spot_at(sdf: pd.DataFrame | None, asof_ts) -> float | None:
    if sdf is None or sdf.empty:
        return None
    asof = to_ny(asof_ts)
    ts = pd.to_datetime(sdf["timestamp"])
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        ts = ts.dt.tz_convert("America/New_York")
    upto = sdf.loc[ts <= asof]
    if upto.empty:
        return None
    px = float(upto.iloc[-1]["close"])
    return px if np.isfinite(px) and px > 0 else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        default="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json",
    )
    ap.add_argument("--start-date", default="2026-02-01")
    ap.add_argument("--end-date", default="2026-07-17")
    ap.add_argument("--hold-minutes", type=int, default=30)
    ap.add_argument("--good-mfe", type=float, default=0.40)
    ap.add_argument("--good-mae-max", type=float, default=0.15)
    ap.add_argument("--toxic-mae", type=float, default=0.30)
    ap.add_argument("--under-good-mfe", type=float, default=0.004)
    ap.add_argument("--under-good-mae-max", type=float, default=0.002)
    ap.add_argument("--under-toxic-mae", type=float, default=0.006)
    ap.add_argument(
        "--out",
        default="maga7/results/lgbm_bouncer/dataset_rule_a.parquet",
    )
    args = ap.parse_args()

    prof = load_profile(args.profile)
    paths = prof["_paths"]
    symbols = list(prof["symbols"])
    months = month_list(args.start_date, args.end_date)
    stock_by: dict[str, pd.DataFrame] = {}
    for sym in list(dict.fromkeys(symbols + ["QQQ"])):
        raw = load_stock_month_files(paths["stock_root"], sym, months)
        if raw.empty:
            continue
        raw = raw[(raw["date"] >= args.start_date) & (raw["date"] <= args.end_date)]
        stock_by[sym] = attach_mf_features(
            raw,
            mf_window=int(prof["signal"].get("mf_window", 10)),
            vol_ma_window=int(prof["signal"].get("vol_ma_window", 20)),
        )
    trade_stock = {s: stock_by[s] for s in symbols if s in stock_by}
    sigs = build_all_first_rule_a_signals(trade_stock, prof["signal"])
    if sigs.empty:
        raise SystemExit("no Rule-A signals in range")

    lock_path = Path(paths.get("open_locked_map") or paths.get("locked_map")).expanduser()
    multi_idx = load_multidte_lock_index(lock_path) if lock_path.is_file() else {}
    quote_root = Path(paths["quote_1s_root"]).expanduser()
    otm_rungs = resolve_otm_rungs(prof, default=5)
    prefer_dte = int((prof.get("lock") or {}).get("prefer_dte", 0))
    allowed_dte = list((prof.get("lock") or {}).get("allowed_dte") or [0, 1, 2])
    clear_otm = float((prof.get("trade") or {}).get("clear_otm_ban_0dte_pct", 0.01) or 0.01)
    entry_frac = float((prof.get("fill") or {}).get("entry_frac", 0.8))
    exit_frac = float((prof.get("fill") or {}).get("exit_frac", 0.8))
    qqq = stock_by.get("QQQ")

    quote_cache: dict[tuple[str, str], pd.DataFrame | None] = {}
    rows = []
    n_opt = n_under = n_none = 0

    for r in sigs.itertuples(index=False):
        sdf = stock_by.get(r.symbol)
        feat = extract_bouncer_features(
            symbol=r.symbol,
            direction=r.dir,
            asof_ts=r.sig_ts,
            stock_df=sdf,
            qqq_df=qqq,
        )
        if feat is None:
            continue
        date = str(r.date) if hasattr(r, "date") else to_ny(r.sig_ts).strftime("%Y-%m-%d")
        if hasattr(r, "date") and pd.isna(getattr(r, "date", None)):
            date = to_ny(r.sig_ts).strftime("%Y-%m-%d")
        # signals may expose date via sig_ts only
        date = to_ny(r.sig_ts).strftime("%Y-%m-%d")

        y_opt = None
        mfe = mae = end_ret = np.nan
        ticker = None
        label_src = "none"

        spot = _spot_at(sdf, r.sig_ts)
        by_dte = multi_idx.get((str(r.symbol), date))
        ticker, dte, src = resolve_open_lock_contract(
            by_dte,
            direction=str(r.dir),
            moneyness="ATM",
            spot=spot,
            prefer_dte=prefer_dte,
            allowed_dte=allowed_dte,
            clear_otm_thresh=clear_otm,
            ladder=True,
            otm_rungs=otm_rungs,
        )
        if ticker:
            qkey = (str(r.symbol), date)
            if qkey not in quote_cache:
                quote_cache[qkey] = load_quotes(quote_root, str(r.symbol), date)
            path = path_for_ticker(quote_cache[qkey], ticker)
            mm = option_path_mfe_mae(
                path,
                r.sig_ts,
                entry_frac=entry_frac,
                exit_frac=exit_frac,
                hold_minutes=args.hold_minutes,
            )
            if mm is not None:
                mfe, mae, end_ret = mm
                y_opt = label_option_ternary(
                    mfe=mfe,
                    mae=mae,
                    good_mfe=args.good_mfe,
                    good_mae_max=args.good_mae_max,
                    toxic_mae=args.toxic_mae,
                )
                label_src = "option"
                n_opt += 1

        y_under = label_underlying_ternary(
            sdf,
            asof_ts=r.sig_ts,
            direction=r.dir,
            horizon_minutes=args.hold_minutes,
            good_mfe=args.under_good_mfe,
            good_mae_max=args.under_good_mae_max,
            toxic_mae=args.under_toxic_mae,
        )
        if y_opt is None and y_under is not None:
            label_src = "underlying"
            n_under += 1
        elif y_opt is None:
            n_none += 1
            continue

        y_ternary = int(y_opt if y_opt is not None else y_under)
        # train target: allow if not toxic
        y_allow = 0 if y_ternary < 0 else 1

        row = {
            "date": date,
            "symbol": str(r.symbol),
            "direction": str(r.dir),
            "sig_ts": str(to_ny(r.sig_ts)),
            "ticker": ticker,
            "lock_source": src if ticker else None,
            "label_src": label_src,
            "y_ternary": y_ternary,
            "y_allow": y_allow,
            "opt_mfe": float(mfe) if np.isfinite(mfe) else None,
            "opt_mae": float(mae) if np.isfinite(mae) else None,
            "opt_end_ret": float(end_ret) if np.isfinite(end_ret) else None,
            "y_under": y_under,
        }
        row.update(feat)
        rows.append(row)

    if not rows:
        raise SystemExit("no labeled rows")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    meta = {
        "n_rows": int(len(df)),
        "n_option_label": int(n_opt),
        "n_underlying_label": int(n_under),
        "n_skip_no_label": int(n_none),
        "feature_cols": list(FEATURE_COLS),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "hold_minutes": args.hold_minutes,
        "option_thresh": {
            "good_mfe": args.good_mfe,
            "good_mae_max": args.good_mae_max,
            "toxic_mae": args.toxic_mae,
        },
        "y_ternary_counts": df["y_ternary"].value_counts().to_dict(),
        "y_allow_rate": float(df["y_allow"].mean()),
        "option_only_allow_rate": float(df.loc[df["label_src"] == "option", "y_allow"].mean())
        if (df["label_src"] == "option").any()
        else None,
        "profile": args.profile,
        "lock_path": str(lock_path),
    }
    out.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
