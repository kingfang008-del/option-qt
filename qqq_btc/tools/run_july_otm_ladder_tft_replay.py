#!/usr/bin/env python3
"""July TFT replay on maga7 open-ladder OTM quotes (NOT open_4bucket polluted tree).

Builds a V4-compatible 4-bucket vehicle from maga7 OTM ladder:
  bucket0 = 1DTE ATM PUT
  bucket1 = 1DTE OTM{rung} PUT
  bucket2 = 1DTE ATM CALL
  bucket3 = 1DTE OTM{rung} CALL

Then: 1s→1m → day_iv → features → frozen infer (V4/FT56) → VX live-aligned replay.

Note: this is an *execution-vehicle* test. Training-true lock is old_style foresight
(``step1_build_target_map_old.py``), not open-ladder. See docs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

NY = "America/New_York"
MAP_DEFAULT = Path.home() / "train_data/locked_targets_map_maga7_googl_qqq_open_ladder_atm5otm_jan_jul.parquet"
SRC_1S = Path("/mnt/s990/data/raw_1s/maga7_mf10_open_ladder_otm5/QQQ")
PY = os.environ.get("PYTHON", str(Path.home() / "anaconda3/envs/ibkr/bin/python"))


def _run(cmd: list[str], *, env: dict | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    merged["PYTHONPATH"] = str(REPO) + (os.pathsep + merged["PYTHONPATH"] if merged.get("PYTHONPATH") else "")
    subprocess.run(cmd, check=True, cwd=str(REPO), env=merged)


def _norm_ticker(t: str) -> str:
    return str(t).replace("O:", "").strip()


def build_remapped_1s(
    *,
    map_path: Path,
    src_1s: Path,
    out_root: Path,
    start: str,
    end: str,
    otm_rung: int,
) -> Path:
    """Write RAW1S/QQQ/QQQ_date.parquet with V4 bucket_ids 0..3 from ladder map."""
    m = pd.read_parquet(map_path)
    m = m[(m["symbol"].astype(str).str.upper() == "QQQ")].copy()
    m["date_str"] = m["date_str"].astype(str)
    m = m[(m["date_str"] >= start) & (m["date_str"] <= end)]
    want_tags = {
        0: f"open_1dte_ATM_p",
        1: f"open_1dte_OTM{otm_rung}_p",
        2: f"open_1dte_ATM_c",
        3: f"open_1dte_OTM{otm_rung}_c",
    }
    out_qqq = out_root / "QQQ"
    out_qqq.mkdir(parents=True, exist_ok=True)
    map_rows = []
    dates = sorted(m["date_str"].unique())
    for d in dates:
        src = src_1s / f"QQQ_{d}.parquet"
        if not src.is_file():
            print(f"skip missing 1s {src}", flush=True)
            continue
        day_m = m[m["date_str"] == d]
        tick_to_b: dict[str, int] = {}
        tag_to_strike = {}
        for b, tag in want_tags.items():
            hit = day_m[day_m["tag"].astype(str) == tag]
            if hit.empty:
                print(f"WARN {d} missing tag {tag}", flush=True)
                continue
            row = hit.iloc[0]
            t = _norm_ticker(row["contract_symbol"])
            tick_to_b[t] = b
            tag_to_strike[tag] = float(row["strike"])
            map_rows.append(
                {
                    "date_str": d,
                    "contract_symbol": str(row["contract_symbol"]),
                    "bucket_id": b,
                    "symbol": "QQQ",
                    "tag": tag,
                    "strike": float(row["strike"]),
                    "dte_mode": "trading",
                    "front_dte": 1,
                }
            )
        if len(tick_to_b) < 4:
            print(f"skip {d}: only {len(tick_to_b)}/4 contracts", flush=True)
            continue
        raw = pd.read_parquet(src)
        raw["ticker_norm"] = raw["ticker"].astype(str).map(_norm_ticker)
        sub = raw[raw["ticker_norm"].isin(tick_to_b)].copy()
        if sub.empty:
            print(f"skip {d}: no overlapping tickers in 1s", flush=True)
            continue
        sub["bucket_id"] = sub["ticker_norm"].map(tick_to_b).astype(int)
        # rewrite tags for downstream
        rev = {v: k for k, v in want_tags.items()}
        sub["tag"] = sub["bucket_id"].map(rev)
        sub["underlying"] = "QQQ"
        keep = [c for c in ("ts", "timestamp", "ticker", "tag", "bucket_id", "underlying", "bid", "ask", "bid_size", "ask_size", "price", "mid_price", "strike") if c in sub.columns]
        out_p = out_qqq / f"QQQ_{d}.parquet"
        sub[keep].to_parquet(out_p, index=False)
        print(f"wrote {out_p} rows={len(sub)} buckets={sorted(sub['bucket_id'].unique())}", flush=True)

    map_out = out_root / f"locked_targets_map_qqq_1dte_atm_otm{otm_rung}_from_ladder.parquet"
    pd.DataFrame(map_rows).to_parquet(map_out, index=False)
    print("map", map_out, "n", len(map_rows), flush=True)
    return map_out


def build_features_and_infer(
    *,
    raw1s: Path,
    exp: Path,
    start: str,
    end: str,
    ym: str,
    ckpt: Path,
    infer_out: Path,
    frozen: Path,
) -> Path:
    opt1m = exp / "options_1m"
    _run(
        [
            PY,
            str(REPO / "preprocess/download/step3_databento_aggregate_1s_to_1m.py"),
            "--input-dir",
            str(raw1s),
            "--output-dir",
            str(opt1m),
            "--symbol",
            "QQQ",
            "--date-from",
            start,
            "--date-to",
            end,
            "--force",
        ]
    )
    # day_iv
    day_iv = exp / "quote_options_day_iv"
    code = r"""
import multiprocessing, os
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass
calc = OptionIVCalculator(
    db_path=os.environ.get("BACKFILL_DB", "/home/kingfang007/notebook/stocks.db"),
    option_root=os.environ["BACKFILL_OPT_1M"],
    data_root=os.environ["BACKFILL_STOCK_RESAMP"],
    iv_option_root=os.environ["BACKFILL_DAY_IV"],
)
calc.run(max_concurrent_stocks=2)
print("day_iv done", os.environ["BACKFILL_DAY_IV"])
"""
    _run(
        [PY, "-c", code],
        env={
            "BACKFILL_OPT_1M": str(opt1m),
            "BACKFILL_DAY_IV": str(day_iv),
            "BACKFILL_STOCK_RESAMP": str(Path.home() / "train_data/spnq_train_resampled"),
        },
    )
    # features + frozen
    feat_cfg = REPO / "qqq_btc/CONFIG/slow_feature_qqq_v2.json"
    norm_cfg = REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"
    feat_code = r"""
import glob, json, os
from pathlib import Path
import pandas as pd
from preprocess.ask_bid.iv_day2month import process_single_symbol
import preprocess.ask_bid.feature_merge_option_raw as fm
from preprocess.ask_bid.options_locked_feature import process_single_file
from qqq_btc.common.frozen_norm import apply_frozen_norm_df

exp = Path(os.environ["EXP"])
ym = os.environ["YM"]
sym = "QQQ"
inp = exp / "quote_options_day_iv"
out_m = exp / "quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/{sym}/**/*.parquet", recursive=True))
print("day_iv files", len(files))
if files:
    print(process_single_symbol((sym, files, str(out_m))))
bucketed = exp / "quote_options_bucketed_v7"
fm.OPTION_MONTHLY_DIR = out_m
fm.AGG_OPTION_MONTHLY_DIR = bucketed
fm.OUTPUT_FEATURES_DIR = exp / "quote_features_raw"
cfg = json.loads(Path(os.environ["FEAT_CFG"]).read_text())
raw_month = out_m / sym / "standard" / f"{ym}.parquet"
if not raw_month.is_file():
    raise SystemExit(f"missing {raw_month}")
print(process_single_file((raw_month, bucketed, sym)))
print(fm.process_stock_month(sym, ym, cfg))
# frozen test
frozen = Path(os.environ["FROZEN"])
norm_cfg = json.loads(Path(os.environ["NORM_CFG"]).read_text())
feats = norm_cfg.get("features") or []
names = [f.get("name") for f in feats if isinstance(f, dict) and f.get("name")]
raw_p = exp / f"quote_features_raw/{sym}/regular/09:30-16:00/1min/{ym}.parquet"
out_dir = exp / f"quote_features_test/{sym}/regular/09:30-16:00/1min"
out_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(raw_p)
apply_frozen_norm_df(df, frozen, feature_names=names).to_parquet(out_dir / f"{ym}.parquet", index=False)
print("features ok", raw_p, "rows", len(df))
"""
    _run(
        [PY, "-c", feat_code],
        env={
            "EXP": str(exp),
            "YM": ym,
            "FEAT_CFG": str(feat_cfg),
            "NORM_CFG": str(norm_cfg),
            "FROZEN": str(frozen),
        },
    )
    # patch vix
    _run(
        [
            PY,
            str(REPO / "qqq_btc/tools/patch_vixy_features.py"),
            "--ym",
            ym,
            "--feature-root",
            str(exp / "quote_features_raw"),
            "--feature-root",
            str(exp / "quote_features_test"),
            "--apply-frozen-on-test",
        ]
    )
    # infer
    tmp = exp / "eval_feat_only"
    if tmp.exists():
        shutil.rmtree(tmp)
    (tmp / "QQQ/regular/09:30-16:00/1min").mkdir(parents=True)
    shutil.copy2(
        exp / f"quote_features_raw/QQQ/regular/09:30-16:00/1min/{ym}.parquet",
        tmp / "QQQ/regular/09:30-16:00/1min" / f"{ym}.parquet",
    )
    infer_out.mkdir(parents=True, exist_ok=True)
    _run(
        [
            PY,
            str(REPO / "qqq_btc/tools/eval_test_set.py"),
            "--checkpoint",
            str(ckpt),
            "--config",
            str(norm_cfg),
            "--feature-root",
            str(tmp),
            "--option-1m-root",
            str(opt1m),
            "--call-bucket",
            "2",
            "--put-bucket",
            "0",
            "--output-dir",
            str(infer_out),
            "--frozen-norm",
            str(frozen),
            "--seed",
            "42",
            "--device",
            "cuda",
            "--live-replay",
        ]
    )
    return infer_out / "test_infer.parquet"


def write_profile(
    *,
    profile_id: str,
    ckpt: str,
    infer: str,
    raw1: str,
    opt1m: str,
    out_path: Path,
    scope: str,
) -> Path:
    blob = {
        "schema_version": 1,
        "profile_id": profile_id,
        "description": f"July TFT on maga7 open-ladder vehicle ({scope}). Not open_4bucket; not training foresight lock.",
        "base_replay": "LIVE_REPLAY",
        "model": {"checkpoint": ckpt},
        "inputs": {
            "infer_by_month": {"2026-07": infer},
            "raw1_by_month": {"2026-07": raw1},
            "opt1m_by_month": {"2026-07": opt1m},
        },
        "replay_overrides": {
            "edge_q10_floor": -0.2,
            "apply_put_entry_quantile": False,
            "next_day_put_quarantine_loss": -0.02,
            "next_day_put_quarantine_vx_slope_min": 0.06,
            "next_day_all_leg_defense_loss": -0.05,
            "next_day_all_leg_defense_position_frac": 0.125,
            "next_day_all_leg_defense_vx_slope_min": 0.06,
        },
        "selector": {
            "mode": "vx",
            "rule_profiles": "qqq_btc/CONFIG/rule_profiles.json",
            "vx_term_structure": "/mnt/s990/data/raw_1m/vix_futures_databento/vx_term_structure_1d.parquet",
            "spot_root": "~/train_data/spnq_train_resampled",
        },
        "execution": {
            "put_gate_mode": "vixy_z",
            "tick_exits": "off",
            "live_label_shift_sec": 60,
            "fill_spread_frac": 0.775,
            "execution_delay_bars": 0,
            "oms_signal_delay_bars": 0,
        },
        "features": {
            "slow_feature_config": "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
            "frozen_norm": "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz",
            "scope_label": scope,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--otm-rung", type=int, default=3, choices=[1, 2, 3, 4, 5])
    ap.add_argument("--start", default="2026-07-01")
    ap.add_argument("--end", default="2026-07-13")
    ap.add_argument("--map", type=Path, default=MAP_DEFAULT)
    ap.add_argument("--src-1s", type=Path, default=SRC_1S)
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("/mnt/s990/data/raw_1s/qqq_tft_otm_ladder_jul"),
    )
    ap.add_argument("--models", default="v4,ft56", help="comma: v4,ft56")
    args = ap.parse_args()
    ym = "2026-07"
    rung = int(args.otm_rung)
    raw1s = Path(args.out_root) / f"otm{rung}_1dte"
    exp = Path.home() / f"train_data/july_tft_ladder_otm{rung}"
    frozen = REPO / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"

    print(f"=== remap OTM{rung} ladder → V4 4-bucket ===", flush=True)
    build_remapped_1s(
        map_path=Path(args.map),
        src_1s=Path(args.src_1s),
        out_root=raw1s,
        start=args.start,
        end=args.end,
        otm_rung=rung,
    )

    models = {
        "v4": REPO / "checkpoint/checkpoints_qqq_v4/best.pth",
        "ft56": REPO / "checkpoint/checkpoints_qqq_ft56_julw1/best.pth",
    }
    for name in [x.strip() for x in args.models.split(",") if x.strip()]:
        ckpt = models[name]
        infer_out = REPO / f"qqq_btc/results/{name}_jul_ladder_otm{rung}_frozen_infer"
        print(f"\n=== features+infer {name} ===", flush=True)
        # rebuild features once; reuse for second model
        if name == args.models.split(",")[0].strip() or not (
            exp / "quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet"
        ).is_file():
            build_features_and_infer(
                raw1s=raw1s,
                exp=exp,
                start=args.start,
                end=args.end,
                ym=ym,
                ckpt=ckpt,
                infer_out=infer_out,
                frozen=frozen,
            )
        else:
            # only re-infer
            tmp = exp / "eval_feat_only"
            infer_out.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    PY,
                    str(REPO / "qqq_btc/tools/eval_test_set.py"),
                    "--checkpoint",
                    str(ckpt),
                    "--config",
                    str(REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"),
                    "--feature-root",
                    str(tmp),
                    "--option-1m-root",
                    str(exp / "options_1m"),
                    "--call-bucket",
                    "2",
                    "--put-bucket",
                    "0",
                    "--output-dir",
                    str(infer_out),
                    "--frozen-norm",
                    str(frozen),
                    "--seed",
                    "42",
                    "--device",
                    "cuda",
                    "--live-replay",
                ]
            )
        prof = REPO / f"qqq_btc/CONFIG/strategy_profiles/{name}_vx_jul_ladder_otm{rung}_v1.json"
        write_profile(
            profile_id=f"{name}_vx_jul_ladder_otm{rung}_v1",
            ckpt=str(ckpt.relative_to(REPO)) if ckpt.is_relative_to(REPO) else str(ckpt),
            infer=str(infer_out.relative_to(REPO) / "test_infer.parquet"),
            raw1=str(exp / "quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-07.parquet"),
            opt1m=str(exp / "options_1m"),
            out_path=prof,
            scope=f"July_ladder_1dte_ATM_OTM{rung}_{args.start}_{args.end}",
        )
        out_name = f"{name}_vx_jul_ladder_otm{rung}_v1_offline"
        _run(
            [
                PY,
                str(REPO / "qqq_btc/tools/replay_offline_live_aligned.py"),
                "--months",
                "2026-07",
                "--strategy-profile",
                str(prof),
                "--out-name",
                out_name,
                "--skip-build",
            ]
        )
        print("DONE", out_name, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
