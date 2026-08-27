#!/usr/bin/env bash
# Rebuild July (available sessions) old_lock features → V4 frozen infer → VX live-aligned replay.
# 07-03 is NYSE holiday (skip). 07-21 requires a valid MASSIVE/POLYGON key (download separately).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
EXP="${EXP:-/mnt/s990/data/raw_1s/dte1_options_old_lock}"
FEAT_ROOT="${FEAT_ROOT:-$HOME/train_data/dte1_options_old_lock_feat}"
FEAT_CFG="${FEAT_CFG:-$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json}"
NORM_CFG="${NORM_CFG:-$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json}"
FROZEN="${FROZEN:-$REPO/qqq_btc/CONFIG/frozen_norm_qqq_daily.npz}"
CKPT="${CKPT:-$REPO/checkpoint/checkpoints_qqq_v4/best.pth}"
INFER_OUT="${INFER_OUT:-$REPO/qqq_btc/results/v4_jul_full_frozen_infer}"
PROFILE="${PROFILE:-$REPO/qqq_btc/CONFIG/strategy_profiles/v4_vx_jul_full_frozen_v1.json}"
OUT_NAME="${OUT_NAME:-v4_vx_jul_full_frozen_v1_offline}"
START="${START:-2026-07-01}"
END="${END:-2026-07-20}"

echo "[1/5] merge by_date lock maps → root map ($START..$END)"
"$PY" - <<PY
from pathlib import Path
import pandas as pd
exp = Path("$EXP")
start, end = "$START", "$END"
rows = []
for d in sorted((exp / "by_date").iterdir()):
    if not d.is_dir():
        continue
    ds = d.name
    if not (start <= ds <= end):
        continue
    lm = d / "lock_map.parquet"
    raw = exp / "QQQ" / f"QQQ_{ds}.parquet"
    if not lm.is_file() or not raw.is_file():
        print("skip incomplete", ds)
        continue
    df = pd.read_parquet(lm)
    rows.append(df)
    # ensure day_iv staged
    hits = sorted((exp / "quote_options_day_iv" / "QQQ").glob(f"**/QQQ_{ds}.parquet"))
    if hits and not (d / "day_iv.parquet").exists():
        tgt = d / "day_iv.parquet"
        try:
            tgt.symlink_to(hits[0])
        except FileExistsError:
            pass
        print("staged day_iv", ds)
if not rows:
    raise SystemExit("no lock maps to merge")
out = pd.concat(rows, ignore_index=True)
out_path = exp / "locked_targets_map_open_4bucket.parquet"
out.to_parquet(out_path, index=False)
print("wrote", out_path, "rows", len(out), "days", sorted(out["date_str"].astype(str).unique()))
PY

echo "[2/5] build raw features + frozen_norm test ($START..$END)"
BACKFILL_EXP="$EXP" \
BACKFILL_FEAT_ROOT="$FEAT_ROOT" \
BACKFILL_FEAT_CFG="$FEAT_CFG" \
BACKFILL_NORM_CFG="$NORM_CFG" \
BACKFILL_START="$START" \
BACKFILL_END="$END" \
BACKFILL_SYM=QQQ \
BACKFILL_FROZEN="$FROZEN" \
BACKFILL_NORM_MODE=frozen \
BACKFILL_FEAT_HISTORY="$HOME/train_data/quote_features_raw" \
"$PY" - <<'PY'
import glob, json, os, shutil
from pathlib import Path
import pandas as pd
from preprocess.ask_bid.iv_day2month import process_single_symbol
import preprocess.ask_bid.feature_merge_option_raw as fm
from preprocess.ask_bid.options_locked_feature import process_single_file
from qqq_btc.common.frozen_norm import apply_frozen_norm_df

exp = Path(os.environ["BACKFILL_EXP"])
feat_root = Path(os.environ["BACKFILL_FEAT_ROOT"])
sym = os.environ["BACKFILL_SYM"]
start = os.environ["BACKFILL_START"]
end = os.environ["BACKFILL_END"]
months = [p.strftime("%Y-%m") for p in pd.period_range(start[:7], end[:7], freq="M")]
inp = exp / "quote_options_day_iv"
out_m = feat_root / "quote_options_monthly_iv"
files = sorted(glob.glob(f"{inp}/{sym}/**/*.parquet", recursive=True))
# keep only files in [start,end]
keep = []
for f in files:
    name = Path(f).stem  # QQQ_2026-07-16
    ds = name.split("_", 1)[-1]
    if start <= ds <= end:
        keep.append(f)
print("day_iv files in range", len(keep), "months", months)
if keep:
    print(process_single_symbol((sym, keep, str(out_m))))
bucketed = feat_root / "quote_options_bucketed_v7"
fm.OPTION_MONTHLY_DIR = out_m
fm.AGG_OPTION_MONTHLY_DIR = bucketed
fm.OUTPUT_FEATURES_DIR = feat_root / "quote_features_raw"
cfg = json.loads(Path(os.environ["BACKFILL_FEAT_CFG"]).read_text())
for ym in months:
    raw_month = out_m / sym / "standard" / f"{ym}.parquet"
    if raw_month.is_file():
        print(process_single_file((raw_month, bucketed, sym)) or f"bucketed ok {ym}")
    else:
        raise SystemExit(f"missing monthly_iv {raw_month}")
    print(fm.process_stock_month(sym, ym, cfg))

frozen = Path(os.environ["BACKFILL_FROZEN"])
norm_cfg = json.loads(Path(os.environ["BACKFILL_NORM_CFG"]).read_text())
feats = norm_cfg.get("features") or norm_cfg.get("slow_features") or []
names = [f.get("name") for f in feats if isinstance(f, dict) and f.get("name")] if feats and isinstance(feats[0], dict) else [str(x) for x in feats]
for res in ("1min", "5min"):
    for ym in months:
        raw_p = feat_root / f"quote_features_raw/{sym}/regular/09:30-16:00/{res}/{ym}.parquet"
        if not raw_p.is_file():
            print("skip frozen missing", raw_p)
            continue
        out_dir = feat_root / f"quote_features_test/{sym}/regular/09:30-16:00/{res}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_p = out_dir / f"{ym}.parquet"
        df = pd.read_parquet(raw_p)
        n_days = None
        if "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            if getattr(ts.dt, "tz", None) is not None:
                day_s = ts.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            else:
                day_s = ts.dt.strftime("%Y-%m-%d")
            mask = (day_s >= start) & (day_s <= end)
            df = df.loc[mask].reset_index(drop=True)
            n_days = int(day_s.loc[mask].nunique())
        normed = apply_frozen_norm_df(df, frozen, feature_names=names)
        normed.to_parquet(out_p, index=False)
        print("frozen_norm wrote", out_p, "rows", len(normed), "days", n_days)
PY

echo "[3/6] fix July stock right-label (best effort)"
"$PY" qqq_btc/tools/fix_qqq_july_right_label_1min.py || true

echo "[4/6] patch VIXY / vix_level into raw+test (required for put_gate)"
"$PY" qqq_btc/tools/patch_vixy_features.py \
  --ym 2026-07 \
  --feature-root "$FEAT_ROOT/quote_features_raw" \
  --feature-root "$FEAT_ROOT/quote_features_test" \
  --apply-frozen-on-test

echo "[5/6] V4 infer with frozen_norm on raw features"
RAW_ROOT="$FEAT_ROOT/quote_features_raw"
TMP="$FEAT_ROOT/eval_feat_july_full_frozen"
rm -rf "$TMP"
mkdir -p "$TMP/QQQ/regular/09:30-16:00/1min" "$TMP/QQQ/regular/09:30-16:00/5min" "$INFER_OUT"
cp "$RAW_ROOT/QQQ/regular/09:30-16:00/1min/2026-07.parquet" "$TMP/QQQ/regular/09:30-16:00/1min/"
if [[ -f "$RAW_ROOT/QQQ/regular/09:30-16:00/5min/2026-07.parquet" ]]; then
  cp "$RAW_ROOT/QQQ/regular/09:30-16:00/5min/2026-07.parquet" "$TMP/QQQ/regular/09:30-16:00/5min/"
fi
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT" \
  --config "$NORM_CFG" \
  --feature-root "$TMP" \
  --option-1m-root "$EXP/options_1m" \
  --call-bucket 2 --put-bucket 0 \
  --output-dir "$INFER_OUT" \
  --frozen-norm "$FROZEN" \
  --seed 42 --device cuda --live-replay

echo "[6/6] offline LIVE-aligned replay (VX + quarantine)"
"$PY" qqq_btc/tools/replay_offline_live_aligned.py \
  --months 2026-07 \
  --strategy-profile "$PROFILE" \
  --out-name "$OUT_NAME" \
  --skip-build

echo "DONE infer=$INFER_OUT/test_infer.parquet"
echo "DONE replay=qqq_btc/results/offline_live_aligned/$OUT_NAME/summary.json"
