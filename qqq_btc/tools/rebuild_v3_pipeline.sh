#!/usr/bin/env bash
# 从 options_databento_v3 重建 QQQ 特征 → split → norm → label → eval
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
cd "$REPO"
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
LOG="${REPO}/qqq_btc/results/rebuild_v3_pipeline.log"
mkdir -p "$(dirname "$LOG")"

exec > >(tee -a "$LOG") 2>&1
echo "=== rebuild_v3_pipeline start $(date -Is) ==="

V3_1M="/mnt/s990/data/raw_1m/options_databento_v3"
DAY_IV="$HOME/train_data/quote_options_day_iv_v3"
MONTHLY_IV="$HOME/train_data/quote_options_monthly_iv_v3"
BUCKETED="$HOME/train_data/quote_options_bucketed_v7_v3"
RAW_FEAT="$HOME/train_data/quote_features_raw_v3"
FEAT_CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
V4_CFG="$FEAT_CFG"
ANCHOR_V3="$REPO/qqq_btc/CONFIG/anchor_qqq_0dte_v3.json"

# ---------- 1. day IV (Greeks) ----------
echo "[1/7] option_cac_day_vectorized_day (v3 1m → day_iv_v3)"
"$PYTHON" - <<'PY'
import multiprocessing
from pathlib import Path
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

calc = OptionIVCalculator(
    db_path="/home/kingfang007/notebook/stocks.db",
    option_root="/mnt/s990/data/raw_1m/options_databento_v3",
    data_root="/home/kingfang007/train_data/spnq_train_resampled",
    iv_option_root=str(Path.home() / "train_data/quote_options_day_iv_v3"),
)
calc.run(max_concurrent_stocks=12)
PY

# ---------- 2. day → month IV ----------
echo "[2/7] iv_day2month (v3)"
"$PYTHON" - <<'PY'
import glob, multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from preprocess.ask_bid.iv_day2month import process_single_symbol, get_target_symbols
from tqdm import tqdm

INPUT_BASE = "/home/kingfang007/train_data/quote_options_day_iv_v3"
OUTPUT_BASE = "/home/kingfang007/train_data/quote_options_monthly_iv_v3"
DB_PATH = "/home/kingfang007/notebook/stocks.db"

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

symbols = get_target_symbols(DB_PATH)
all_files = glob.glob(f"{INPUT_BASE}/**/*.parquet", recursive=True)
symbol_to_files = defaultdict(list)
for f in all_files:
    name = Path(f).stem
    sym = name.rsplit("_", 1)[0]
    if sym in symbols:
        symbol_to_files[sym].append(f)
tasks = [(sym, files, OUTPUT_BASE) for sym, files in symbol_to_files.items()]
with ProcessPoolExecutor(max_workers=16) as pool:
    for fut in tqdm(as_completed({pool.submit(process_single_symbol, t): t for t in tasks}), total=len(tasks)):
        print(fut.result())
PY

# ---------- 3. bucketed monthly features ----------
echo "[3/7] options_locked_feature (v3)"
"$PYTHON" - <<'PY'
import concurrent.futures, logging
from pathlib import Path
from tqdm import tqdm
from preprocess.ask_bid.options_locked_feature import process_single_file

RAW_DIR = Path.home() / "train_data/quote_options_monthly_iv_v3"
OUTPUT_DIR = Path.home() / "train_data/quote_options_bucketed_v7_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tasks = []
for sym in ["QQQ"]:
    src = RAW_DIR / sym / "standard"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    for p in src.glob("*.parquet"):
        tasks.append((p, OUTPUT_DIR, sym))
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(process_single_file, t): t for t in tasks}
    for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
        r = f.result()
        if r:
            logging.warning(r)
PY

# ---------- 4. feature_merge ----------
echo "[4/7] feature_merge_option_raw → quote_features_raw_v3"
"$PYTHON" - <<'PY'
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

fm.OUTPUT_FEATURES_DIR = Path.home() / "train_data/quote_features_raw_v3"
fm.OPTION_MONTHLY_DIR = Path.home() / "train_data/quote_options_monthly_iv_v3"
fm.AGG_OPTION_MONTHLY_DIR = Path.home() / "train_data/quote_options_bucketed_v7_v3"
fm.main()
PY

# ---------- 5. split ----------
echo "[5/7] split_raw_features → *_v3"
"$PYTHON" - <<'PY'
import shutil
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from preprocess.ask_bid.split_raw_features import (
    get_destination_path, process_and_copy_file, get_valid_symbols,
)

SOURCE = Path.home() / "train_data/quote_features_raw_v3"
TRAIN = Path.home() / "train_data/quote_features_train_v3"
VAL = Path.home() / "train_data/quote_features_val_v3"
TEST = Path.home() / "train_data/quote_features_test_v3"
train_r = (pd.Timestamp("2023-03-01"), pd.Timestamp("2025-12-31"))
val_r = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
test_r = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))

symbols = get_valid_symbols()
tasks = []
for sym in symbols:
    sp = SOURCE / sym
    if sp.exists():
        tasks.extend(sp.glob("**/*.parquet"))
worker = partial(
    process_and_copy_file,
    source_dir=SOURCE, train_dir=TRAIN, val_dir=VAL, test_dir=TEST,
    train_range_ts=train_r, val_range_ts=val_r, test_range_ts=test_r,
)
with ProcessPoolExecutor(max_workers=32) as ex:
    list(tqdm(ex.map(worker, tasks), total=len(tasks), desc="split v3"))
print(f"split done: {len(tasks)} files")
PY

# ---------- 6. rolling norm ----------
echo "[6/7] apply_rolling_norm (v3 stages)"
export FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
"$PYTHON" - <<'PY'
import os
from pathlib import Path
import preprocess.ask_bid.apply_rolling_norm_standalone as arn

arn.BASE_ROOT = Path.home() / "train_data"
arn.STAGES = ["train_v3", "val_v3", "test_v3"]

# patch stage dir naming: quote_features_{stage} → quote_features_train_v3 etc.
_orig = arn.main

def _patched_main():
    import logging
    from tqdm import tqdm
    import concurrent.futures
    norm_cols = arn.load_target_features(arn.CONFIG_PATH)
    for stage in arn.STAGES:
        stage_root = arn.BASE_ROOT / f"quote_features_{stage}"
        print(f"Norm stage {stage} -> {stage_root}")
        if not stage_root.exists():
            logging.error("missing %s", stage_root)
            continue
        target_dirs = arn.find_leaf_directories(stage_root)
        tasks = [(d, norm_cols) for d in target_dirs]
        with concurrent.futures.ProcessPoolExecutor(max_workers=arn.MAX_WORKERS) as ex:
            for res in tqdm(ex.map(arn.process_single_directory, tasks), total=len(tasks)):
                if res and res.startswith("ERROR"):
                    logging.error(res)
        arn.verify_data_quality(stage_root, norm_cols)
    print("rolling norm v3 done")

_patched_main()
PY

# ---------- 7. label_pipeline ----------
echo "[7/7] label_pipeline (v3 anchor + fill labels)"
for stage in train_v3 val_v3 test_v3; do
  "$PYTHON" "$REPO/qqq_btc/tools/label_pipeline.py" \
    --input "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --output "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --symbol QQQ \
    --anchor-config "$ANCHOR_V3" \
    --report "/tmp/label_report_${stage}_v3.json"
done

# ---------- 8. V4 eval ----------
echo "[eval] V4 strict replay on v3 features"
"$PYTHON" "$REPO/qqq_btc/tools/eval_test_set.py" \
  --checkpoint "$REPO/checkpoints_qqq_v4/best.pth" \
  --config "$V4_CFG" \
  --feature-root "$HOME/train_data/quote_features_test_v3" \
  --option-1m-root "$V3_1M" \
  --output-dir /tmp/qqq_btc_test_eval_v4_v3 \
  --device auto

echo "=== rebuild_v3_pipeline done $(date -Is) ==="
