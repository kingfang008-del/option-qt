#!/usr/bin/env bash
# V4 全链路重建：Polygon dte2_options 8-bucket ladder 1s → 1m → dynamic best-label → LMDB → pretrain
#
# 数据切分:
#   train 2023-03 ~ 2025-12
#   val   2026-01 ~ 2026-02
#   test  2026-03 ~ 2026-04
# 股票数据源: spnq_train_resampled（不变）
# anchor: qqq_2dte_ladder (2DTE 8-bucket dynamic best executable label)
set -euo pipefail

REPO="/home/kingfang007/文档/GitHub/option-qt"
PY="/home/kingfang007/anaconda3/envs/ibkr/bin/python"
LOG="$REPO/qqq_btc/results/rebuild_train_v4_dte2_ladder_conservative.log"
MANIFEST="$HOME/train_data/builds/v4_dte2_ladder_conservative/manifest.json"
CKPT_DIR="$REPO/checkpoints_qqq_v4_dte2_ladder_conservative"
LMDB_ROOT="$HOME/train_data/lmdb"
EVAL_OUT="/tmp/qqq_btc_test_eval_v4_dte2_ladder_conservative"
STOCK_ROOT="/home/kingfang007/train_data/spnq_train_resampled"
TRAIN_START="2023-03-01"
TRAIN_END="2025-12-31"
VAL_START="2026-01-01"
VAL_END="2026-02-28"
TEST_START="2026-03-01"
TEST_END="2026-04-30"
FEATURE_CFG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4_dte2_ladder_conservative.json"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
ANCHOR_CFG="$REPO/qqq_btc/CONFIG/anchor_qqq_2dte_ladder_conservative.json"
OPTION_1S="/mnt/s990/data/raw_1s/dte2_options"
OPTION_1M="/mnt/s990/data/raw_1m/dte2_options"
DAY_IV="$HOME/train_data/quote_options_day_iv_dte2_ladder"
MONTHLY_IV="$HOME/train_data/quote_options_monthly_iv_dte2_ladder"
BUCKETED="$HOME/train_data/quote_options_bucketed_v7_dte2_ladder"
RAW_FEAT="$HOME/train_data/quote_features_raw_dte2_ladder_conservative"
TRAIN_FEAT="$HOME/train_data/quote_features_train_dte2_ladder_conservative"
VAL_FEAT="$HOME/train_data/quote_features_val_dte2_ladder_conservative"
TEST_FEAT="$HOME/train_data/quote_features_test_dte2_ladder_conservative"

export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export FEATURE_CONFIG="$FEATURE_CFG"
export LOCKED_TARGETS_MAP="$HOME/train_data/locked_targets_map_2dte_ladder.parquet"

MAP_MAX_DATE="$("$PY" - <<'PY'
import os
from pathlib import Path

import pandas as pd

p = Path(os.environ["LOCKED_TARGETS_MAP"]).expanduser()
df = pd.read_parquet(p, columns=["date_str"])
print(str(df["date_str"].astype(str).max()))
PY
)"
if [[ "$MAP_MAX_DATE" < "$TEST_START" ]]; then
  echo "locked map max date ${MAP_MAX_DATE} is before TEST_START ${TEST_START}" >&2
  exit 1
fi
if [[ "$MAP_MAX_DATE" < "$TEST_END" ]]; then
  TEST_END="$MAP_MAX_DATE"
fi

cd "$REPO"
mkdir -p "$(dirname "$MANIFEST")" "$CKPT_DIR" "$(dirname "$LOG")"

: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
log() { echo "[$(date '+%F %T')] $*"; }

log "=== V4 dte2_ladder_conservative full rebuild start ==="
log "git HEAD: $(git rev-parse --short HEAD)"
log "splits: train ${TRAIN_START}..${TRAIN_END} | val ${VAL_START}..${VAL_END} | test ${TEST_START}..${TEST_END}"
log "stock: $STOCK_ROOT | option_1s: $OPTION_1S | anchor: $ANCHOR_CFG"

log "=== [0/11] step3: dte2_ladder_conservative 1s → 1m (--force) ==="
"$PY" preprocess/download/step3_databento_aggregate_1s_to_1m.py \
  --input-dir "$OPTION_1S" \
  --output-dir "$OPTION_1M" \
  --symbol QQQ \
  --force

log "=== purge dte2_ladder_conservative pipeline outputs ==="
for sub in \
  "quote_features_raw_dte2_ladder_conservative/QQQ" \
  "quote_features_train_dte2_ladder_conservative/QQQ" \
  "quote_features_val_dte2_ladder_conservative/QQQ" \
  "quote_features_test_dte2_ladder_conservative/QQQ"; do
  rm -rf "$HOME/train_data/${sub}"
done
rm -rf "$EVAL_OUT" "$CKPT_DIR" \
  "$LMDB_ROOT/train_qqq_v4_dte2_ladder_conservative.lmdb" \
  "$LMDB_ROOT/val_qqq_v4_dte2_ladder_conservative.lmdb" \
  "$LMDB_ROOT/test_qqq_v4_dte2_ladder_conservative.lmdb"
mkdir -p "$CKPT_DIR"

log "=== [1/11] option_cac_day_vectorized_day ==="
"$PY" - <<PY
import multiprocessing
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

calc = OptionIVCalculator(
    db_path="/home/kingfang007/notebook/stocks.db",
    option_root="$OPTION_1M",
    data_root="$STOCK_ROOT",
    iv_option_root="$DAY_IV",
)
calc.run(max_concurrent_stocks=12)
PY

log "=== [2/11] iv_day2month ==="
"$PY" - <<PY
import glob, multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from preprocess.ask_bid.iv_day2month import process_single_symbol, get_target_symbols
from tqdm import tqdm

INPUT_BASE = "$DAY_IV"
OUTPUT_BASE = "$MONTHLY_IV"
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

log "=== [3/11] options_locked_feature ==="
"$PY" - <<PY
import concurrent.futures, logging
from pathlib import Path
from tqdm import tqdm
from preprocess.ask_bid.options_locked_feature import process_single_file

RAW_DIR = Path("$MONTHLY_IV")
OUTPUT_DIR = Path("$BUCKETED")
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

log "=== [4/11] feature_merge_option_raw ==="
"$PY" - <<PY
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

fm.OUTPUT_FEATURES_DIR = Path("$RAW_FEAT")
fm.OPTION_MONTHLY_DIR = Path("$MONTHLY_IV")
fm.AGG_OPTION_MONTHLY_DIR = Path("$BUCKETED")
fm.main()
PY

log "=== [5/11] split_raw_features ==="
"$PY" - <<PY
from functools import partial
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from preprocess.ask_bid.split_raw_features import (
    get_destination_path, process_and_copy_file, get_valid_symbols,
)

SOURCE = Path("$RAW_FEAT")
TRAIN = Path("$TRAIN_FEAT")
VAL = Path("$VAL_FEAT")
TEST = Path("$TEST_FEAT")
train_r = (pd.Timestamp("$TRAIN_START"), pd.Timestamp("$TRAIN_END"))
val_r = (pd.Timestamp("$VAL_START"), pd.Timestamp("$VAL_END"))
test_r = (pd.Timestamp("$TEST_START"), pd.Timestamp("$TEST_END"))

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
    list(tqdm(ex.map(worker, tasks), total=len(tasks), desc="split dte2_ladder_conservative"))
print(f"split done: {len(tasks)} files")
PY

log "=== [6/11] apply_rolling_norm ==="
"$PY" - <<PY
import concurrent.futures, logging
from pathlib import Path
from tqdm import tqdm
import preprocess.ask_bid.apply_rolling_norm_standalone as arn

norm_cols = arn.load_target_features(arn.CONFIG_PATH)
for stage in ["train_dte2_ladder_conservative", "val_dte2_ladder_conservative", "test_dte2_ladder_conservative"]:
    stage_root = Path.home() / f"train_data/quote_features_{stage}"
    print(f"Norm stage {stage} -> {stage_root}")
    if not stage_root.exists():
        raise SystemExit(f"missing {stage_root}")
    target_dirs = arn.find_leaf_directories(stage_root)
    tasks = [(d, norm_cols) for d in target_dirs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=arn.MAX_WORKERS) as ex:
        for res in tqdm(ex.map(arn.process_single_directory, tasks), total=len(tasks)):
            if res and res.startswith("ERROR"):
                logging.error(res)
    arn.verify_data_quality(stage_root, norm_cols)
print("rolling norm dte2_ladder_conservative done")
PY

log "=== [7/11] label_pipeline dynamic ladder ==="
for stage in train_dte2_ladder_conservative val_dte2_ladder_conservative test_dte2_ladder_conservative; do
  log "label_pipeline $stage"
  "$PY" qqq_btc/tools/label_pipeline.py \
    --input "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --output "$HOME/train_data/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
    --symbol QQQ \
    --anchor-config "$ANCHOR_CFG" \
    --entry-delay-seconds 60 \
    --report "/tmp/label_report_v4_dte2_ladder_conservative_${stage}.json"
done

log "=== label stats ==="
"$PY" - <<'PY'
import json
for stage in ("train_dte2_ladder_conservative", "val_dte2_ladder_conservative", "test_dte2_ladder_conservative"):
    rep = json.load(open(f"/tmp/label_report_v4_dte2_ladder_conservative_{stage}.json"))
    print(stage, "files", rep["files"], "avg_net_std", round(rep["avg_net_std"], 4))
PY

log "=== [8/11] build LMDB ==="
for stage in train val test; do
  "$PY" qqq_btc/tools/build_lmdb.py \
    --feature-root "$HOME/train_data/quote_features_${stage}_dte2_ladder_conservative" \
    --config "$FEATURE_CFG" \
    --symbol-map "$SYM_MAP" \
    --output "$LMDB_ROOT/${stage}_qqq_v4_dte2_ladder_conservative.lmdb" \
    --symbols QQQ \
    --window-step 1
done

log "=== [9/11] pretrain V4 dte2_ladder_conservative ==="
"$PY" -m qqq_btc.model.train \
  --mode pretrain \
  --config "$FEATURE_CFG" \
  --data-root "$LMDB_ROOT" \
  --train-lmdb train_qqq_v4_dte2_ladder_conservative.lmdb \
  --val-lmdbs val_qqq_v4_dte2_ladder_conservative.lmdb \
  --checkpoint-dir "$CKPT_DIR" \
  --epochs 6 \
  --batch-size 1024 \
  --num-workers 8 \
  --device cuda \
  2>&1 | tee "$CKPT_DIR/train.log"

log "=== [10/11] test eval + strict replay ==="
"$PY" qqq_btc/tools/eval_test_set.py \
  --checkpoint "$CKPT_DIR/best.pth" \
  --config "$FEATURE_CFG" \
  --feature-root "$TEST_FEAT" \
  --option-1m-root "$OPTION_1M" \
  --output-dir "$EVAL_OUT" \
  --device cuda

log "=== write manifest ==="
"$PY" - <<PY
import json, subprocess
from pathlib import Path
from datetime import datetime, timezone

def count_parquet(root):
    return len(list(Path(root).glob("**/*.parquet"))) if Path(root).exists() else 0

manifest = {
    "build_id": "v4_dte2_ladder_conservative",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "splits": {
        "train": "$TRAIN_START..$TRAIN_END",
        "val": "$VAL_START..$VAL_END",
        "test": "$TEST_START..$TEST_END",
    },
    "note": "strict 2DTE 8-bucket ladder polygon quotes; conservative entry-time value-score label; stock source unchanged",
    "paths": {
        "option_1s": "$OPTION_1S",
        "option_1m": "$OPTION_1M",
        "stock": "$STOCK_ROOT",
        "day_iv": "$DAY_IV",
        "monthly_iv": "$MONTHLY_IV",
        "bucketed": "$BUCKETED",
        "features_test": "$TEST_FEAT",
        "checkpoint": "$CKPT_DIR/best.pth",
        "eval_out": "$EVAL_OUT",
        "anchor": "$ANCHOR_CFG",
    },
    "counts": {
        "option_1m_qqq": count_parquet(Path("$OPTION_1M/QQQ")),
        "day_iv_qqq": count_parquet(Path("$DAY_IV/QQQ")),
        "test_1min": count_parquet(Path("$TEST_FEAT/QQQ/regular/09:30-16:00/1min")),
    },
    "log": "$LOG",
}
if Path("$EVAL_OUT/replay_summary.json").exists():
    manifest["replay_summary"] = json.loads(Path("$EVAL_OUT/replay_summary.json").read_text())
Path("$MANIFEST").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
PY

log "=== V4 dte2_ladder_conservative DONE ==="
log "checkpoint: $CKPT_DIR/best.pth"
log "manifest: $MANIFEST"
