#!/usr/bin/env bash
# 真正 trading 0DTE（旧锁约算法 + Databento 1s）→ 按 V4 同款逻辑重建特征并重训
# 不依赖 standard_old_v2 / 1DTE / bak 特征；全新隔离目录。
#
# 切分: train 2023-04~2025-12 | val 2026-01~03 | test 2026-04~06
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
export PYTHONPATH="${REPO}${PYTHONPATH:+:$PYTHONPATH}"
export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/tmp/numba_cache}"
export FEATURE_CONFIG="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v2.json"

RAW_1S="${RAW_1S:-/mnt/s990/data/raw_1s/dte0_options_old_lock_databento}"
RAW_1M="${RAW_1M:-/mnt/s990/data/raw_1m/dte0_options_old_lock_databento}"
OUT_ROOT="${OUT_ROOT:-$HOME/train_data/builds/v4_true_0dte_old_lock}"
LMDB_ROOT="${LMDB_ROOT:-$HOME/train_data/lmdb}"
CKPT_DIR="${CKPT_DIR:-$REPO/checkpoint/checkpoints_qqq_v4_true_0dte_old_lock}"
FEATURE_CFG_TRAIN="$REPO/qqq_btc/CONFIG/slow_feature_qqq_v4.json"
ANCHOR_CFG="$REPO/qqq_btc/CONFIG/anchor_qqq_true_0dte_old_lock.json"
SYM_MAP="$REPO/qqq_btc/CONFIG/symbol_map.json"
STOCK_DB="${STOCK_DB:-/home/kingfang007/notebook/stocks.db}"
STOCK_ROOT="${STOCK_ROOT:-$HOME/train_data/spnq_train_resampled}"
LOG="$REPO/qqq_btc/results/train_v4_true_0dte_old_lock.log"
EVAL_VAL="$REPO/qqq_btc/results/v4_true_0dte_old_lock_val"
EVAL_TEST="$REPO/qqq_btc/results/v4_true_0dte_old_lock_test"

SKIP_AGG="${SKIP_AGG:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
START_STEP="${START_STEP:-1}"

mkdir -p "$OUT_ROOT" "$CKPT_DIR" "$LMDB_ROOT" "$(dirname "$LOG")" "$RAW_1M"
exec > >(tee -a "$LOG") 2>&1
cd "$REPO"

echo "=== OUT_ROOT=$OUT_ROOT ==="
echo "=== RAW_1S=$RAW_1S ==="
echo "=== RAW_1M=$RAW_1M ==="
echo "=== CKPT_DIR=$CKPT_DIR ==="
echo "=== START_STEP=$START_STEP $(date -Is) ==="

run_step() {
  local n="$1"
  shift
  if (( START_STEP > n )); then
    echo "=== skip step $n ==="
    return 0
  fi
  echo "=== [$n] $* ==="
}

# ---------- 1) 1s → 1m ----------
run_step 1 "1s→1m aggregate"
if (( START_STEP <= 1 )) && [[ "$SKIP_AGG" != "1" ]]; then
  "$PY" "$REPO/preprocess/download/step3_databento_aggregate_1s_to_1m.py" \
    --input-dir "$RAW_1S" \
    --output-dir "$RAW_1M" \
    --symbol QQQ \
    --max-workers 24
  echo "1m files=$(ls "$RAW_1M/QQQ" 2>/dev/null | wc -l)"
fi

# ---------- 2) day IV ----------
run_step 2 "option_cac_day_vectorized_day → day_iv"
if (( START_STEP <= 2 )); then
  DAY_IV="$OUT_ROOT/quote_options_day_iv"
  mkdir -p "$DAY_IV"
  "$PY" - <<PY
import multiprocessing
from pathlib import Path
from preprocess.ask_bid.option_cac_day_vectorized_day import OptionIVCalculator

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

calc = OptionIVCalculator(
    db_path="$STOCK_DB",
    option_root="$RAW_1M",
    data_root="$STOCK_ROOT",
    iv_option_root="$DAY_IV",
)
calc.run(max_concurrent_stocks=12)
n = len(list(Path("$DAY_IV/QQQ/standard").glob("QQQ_*.parquet")))
print("day_iv files", n)
PY
fi

# ---------- 3) monthly ----------
run_step 3 "iv_day2month"
if (( START_STEP <= 3 )); then
  MONTHLY="$OUT_ROOT/quote_options_monthly_iv"
  mkdir -p "$MONTHLY"
  "$PY" - <<PY
import glob, multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from preprocess.ask_bid.iv_day2month import process_single_symbol
from tqdm import tqdm

INPUT_BASE = "$OUT_ROOT/quote_options_day_iv"
OUTPUT_BASE = "$MONTHLY"
try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

all_files = glob.glob(f"{INPUT_BASE}/**/*.parquet", recursive=True)
symbol_to_files = defaultdict(list)
for f in all_files:
    name = Path(f).stem
    sym = name.rsplit("_", 1)[0]
    if sym == "QQQ":
        symbol_to_files[sym].append(f)
tasks = [(sym, files, OUTPUT_BASE) for sym, files in symbol_to_files.items()]
print("monthly tasks", [(s, len(fs)) for s, fs, _ in tasks])
with ProcessPoolExecutor(max_workers=8) as pool:
    futs = {pool.submit(process_single_symbol, t): t for t in tasks}
    for fut in tqdm(as_completed(futs), total=len(futs)):
        print(fut.result())
print("monthly files", len(list(Path(OUTPUT_BASE).rglob("*.parquet"))))
PY
fi

# ---------- 4) bucketed ----------
run_step 4 "options_locked_feature"
if (( START_STEP <= 4 )); then
  BUCKETED="$OUT_ROOT/quote_options_bucketed_v7"
  mkdir -p "$BUCKETED"
  "$PY" - <<PY
import concurrent.futures, logging
from pathlib import Path
from tqdm import tqdm
from preprocess.ask_bid.options_locked_feature import process_single_file

RAW_DIR = Path("$OUT_ROOT/quote_options_monthly_iv")
OUTPUT_DIR = Path("$BUCKETED")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tasks = []
src = RAW_DIR / "QQQ" / "standard"
if not src.exists():
    raise SystemExit(f"missing {src}")
for p in src.glob("*.parquet"):
    tasks.append((p, OUTPUT_DIR, "QQQ"))
print("bucketed tasks", len(tasks))
with concurrent.futures.ProcessPoolExecutor(max_workers=16) as ex:
    futs = {ex.submit(process_single_file, t): t for t in tasks}
    for f in tqdm(concurrent.futures.as_completed(futs), total=len(futs)):
        r = f.result()
        if r:
            logging.warning(r)
print("bucketed months", len(list((OUTPUT_DIR / "QQQ").glob("*.parquet"))))
PY
fi

# ---------- 5) feature merge ----------
run_step 5 "feature_merge_option_raw"
if (( START_STEP <= 5 )); then
  "$PY" - <<PY
import json
from pathlib import Path
import preprocess.ask_bid.feature_merge_option_raw as fm

out_root = Path("$OUT_ROOT")
cfg_path = Path("$FEATURE_CONFIG")
with open(cfg_path, encoding="utf-8") as f:
    config = json.load(f)

fm.OPTION_MONTHLY_DIR = out_root / "quote_options_monthly_iv"
fm.AGG_OPTION_MONTHLY_DIR = out_root / "quote_options_bucketed_v7"
fm.OUTPUT_FEATURES_DIR = out_root / "quote_features_raw"
fm.CONFIG_FILE = str(cfg_path)
fm.OVERWRITE_EXISTING = True
fm.OUTPUT_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

months = sorted({p.stem for p in (fm.OPTION_MONTHLY_DIR / "QQQ" / "standard").glob("*.parquet")})
print("feature_merge months", len(months), months[0] if months else None, "->", months[-1] if months else None)
for mo in months:
    msg = fm.process_stock_month("QQQ", mo, config)
    print(mo, msg)

for fn in (
    "generate_vix_level_global",
    "update_vol_vix_abs",
    "update_cat_features_in_files",
    "update_new_labels_in_files",
):
    try:
        getattr(fm, fn)(config)
        print("ok", fn)
    except Exception as e:
        print("warn", fn, e)

print("raw feature files", len(list(fm.OUTPUT_FEATURES_DIR.rglob("*.parquet"))))
PY
fi

# ---------- 6+7) split + rolling norm ----------
run_step 6 "split + rolling_norm"
if (( START_STEP <= 6 )); then
  "$PY" - <<PY
import os, shutil
from pathlib import Path
import pandas as pd
import preprocess.ask_bid.apply_rolling_norm_standalone as norm

out_root = Path("$OUT_ROOT")
symbol = "QQQ"
raw_root = out_root / "quote_features_raw" / symbol
if not raw_root.exists():
    raise SystemExit(f"missing raw features: {raw_root}")

train_range = (pd.Timestamp("2023-04-01"), pd.Timestamp("2025-12-31"))
val_range = (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31"))
test_range = (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-30"))

stage_roots = {
    "train": out_root / "quote_features_train" / symbol,
    "val": out_root / "quote_features_val" / symbol,
    "test": out_root / "quote_features_test" / symbol,
}
for d in stage_roots.values():
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

copied = {"train": 0, "val": 0, "test": 0, "skip": 0}
for fp in sorted(raw_root.rglob("*.parquet")):
    try:
        ym = pd.Timestamp(fp.stem + "-01")
    except Exception:
        copied["skip"] += 1
        continue
    if train_range[0] <= ym <= train_range[1]:
        stage = "train"
    elif val_range[0] <= ym <= val_range[1]:
        stage = "val"
    elif test_range[0] <= ym <= test_range[1]:
        stage = "test"
    else:
        copied["skip"] += 1
        continue
    rel = fp.relative_to(raw_root)
    dst = stage_roots[stage] / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(fp, dst)
    copied[stage] += 1
print("copied", copied)

os.environ["FEATURE_CONFIG"] = str(Path("$FEATURE_CONFIG").resolve())
norm.CONFIG_PATH = Path(os.environ["FEATURE_CONFIG"])
targets = norm.load_target_features(norm.CONFIG_PATH)
for stage, root in stage_roots.items():
    for leaf in norm.find_leaf_directories(root):
        msg = norm.process_single_directory((leaf, targets))
        print("norm", stage, leaf, msg, "n", len(list(leaf.glob("*.parquet"))))
print("split+norm done")
PY
fi

# ---------- 7) labels ----------
run_step 7 "label_pipeline"
if (( START_STEP <= 7 )); then
  for stage in train val test; do
    "$PY" "$REPO/qqq_btc/tools/label_pipeline.py" \
      --input "$OUT_ROOT/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
      --output "$OUT_ROOT/quote_features_${stage}/QQQ/regular/09:30-16:00/1min" \
      --symbol QQQ \
      --anchor-config "$ANCHOR_CFG" \
      --report "$OUT_ROOT/label_report_${stage}.json"
  done
fi

# ---------- 8) LMDB ----------
run_step 8 "build LMDB"
if (( START_STEP <= 8 )); then
  for stage in train val test; do
    out="$LMDB_ROOT/${stage}_qqq_v4_true_0dte_old_lock.lmdb"
    rm -rf "$out"
    "$PY" "$REPO/qqq_btc/tools/build_lmdb.py" \
      --feature-root "$OUT_ROOT/quote_features_${stage}" \
      --config "$FEATURE_CFG_TRAIN" \
      --symbol-map "$SYM_MAP" \
      --output "$out" \
      --symbols QQQ \
      --window-step 1
  done
fi

# ---------- 9) train ----------
run_step 9 "pretrain V4"
if (( START_STEP <= 9 )) && [[ "$SKIP_TRAIN" != "1" ]]; then
  "$PY" -m qqq_btc.model.train \
    --mode pretrain \
    --config "$FEATURE_CFG_TRAIN" \
    --data-root "$LMDB_ROOT" \
    --train-lmdb train_qqq_v4_true_0dte_old_lock.lmdb \
    --val-lmdbs val_qqq_v4_true_0dte_old_lock.lmdb \
    --checkpoint-dir "$CKPT_DIR" \
    --epochs 20 \
    --batch-size 1024 \
    --num-workers 8 \
    --device cuda

  "$PY" - <<PY
import json, torch
from pathlib import Path
st = torch.load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
out = Path("$OUT_ROOT/slow_feature_from_ckpt.json")
out.write_text(json.dumps(st["config"], indent=2))
print("wrote", out, "best_ic", st.get("best_ic"), "epoch", st.get("epoch"))
PY

  "$PY" "$REPO/qqq_btc/tools/eval_test_set.py" \
    --checkpoint "$CKPT_DIR/best.pth" \
    --config "$OUT_ROOT/slow_feature_from_ckpt.json" \
    --feature-root "$OUT_ROOT/quote_features_val" \
    --option-1m-root "$RAW_1M" \
    --strategy-config qqq_btc.qqq.config_true_0dte \
    --output-dir "$EVAL_VAL" \
    --device cuda

  "$PY" "$REPO/qqq_btc/tools/eval_test_set.py" \
    --checkpoint "$CKPT_DIR/best.pth" \
    --config "$OUT_ROOT/slow_feature_from_ckpt.json" \
    --feature-root "$OUT_ROOT/quote_features_test" \
    --option-1m-root "$RAW_1M" \
    --strategy-config qqq_btc.qqq.config_true_0dte \
    --output-dir "$EVAL_TEST" \
    --device cuda

  "$PY" - <<PY
import json
from pathlib import Path
ck = __import__("torch").load("$CKPT_DIR/best.pth", map_location="cpu", weights_only=False)
print("train best_ic", ck.get("best_ic"), "epoch", ck.get("epoch"))
for name in ["$EVAL_VAL", "$EVAL_TEST"]:
    p = Path(name) / "replay_summary.json"
    if not p.exists():
        print(name, "MISSING"); continue
    s = json.load(open(p))
    print(Path(name).name,
          "ic=", s.get("label_metrics", {}).get("ic"),
          "acct25=", s.get("total_net_return"),
          "trades=", s.get("trades"))
PY
fi

echo "DONE $(date -Is)"
