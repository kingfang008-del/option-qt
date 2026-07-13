#!/usr/bin/env bash
# 下载旧锁约算法 + 真正 trading 0DTE 的 Polygon 1s quote
# 用法:
#   bash preprocess/download/run_step2_true_0dte_old_lock.sh
#   bash preprocess/download/run_step2_true_0dte_old_lock.sh --force
#   MAP=... OUT=... WORKERS=20 bash preprocess/download/run_step2_true_0dte_old_lock.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"

# 默认用 clean map（已剔除缺 CALL / 非当日到期的天）
MAP="${MAP:-$HOME/train_data/locked_targets_map_old_style_true_0dte_clean.parquet}"
OUT="${OUT:-/mnt/s990/data/raw_1s/dte0_options_old_lock}"
WORKERS="${WORKERS:-30}"
LOG="${LOG:-/tmp/step2_dte0_old_lock.log}"

# 正股一般已有缓存；若缺再去掉 --no-download-stock
EXTRA_ARGS=(--no-download-stock)
if [[ "${DOWNLOAD_STOCK:-0}" == "1" ]]; then
  EXTRA_ARGS=()
fi

mkdir -p "$OUT"
cd "$REPO"

echo "MAP=$MAP"
echo "OUT=$OUT"
echo "WORKERS=$WORKERS"
echo "LOG=$LOG"
echo "extra: ${EXTRA_ARGS[*]:-} $*"

exec "$PY" -u preprocess/download/step2_polygon_second_sniper_v1.py \
  --target-map "$MAP" \
  --output-dir "$OUT" \
  --max-workers "$WORKERS" \
  "${EXTRA_ARGS[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
