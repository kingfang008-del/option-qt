#!/usr/bin/env bash
# 用 Databento 下载：旧锁约算法 + 真正 trading 0DTE
# 用法:
#   bash preprocess/download/run_step2_true_0dte_old_lock_databento.sh
#   bash preprocess/download/run_step2_true_0dte_old_lock_databento.sh --force
#   LIMIT_DAYS=3 bash preprocess/download/run_step2_true_0dte_old_lock_databento.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PY:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"

MAP="${MAP:-$HOME/train_data/locked_targets_map_old_style_true_0dte_clean.parquet}"
OUT="${OUT:-/mnt/s990/data/raw_1s/dte0_options_old_lock_databento}"
WORKERS="${WORKERS:-8}"
LOG="${LOG:-/tmp/step2_dte0_old_lock_databento.log}"
KEY_FILE="${KEY_FILE:-$HOME/api_key.txt}"

if [[ -z "${DATABENTO_API_KEY:-}" ]]; then
  if [[ ! -f "$KEY_FILE" ]]; then
    echo "缺少 API key: 设置 DATABENTO_API_KEY 或提供 $KEY_FILE" >&2
    exit 1
  fi
  DATABENTO_API_KEY="$(tr -d ' \n\r' < "$KEY_FILE")"
  export DATABENTO_API_KEY
fi

EXTRA=()
if [[ -n "${LIMIT_DAYS:-}" ]]; then
  EXTRA+=(--limit-days "$LIMIT_DAYS")
fi
if [[ -n "${DATE_FROM:-}" ]]; then
  EXTRA+=(--date-from "$DATE_FROM")
fi
if [[ -n "${DATE_TO:-}" ]]; then
  EXTRA+=(--date-to "$DATE_TO")
fi

mkdir -p "$OUT"
cd "$REPO"

echo "MAP=$MAP"
echo "OUT=$OUT"
echo "WORKERS=$WORKERS"
echo "LOG=$LOG"
echo "extra: ${EXTRA[*]:-} $*"

exec "$PY" -u preprocess/download/step2_databento_second_sniper_v1.py \
  --target-map "$MAP" \
  --output-dir "$OUT" \
  --api-key "$DATABENTO_API_KEY" \
  --max-workers "$WORKERS" \
  "${EXTRA[@]}" \
  "$@" \
  2>&1 | tee "$LOG"
