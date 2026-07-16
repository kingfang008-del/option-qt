#!/usr/bin/env bash
# Mag7 open-lock 锁约 → seed → miss → 1s 下载 → merge
# 加标的示例:
#   bash maga7/tools/prepare_open_lock_quotes.sh --add-symbols GOOGL,GOOG
#   bash maga7/tools/prepare_open_lock_quotes.sh --symbols NVDA,TSLA,AAPL,AMZN,META,MSFT,AMD,GOOGL
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="${PYTHON:-/home/kingfang007/anaconda3/envs/ibkr/bin/python}"
PROFILE="${PROFILE:-$ROOT/maga7/CONFIG/strategy_profiles/m5c_qqq_onlywin_open_lock_research_v1.json}"
STEP="${STEP:-all}"
WORKERS="${MAX_WORKERS:-12}"
CONTRACT_WORKERS="${CONTRACT_WORKERS:-4}"

if [[ -z "${MASSIVE_API_KEY:-}" && -z "${POLYGON_API_KEY:-}" ]]; then
  # status/lock/seed/miss/merge 不需要 key；quotes/all 才需要
  if [[ "$STEP" == "quotes" || "$STEP" == "all" ]]; then
    echo "set MASSIVE_API_KEY or POLYGON_API_KEY" >&2
    exit 1
  fi
fi

export PYTHONPATH="${ROOT}:${ROOT}/preprocess/download:${PYTHONPATH:-}"
exec "$PY" -u -m maga7.tools.prepare_open_lock_quotes \
  --profile "$PROFILE" \
  --step "$STEP" \
  --max-workers "$WORKERS" \
  --contract-workers "$CONTRACT_WORKERS" \
  "$@"
