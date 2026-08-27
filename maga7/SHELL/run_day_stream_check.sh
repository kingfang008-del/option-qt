#!/bin/bash
# 流式对拍：历史 1m 特征轨（含 lookback，对齐 offline）→ trade_log → 对 offline。
# 成交仍用 1s quote；默认按日跑 7 月（2026-07-01..2026-07-17）。
# 单日：./run_day_stream_check.sh 2026-07-02 --force-local
# 整窗单进程（易串日）：./run_day_stream_check.sh --range --force-local
set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PY="${MAG7_PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "$HOME/anaconda3/envs/ibkr/bin/python" ]]; then
        PY="$HOME/anaconda3/envs/ibkr/bin/python"
    else
        PY="python3"
    fi
fi

PROFILE="${MAG7_PROFILE:-maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json}"
STOCK_1S="${MAG7_STOCK_1S:-/mnt/s990/data/raw_1s/stocks}"
JULY_START="${MAG7_HARDEN_DATE:-2026-07-01}"
JULY_END="${MAG7_HARDEN_END:-2026-07-17}"

run_one() {
  local start="$1" end="$2"
  shift 2
  "$PY" -m maga7.tools.run_day_stream_check \
    --profile "$PROFILE" \
    --date "$start" \
    --end-date "$end" \
    "$@"
}

if [[ "${1:-}" == "--range" ]]; then
  shift
  run_one "$JULY_START" "$JULY_END" "$@"
  exit $?
fi

# Single / explicit range if first arg is a date
if [[ $# -ge 1 && "${1:-}" != -* ]]; then
  START="$1"
  shift
  END="$START"
  if [[ $# -ge 1 && "${1:-}" != -* ]]; then
    END="$1"
    shift
  fi
  run_one "$START" "$END" "$@"
  exit $?
fi

# Default: per-day July (discover from 1s disk)
SYM_PROBE="${MAG7_PROBE_SYMBOL:-MSFT}"
mapfile -t DAYS < <(
  ls "$STOCK_1S/$SYM_PROBE" 2>/dev/null \
    | grep -oE '2026-07-[0-9]{2}' \
    | sort -u \
    | awk -v a="$JULY_START" -v b="$JULY_END" '$0>=a && $0<=b'
)
if [[ ${#DAYS[@]} -eq 0 ]]; then
  echo "No July 1s days under $STOCK_1S/$SYM_PROBE ($JULY_START..$JULY_END)" >&2
  exit 1
fi

echo "==> July day-stream check (${#DAYS[@]} days) $JULY_START..$JULY_END $*"
pass=0
fail=0
crash=0
# set +e so one FAIL day does not abort the loop
set +e
for d in "${DAYS[@]}"; do
  echo "----- $d -----"
  out="$(run_one "$d" "$d" "$@" 2>&1)"
  rc=$?
  echo "$out" | tail -n 8
  if echo "$out" | grep -q '"ok": true'; then
    pass=$((pass + 1))
    echo "PASS $d"
  elif echo "$out" | grep -q '"ok": false'; then
    fail=$((fail + 1))
    echo "FAIL $d"
  else
    crash=$((crash + 1))
    echo "CRASH $d rc=$rc"
  fi
done
set -e

echo "==> July summary: pass=$pass fail=$fail crash=$crash / ${#DAYS[@]}"
if [[ "$fail" -gt 0 || "$crash" -gt 0 ]]; then
  exit 2
fi
exit 0
