#!/bin/bash
# Mag7 盘前加固：故障注入 + OMS dry（可选 S5 / stream parity）
# 用法见 menu.md 或: ./run_premarket_hardening.sh help

set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
LOG_DIR="${MAG7_LOG_DIR:-$PROJECT_ROOT/logs/maga7}"
mkdir -p "$LOG_DIR"

DEFAULT_PROFILE="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
PROFILE="${MAG7_PROFILE:-$DEFAULT_PROFILE}"
DATE="${MAG7_HARDEN_DATE:-2026-05-28}"
END_DATE="${MAG7_HARDEN_END:-}"
SCHEME="${MAG7_SCHEME:-}"
WITH_S5=0
WITH_PARITY=0
FAULTS_ONLY=0
SKIP_DRY=0
ALLOW_DRY_MISMATCH=0
EXTRA=()

usage() {
    cat <<EOF
Mag7 premarket hardening (avoid burning a live day)

Usage:
  $(basename "$0") [options]
  $(basename "$0") faults-only
  $(basename "$0") help

Default stages:
  1) pytest fault injection + risk_guards
  2) OMS dry-run --compare-offline (freeze profile, 1s ingest)

Optional:
  --with-s5       Redis S5 fused sim (needs Redis on MAG7_REDIS_*)
  --with-parity   G2 stream parity on the same window
  --date YYYY-MM-DD          default 2026-05-28 (dry↔offline golden)
  --end-date YYYY-MM-DD      default = --date
  --profile PATH
  --scheme NAME              default: profile recommended_scheme (single)
  --faults-only              only unit fault tests
  --skip-dry
  --allow-dry-mismatch       do not fail on dry↔offline trade diffs

Examples:
  $(basename "$0")
  $(basename "$0") --date 2026-05-28 --with-parity
  $(basename "$0") --date 2026-05-28 --with-s5
  $(basename "$0") faults-only

Env:
  MAG7_PROFILE  MAG7_HARDEN_DATE  MAG7_SCHEME
  MAG7_REDIS_HOST  MAG7_REDIS_PORT  MAG7_REDIS_DB (S5 default db=1)
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        help|-h|--help) usage; exit 0 ;;
        faults-only|--faults-only) FAULTS_ONLY=1; shift ;;
        --with-s5) WITH_S5=1; shift ;;
        --with-parity) WITH_PARITY=1; shift ;;
        --skip-dry) SKIP_DRY=1; shift ;;
        --allow-dry-mismatch) ALLOW_DRY_MISMATCH=1; shift ;;
        --date) DATE="$2"; shift 2 ;;
        --end-date) END_DATE="$2"; shift 2 ;;
        --profile) PROFILE="$2"; shift 2 ;;
        --scheme) SCHEME="$2"; shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Prefer project conda env if present
PY="${MAG7_PYTHON:-}"
if [[ -z "$PY" ]]; then
    if [[ -x "$HOME/anaconda3/envs/ibkr/bin/python" ]]; then
        PY="$HOME/anaconda3/envs/ibkr/bin/python"
    else
        PY="python3"
    fi
fi

CMD=(
    "$PY" -m maga7.tools.run_premarket_hardening
    --profile "$PROFILE"
    --date "$DATE"
)

if [[ -n "$END_DATE" ]]; then
    CMD+=(--end-date "$END_DATE")
fi
if [[ -n "$SCHEME" ]]; then
    CMD+=(--scheme "$SCHEME")
fi
if [[ "$FAULTS_ONLY" -eq 1 ]]; then
    CMD+=(--faults-only)
fi
if [[ "$SKIP_DRY" -eq 1 ]]; then
    CMD+=(--skip-dry)
fi
if [[ "$ALLOW_DRY_MISMATCH" -eq 1 ]]; then
    CMD+=(--allow-dry-mismatch)
fi
if [[ "$WITH_S5" -eq 1 ]]; then
    CMD+=(--with-s5)
fi
if [[ "$WITH_PARITY" -eq 1 ]]; then
    CMD+=(--with-parity)
fi
if [[ ${#EXTRA[@]} -gt 0 ]]; then
    CMD+=("${EXTRA[@]}")
fi

LOG_FILE="$LOG_DIR/premarket_hardening.log"
echo "[$(date -Is)] ${CMD[*]}" | tee -a "$LOG_FILE"
set +e
"${CMD[@]}" 2>&1 | tee -a "$LOG_FILE"
rc=${PIPESTATUS[0]}
set -e
exit "$rc"
