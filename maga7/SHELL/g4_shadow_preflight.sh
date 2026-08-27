#!/bin/bash
# G4 Shadow 盘前一键体检（不启 IB 会话；列出阻塞项）
# 用法: ./g4_shadow_preflight.sh [--with-day-check] [--date YYYY-MM-DD]

set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PROFILE="${MAG7_PROFILE:-maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json}"
DAY_CHECK=0
CHECK_DATE="${MAG7_HARDEN_DATE:-2026-07-17}"
REDIS_HOST="${MAG7_REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${MAG7_REDIS_PORT:-6379}"
REDIS_DB="${MAG7_REDIS_DB:-0}"
PY="${MAG7_PYTHON:-$HOME/anaconda3/envs/ibkr/bin/python}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || command -v python)"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --with-day-check) DAY_CHECK=1; shift ;;
    --date) CHECK_DATE="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: $(basename "$0") [--with-day-check] [--date YYYY-MM-DD]"
      exit 0
      ;;
    *) echo "unknown arg: $1"; exit 2 ;;
  esac
done

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
ok() { echo -e "${GREEN}PASS${NC}  $*"; }
warn() { echo -e "${YELLOW}WARN${NC}  $*"; }
fail() { echo -e "${RED}FAIL${NC}  $*"; }
info() { echo -e "${CYAN}----${NC} $*"; }

BLOCKERS=0
WARNINGS=0

info "G4 Shadow preflight  $(date -Is)  NY=$(TZ=America/New_York date '+%Y-%m-%d %H:%M')"
info "profile=$PROFILE"

# 1) profile exists
if [[ -f "$PROFILE" ]]; then
  ok "profile exists"
else
  fail "profile missing: $PROFILE"
  BLOCKERS=$((BLOCKERS + 1))
fi

# 2) stock 1s coverage
python - <<'PY' || true
import sys
from pathlib import Path
syms = ["NVDA","TSLA","AAPL","AMZN","META","MSFT","AMD","GOOGL","QQQ"]
root = Path("/mnt/s990/data/raw_1s/stocks")
last = {}
miss = []
for s in syms:
    files = sorted(root.joinpath(s).glob(f"{s}_*.parquet"))
    if not files:
        miss.append(s); continue
    last[s] = files[-1].stem.split("_", 1)[1]
print("STOCK_LAST", min(last.values()) if last else "NONE", "→", max(last.values()) if last else "NONE")
if miss:
    print("STOCK_MISS", ",".join(miss))
    sys.exit(2)
# warn if last < today-ish calendar (trading) — soft
from datetime import date
end = max(last.values())
print("STOCK_END", end)
if end < "2026-07-17":
    sys.exit(3)
PY
st=$?
if [[ $st -eq 0 ]]; then
  ok "stock 1s present (all Mag7+QQQ); end=$(python - <<'PY'
from pathlib import Path
files=sorted(Path('/mnt/s990/data/raw_1s/stocks/NVDA').glob('NVDA_*.parquet'))
print(files[-1].stem.split('_',1)[1] if files else '?')
PY
)"
  # soft warn if stale vs wall clock NY
  NY_DATE=$(TZ=America/New_York date +%F)
  STOCK_END=$(python - <<'PY'
from pathlib import Path
files=sorted(Path('/mnt/s990/data/raw_1s/stocks/NVDA').glob('NVDA_*.parquet'))
print(files[-1].stem.split('_',1)[1] if files else '')
PY
)
  if [[ -n "$STOCK_END" && "$STOCK_END" < "$NY_DATE" ]]; then
    warn "stock 1s ends $STOCK_END < NY today $NY_DATE — backfill before relying on offline/day_stream past that date"
    WARNINGS=$((WARNINGS + 1))
  fi
else
  fail "stock 1s incomplete"
  BLOCKERS=$((BLOCKERS + 1))
fi

# 3) lock map + quote root
python - <<PY
import json, sys
from pathlib import Path
prof = json.loads(Path("$PROFILE").read_text())
paths = prof.get("paths") or {}
lm = Path(paths.get("open_locked_map") or "").expanduser()
qr = Path(paths.get("quote_1s_root") or "").expanduser()
ok = True
if not lm.is_file():
    print("LOCK_MISS", lm); ok = False
else:
    print("LOCK_OK", lm)
if not qr.is_dir():
    print("QUOTE_MISS", qr); ok = False
else:
    print("QUOTE_OK", qr)
sys.exit(0 if ok else 2)
PY
if [[ $? -eq 0 ]]; then
  ok "open_locked_map + quote_1s_root"
else
  fail "lock map or quote root missing"
  BLOCKERS=$((BLOCKERS + 1))
fi

# 4) event calendar
CAL="maga7/CONFIG/event_calendar_live.json"
if [[ -f "$CAL" ]]; then
  ok "event_calendar_live.json present"
else
  warn "event_calendar_live.json missing — run: ./start_maga7_live_session.sh sync-calendar"
  WARNINGS=$((WARNINGS + 1))
fi

# 5) Redis
if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" ping 2>/dev/null | grep -q PONG; then
  ok "Redis $REDIS_HOST:$REDIS_PORT db=$REDIS_DB"
else
  fail "Redis not reachable ($REDIS_HOST:$REDIS_PORT db=$REDIS_DB)"
  BLOCKERS=$((BLOCKERS + 1))
fi

# 6) IB Gateway ports (shadow dry on :4002 or :4001 live-MD)
IB_OK=0
IB_SEEN=""
for p in 4001 4002 7497 7496; do
  if ss -ltn 2>/dev/null | grep -q ":$p "; then
    IB_SEEN="${IB_SEEN:+$IB_SEEN,}$p"
    IB_OK=1
  fi
done
if [[ $IB_OK -eq 1 ]]; then
  ok "IB listener on :$IB_SEEN"
  if echo "$IB_SEEN" | grep -q '4001'; then
    info "tip: use './start_maga7_live_session.sh start dry' for live-MD :4001 + dry OMS"
  fi
  SMOKE_PORT="${MAG7_IB_PORT:-}"
  if [[ -z "$SMOKE_PORT" ]]; then
    SMOKE_PORT="$(echo "$IB_SEEN" | tr ',' '\n' | awk '$1 == 4001 {print; exit} $1 == 4002 {fallback=$1} END {if (!NR && fallback) print fallback}')"
    [[ -n "$SMOKE_PORT" ]] || SMOKE_PORT="${IB_SEEN%%,*}"
  fi
  info "IB read-only stock+option smoke on :$SMOKE_PORT…"
  if "$PY" -m maga7.tools.ibkr_readonly_smoke \
      --host "${MAG7_IB_HOST:-127.0.0.1}" \
      --port "$SMOKE_PORT" \
      --client-id "${MAG7_PREFLIGHT_CLIENT_ID:-912}" \
      >/tmp/mag7_ibkr_smoke.log 2>&1; then
    ok "$(cat /tmp/mag7_ibkr_smoke.log)"
  else
    fail "IBKR account/LIVE stock/option smoke failed — see /tmp/mag7_ibkr_smoke.log"
    BLOCKERS=$((BLOCKERS + 1))
  fi
  info "Mag7 lock/subscription preflight on :$SMOKE_PORT…"
  LOCK_PRE_ARGS=(
    --host "${MAG7_IB_HOST:-127.0.0.1}"
    --port "$SMOKE_PORT"
    --client-id "${MAG7_LOCK_PREFLIGHT_CLIENT_ID:-913}"
    --profile "$PROFILE"
    --json-out /tmp/mag7_mag7_lock_preflight.json
  )
  if [[ "${MAG7_REQUIRE_OPTION_QUOTES:-0}" == "1" ]]; then
    LOCK_PRE_ARGS+=(--require-quotes)
  fi
  if "$PY" -m maga7.tools.ibkr_mag7_lock_preflight \
      "${LOCK_PRE_ARGS[@]}" \
      >/tmp/mag7_mag7_lock_preflight.log 2>&1; then
    ok "Mag7 lock preflight — $(rg -N 'MAG7_LOCK_PREFLIGHT_OK' /tmp/mag7_mag7_lock_preflight.log | tail -1)"
    if rg -q 'nearest_fallback|ticker_alive_no_nbbo|awaiting_nbbo|NO_NBBO' /tmp/mag7_mag7_lock_preflight.log; then
      warn "nearest-DTE fallback and/or quote diagnosis pending — see /tmp/mag7_mag7_lock_preflight.log"
      WARNINGS=$((WARNINGS + 1))
    fi
  else
    fail "Mag7 lock/subscription preflight failed — see /tmp/mag7_mag7_lock_preflight.log"
    BLOCKERS=$((BLOCKERS + 1))
  fi
else
  fail "IB Gateway/TWS API not listening (dry→:4001 or shadow→:4002)"
  BLOCKERS=$((BLOCKERS + 1))
fi

# 7) live session not already stuck
if [[ -f "$PROJECT_ROOT/logs/maga7/live_session.pid" ]]; then
  pid=$(cat "$PROJECT_ROOT/logs/maga7/live_session.pid" 2>/dev/null || true)
  if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
    warn "live_session already running pid=$pid — status/stop before new Shadow"
    WARNINGS=$((WARNINGS + 1))
  else
    ok "no active live_session pid"
  fi
else
  ok "no live_session.pid"
fi

# 8) prior Shadow evidence
LS="${MAG7_LIVE_SESSIONS_DIR:-/mnt/s990/data/maga7/live_sessions}"
if [[ -d "$LS" ]] && [[ -n "$(ls -A "$LS" 2>/dev/null || true)" ]]; then
  ok "prior live_sessions artifacts exist under $LS"
else
  warn "no G4 evidence yet under $LS (expected until first Shadow day)"
  WARNINGS=$((WARNINGS + 1))
fi

# 9) fault tests
info "running faults-only hardening…"
if "$SHELL_DIR/run_premarket_hardening.sh" faults-only >/tmp/mag7_preflight_faults.log 2>&1; then
  ok "fault injection + risk_guards"
else
  fail "faults-only failed — see /tmp/mag7_preflight_faults.log"
  BLOCKERS=$((BLOCKERS + 1))
fi

info "running complete live safety regression…"
if PYTHONPATH="$PROJECT_ROOT" "$PY" -m pytest -q \
    maga7/tests/test_live_safety.py \
    maga7/tests/test_live_fault_injection.py \
    maga7/tests/test_risk_guards.py \
    maga7/tests/test_requote.py \
    maga7/tests/test_iceberg.py \
    maga7/tests/test_oms_hold_watchdog.py \
    maga7/tests/test_oms_trade_toxic.py \
    >/tmp/mag7_preflight_live_tests.log 2>&1; then
  ok "live safety regression"
else
  fail "live safety regression failed — see /tmp/mag7_preflight_live_tests.log"
  BLOCKERS=$((BLOCKERS + 1))
fi

# 10) optional day stream
if [[ $DAY_CHECK -eq 1 ]]; then
  info "day_stream_check $CHECK_DATE…"
  if "$SHELL_DIR/run_day_stream_check.sh" "$CHECK_DATE" --force-local >/tmp/mag7_preflight_day.log 2>&1; then
    ok "day_stream_check $CHECK_DATE"
  else
    # script may return non-zero; inspect json
    if rg -q '"ok"\s*:\s*true' /tmp/mag7_preflight_day.log 2>/dev/null; then
      ok "day_stream_check $CHECK_DATE (ok in log)"
    else
      fail "day_stream_check $CHECK_DATE — see /tmp/mag7_preflight_day.log"
      BLOCKERS=$((BLOCKERS + 1))
    fi
  fi
fi

echo
info "summary: blockers=$BLOCKERS warnings=$WARNINGS"
if [[ $BLOCKERS -eq 0 ]]; then
  echo -e "${GREEN}READY for G4 Shadow / dry${NC}:"
  echo "  cd maga7/SHELL"
  echo "  ./start_maga7_live_session.sh sync-calendar"
  echo "  ./start_maga7_live_session.sh start dry     # :4001 live MD + dry OMS (推荐)"
  echo "  # 或: ./start_maga7_live_session.sh start shadow --ib-port 4001"
  echo "  # 或: ./start_maga7_live_session.sh start shadow   # :4002 Paper Gateway"
  echo "  ./start_maga7_live_session.sh status"
  echo "  python dash/run.py               # Live board"
  exit 0
fi
echo -e "${RED}NOT READY${NC} — clear blockers before start shadow"
exit 1
