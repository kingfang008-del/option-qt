#!/bin/bash
# Mag7 live session 一键启动（G4 Shadow / G5 Paper / G6 Live）
# 位置: maga7/SHELL/（不要放进 production/，那是旧截面排序备用栈）
# 用法见同目录 menu.md 或: ./start_maga7_live_session.sh help

set -euo pipefail

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# maga7/SHELL -> repo root
PROJECT_ROOT="$(cd "$SHELL_DIR/../.." && pwd)"
LOG_DIR="${MAG7_LOG_DIR:-$PROJECT_ROOT/logs/maga7}"
PID_FILE="$LOG_DIR/live_session.pid"
STOCK_PID_FILE="$LOG_DIR/live_session_stock.pid"
SESSION_ID_FILE="$LOG_DIR/live_session_id"
GUARD_PID_FILE="$LOG_DIR/live_session_guard.pid"
STOP_MARKER="$LOG_DIR/live_session.stop_requested"
MODULE="maga7.tools.run_live_session"
STOCK_MODULE="maga7.tools.run_stock_md"

DEFAULT_PROFILE="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
PROFILE="${MAG7_PROFILE:-$DEFAULT_PROFILE}"
MODE="${MAG7_MODE:-shadow}"
SCHEME="${MAG7_SCHEME:-m5_circuit}"
PRELOCK_TIME="${MAG7_PRELOCK_TIME:-auto}"
LOCK_TIME="${MAG7_LOCK_TIME:-auto}"
END_TIME="${MAG7_END_TIME:-auto}"
REDIS_HOST="${MAG7_REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${MAG7_REDIS_PORT:-6379}"
REDIS_DB="${MAG7_REDIS_DB:-0}"
IB_HOST="${MAG7_IB_HOST:-127.0.0.1}"
IB_PORT="${MAG7_IB_PORT:-}"
ACCOUNT="${MAG7_ACCOUNT:-}"
CLIENT_ID="${MAG7_CLIENT_ID:-212}"
STOCK_CLIENT_ID="${MAG7_STOCK_CLIENT_ID:-212}"
OPTION_CLIENT_ID="${MAG7_OPTION_CLIENT_ID:-213}"
# Split stock MD / options+OMS into two IB clients (default on).
SPLIT_MD="${MAG7_SPLIT_MD:-1}"
MAX_QTY="${MAG7_MAX_QTY:-1}"
MARKET_DATA_TYPE="${MAG7_MARKET_DATA_TYPE:-1}"
SESSION_ID="${MAG7_SESSION_ID:-}"
FOREGROUND=0
RESUME=0
LIVE_ORDERS=0
PREPARE_ONLY=0
# dry = shadow OMS (model fill, no placeOrder) + default IB live gateway :4001
DRY_MD=0
EXTRA_ARGS=()

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m'

mkdir -p "$LOG_DIR"

usage() {
    cat <<EOF
Mag7 live session launcher

Usage:
  $(basename "$0") start [shadow|dry|paper|live] [options]
  $(basename "$0") stop
  $(basename "$0") status
  $(basename "$0") restart-options   # 股价健康时只重启期权/OMS（不重启 StockMD）
  $(basename "$0") resume <session_id> [shadow|dry|paper|live]
  $(basename "$0") sync-calendar [YYYY-MM-DD YYYY-MM-DD]
  $(basename "$0") help

Modes:
  shadow   G4 Shadow：OMS dry（模型成交、不 placeOrder）；默认 IB :4002
  dry      同 shadow OMS dry，但默认接 IB Live Gateway :4001（实盘行情 / 不发单）
  paper    G5 IBKR Paper（需 --account / MAG7_ACCOUNT；会发 Paper 单）
  live     G6 Live（另需 MAG7_LIVE_TRADING / MAG7_LIVE_CONFIRM）

Split MD (default MAG7_SPLIT_MD=1):
  StockMD  client_id=${MAG7_STOCK_CLIENT_ID:-212}  — 股价订阅/tape/rth_opens
  OptionMD client_id=${MAG7_OPTION_CLIENT_ID:-213} — 期权订阅 + Scanner/OMS
  设 MAG7_SPLIT_MD=0 可回退单进程 combined

Common options:
  --profile PATH          策略 profile（默认 full_day peer3）
  --account ACC           Paper/Live 账户
  --ib-port PORT          覆盖默认端口（dry→4001, shadow/paper→4002, live→4001）
  --redis-db N            默认 0
  --scheme NAME           默认 m5_circuit
  --prelock-time HH:MM|auto|off  默认auto（锁定前10分钟预取合约元数据）
  --lock-time HH:MM|auto  默认 auto
  --end-time HH:MM|auto   默认 auto
  --session-id ID         指定 session id
  --fg                    前台运行（默认 nohup 后台）
  --prepare-only          只准备不进入交易循环
  --live-orders           Live 模式显式开单（G6）
  --allow-code-drift      resume 时显式接受代码指纹变化

Examples:
  # G4：Paper Gateway 行情 + dry OMS
  $(basename "$0") start shadow

  # 实盘 Gateway :4001 行情 + dry OMS（不发单）
  $(basename "$0") start dry
  # 等价: $(basename "$0") start shadow --ib-port 4001

  # G5 Paper
  MAG7_ACCOUNT=DUxxxxxx $(basename "$0") start paper --account DUxxxxxx

  # 看日志
  tail -f $LOG_DIR/live_session.log
EOF
}

log_info() { echo -e "${CYAN}$*${NC}"; }
log_ok() { echo -e "${GREEN}$*${NC}"; }
log_warn() { echo -e "${YELLOW}$*${NC}"; }
log_err() { echo -e "${RED}$*${NC}"; }

default_ib_port() {
    # dry alias → live MD port; OMS still shadow (no broker orders)
    if [ "$DRY_MD" -eq 1 ] || [ "$MODE" = "dry" ]; then
        echo 4001
        return
    fi
    case "$MODE" in
        live) echo 4001 ;;
        *) echo 4002 ;;
    esac
}

normalize_mode() {
    # Map launcher alias → python --mode (only shadow|paper|live)
    if [ "$MODE" = "dry" ]; then
        DRY_MD=1
        MODE="shadow"
    fi
}

resolve_profile() {
    local p="$1"
    if [[ "$p" = /* ]]; then
        echo "$p"
    else
        echo "$PROJECT_ROOT/$p"
    fi
}

get_session_pids() {
    # Match real python -m workers only (exclude bash wrappers that embed the cmdline).
    ps -eo pid=,args= 2>/dev/null | awk -v n="-m ${MODULE}" '
        index($0, n) && $0 ~ /(^|\/)python[0-9.]* / && $0 !~ /extglob/ { print $1 }
    ' || true
}

resolve_python() {
    if [ -n "${MAG7_PYTHON:-}" ]; then
        echo "$MAG7_PYTHON"
        return
    fi
    if [ -x "$HOME/anaconda3/envs/ibkr/bin/python" ]; then
        echo "$HOME/anaconda3/envs/ibkr/bin/python"
        return
    fi
    if [ -x "$HOME/miniconda3/envs/ibkr/bin/python" ]; then
        echo "$HOME/miniconda3/envs/ibkr/bin/python"
        return
    fi
    command -v python3 >/dev/null 2>&1 && { command -v python3; return; }
    command -v python >/dev/null 2>&1 && { command -v python; return; }
    echo ""
}

check_python() {
    local py
    py="$(resolve_python)"
    if [ -z "$py" ] || [ ! -x "$py" ]; then
        log_err "❌ Python not found. Activate conda env ibkr or set MAG7_PYTHON."
        exit 1
    fi
    if ! "$py" -c "import ib_insync" >/dev/null 2>&1; then
        log_err "❌ $py missing ib_insync — use conda env ibkr or MAG7_PYTHON=..."
        exit 1
    fi
    export MAG7_PYTHON="$py"
    log_ok "✅ Python: $py"
}

check_redis() {
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
        log_ok "✅ Redis ready ($REDIS_HOST:$REDIS_PORT db=$REDIS_DB)"
        return 0
    fi
    log_warn "Redis not responding; trying redis-server --daemonize yes ..."
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes || true
        sleep 1
    fi
    if redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
        log_ok "✅ Redis ready"
    else
        log_err "❌ Redis unavailable"
        exit 1
    fi
}

check_ib_port() {
    local port="$1"
    if ss -ltn 2>/dev/null | grep -q ":${port} "; then
        log_ok "✅ IBKR API port $port is listening"
        return 0
    fi
    if command -v nc >/dev/null 2>&1 && nc -z "$IB_HOST" "$port" >/dev/null 2>&1; then
        log_ok "✅ IBKR API $IB_HOST:$port reachable"
        return 0
    fi
    log_err "❌ IBKR API not listening on $IB_HOST:$port"
    log_err "   Start IB Gateway/TWS (Paper=4002, Live=4001) before G4/G5/G6."
    exit 1
}

check_mag7_lock_preflight() {
    local port="$1"
    if [ "${MAG7_SKIP_LOCK_PREFLIGHT:-0}" = "1" ]; then
        log_warn "⚠️  MAG7_SKIP_LOCK_PREFLIGHT=1 — skipping Mag7 lock/quote preflight"
        return 0
    fi
    local py
    py="${MAG7_PYTHON:-}"
    if [ -z "$py" ] || [ ! -x "$py" ]; then
        py="$(command -v python3 || command -v python)"
    fi
    local out="$LOG_DIR/mag7_lock_preflight_$(date -u '+%Y%m%dT%H%M%SZ').log"
    log_info "Mag7 lock/subscription preflight on :$port…"
    local extra=()
    if [ "${MAG7_REQUIRE_OPTION_QUOTES:-0}" = "1" ]; then
        extra+=(--require-quotes)
    fi
    if PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
        "$py" -m maga7.tools.ibkr_mag7_lock_preflight \
        --host "$IB_HOST" \
        --port "$port" \
        --client-id "${MAG7_LOCK_PREFLIGHT_CLIENT_ID:-913}" \
        --profile "$PROFILE" \
        --json-out "$LOG_DIR/mag7_lock_preflight_latest.json" \
        "${extra[@]}" \
        >"$out" 2>&1; then
        log_ok "✅ Mag7 lock preflight OK"
        if command -v rg >/dev/null 2>&1 && rg -q 'nearest_fallback|ticker_alive_no_nbbo|awaiting_nbbo|NO_NBBO' "$out"; then
            log_warn "⚠️  nearest-DTE fallback and/or quote diagnosis pending — see $out"
        fi
        return 0
    fi
    log_err "❌ Mag7 lock/subscription preflight FAILED — see $out"
    log_err "   Fix IB/OPRA/chain selection before start, or MAG7_SKIP_LOCK_PREFLIGHT=1 to bypass."
    exit 1
}

check_ib_exposure() {
    local port="$1"
    local listeners
    listeners="$(ss -ltnH 2>/dev/null | awk -v p=":${port}" '$4 ~ p"$" {print $4}')"
    if echo "$listeners" | awk '
        /^\*:/ || /^0\.0\.0\.0:/ || /^\[::\]:/ { exposed=1 }
        END { exit exposed ? 0 : 1 }
    '; then
        if [ "$MODE" = "paper" ] || [ "$MODE" = "live" ]; then
            if [ "${MAG7_ALLOW_WILDCARD_IB:-}" != "1" ]; then
                log_err "❌ IBKR API :$port listens on all interfaces: $listeners"
                log_err "   Bind Gateway to localhost / firewall the port, or explicitly set MAG7_ALLOW_WILDCARD_IB=1 after review."
                exit 1
            fi
            log_warn "⚠️  wildcard IB listener explicitly accepted: $listeners"
        else
            log_warn "⚠️  IBKR API :$port is externally bound: $listeners (Shadow only)"
        fi
    fi
}

ny_now() {
    TZ=America/New_York date '+%Y-%m-%d %H:%M:%S %Z'
}

cmd_status() {
    local pids
    pids="$(get_session_pids)"
    echo "=================================================="
    echo "Mag7 live session status"
    echo "Root: $PROJECT_ROOT"
    echo "NY now: $(ny_now)"
    echo "=================================================="
    if [ -n "$pids" ]; then
        log_ok "✅ running PIDs: $pids"
    else
        log_warn "ℹ️  not running"
    fi
    if [ -f "$PID_FILE" ]; then
        echo "pidfile: $PID_FILE -> $(cat "$PID_FILE")"
    fi
    if [ -f "$GUARD_PID_FILE" ]; then
        echo "guard pid: $(cat "$GUARD_PID_FILE")"
    fi
    if [ -f "$LOG_DIR/live_session.log" ]; then
        echo "log: $LOG_DIR/live_session.log"
        echo "---- last useful log lines ----"
        # Legacy shared logs may contain old tracebacks. If this run explicitly
        # resumed after a code review, show only the current-run suffix.
        awk '
            /explicit --allow-code-drift/ { suffix=""; current=1 }
            { suffix = suffix $0 ORS }
            END { printf "%s", suffix }
        ' "$LOG_DIR/live_session.log" 2>/dev/null \
            | grep -Ev 'ConnectionRefusedError|Make sure API port on TWS/IBG' \
            | tail -n 12 || true
    fi
    local latest=""
    local live_root="${MAG7_LIVE_SESSIONS_DIR:-/mnt/s990/data/maga7/live_sessions}"
    if [ -d "$live_root" ]; then
        latest="$(ls -dt "$live_root"/*/* 2>/dev/null | head -1 || true)"
    fi
    # Fallback: legacy in-tree path (pre path migration)
    if [ -z "${latest:-}" ]; then
        latest="$(ls -dt "$PROJECT_ROOT"/maga7/results/live_sessions/*/* 2>/dev/null | head -1 || true)"
    fi
    if [ -n "${latest:-}" ]; then
        echo "latest session dir: $latest"
        if [ -f "$latest/manifest.json" ]; then
            python - <<PY
import json
import time
from pathlib import Path

m = json.loads(Path("$latest").joinpath("manifest.json").read_text())
c = m.get("connector") or {}
e = m.get("engine_metrics") or {}
print(f"  state={m.get('state')} mode={m.get('mode')} error={m.get('error')!r}")
print(
    f"  data_mode={c.get('data_mode')} lock={c.get('lock_status')} "
    f"frames={e.get('frames')} rejected={e.get('rejected')} foreign={e.get('foreign')}"
)
for name in ("locks.json", "tape_parity.json"):
    p = Path("$latest") / name
    if not p.is_file():
        continue
    try:
        d = json.loads(p.read_text())
    except Exception:
        continue
    if name == "locks.json":
        print(f"  locks={d.get('status')} n={len(d.get('locks') or {})}")
    else:
        print(f"  parity ok={d.get('ok')} stage={d.get('stage')} issues={d.get('issues')}")

# Live Redis heartbeat (authoritative while manifest still STARTING)
try:
    import msgpack
    import redis

    sid = str(m.get("session_id") or Path("$latest").name)
    r = redis.Redis(host="${REDIS_HOST}", port=int("${REDIS_PORT}"), db=int("${REDIS_DB}"), decode_responses=False)
    raw = r.hget(f"live_ibkr_connector:maga7:{sid}", "status")
    if raw:
        st = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        if isinstance(st, dict) and st.get("__maga7_wire__") == 1:
            st = st.get("payload") or {}
        now = time.time()
        # Prefer feed_health (1s) over status snapshot (heartbeat ~15s).
        feed_raw = r.hget(f"live_ibkr_connector:maga7:{sid}", "feed_health")
        feed = msgpack.unpackb(feed_raw, raw=False, strict_map_key=False) if feed_raw else st
        if isinstance(feed, dict) and feed.get("__maga7_wire__") == 1:
            feed = feed.get("payload") or {}
        ages = []
        for info in (feed.get("stock_feed") or {}).values():
            try:
                ages.append(now - float(info.get("last_ts") or 0))
            except Exception:
                pass
        max_age = max(ages) if ages else None
        print(
            f"  redis state={st.get('state')} connected={st.get('connected')} "
            f"data_mode={st.get('data_mode')} phase={st.get('session_phase')} "
            f"lock={st.get('lock_status')} stocks={feed.get('stock_live_symbols', st.get('stock_live_symbols'))} "
            f"tape_writes={feed.get('tape_writes', st.get('tape_writes'))} "
            f"max_stock_age={None if max_age is None else round(max_age,1)}s"
        )
        pre = r.xlen(st.get("stream_pre") or f"fused_market_stream:maga7:{sid}:pre")
        rth = r.xlen(st.get("stream") or f"fused_market_stream:maga7:{sid}")
        print(f"  redis streams pre={pre} rth={rth}")
except Exception as exc:
    print(f"  redis status unavailable: {exc}")
PY
        fi
    fi
}

cmd_stop() {
    local pids stock_pid
    pids="$(get_session_pids)"
    stock_pid=""
    if [ -f "$STOCK_PID_FILE" ]; then
        stock_pid="$(cat "$STOCK_PID_FILE" 2>/dev/null || true)"
    fi
    if [ -z "$pids" ] && [ -z "${stock_pid:-}" ]; then
        log_warn "ℹ️  Mag7 live session not running."
        rm -f "$PID_FILE" "$STOCK_PID_FILE"
        if [ -f "$GUARD_PID_FILE" ]; then
            kill "$(cat "$GUARD_PID_FILE" 2>/dev/null || true)" 2>/dev/null || true
            rm -f "$GUARD_PID_FILE"
        fi
        return 0
    fi
    log_warn "🛑 Stopping Mag7 live session (engine PIDs: ${pids:-none}; stock: ${stock_pid:-none})..."
    : >"$STOP_MARKER"
    # shellcheck disable=SC2086
    [ -n "$pids" ] && kill $pids 2>/dev/null || true
    [ -n "${stock_pid:-}" ] && kill "$stock_pid" 2>/dev/null || true
    sleep 2
    pids="$(get_session_pids)"
    if [ -n "$pids" ]; then
        log_warn "force kill engine..."
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
    if [ -n "${stock_pid:-}" ] && kill -0 "$stock_pid" 2>/dev/null; then
        kill -9 "$stock_pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE" "$STOCK_PID_FILE"
    if [ -f "$GUARD_PID_FILE" ]; then
        local guard_pid
        guard_pid="$(cat "$GUARD_PID_FILE" 2>/dev/null || true)"
        if [ -n "$guard_pid" ]; then
            kill "$guard_pid" 2>/dev/null || true
        fi
        rm -f "$GUARD_PID_FILE"
    fi
    log_ok "✅ stopped"
}

stock_md_healthy() {
    local sid="${1:-}"
    if [ -z "$sid" ] && [ -f "$SESSION_ID_FILE" ]; then
        sid="$(cat "$SESSION_ID_FILE" 2>/dev/null || true)"
    fi
    [ -n "$sid" ] || return 1
    if [ -f "$STOCK_PID_FILE" ]; then
        local spid
        spid="$(cat "$STOCK_PID_FILE" 2>/dev/null || true)"
        if [ -n "$spid" ] && kill -0 "$spid" 2>/dev/null; then
            :
        else
            return 1
        fi
    else
        return 1
    fi
    redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB" EXISTS "maga7:feed_health_stock:$sid" 2>/dev/null | grep -q 1
}

cmd_restart_options() {
    # Restart options/OMS only when stock MD is healthy.
    if [ "$SPLIT_MD" != "1" ]; then
        log_err "restart-options requires MAG7_SPLIT_MD=1"
        exit 1
    fi
    if [ -z "$SESSION_ID" ] && [ -f "$SESSION_ID_FILE" ]; then
        SESSION_ID="$(cat "$SESSION_ID_FILE")"
    fi
    if [ -z "$SESSION_ID" ]; then
        log_err "no SESSION_ID — cannot restart options against running stock MD"
        exit 1
    fi
    if ! stock_md_healthy "$SESSION_ID"; then
        log_err "stock MD unhealthy/missing — refusing options-only restart (fix stock first)"
        exit 1
    fi
    log_ok "stock MD healthy for $SESSION_ID — restarting options/OMS only"
    local pids
    pids="$(get_session_pids)"
    : >"$STOP_MARKER"
    # shellcheck disable=SC2086
    [ -n "$pids" ] && kill $pids 2>/dev/null || true
    sleep 2
    pids="$(get_session_pids)"
    # shellcheck disable=SC2086
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
    rm -f "$PID_FILE" "$STOP_MARKER"
    RESUME=1
    FOREGROUND=0
    # Keep stock process; start options engine with same session id.
    SPLIT_MD=1
    cmd_start
}

build_cmd() {
    local profile_path
    profile_path="$(resolve_profile "$PROFILE")"
    if [ ! -f "$profile_path" ]; then
        log_err "❌ profile not found: $profile_path"
        exit 1
    fi

    if [ -z "$IB_PORT" ]; then
        IB_PORT="$(default_ib_port)"
    fi

    local py
    py="${MAG7_PYTHON:-$(resolve_python)}"
    local -a cmd=(
        "$py" -u -m "$MODULE"
        --profile "$profile_path"
        --mode "$MODE"
        --scheme "$SCHEME"
        --prelock-time "$PRELOCK_TIME"
        --lock-time "$LOCK_TIME"
        --end-time "$END_TIME"
        --ib-host "$IB_HOST"
        --ib-port "$IB_PORT"
        --client-id "$CLIENT_ID"
        --redis-host "$REDIS_HOST"
        --redis-port "$REDIS_PORT"
        --redis-db "$REDIS_DB"
        --max-qty "$MAX_QTY"
        --market-data-type "$MARKET_DATA_TYPE"
    )

    if [ -n "$ACCOUNT" ]; then
        cmd+=(--account "$ACCOUNT")
    fi
    if [ -n "$SESSION_ID" ]; then
        cmd+=(--session-id "$SESSION_ID")
    fi
    if [ "$RESUME" -eq 1 ]; then
        cmd+=(--resume)
    fi
    if [ "$SPLIT_MD" = "1" ]; then
        # Override single CLIENT_ID with dedicated options client.
        local rebuilt=()
        local skip_next=0
        local i=0
        for ((i=0; i<${#cmd[@]}; i++)); do
            if [ "$skip_next" -eq 1 ]; then
                skip_next=0
                continue
            fi
            if [ "${cmd[$i]}" = "--client-id" ]; then
                rebuilt+=(--client-id "$OPTION_CLIENT_ID")
                skip_next=1
                continue
            fi
            rebuilt+=("${cmd[$i]}")
        done
        cmd=("${rebuilt[@]}")
        cmd+=(--md-role options)
    fi
    if [ "$PREPARE_ONLY" -eq 1 ]; then
        cmd+=(--prepare-only)
    fi
    if [ "$LIVE_ORDERS" -eq 1 ]; then
        cmd+=(--live-orders)
    fi
    if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
        cmd+=("${EXTRA_ARGS[@]}")
    fi

    printf '%q ' "${cmd[@]}"
}

build_stock_cmd() {
    local profile_path
    profile_path="$(resolve_profile "$PROFILE")"
    local py
    py="${MAG7_PYTHON:-$(resolve_python)}"
    if [ -z "$IB_PORT" ]; then
        IB_PORT="$(default_ib_port)"
    fi
    local -a cmd=(
        "$py" -u -m "$STOCK_MODULE"
        --profile "$profile_path"
        --session-id "$SESSION_ID"
        --ib-host "$IB_HOST"
        --ib-port "$IB_PORT"
        --client-id "$STOCK_CLIENT_ID"
        --redis-host "$REDIS_HOST"
        --redis-port "$REDIS_PORT"
        --redis-db "$REDIS_DB"
        --market-data-type "$MARKET_DATA_TYPE"
        --end-time "$END_TIME"
    )
    if [ "$RESUME" -eq 1 ]; then
        cmd+=(--resume)
    fi
    printf '%q ' "${cmd[@]}"
}

validate_mode() {
    normalize_mode
    case "$MODE" in
        shadow|paper|live) ;;
        *)
            log_err "❌ unknown mode: $MODE (use shadow|dry|paper|live)"
            exit 1
            ;;
    esac
    if [ "$MODE" = "paper" ] || [ "$MODE" = "live" ]; then
        if [ -z "$ACCOUNT" ]; then
            log_err "❌ $MODE requires --account or MAG7_ACCOUNT"
            exit 1
        fi
    fi
    if [ "$MODE" = "live" ]; then
        if [ "${MAG7_LIVE_TRADING:-}" != "1" ]; then
            log_err "❌ live mode requires MAG7_LIVE_TRADING=1"
            exit 1
        fi
        if [ -z "${MAG7_LIVE_CONFIRM:-}" ]; then
            log_err "❌ live mode requires MAG7_LIVE_CONFIRM=<NY_date>:<profile_hash12>"
            exit 1
        fi
        if [ "$LIVE_ORDERS" -ne 1 ]; then
            log_err "❌ live mode requires --live-orders"
            exit 1
        fi
    fi
    # Safety: dry/shadow must never enable live-orders flag
    if [ "$MODE" = "shadow" ] && [ "$LIVE_ORDERS" -eq 1 ]; then
        log_err "❌ shadow/dry OMS cannot use --live-orders"
        exit 1
    fi
}

cmd_start() {
    check_python
    validate_mode
    cd "$PROJECT_ROOT"

    local existing
    existing="$(get_session_pids)"
    if [ -n "$existing" ]; then
        log_warn "⚠️  already running (PIDs: $existing). Stop it before start/resume."
        exit 1
    fi
    rm -f "$STOP_MARKER"
    if [ -z "$SESSION_ID" ]; then
        SESSION_ID="live_$(TZ=America/New_York date '+%Y%m%d_%H%M%S')_${RANDOM}${RANDOM}"
    fi

    if [ -z "$IB_PORT" ]; then
        IB_PORT="$(default_ib_port)"
    fi

    echo "=================================================="
    echo "Mag7 live session start"
    echo "Root:    $PROJECT_ROOT"
    if [ "$DRY_MD" -eq 1 ]; then
        echo "Mode:    shadow (dry OMS / no placeOrder)  [alias=dry]"
    else
        echo "Mode:    $MODE"
    fi
    echo "IB port: $IB_PORT"
    echo "OMS:     $([ "$MODE" = "shadow" ] && echo 'DRY model-fill' || echo "broker $MODE")"
    echo "Profile: $PROFILE"
    echo "NY now:  $(ny_now)"
    echo "=================================================="

    check_redis
    if [ -z "$IB_PORT" ]; then
        IB_PORT="$(default_ib_port)"
    fi
    check_ib_port "$IB_PORT"
    check_ib_exposure "$IB_PORT"
    check_mag7_lock_preflight "$IB_PORT"

    if [ -n "${MAG7_EVENT_CALENDAR_PATH:-}" ]; then
        log_info "EVENT_CALENDAR_PATH=$MAG7_EVENT_CALENDAR_PATH"
    elif [ -f "$PROJECT_ROOT/maga7/CONFIG/event_calendar_live.json" ]; then
        export MAG7_EVENT_CALENDAR_PATH="$PROJECT_ROOT/maga7/CONFIG/event_calendar_live.json"
        log_info "EVENT_CALENDAR_PATH=$MAG7_EVENT_CALENDAR_PATH"
    fi

    local cmd_str log_file legacy_log stock_cmd stock_log
    echo "$SESSION_ID" >"$SESSION_ID_FILE"
    if [ "$SPLIT_MD" = "1" ]; then
        if stock_md_healthy "$SESSION_ID"; then
            log_ok "stock MD already healthy — keeping client_id=$STOCK_CLIENT_ID"
        else
            stock_cmd="$(build_stock_cmd)"
            stock_log="$LOG_DIR/live_session_stock_${SESSION_ID}_$(date -u '+%Y%m%dT%H%M%SZ').log"
            log_info "STOCK MD CMD: $stock_cmd"
            log_info "STOCK MD LOG: $stock_log"
            # shellcheck disable=SC2086
            nohup bash -c "cd \"$PROJECT_ROOT\" && $stock_cmd" >>"$stock_log" 2>&1 &
            local stock_pid=$!
            echo "$stock_pid" >"$STOCK_PID_FILE"
            sleep 2
            if ! ps -p "$stock_pid" >/dev/null 2>&1; then
                log_err "❌ stock MD failed to stay up. Last log:"
                tail -n 40 "$stock_log" || true
                rm -f "$STOCK_PID_FILE"
                exit 1
            fi
            log_ok "✅ stock MD STARTED (PID: $stock_pid client_id=$STOCK_CLIENT_ID)"
            sleep 2
        fi
    fi

    cmd_str="$(build_cmd)"
    log_file="$LOG_DIR/live_session_${SESSION_ID}_$(date -u '+%Y%m%dT%H%M%SZ').log"
    legacy_log="$LOG_DIR/live_session.log"
    if [ -e "$legacy_log" ] && [ ! -L "$legacy_log" ]; then
        mv "$legacy_log" "$LOG_DIR/live_session_legacy_$(date -u '+%Y%m%dT%H%M%SZ').log"
    fi
    ln -sfn "$(basename "$log_file")" "$legacy_log"
    log_info "CMD: $cmd_str"
    log_info "LOG: $log_file"

    if [ "$FOREGROUND" -eq 1 ]; then
        # shellcheck disable=SC2086
        eval "$cmd_str"
        return $?
    fi

    # shellcheck disable=SC2086
    nohup bash -c "cd \"$PROJECT_ROOT\" && $cmd_str" >>"$log_file" 2>&1 &
    local pid=$!
    echo "$pid" >"$PID_FILE"
    sleep 2
    if ps -p "$pid" >/dev/null 2>&1; then
        local live_root guard_log guard_pid
        live_root="${MAG7_LIVE_SESSIONS_DIR:-/mnt/s990/data/maga7/live_sessions}"
        guard_log="$LOG_DIR/live_session_guard_${SESSION_ID}.log"
        nohup "${MAG7_PYTHON:-$(resolve_python)}" -u -m maga7.tools.live_session_guard \
            --pid "$pid" \
            --session-id "$SESSION_ID" \
            --log-dir "$LOG_DIR" \
            --live-root "$live_root" \
            >>"$guard_log" 2>&1 &
        guard_pid=$!
        echo "$guard_pid" >"$GUARD_PID_FILE"
        log_ok "✅ options/OMS STARTED (PID: $pid client_id=$OPTION_CLIENT_ID)"
        log_info "   guard PID: $guard_pid"
        log_info "   tail -f $log_file"
        log_info "   artifacts: ${MAG7_LIVE_SESSIONS_DIR:-/mnt/s990/data/maga7/live_sessions}/"
        if [ "$SPLIT_MD" = "1" ]; then
            log_info "   split MD: stock client=$STOCK_CLIENT_ID options client=$OPTION_CLIENT_ID"
            log_info "   restart options only: $0 restart-options"
        fi
    else
        log_err "❌ FAILED to stay up. Last log:"
        tail -n 30 "$log_file" || true
        rm -f "$PID_FILE"
        exit 1
    fi
}

cmd_sync_calendar() {
    check_python
    cd "$PROJECT_ROOT"
    local start_d end_d
    start_d="${1:-}"
    end_d="${2:-}"
    local out="$PROJECT_ROOT/maga7/CONFIG/event_calendar_live.json"
    local audit="$PROJECT_ROOT/maga7/CONFIG/event_news_audit.json"
    local -a args=(
        "$(resolve_python)" -u -m maga7.tools.sync_event_calendar
        --out "$out"
        --news-audit-out "$audit"
        --news-mode "${MAG7_NEWS_MODE:-hard_risk}"
    )
    if [ -n "$start_d" ] && [ -n "$end_d" ]; then
        args+=(--start "$start_d" --end "$end_d")
    fi
    # Company news (Finnhub + Investing RSS) on by default; set MAG7_NO_NEWS=1 to skip.
    if [ "${MAG7_NO_NEWS:-0}" = "1" ]; then
        args+=(--no-news)
    fi
    log_info "sync event calendar (+company news) -> $out"
    "${args[@]}"
}

parse_options() {
    while [ $# -gt 0 ]; do
        case "$1" in
            shadow|dry|paper|live)
                MODE="$1"
                if [ "$1" = "dry" ]; then
                    DRY_MD=1
                fi
                shift
                ;;
            --profile)
                PROFILE="$2"; shift 2
                ;;
            --account)
                ACCOUNT="$2"; shift 2
                ;;
            --ib-port)
                IB_PORT="$2"; shift 2
                ;;
            --redis-db)
                REDIS_DB="$2"; shift 2
                ;;
            --scheme)
                SCHEME="$2"; shift 2
                ;;
            --prelock-time)
                PRELOCK_TIME="$2"; shift 2
                ;;
            --lock-time)
                LOCK_TIME="$2"; shift 2
                ;;
            --end-time)
                END_TIME="$2"; shift 2
                ;;
            --session-id)
                SESSION_ID="$2"; shift 2
                ;;
            --fg|--foreground)
                FOREGROUND=1; shift
                ;;
            --resume)
                RESUME=1; shift
                ;;
            --prepare-only)
                PREPARE_ONLY=1; shift
                ;;
            --live-orders)
                LIVE_ORDERS=1; shift
                ;;
            --)
                shift
                EXTRA_ARGS+=("$@")
                break
                ;;
            -h|--help|help)
                usage; exit 0
                ;;
            *)
                EXTRA_ARGS+=("$1"); shift
                ;;
        esac
    done
}

main() {
    local action="${1:-start}"
    if [ $# -gt 0 ]; then
        shift
    fi

    case "$action" in
        start)
            parse_options "$@"
            cmd_start
            ;;
        stop)
            cmd_stop
            ;;
        restart-options|restart-option|restart-engine)
            parse_options "$@"
            cmd_restart_options
            ;;
        status)
            cmd_status
            ;;
        resume)
            if [ $# -lt 1 ]; then
                log_err "usage: $0 resume <session_id> [mode]"
                exit 1
            fi
            SESSION_ID="$1"
            shift
            RESUME=1
            parse_options "$@"
            cmd_start
            ;;
        sync-calendar)
            cmd_sync_calendar "$@"
            ;;
        help|-h|--help)
            usage
            ;;
        *)
            # allow: ./script shadow   (implicit start)
            if [[ "$action" =~ ^(shadow|dry|paper|live)$ ]]; then
                parse_options "$action" "$@"
                cmd_start
            else
                log_err "unknown action: $action"
                usage
                exit 1
            fi
            ;;
    esac
}

main "$@"
