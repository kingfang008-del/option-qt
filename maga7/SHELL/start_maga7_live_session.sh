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
MODULE="maga7.tools.run_live_session"

DEFAULT_PROFILE="maga7/CONFIG/strategy_profiles/single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
PROFILE="${MAG7_PROFILE:-$DEFAULT_PROFILE}"
MODE="${MAG7_MODE:-shadow}"
SCHEME="${MAG7_SCHEME:-m5_circuit}"
LOCK_TIME="${MAG7_LOCK_TIME:-auto}"
END_TIME="${MAG7_END_TIME:-auto}"
REDIS_HOST="${MAG7_REDIS_HOST:-127.0.0.1}"
REDIS_PORT="${MAG7_REDIS_PORT:-6379}"
REDIS_DB="${MAG7_REDIS_DB:-0}"
IB_HOST="${MAG7_IB_HOST:-127.0.0.1}"
IB_PORT="${MAG7_IB_PORT:-}"
ACCOUNT="${MAG7_ACCOUNT:-}"
CLIENT_ID="${MAG7_CLIENT_ID:-212}"
MAX_QTY="${MAG7_MAX_QTY:-1}"
MARKET_DATA_TYPE="${MAG7_MARKET_DATA_TYPE:-1}"
SESSION_ID="${MAG7_SESSION_ID:-}"
FOREGROUND=0
RESUME=0
LIVE_ORDERS=0
PREPARE_ONLY=0
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
  $(basename "$0") start [shadow|paper|live] [options]
  $(basename "$0") stop
  $(basename "$0") status
  $(basename "$0") resume <session_id> [shadow|paper|live]
  $(basename "$0") sync-calendar [YYYY-MM-DD YYYY-MM-DD]
  $(basename "$0") help

Modes:
  shadow   G4 真实行情 Shadow（默认，不发单）
  paper    G5 IBKR Paper（需 --account / MAG7_ACCOUNT）
  live     G6 Live（另需 MAG7_LIVE_TRADING / MAG7_LIVE_CONFIRM）

Common options:
  --profile PATH          策略 profile（默认 full_day peer3）
  --account ACC           Paper/Live 账户
  --ib-port PORT          默认 paper=4002, live=4001, shadow=4002
  --redis-db N            默认 0
  --scheme NAME           默认 m5_circuit
  --lock-time HH:MM|auto  默认 auto
  --end-time HH:MM|auto   默认 auto
  --session-id ID         指定 session id
  --fg                    前台运行（默认 nohup 后台）
  --prepare-only          只准备不进入交易循环
  --live-orders           Live 模式显式开单（G6）

Examples:
  # G4：开盘前一键 Shadow
  $(basename "$0") start shadow

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
    case "$MODE" in
        live) echo 4001 ;;
        *) echo 4002 ;;
    esac
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
    pgrep -f "$MODULE" 2>/dev/null || true
}

check_python() {
    if ! command -v python >/dev/null 2>&1; then
        log_err "❌ Python not found. Activate conda/env first."
        exit 1
    fi
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
    if [ -f "$LOG_DIR/live_session.log" ]; then
        echo "log: $LOG_DIR/live_session.log"
        echo "---- last 12 log lines ----"
        tail -n 12 "$LOG_DIR/live_session.log" || true
    fi
    local latest
    latest="$(ls -dt "$PROJECT_ROOT"/maga7/results/live_sessions/*/* 2>/dev/null | head -1 || true)"
    if [ -n "${latest:-}" ]; then
        echo "latest session dir: $latest"
        if [ -f "$latest/manifest.json" ]; then
            python - <<PY
import json
from pathlib import Path
m=json.loads(Path("$latest/manifest.json").read_text())
c=m.get("connector") or {}
e=m.get("engine_metrics") or {}
print(f"  state={m.get('state')} mode={m.get('mode')} error={m.get('error')!r}")
print(f"  data_mode={c.get('data_mode')} lock={c.get('lock_status')} frames={e.get('frames')} rejected={e.get('rejected')} foreign={e.get('foreign')}")
PY
        fi
    fi
}

cmd_stop() {
    local pids
    pids="$(get_session_pids)"
    if [ -z "$pids" ]; then
        log_warn "ℹ️  Mag7 live session not running."
        rm -f "$PID_FILE"
        return 0
    fi
    log_warn "🛑 Stopping Mag7 live session (PIDs: $pids)..."
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 2
    pids="$(get_session_pids)"
    if [ -n "$pids" ]; then
        log_warn "force kill..."
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    log_ok "✅ stopped"
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

    local -a cmd=(
        python -u -m "$MODULE"
        --profile "$profile_path"
        --mode "$MODE"
        --scheme "$SCHEME"
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

validate_mode() {
    case "$MODE" in
        shadow|paper|live) ;;
        *)
            log_err "❌ unknown mode: $MODE"
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
}

cmd_start() {
    check_python
    validate_mode
    cd "$PROJECT_ROOT"

    local existing
    existing="$(get_session_pids)"
    if [ -n "$existing" ] && [ "$RESUME" -eq 0 ]; then
        log_warn "⚠️  already running (PIDs: $existing). Use status/stop, or resume."
        exit 1
    fi

    echo "=================================================="
    echo "Mag7 live session start"
    echo "Root:    $PROJECT_ROOT"
    echo "Mode:    $MODE"
    echo "Profile: $PROFILE"
    echo "NY now:  $(ny_now)"
    echo "=================================================="

    check_redis
    if [ -z "$IB_PORT" ]; then
        IB_PORT="$(default_ib_port)"
    fi
    check_ib_port "$IB_PORT"

    if [ -n "${MAG7_EVENT_CALENDAR_PATH:-}" ]; then
        log_info "EVENT_CALENDAR_PATH=$MAG7_EVENT_CALENDAR_PATH"
    elif [ -f "$PROJECT_ROOT/maga7/CONFIG/event_calendar_live.json" ]; then
        export MAG7_EVENT_CALENDAR_PATH="$PROJECT_ROOT/maga7/CONFIG/event_calendar_live.json"
        log_info "EVENT_CALENDAR_PATH=$MAG7_EVENT_CALENDAR_PATH"
    fi

    local cmd_str log_file
    cmd_str="$(build_cmd)"
    log_file="$LOG_DIR/live_session.log"
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
        log_ok "✅ STARTED (PID: $pid)"
        log_info "   tail -f $log_file"
        log_info "   artifacts: $PROJECT_ROOT/maga7/results/live_sessions/"
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
    local -a args=(
        python -u -m maga7.tools.sync_event_calendar
        --out "$out"
    )
    if [ -n "$start_d" ] && [ -n "$end_d" ]; then
        args+=(--start "$start_d" --end "$end_d")
    fi
    log_info "sync event calendar -> $out"
    "${args[@]}"
}

parse_options() {
    while [ $# -gt 0 ]; do
        case "$1" in
            shadow|paper|live)
                MODE="$1"
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
            if [[ "$action" =~ ^(shadow|paper|live)$ ]]; then
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
