#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: oms_state_recovery.py
描述  : OMS / SE 状态快速诊断与清理工具

使用场景:
  1. 每日开盘前 (盘前健康检查)
     -> 检查 Redis Stream 积压 / PG 里是否有跨天残留的 _GLOBAL_STATE_
     -> 一键清理, 保证盘中 fresh start
  2. 盘中进程意外重启后
     -> 主动丢弃 Redis Stream 里的历史积压, 避免 OMS 把旧信号重放
        导致 "重复下单 / 账户资金莫名漂移" 的问题

行为:
  - 默认只做诊断, 不动任何数据 (inspect-only, read-only)
  - 需要真正清理时必须显式传开关:
      --drop-backlog        推进 OMS / SE 消费者组 last-delivered-id 到 $
      --reset-global-state  清除 PG 中 _GLOBAL_STATE_ 行 (mock_cash 漂移修复)
      --purge-stale         清除 PG 中非当天 (NY 日) 的 symbol_state 行
      --purge-fcs-stale     清除 fcs_state_snapshot 中非当天的 namespace (FCS 快照)
      --all                 = 以上四者同时执行
  - 默认会做 y/N 确认; --yes 可跳过

典型用法:
  # 开盘前健康检查 (只诊断)
  python production/scripts/oms_state_recovery.py

  # 进程重启后快速修复
  python production/scripts/oms_state_recovery.py --all --yes

  # 只清 Redis 积压
  python production/scripts/oms_state_recovery.py --drop-backlog --yes

  # 只清 PG 中跨天残留
  python production/scripts/oms_state_recovery.py --purge-stale --reset-global-state --yes

  # 指定目标 Redis DB (默认读 config: realtime=0 / simulate=1)
  python production/scripts/oms_state_recovery.py --redis-db 0
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASELINE_DIR = PROJECT_ROOT / "production" / "baseline"
if str(BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_DIR))

try:
    from config import (  # type: ignore
        REDIS_CFG,
        STREAM_ORCH_SIGNAL,
        STREAM_FUSED_MARKET,
        STREAM_INFERENCE,
        GROUP_OMS,
        GROUP_ORCH,
        PG_DB_URL,
        NY_TZ,
        RUN_MODE,
    )
except Exception as e:
    print(f"[FATAL] 无法导入 production/baseline/config.py: {e}")
    sys.exit(2)

try:
    import redis  # type: ignore
except ImportError:
    print("[FATAL] 缺少 redis-py, 请先 pip install redis")
    sys.exit(2)

try:
    import psycopg2  # type: ignore
except ImportError:
    print("[FATAL] 缺少 psycopg2, 请先 pip install psycopg2-binary")
    sys.exit(2)


# ============================================================
# 受本脚本管理的 (stream, group) 二元组
# ============================================================
# OMS 消费的: orch_trade_signals (+ 在实盘/DRY 下还有 fused_market_stream)
# SE  消费的: fused_market_stream / unified_inference_stream
# 我们统一处理这几个, 盘前一把梭不会遗漏
TARGET_GROUPS = [
    (STREAM_ORCH_SIGNAL,  GROUP_OMS),
    (STREAM_FUSED_MARKET, GROUP_OMS),
    (STREAM_FUSED_MARKET, GROUP_ORCH),
    (STREAM_INFERENCE,    GROUP_ORCH),
]


# ============================================================
# Redis helpers
# ============================================================
def _build_redis(db_override: int | None) -> "redis.Redis":
    cfg = dict(REDIS_CFG)
    if db_override is not None:
        cfg['db'] = db_override
    return redis.Redis(
        host=cfg['host'],
        port=cfg['port'],
        db=cfg['db'],
        socket_timeout=5.0,
        socket_connect_timeout=5.0,
    )


def _stream_exists(r, stream: str) -> bool:
    try:
        r.xinfo_stream(stream)
        return True
    except redis.exceptions.ResponseError:
        return False
    except Exception:
        return False


def _find_group(r, stream: str, group: str):
    try:
        infos = r.xinfo_groups(stream)
    except redis.exceptions.ResponseError:
        return None
    except Exception:
        return None
    for g in infos:
        name = g.get(b'name') or g.get('name')
        if isinstance(name, bytes):
            name = name.decode('utf-8', errors='ignore')
        if name == group:
            return g
    return None


def _fmt_bytes(v):
    if isinstance(v, bytes):
        return v.decode('utf-8', errors='ignore')
    return v


def inspect_redis(r, db_idx: int) -> list[dict]:
    print("")
    print("=" * 70)
    print(f"📡 Redis 状态诊断 (host={REDIS_CFG['host']} port={REDIS_CFG['port']} db={db_idx})")
    print("=" * 70)
    report = []
    for stream, group in TARGET_GROUPS:
        row = {
            'stream': stream,
            'group':  group,
            'stream_exists': False,
            'group_exists':  False,
            'last_delivered_id': None,
            'pending':  None,
            'lag':      None,
            'stream_last_id': None,
            'stream_len':     None,
        }
        if not _stream_exists(r, stream):
            print(f"  🧊 [{stream:<28}] stream 不存在 (无积压风险)")
            report.append(row)
            continue
        row['stream_exists'] = True
        try:
            s_info = r.xinfo_stream(stream)
            row['stream_last_id'] = _fmt_bytes(s_info.get(b'last-generated-id') or s_info.get('last-generated-id'))
            row['stream_len']     = s_info.get(b'length') or s_info.get('length')
        except Exception:
            pass

        g = _find_group(r, stream, group)
        if g is None:
            print(f"  🟡 [{stream:<28}] 存在但组 '{group}' 未创建 (lag=全量, 由 OMS/SE 启动时自己处理)")
            report.append(row)
            continue
        row['group_exists']     = True
        row['last_delivered_id'] = _fmt_bytes(g.get(b'last-delivered-id') or g.get('last-delivered-id'))
        row['pending']           = g.get(b'pending')    or g.get('pending')
        row['lag']               = g.get(b'lag')        or g.get('lag')

        warn = ""
        try:
            if row['lag'] is not None and int(row['lag']) > 0:
                warn = f"  ⚠️ lag={row['lag']} 条历史信号积压"
        except Exception:
            pass
        try:
            if row['last_delivered_id'] and row['last_delivered_id'].startswith("0-0"):
                warn += "  ⚠️ last-delivered-id=0-0 (从未成功消费过, 重启会全量回放!)"
        except Exception:
            pass

        print(
            f"  🔎 [{stream:<28}] group={group:<18} "
            f"last-delivered={row['last_delivered_id']!s:<22} "
            f"pending={row['pending']!s:<5} lag={row['lag']!s:<6} "
            f"stream_last={row['stream_last_id']!s} len={row['stream_len']!s}"
            f"{warn}"
        )
        report.append(row)
    return report


def drop_backlog(r, report: list[dict], yes: bool) -> None:
    print("")
    print("=" * 70)
    print("🧹 操作: 推进消费者组 last-delivered-id 到 $ (丢弃启动前积压)")
    print("=" * 70)
    actionable = [row for row in report if row['group_exists']]
    if not actionable:
        print("  (无已存在的消费者组, 无需操作)")
        return
    for row in actionable:
        print(f"  → XGROUP SETID {row['stream']} {row['group']} $")
    if not yes:
        ans = input("确认执行? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  已取消。")
            return
    for row in actionable:
        try:
            r.xgroup_setid(row['stream'], row['group'], "$")
            print(f"  ✅ SETID 完成: {row['stream']} / {row['group']}")
        except Exception as e:
            print(f"  ❌ SETID 失败 {row['stream']} / {row['group']}: {e}")


# ============================================================
# PostgreSQL helpers
# ============================================================
def _pg_conn():
    return psycopg2.connect(PG_DB_URL)


def inspect_fcs_state() -> list[dict]:
    """诊断 fcs_state_snapshot 按 namespace 的快照状态."""
    out = []
    try:
        conn = _pg_conn(); c = conn.cursor()
        c.execute("SELECT to_regclass('public.fcs_state_snapshot')")
        if c.fetchone()[0] is None:
            conn.close()
            return out
        c.execute("""
            SELECT namespace,
                   COUNT(*) AS rows,
                   COUNT(*) FILTER (WHERE symbol <> '_META_') AS symbol_count,
                   MAX(schema_version) AS schema_version,
                   MAX(updated_at)     AS latest_saved
            FROM fcs_state_snapshot
            GROUP BY namespace
            ORDER BY MAX(updated_at) DESC
        """)
        rows = c.fetchall()
        conn.close()
    except Exception as e:
        print(f"  ❌ 无法查询 fcs_state_snapshot: {e}")
        return out

    now_ny_date = datetime.now(NY_TZ).date()
    print("")
    print("=" * 70)
    print("📦 FCS State Snapshot 诊断")
    print("=" * 70)
    if not rows:
        print("  (暂无 fcs_state_snapshot 数据)")
        return out
    for ns, n_rows, syms, ver, latest in rows:
        is_today = False
        age_h = None
        saved_str = "-"
        try:
            if latest is not None:
                ts_dt = datetime.fromtimestamp(float(latest), NY_TZ)
                is_today = (ts_dt.date() == now_ny_date)
                age_h = (datetime.now(NY_TZ).timestamp() - float(latest)) / 3600.0
                saved_str = ts_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception: pass
        tag = "当日" if is_today else "⚠️ 跨天"
        print(
            f"  ns={ns:<32} rows={n_rows:<4} syms={syms:<4} v{ver} "
            f"saved={saved_str} age={age_h if age_h is None else f'{age_h:.2f}h':<8} [{tag}]"
        )
        out.append({
            'namespace': ns, 'rows': n_rows, 'symbol_count': syms,
            'schema_version': ver, 'latest_saved': latest, 'is_today': is_today,
        })
    return out


def purge_fcs_stale(fcs_rows: list[dict], yes: bool) -> None:
    print("")
    print("=" * 70)
    print("🧹 操作: 删除 fcs_state_snapshot 中跨天 (非当日 NY 日) 的整个 namespace")
    print("=" * 70)
    stale = [r for r in fcs_rows if not r.get('is_today')]
    if not stale:
        print("  没有跨天 FCS 快照, 跳过。")
        return
    for r in stale:
        print(f"  → DELETE namespace={r['namespace']} rows={r['rows']}")
    if not yes:
        ans = input("确认删除? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  已取消。")
            return
    try:
        conn = _pg_conn(); c = conn.cursor()
        for r in stale:
            c.execute("DELETE FROM fcs_state_snapshot WHERE namespace = %s", (r['namespace'],))
        conn.commit()
        print(f"  ✅ 已删除 {len(stale)} 个跨天 namespace 的全部行")
        conn.close()
    except Exception as e:
        print(f"  ❌ 删除失败: {e}")


def inspect_pg() -> dict:
    print("")
    print("=" * 70)
    print(f"🐘 PostgreSQL 状态诊断 ({PG_DB_URL.split('host=')[-1].split()[0]})")
    print("=" * 70)
    result = {
        'table_exists':       False,
        'total_rows':          0,
        'global_state_row':    None,
        'stale_symbol_rows':   [],
        'today_symbol_rows':   0,
    }
    try:
        conn = _pg_conn()
        c = conn.cursor()
    except Exception as e:
        print(f"  ❌ 无法连接 PG: {e}")
        return result

    try:
        c.execute("SELECT to_regclass('public.symbol_state')")
        if c.fetchone()[0] is None:
            print("  🧊 symbol_state 表不存在 (首次启动即自动创建)")
            conn.close()
            return result
        result['table_exists'] = True

        c.execute("SELECT COUNT(*) FROM symbol_state")
        result['total_rows'] = c.fetchone()[0]

        now_ny_date = datetime.now(NY_TZ).date()

        c.execute("SELECT symbol, updated_at FROM symbol_state ORDER BY updated_at DESC")
        rows = c.fetchall()

        for sym, updated_at in rows:
            try:
                ts_dt = datetime.fromtimestamp(float(updated_at), NY_TZ).date()
            except Exception:
                ts_dt = None
            is_today = (ts_dt == now_ny_date)
            if sym == '_GLOBAL_STATE_':
                result['global_state_row'] = {
                    'symbol':     sym,
                    'updated_at': updated_at,
                    'ts_ny':      str(ts_dt),
                    'is_today':   is_today,
                }
            else:
                if is_today:
                    result['today_symbol_rows'] += 1
                else:
                    result['stale_symbol_rows'].append((sym, ts_dt))

        print(f"  📦 总行数: {result['total_rows']}")
        print(f"  📅 当日 (NY={now_ny_date}) symbol 状态行: {result['today_symbol_rows']}")
        print(f"  🕰  跨天残留 symbol 状态行: {len(result['stale_symbol_rows'])}")
        if result['global_state_row']:
            gs = result['global_state_row']
            tag = "当日有效" if gs['is_today'] else "⚠️ 跨天残留 (会污染 mock_cash)"
            print(f"  💰 _GLOBAL_STATE_: updated_at={gs['updated_at']} (NY={gs['ts_ny']})  [{tag}]")
        else:
            print("  💰 _GLOBAL_STATE_: 不存在 (fresh)")

        if result['stale_symbol_rows'][:5]:
            preview = ", ".join(f"{s}@{d}" for s, d in result['stale_symbol_rows'][:5])
            suffix = " ..." if len(result['stale_symbol_rows']) > 5 else ""
            print(f"     stale 预览: {preview}{suffix}")

        conn.close()
    except Exception as e:
        print(f"  ❌ PG 诊断失败: {e}")
        try: conn.close()
        except Exception: pass
    return result


def reset_global_state(pg_report: dict, yes: bool) -> None:
    print("")
    print("=" * 70)
    print("🧹 操作: 清除 PG 中 _GLOBAL_STATE_ 行 (修复 mock_cash 漂移)")
    print("=" * 70)
    if not pg_report.get('table_exists'):
        print("  symbol_state 表不存在, 跳过。")
        return
    gs = pg_report.get('global_state_row')
    if not gs:
        print("  _GLOBAL_STATE_ 不存在, 无需清理。")
        return
    if gs.get('is_today'):
        print(f"  ℹ️ _GLOBAL_STATE_ 是当日数据 (ts={gs['ts_ny']})。")
        print("  ℹ️ 清理会使 OMS 重启后以 INITIAL_ACCOUNT 作为 mock_cash。")
    if not yes:
        ans = input("确认删除 _GLOBAL_STATE_ ? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  已取消。")
            return
    try:
        conn = _pg_conn()
        c = conn.cursor()
        c.execute("DELETE FROM symbol_state WHERE symbol = %s", ('_GLOBAL_STATE_',))
        conn.commit()
        print(f"  ✅ 已删除 _GLOBAL_STATE_ ({c.rowcount} 行)")
        conn.close()
    except Exception as e:
        print(f"  ❌ 删除失败: {e}")


def purge_stale(pg_report: dict, yes: bool) -> None:
    print("")
    print("=" * 70)
    print("🧹 操作: 删除 PG 中跨天 (非当日 NY 日) 的 symbol_state 行")
    print("=" * 70)
    if not pg_report.get('table_exists'):
        print("  symbol_state 表不存在, 跳过。")
        return
    stale = pg_report.get('stale_symbol_rows') or []
    if not stale:
        print("  没有跨天残留, 跳过。")
        return
    print(f"  将删除 {len(stale)} 行 (不含 _GLOBAL_STATE_, _GLOBAL_STATE_ 走 --reset-global-state)")
    if not yes:
        ans = input("确认删除? [y/N]: ").strip().lower()
        if ans != 'y':
            print("  已取消。")
            return
    try:
        conn = _pg_conn()
        c = conn.cursor()
        now_ny_date = datetime.now(NY_TZ).date()
        # 取当天 NY 00:00 的 epoch, 删除 updated_at 小于它且 symbol != _GLOBAL_STATE_ 的行
        start_of_day = NY_TZ.localize(datetime.combine(now_ny_date, datetime.min.time())).timestamp()
        c.execute(
            "DELETE FROM symbol_state WHERE symbol <> %s AND updated_at < %s",
            ('_GLOBAL_STATE_', start_of_day),
        )
        conn.commit()
        print(f"  ✅ 已删除跨天 symbol_state {c.rowcount} 行")
        conn.close()
    except Exception as e:
        print(f"  ❌ 删除失败: {e}")


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--redis-db', type=int, default=None, help='强制指定 Redis DB (默认读 config)')
    parser.add_argument('--drop-backlog', action='store_true', help='推进消费者组 last-delivered-id 到 $')
    parser.add_argument('--reset-global-state', action='store_true', help='清除 PG 中 _GLOBAL_STATE_ 行')
    parser.add_argument('--purge-stale', action='store_true', help='删除 PG 中跨天 symbol_state 行')
    parser.add_argument('--purge-fcs-stale', action='store_true', help='删除 fcs_state_snapshot 中跨天的 namespace (FCS 快照)')
    parser.add_argument('--all', action='store_true', help='等价于 --drop-backlog --reset-global-state --purge-stale --purge-fcs-stale')
    parser.add_argument('--yes', action='store_true', help='跳过确认提示 (用于 crontab)')
    args = parser.parse_args()

    do_drop      = args.drop_backlog       or args.all
    do_reset_gs  = args.reset_global_state or args.all
    do_purge     = args.purge_stale        or args.all
    do_purge_fcs = args.purge_fcs_stale    or args.all

    print("")
    print(f"🚦 OMS 状态诊断 / 清理工具   RUN_MODE={RUN_MODE}")
    print(f"   时间 (NY): {datetime.now(NY_TZ).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    mode_desc = "dry-run (仅诊断, 不做任何变更)" if not (do_drop or do_reset_gs or do_purge or do_purge_fcs) else "执行清理"
    print(f"   模式      : {mode_desc}")

    db_idx = args.redis_db if args.redis_db is not None else REDIS_CFG['db']
    try:
        r = _build_redis(args.redis_db)
        r.ping()
    except Exception as e:
        print(f"[FATAL] 无法连接 Redis (db={db_idx}): {e}")
        sys.exit(3)

    redis_report = inspect_redis(r, db_idx)
    pg_report    = inspect_pg()
    fcs_rows     = inspect_fcs_state()

    if do_drop:
        drop_backlog(r, redis_report, args.yes)
    if do_reset_gs:
        reset_global_state(pg_report, args.yes)
    if do_purge:
        purge_stale(pg_report, args.yes)
    if do_purge_fcs:
        purge_fcs_stale(fcs_rows, args.yes)

    if do_drop or do_reset_gs or do_purge or do_purge_fcs:
        print("")
        print("=" * 70)
        print("🔁 清理完成后重新诊断 (verify)")
        print("=" * 70)
        inspect_redis(r, db_idx)
        inspect_pg()
        inspect_fcs_state()

    print("")
    print("✅ 完成。")
    print("")


if __name__ == '__main__':
    main()
