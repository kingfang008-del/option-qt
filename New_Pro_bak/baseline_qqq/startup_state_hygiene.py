#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文件名: startup_state_hygiene.py
描述:
    启动期状态卫生 (Startup State Hygiene)。

    新架构原则:
    - OMS 是唯一交易状态权威；Redis 中的 `oms:live_positions` 只允许作为
      Dashboard/诊断的只读投影。
    - SE 启动绝不能再根据 Redis 心跳/仓位去清理 PG `symbol_state` 或影响
      OMS 持仓/现金状态。
    - OMS 启动可以清理 Redis 投影，随后由 OMS 自己重新发布 fresh projection。
    - 仿真/回测模式 (IS_SIMULATED) 下彻底跳过, 避免影响回放流水线.

    对外接口:
    - run_startup_cleanup(role='se'|'oms', dry_run=False)
"""

import json
import logging
import os
import time
from typing import Optional

logger = logging.getLogger("StartupStateHygiene")

OMS_LIVE_POSITIONS_KEY = "oms:live_positions"
SYSTEM_CASH_FIELD = "____SYSTEM_CASH____"
DEFAULT_HEARTBEAT_TTL_SEC = 30.0


def _make_redis_client():
    """按当前运行时配置构造一个 Redis 客户端"""
    import redis
    from config import REDIS_CFG
    return redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ("host", "port", "db")})


def _read_system_cash_ts(r) -> Optional[float]:
    """读取 OMS 最近一次广播的 __SYSTEM_CASH__.ts"""
    try:
        raw = r.hget(OMS_LIVE_POSITIONS_KEY, SYSTEM_CASH_FIELD)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        ts = data.get("ts")
        return float(ts) if ts is not None else None
    except Exception as e:
        logger.warning(f"[Hygiene] 读取 OMS 心跳失败: {e}")
        return None


def _heartbeat_ttl_sec() -> float:
    try:
        return float(os.environ.get("OMS_HEARTBEAT_TTL_SEC", DEFAULT_HEARTBEAT_TTL_SEC))
    except (TypeError, ValueError):
        return DEFAULT_HEARTBEAT_TTL_SEC


def is_oms_heartbeat_fresh(r, ttl_sec: Optional[float] = None) -> bool:
    """OMS 在 TTL 窗口内是否仍在广播"""
    ttl = ttl_sec if ttl_sec is not None else _heartbeat_ttl_sec()
    ts = _read_system_cash_ts(r)
    if ts is None:
        return False
    age = time.time() - ts
    return age <= ttl


def _count_redis_phantoms(r) -> int:
    try:
        total = r.hlen(OMS_LIVE_POSITIONS_KEY) or 0
        if r.hexists(OMS_LIVE_POSITIONS_KEY, SYSTEM_CASH_FIELD):
            total = max(total - 1, 0)
        return int(total)
    except Exception as e:
        logger.warning(f"[Hygiene] 统计 Redis 幽灵仓位失败: {e}")
        return 0


def _clean_redis_positions(r, dry_run: bool) -> int:
    """删除整张 oms:live_positions hash. 返回被清理的 symbol 数量 (不含 SYSTEM_CASH)."""
    n = _count_redis_phantoms(r)
    if n == 0 and not r.exists(OMS_LIVE_POSITIONS_KEY):
        return 0
    if dry_run:
        logger.info(f"[Hygiene][DRY] 将删除 Redis 键 {OMS_LIVE_POSITIONS_KEY} (含 {n} 个幽灵 symbol).")
        return n
    try:
        r.delete(OMS_LIVE_POSITIONS_KEY)
        logger.info(f"🧹 [Hygiene] 已删除 Redis 键 {OMS_LIVE_POSITIONS_KEY} (清理 {n} 个幽灵 symbol).")
    except Exception as e:
        logger.error(f"[Hygiene] 删除 Redis 幽灵仓位失败: {e}")
    return n


def _clean_pg_phantom_rows(dry_run: bool) -> int:
    """清理 PG `symbol_state` 表中 position != 0 的当日陈旧行.

    说明:
    - 只清 position != 0 的行; warmup buffer (position == 0) 保留, 避免冷启动重算 Alpha.
    - 保留 _GLOBAL_STATE_ (mock_cash 等), 不动.
    - 返回被清理的行数.
    """
    try:
        import psycopg2
        from config import PG_DB_URL, OMS_STATE_NAMESPACE
    except Exception as e:
        logger.error(f"[Hygiene] 无法 import psycopg2/config: {e}")
        return 0

    n_affected = 0
    conn = None
    try:
        conn = psycopg2.connect(PG_DB_URL)
        c = conn.cursor()

        c.execute("SELECT to_regclass('public.symbol_state')")
        if c.fetchone()[0] is None:
            logger.info("[Hygiene] PG 表 symbol_state 不存在, 跳过 PG 清理.")
            return 0

        c.execute("""
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='symbol_state' AND column_name='namespace'
        """)
        has_namespace = c.fetchone() is not None
        ns_filter = "namespace = %s AND " if has_namespace else ""
        ns_params = (OMS_STATE_NAMESPACE,) if has_namespace else ()

        c.execute(f"""
            SELECT symbol, data
            FROM symbol_state
            WHERE {ns_filter}symbol <> '_GLOBAL_STATE_'
        """, ns_params)
        rows = c.fetchall()

        phantom_symbols = []
        for sym, data_str in rows:
            try:
                data = json.loads(data_str) if isinstance(data_str, str) else (data_str or {})
                pos = int(data.get("position", 0) or 0)
                if pos != 0:
                    phantom_symbols.append((sym, pos, data.get("entry_price", 0.0)))
            except Exception:
                continue

        if not phantom_symbols:
            logger.info("[Hygiene] PG symbol_state 无 position != 0 的幽灵行.")
            return 0

        preview = ", ".join(f"{s}(pos={p},ep={ep})" for s, p, ep in phantom_symbols[:8])
        more = "" if len(phantom_symbols) <= 8 else f" ...(+{len(phantom_symbols) - 8})"
        logger.warning(f"[Hygiene] 发现 PG 幽灵仓位 {len(phantom_symbols)} 个: {preview}{more}")

        if dry_run:
            logger.info("[Hygiene][DRY] 将 UPDATE 这些行的 data.position=0 (保留 warmup buffer).")
            return len(phantom_symbols)

        import psycopg2.extras
        payloads = []
        ts_now = time.time()
        for sym, _pos, _ep in phantom_symbols:
            if has_namespace:
                c.execute(
                    "SELECT data FROM symbol_state WHERE namespace=%s AND symbol=%s",
                    (OMS_STATE_NAMESPACE, sym),
                )
            else:
                c.execute("SELECT data FROM symbol_state WHERE symbol=%s", (sym,))
            row = c.fetchone()
            if not row:
                continue
            try:
                d = json.loads(row[0]) if isinstance(row[0], str) else (row[0] or {})
            except Exception:
                d = {}
            d["position"] = 0
            d["qty"] = 0
            d["entry_price"] = 0.0
            d["entry_stock"] = 0.0
            d["entry_ts"] = 0.0
            d["max_roi"] = -1.0
            payloads.append((sym, json.dumps(d), ts_now))

        if payloads:
            if has_namespace:
                psycopg2.extras.execute_batch(c, """
                    UPDATE symbol_state
                    SET data=%s, updated_at=%s
                    WHERE namespace=%s AND symbol=%s
                """, [(p[1], p[2], OMS_STATE_NAMESPACE, p[0]) for p in payloads])
            else:
                psycopg2.extras.execute_batch(c, """
                    UPDATE symbol_state SET data=%s, updated_at=%s WHERE symbol=%s
                """, [(p[1], p[2], p[0]) for p in payloads])
            conn.commit()
            n_affected = len(payloads)
            logger.info(f"🧹 [Hygiene] 已将 {n_affected} 个 PG 幽灵仓位置零 (保留 warmup).")

    except Exception as e:
        logger.error(f"[Hygiene] 清理 PG 幽灵仓位失败: {e}")
        try:
            if conn:
                conn.rollback()
        except Exception:
            pass
    finally:
        try:
            if conn:
                conn.close()
        except Exception:
            pass
    return n_affected


def run_startup_cleanup(role: str = "se", dry_run: bool = False) -> dict:
    """启动期幽灵仓位清理入口.

    参数:
        role: 'se'  — Signal Engine 启动前调用；不读/不清交易状态。
              'oms' — Execution Engine 启动前调用；只清 Redis projection，不动 PG。
        dry_run: True 仅预演, 不实际写入.

    返回: {'skipped': bool, 'reason': str, 'heartbeat_age_sec': float|None,
           'redis_cleared': int, 'pg_cleared': int}
    """
    result = {
        "skipped": False,
        "reason": "",
        "heartbeat_age_sec": None,
        "redis_cleared": 0,
        "pg_cleared": 0,
    }

    role = (role or "se").lower()
    if role not in ("se", "oms"):
        result["skipped"] = True
        result["reason"] = f"unknown_role={role}"
        logger.warning(f"[Hygiene] 未知角色 {role}, 跳过.")
        return result

    if os.environ.get("SKIP_STARTUP_CLEANUP", "").strip().lower() in ("1", "true", "yes"):
        result["skipped"] = True
        result["reason"] = "SKIP_STARTUP_CLEANUP=1"
        logger.info("[Hygiene] SKIP_STARTUP_CLEANUP=1, 跳过启动期清理.")
        return result

    try:
        from config import IS_SIMULATED, RUN_MODE
    except Exception as e:
        result["skipped"] = True
        result["reason"] = f"config_import_failed:{e}"
        logger.error(f"[Hygiene] import config 失败, 跳过: {e}")
        return result

    if IS_SIMULATED:
        result["skipped"] = True
        result["reason"] = f"simulated_mode={RUN_MODE}"
        logger.info(f"[Hygiene] 仿真模式 ({RUN_MODE}) 跳过启动期清理.")
        return result

    if role == "se":
        result["skipped"] = True
        result["reason"] = "se_no_trading_state_cleanup"
        logger.info(
            "[Hygiene] SE no longer participates in trading-state cleanup; "
            "OMS memory/PG snapshot remain the only trading-state authority."
        )
        return result

    try:
        r = _make_redis_client()
    except Exception as e:
        result["skipped"] = True
        result["reason"] = f"redis_init_failed:{e}"
        logger.error(f"[Hygiene] 初始化 Redis 失败, 跳过: {e}")
        return result

    ts = _read_system_cash_ts(r)
    ttl = _heartbeat_ttl_sec()
    if ts is not None:
        age = time.time() - ts
        result["heartbeat_age_sec"] = round(age, 2)
    else:
        age = None

    logger.info(
        f"[Hygiene] role={role} | RUN_MODE={RUN_MODE} | "
        f"OMS heartbeat age={age if age is None else f'{age:.1f}s'} | ttl={ttl:.0f}s | "
        f"dry_run={dry_run}"
    )

    if role == "oms":
        # OMS 启动期: 只清 Redis 只读投影，OMS 会在首次 broadcast 重建。
        # PG symbol_state 是 OMS 自己的恢复快照，绝不能由 Redis 心跳间接清理。
        result["redis_cleared"] = _clean_redis_positions(r, dry_run=dry_run)
        result["reason"] = "oms_projection_fresh_start"
        return result

    return result


if __name__ == "__main__":
    # CLI: python startup_state_hygiene.py [--role se|oms] [--dry-run]
    import argparse
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [Hygiene] - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="启动期幽灵仓位清理")
    parser.add_argument("--role", choices=["se", "oms"], default="se")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_startup_cleanup(role=args.role, dry_run=args.dry_run)
