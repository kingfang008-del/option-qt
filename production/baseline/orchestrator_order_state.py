import json
import logging
import threading
import time

import psycopg2
import redis
from psycopg2 import extras

from config import OMS_STATE_NAMESPACE, PG_DB_URL, REDIS_CFG, RUN_MODE


logger = logging.getLogger("V8_Orchestrator.OrderState")

ORDER_STATE_TABLE = "oms_order_state"
ORDER_STATE_REDIS_KEY = "oms:pending_orders"
TERMINAL_ORDER_STATUSES = {
    "FILLED",
    "CANCELLED",
    "REJECTED",
    "ERROR",
    "EXPIRED",
    "INACTIVE",
    "API_CANCELLED",
}


def namespaced_pending_orders_key(namespace: str = OMS_STATE_NAMESPACE) -> str:
    ns = str(namespace or "").strip()
    if not ns:
        return ORDER_STATE_REDIS_KEY
    return f"{ORDER_STATE_REDIS_KEY}:{ns}"


class OrchestratorOrderStateManager:
    def __init__(self, orchestrator):
        self.orch = orchestrator
        self.state_namespace = OMS_STATE_NAMESPACE
        self.redis_key = namespaced_pending_orders_key(self.state_namespace)
        if not hasattr(self.orch, "pending_orders") or not isinstance(getattr(self.orch, "pending_orders", None), dict):
            self.orch.pending_orders = {}

    def _get_pg_conn(self):
        return psycopg2.connect(PG_DB_URL)

    def _get_redis(self):
        return redis.Redis(**{k: v for k, v in REDIS_CFG.items() if k in ["host", "port", "db"]}, decode_responses=False)

    def _init_db(self):
        if getattr(self.orch, "mode", "") == "backtest":
            return
        try:
            conn = self._get_pg_conn()
            c = conn.cursor()
            c.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {ORDER_STATE_TABLE} (
                    namespace TEXT NOT NULL,
                    order_key TEXT NOT NULL,
                    symbol TEXT,
                    intent TEXT,
                    status TEXT,
                    is_terminal BOOLEAN DEFAULT FALSE,
                    data TEXT,
                    updated_at DOUBLE PRECISION,
                    PRIMARY KEY (namespace, order_key)
                )
                """
            )
            c.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{ORDER_STATE_TABLE}_ns_updated
                ON {ORDER_STATE_TABLE} (namespace, updated_at)
                """
            )
            c.execute(
                f"""
                CREATE INDEX IF NOT EXISTS idx_{ORDER_STATE_TABLE}_ns_terminal
                ON {ORDER_STATE_TABLE} (namespace, is_terminal, updated_at)
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"❌ Order state DB init failed: {e}")

    def restore_active_orders(self):
        if getattr(self.orch, "mode", "") == "backtest":
            self.orch.pending_orders = {}
            return {}
        try:
            self._init_db()
            conn = self._get_pg_conn()
            c = conn.cursor()
            c.execute(
                f"""
                SELECT order_key, data
                FROM {ORDER_STATE_TABLE}
                WHERE namespace = %s AND is_terminal = FALSE
                ORDER BY updated_at DESC
                """,
                (self.state_namespace,),
            )
            rows = c.fetchall()
            conn.close()
            restored = {}
            for order_key, data_txt in rows:
                try:
                    payload = json.loads(data_txt)
                except Exception:
                    continue
                restored[str(order_key)] = payload
            self.orch.pending_orders = restored
            self.publish_active_snapshot()
            return restored
        except Exception as e:
            logger.warning(f"⚠️ restore_active_orders failed: {e}")
            return {}

    def _save_payload_async(self, order_key: str, payload: dict, is_terminal: bool):
        def _write_task():
            try:
                self._init_db()
                conn = self._get_pg_conn()
                c = conn.cursor()
                extras.execute_batch(
                    c,
                    f"""
                    INSERT INTO {ORDER_STATE_TABLE}
                        (namespace, order_key, symbol, intent, status, is_terminal, data, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (namespace, order_key) DO UPDATE
                    SET symbol = EXCLUDED.symbol,
                        intent = EXCLUDED.intent,
                        status = EXCLUDED.status,
                        is_terminal = EXCLUDED.is_terminal,
                        data = EXCLUDED.data,
                        updated_at = EXCLUDED.updated_at
                    WHERE {ORDER_STATE_TABLE}.updated_at IS NULL
                       OR {ORDER_STATE_TABLE}.updated_at <= EXCLUDED.updated_at
                    """,
                    [(
                        self.state_namespace,
                        str(order_key),
                        str(payload.get("symbol", "") or ""),
                        str(payload.get("intent", "") or ""),
                        str(payload.get("status", "") or ""),
                        bool(is_terminal),
                        json.dumps(payload, ensure_ascii=True),
                        float(payload.get("last_update_ts", time.time()) or time.time()),
                    )],
                )
                conn.commit()
                conn.close()
            except Exception as e:
                logger.error(f"❌ Order state save failed for {order_key}: {e}")

        threading.Thread(target=_write_task, daemon=True).start()

    def publish_active_snapshot(self):
        try:
            client = self._get_redis()
            pipe = client.pipeline()
            pipe.delete(self.redis_key)
            active_map = {}
            for order_key, payload in (getattr(self.orch, "pending_orders", {}) or {}).items():
                if not isinstance(payload, dict):
                    continue
                active_map[str(order_key)] = json.dumps(payload, ensure_ascii=True)
            if active_map:
                pipe.hset(self.redis_key, mapping=active_map)
            pipe.expire(self.redis_key, 24 * 3600)
            pipe.execute()
        except Exception as e:
            logger.warning(f"⚠️ publish_active_snapshot failed: {e}")

    def upsert(self, order_key: str, payload: dict):
        if not order_key or not isinstance(payload, dict):
            return
        merged = dict(getattr(self.orch, "pending_orders", {}).get(order_key, {}) or {})
        merged.update(payload)
        merged["order_key"] = str(order_key)
        merged["run_mode"] = RUN_MODE
        merged["state_namespace"] = self.state_namespace
        merged["last_update_ts"] = float(merged.get("last_update_ts", time.time()) or time.time())
        status = str(merged.get("status", "") or "").upper()
        merged["status"] = status
        is_terminal = bool(merged.get("is_terminal", False) or status in TERMINAL_ORDER_STATUSES)
        merged["is_terminal"] = is_terminal

        if is_terminal:
            self.orch.pending_orders.pop(order_key, None)
        else:
            self.orch.pending_orders[str(order_key)] = merged

        self._save_payload_async(str(order_key), merged, is_terminal=is_terminal)
        self.publish_active_snapshot()
