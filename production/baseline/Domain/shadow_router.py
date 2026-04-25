from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .adapters import (
    alpha_frame_from_legacy,
    execution_quote_from_legacy_payload,
    position_snapshot_from_legacy_state,
)

logger = logging.getLogger("DomainShadowRouter")


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class DomainShadowRouter:
    """Sidecar validation route for Domain models.

    设计目标：
    - 默认关闭，主线零影响
    - 打开后只做旁路转换 / validate / 可选落样本
    - 任意异常都吞掉并打日志，不允许影响交易主链路
    """

    def __init__(
        self,
        *,
        enabled: Optional[bool] = None,
        dump_dir: Optional[str] = None,
        dump_payloads: Optional[bool] = None,
        ok_log_every: Optional[int] = None,
    ) -> None:
        self.enabled = _env_flag("DOMAIN_SHADOW_ROUTER_ENABLE", False) if enabled is None else bool(enabled)
        self.dump_payloads = _env_flag("DOMAIN_SHADOW_ROUTER_DUMP_PAYLOADS", False) if dump_payloads is None else bool(dump_payloads)
        self.ok_log_every = max(
            1,
            int(
                os.environ.get("DOMAIN_SHADOW_ROUTER_LOG_OK_EVERY", "50")
                if ok_log_every is None
                else ok_log_every
            ),
        )
        raw_dump_dir = dump_dir if dump_dir is not None else os.environ.get("DOMAIN_SHADOW_ROUTER_DUMP_DIR", "").strip()
        self.dump_dir = Path(raw_dump_dir).expanduser() if raw_dump_dir else None
        self._lock = threading.Lock()
        self._counts: Dict[str, Dict[str, int]] = {
            "alpha_frame": {"ok": 0, "error": 0},
            "execution_quote": {"ok": 0, "error": 0},
            "position_state": {"ok": 0, "error": 0},
        }

    def _bump(self, kind: str, has_errors: bool) -> Dict[str, int]:
        with self._lock:
            bucket = self._counts.setdefault(kind, {"ok": 0, "error": 0})
            bucket["error" if has_errors else "ok"] += 1
            return dict(bucket)

    def stats(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {k: dict(v) for k, v in self._counts.items()}

    def _maybe_dump(self, kind: str, identity: str, raw_payload: Any, converted_payload: Optional[Dict[str, Any]], errors: list[str]) -> None:
        if not self.enabled or not self.dump_payloads or self.dump_dir is None:
            return
        try:
            target_dir = self.dump_dir / kind
            target_dir.mkdir(parents=True, exist_ok=True)
            safe_identity = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (identity or "unknown"))[:80]
            ts_ms = int(time.time() * 1000.0)
            out_path = target_dir / f"{ts_ms}_{safe_identity}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "kind": kind,
                        "identity": identity,
                        "errors": list(errors or []),
                        "raw_payload": raw_payload,
                        "converted_payload": converted_payload,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning(f"[DomainShadow] failed to dump {kind} sample: {exc}")

    def _log_result(self, kind: str, identity: str, errors: list[str], detail: str) -> None:
        counts = self._bump(kind, bool(errors))
        if errors:
            logger.warning(
                f"[DomainShadow] {kind} invalid identity={identity or 'N/A'} "
                f"errors={len(errors)} detail={detail} first={errors[0]}"
            )
            return
        if counts["ok"] == 1 or counts["ok"] % self.ok_log_every == 0:
            logger.info(
                f"[DomainShadow] {kind} ok identity={identity or 'N/A'} "
                f"ok={counts['ok']} error={counts['error']} detail={detail}"
            )

    def on_alpha_frame(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            frame = alpha_frame_from_legacy(payload)
            errors = frame.validate()
            identity = frame.frame_id or str(frame.minute_ts or "")
            detail = f"minute_ts={frame.minute_ts} items={len(frame.items)}"
            self._log_result("alpha_frame", identity, errors, detail)
            self._maybe_dump("alpha_frame", identity, dict(payload), frame.to_dict(), errors)
        except Exception as exc:
            logger.warning(f"[DomainShadow] alpha_frame route failed: {exc}")

    def on_execution_quote(self, symbol: str, payload: Mapping[str, Any], *, legacy_position: Optional[int] = None) -> None:
        if not self.enabled:
            return
        try:
            quote = execution_quote_from_legacy_payload(
                symbol,
                payload,
                legacy_position=legacy_position,
                fallback_ts=float(payload.get("ts", 0.0) or 0.0),
            )
            errors = quote.validate()
            identity = f"{symbol}_{int(float(quote.ts or 0.0))}"
            detail = (
                f"ts={quote.ts:.3f} bid={quote.best_bid:.4f} ask={quote.best_ask:.4f} "
                f"source={quote.source_kind.value}"
            )
            self._log_result("execution_quote", identity, errors, detail)
            self._maybe_dump("execution_quote", identity, dict(payload), quote.to_dict(), errors)
        except Exception as exc:
            logger.warning(f"[DomainShadow] execution_quote route failed for {symbol}: {exc}")

    def on_state_snapshot(self, state_data: Mapping[str, Any], *, namespace: str = "", run_mode: str = "") -> None:
        if not self.enabled:
            return
        for sym, row in state_data.items():
            if sym == "_GLOBAL_STATE_" or not isinstance(row, Mapping):
                continue
            try:
                position = position_snapshot_from_legacy_state(row)
                errors = position.validate()
                identity = f"{namespace}:{sym}" if namespace else sym
                detail = (
                    f"run_mode={run_mode or 'N/A'} side={position.side.value} "
                    f"qty={position.quantity:.4f} is_open={position.is_open}"
                )
                self._log_result("position_state", identity, errors, detail)
                self._maybe_dump("position_state", identity, dict(row), position.to_dict(), errors)
            except Exception as exc:
                logger.warning(f"[DomainShadow] position_state route failed for {sym}: {exc}")


_ROUTER_SINGLETON: Optional[DomainShadowRouter] = None


def get_domain_shadow_router() -> DomainShadowRouter:
    global _ROUTER_SINGLETON
    if _ROUTER_SINGLETON is None:
        _ROUTER_SINGLETON = DomainShadowRouter()
    return _ROUTER_SINGLETON


__all__ = ["DomainShadowRouter", "get_domain_shadow_router"]
