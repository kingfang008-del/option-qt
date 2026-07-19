"""Read-only data sources for the repository-wide operations dashboard."""
from __future__ import annotations

import json
import pickle
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from maga7.common.provenance import code_fingerprint

REPO = Path(__file__).resolve().parents[1]
MAGA7_RESULTS = REPO / "maga7" / "results"
PROD_PROFILE = (
    REPO
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)

SHARED_STREAMS = (
    "fused_market_stream",
    "unified_inference_stream",
    "orch_trade_signals",
    "trade_log_stream",
)
LIVE_HASHES = (
    "live_ibkr_connector",
    "live_account_info",
    "meta:global_gates",
    "meta:oms_ledger",
    "monitor:warmup:norm",
    "monitor:warmup:orch",
)


@dataclass
class RunArtifact:
    path: Path
    name: str
    stage: str
    mtime: float
    summary: dict[str, Any] = field(default_factory=dict)
    compare: dict[str, Any] = field(default_factory=dict)
    parity: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool | None:
        if self.stage == "redis_replay":
            return bool(
                self.summary.get("frame_integrity_ok")
                and self.compare.get("ok")
                and self.summary.get("prefer_redis_quotes")
                and int(self.summary.get("n_path_disk") or 0) == 0
            )
        if self.stage == "stream_parity":
            return bool(self.parity.get("ok"))
        if self.stage in {"offline", "shadow", "dry"}:
            return bool(self.summary)
        return None


@dataclass
class LiveSessionArtifact:
    path: Path
    manifest: dict[str, Any]
    mtime: float
    event_counts: dict[str, int] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return str(self.path.relative_to(MAGA7_RESULTS))

    @property
    def mode(self) -> str:
        return str(self.manifest.get("mode") or "").lower()

    @property
    def base_ok(self) -> bool:
        connector = self.manifest.get("connector") or {}
        engine = self.manifest.get("engine_metrics") or {}
        oms = self.manifest.get("oms") or {}
        return bool(
            self.manifest.get("state") == "DONE"
            and not self.manifest.get("error")
            and connector.get("connected")
            and connector.get("data_mode") == "LIVE"
            and connector.get("lock_status") == "LOCKED"
            and int(connector.get("option_quote_symbols") or 0)
            >= int(connector.get("trade_symbols") or 1)
            and int(engine.get("frames") or 0) > 0
            and int(engine.get("rejected") or 0) == 0
            and int(engine.get("foreign") or 0) == 0
            and int(oms.get("positions") or 0) == 0
        )

    @property
    def broker_lifecycle_ok(self) -> bool:
        required = (
            "ORDER_SUBMITTED",
            "ORDER_STATUS",
            "FILL",
            "COMMISSION",
            "RECONCILE",
        )
        return bool(
            self.base_ok
            and (self.manifest.get("oms") or {}).get("reconcile_ok")
            and all(int(self.event_counts.get(kind) or 0) > 0 for kind in required)
        )


def discover_live_sessions(limit: int = 100) -> list[LiveSessionArtifact]:
    root = MAGA7_RESULTS / "live_sessions"
    if not root.is_dir():
        return []
    rows = []
    for path in root.rglob("manifest.json"):
        manifest = read_json(path)
        if not manifest:
            continue
        counts: dict[str, int] = {}
        event_path = path.parent / "order_events.jsonl"
        if event_path.is_file():
            try:
                with event_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            kind = str(json.loads(line).get("kind") or "")
                        except Exception:
                            continue
                        if kind:
                            counts[kind] = counts.get(kind, 0) + 1
            except OSError:
                pass
        rows.append(
            LiveSessionArtifact(
                path=path.parent,
                manifest=manifest,
                mtime=path.stat().st_mtime,
                event_counts=counts,
            )
        )
    return sorted(rows, key=lambda row: row.mtime, reverse=True)[:limit]


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _run_stage(path: Path, summary: dict[str, Any], parity: dict[str, Any]) -> str:
    mode = str(summary.get("mode") or "").upper()
    if mode == "MAG7_S5_REDIS" or "s5_" in path.as_posix().lower():
        return "redis_replay"
    if parity or "parity" in path.name.lower():
        return "stream_parity"
    if "SHADOW" in mode:
        return "shadow"
    if "DRY" in mode or "STUB" in mode:
        return "dry"
    return "offline"


def discover_maga7_runs(limit: int = 100) -> list[RunArtifact]:
    """Discover both legacy flat runs and new ``tag/run_id`` runs."""
    if not MAGA7_RESULTS.is_dir():
        return []
    dirs: set[Path] = set()
    for filename in ("summary.json", "parity_summary.json", "offline_summary.json"):
        for path in MAGA7_RESULTS.rglob(filename):
            if "_dash" not in path.parts:
                dirs.add(path.parent)
    rows: list[RunArtifact] = []
    for path in dirs:
        summary = read_json(path / "summary.json") or read_json(path / "offline_summary.json")
        parity = read_json(path / "parity_summary.json")
        compare = read_json(path / "compare_summary.json")
        try:
            mtime = max(
                p.stat().st_mtime
                for p in (
                    path / "summary.json",
                    path / "parity_summary.json",
                    path / "offline_summary.json",
                )
                if p.is_file()
            )
        except (OSError, ValueError):
            continue
        rows.append(
            RunArtifact(
                path=path,
                name=str(path.relative_to(MAGA7_RESULTS)),
                stage=_run_stage(path, summary, parity),
                mtime=mtime,
                summary=summary,
                compare=compare,
                parity=parity,
            )
        )
    return sorted(rows, key=lambda row: row.mtime, reverse=True)[:limit]


def latest_by_stage(runs: Iterable[RunArtifact]) -> dict[str, RunArtifact]:
    out: dict[str, RunArtifact] = {}
    for run in runs:
        out.setdefault(run.stage, run)
    return out


def _resolve_profile_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else REPO / path


def profile_snapshot(profile_path: Path = PROD_PROFILE) -> dict[str, Any]:
    profile = read_json(profile_path)
    paths = profile.get("paths") or {}
    path_rows = []
    for key, raw in paths.items():
        path = _resolve_profile_path(str(raw))
        detail = ""
        if key in {"open_locked_map", "locked_map"} and path.is_file():
            try:
                lock_df = pd.read_parquet(path)
                date_col = next(
                    (col for col in ("date", "trade_date", "session_date") if col in lock_df),
                    None,
                )
                max_date = str(lock_df[date_col].max()) if date_col and len(lock_df) else "-"
                detail = f"rows={len(lock_df)}, max_date={max_date}"
            except Exception as exc:
                detail = f"metadata error: {exc}"
        path_rows.append(
            {
                "name": key,
                "path": str(path),
                "exists": path.exists(),
                "size_mb": round(path.stat().st_size / 1e6, 2) if path.is_file() else None,
                "mtime": (
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
                    if path.exists()
                    else ""
                ),
                "detail": detail,
            }
        )
    return {
        "path": str(profile_path),
        "exists": profile_path.is_file(),
        "mtime": profile_path.stat().st_mtime if profile_path.is_file() else 0.0,
        "strategy_fingerprint": (
            code_fingerprint(profile_path) if profile_path.is_file() else ""
        ),
        "live_fingerprint": (
            code_fingerprint(profile_path, live=True) if profile_path.is_file() else ""
        ),
        "profile": profile,
        "paths": path_rows,
    }


def pipeline_gates(
    runs: list[RunArtifact],
    profile: dict[str, Any],
    live_sessions: list[LiveSessionArtifact] | None = None,
) -> list[dict[str, Any]]:
    strategy_fingerprint = str(profile.get("strategy_fingerprint") or "")
    live_fingerprint = str(profile.get("live_fingerprint") or "")
    parity = next(
        (
            run
            for run in runs
            if run.stage == "stream_parity"
            and run.ok
            and run.parity.get("strategy_fingerprint") == strategy_fingerprint
        ),
        None,
    )
    redis_run = next(
        (
            run
            for run in runs
            if run.stage == "redis_replay"
            and run.ok
            and run.summary.get("strategy_fingerprint") == strategy_fingerprint
            and run.compare.get("strategy_fingerprint") == strategy_fingerprint
        ),
        None,
    )
    path_rows = profile.get("paths") or []
    lock_ready = any(
        row["name"] == "open_locked_map" and row["exists"] for row in path_rows
    )
    quote_ready = any(
        row["name"] == "quote_1s_root" and row["exists"] for row in path_rows
    )
    live_sessions = live_sessions or discover_live_sessions()
    shadow = next(
        (
            session
            for session in live_sessions
            if session.mode == "shadow"
            and session.base_ok
            and session.manifest.get("live_fingerprint") == live_fingerprint
        ),
        None,
    )
    paper = next(
        (
            session
            for session in live_sessions
            if session.mode == "paper"
            and session.broker_lifecycle_ok
            and session.manifest.get("live_fingerprint") == live_fingerprint
        ),
        None,
    )
    live = next(
        (
            session
            for session in live_sessions
            if session.mode == "live"
            and session.broker_lifecycle_ok
            and session.manifest.get("live_fingerprint") == live_fingerprint
            and int(session.event_counts.get("POSITION_CLOSE") or 0) > 0
        ),
        None,
    )
    return [
        {
            "stage": "G0 配置/数据",
            "status": "PASS" if profile.get("exists") and lock_ready and quote_ready else "BLOCK",
            "evidence": "生产 profile + open_locked_map + quote_1s_root",
        },
        {
            "stage": "G1 Offline",
            "status": "PASS" if parity and parity.parity.get("n_offline", 0) else "BLOCK",
            "evidence": parity.name if parity else "缺少当前 profile 修改后的 offline 金标",
        },
        {
            "stage": "G2 Stream parity",
            "status": "PASS" if parity and parity.ok else "BLOCK",
            "evidence": parity.name if parity else "缺少当前 profile 的 parity_summary.json",
        },
        {
            "stage": "G3 Redis S5",
            "status": "PASS" if redis_run and redis_run.ok else "BLOCK",
            "evidence": redis_run.name if redis_run else "缺少当前 profile 的隔离 Redis 对拍",
        },
        {
            "stage": "G4 Shadow live",
            "status": "PASS" if shadow else "BLOCK",
            "evidence": shadow.name if shadow else "需要完整真实行情 Shadow session 证据",
        },
        {
            "stage": "G5 Paper broker",
            "status": "PASS" if paper and shadow else "BLOCK",
            "evidence": paper.name if paper and shadow else "需要 G4 + Paper order/status/fill/reconcile",
        },
        {
            "stage": "G6 Live",
            "status": "PASS" if live and shadow and paper else "BLOCK",
            "evidence": (
                live.name
                if live and shadow and paper
                else "需要 G4/G5 + 显式武装的小仓 Live 完整平仓证据"
            ),
        },
    ]


def _decode(value: Any) -> Any:
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            try:
                return pickle.loads(value)
            except Exception:
                return f"<bytes:{len(value)}>"
    return value


def stream_probe(client: Any, stream: str) -> dict[str, Any]:
    try:
        info = client.xinfo_stream(stream)
        groups = client.xinfo_groups(stream)
    except Exception as exc:
        return {"stream": stream, "error": str(exc)}
    last = info.get(b"last-entry") or info.get("last-entry")
    last_id = _decode(last[0]) if last else ""
    age = None
    if last_id:
        try:
            age = max(0.0, time.time() - int(str(last_id).split("-")[0]) / 1000.0)
        except Exception:
            age = None
    pending = sum(int(g.get(b"pending", g.get("pending", 0)) or 0) for g in groups)
    lag = sum(int(g.get(b"lag", g.get("lag", 0)) or 0) for g in groups)
    return {
        "stream": stream,
        "length": int(info.get(b"length", info.get("length", 0)) or 0),
        "last_id": last_id,
        "age_sec": round(age, 2) if age is not None else None,
        "groups": len(groups),
        "pending": pending,
        "lag": lag,
        "error": "",
    }


def redis_snapshot(client: Any, limit_runs: int = 20) -> dict[str, Any]:
    rows = [stream_probe(client, stream) for stream in SHARED_STREAMS]
    isolated: list[str] = []
    try:
        for key in client.scan_iter(match="fused_market_stream:maga7:*", count=100):
            isolated.append(str(_decode(key)))
    except Exception:
        pass
    isolated = sorted(isolated)[-limit_runs:]
    rows.extend(stream_probe(client, stream) for stream in isolated)

    statuses = []
    try:
        for key in client.scan_iter(match="replay:status:*", count=100):
            key_s = str(_decode(key))
            statuses.append({"key": key_s, "value": _decode(client.get(key))})
    except Exception:
        pass
    live_keys = list(LIVE_HASHES)
    live_keys.append("meta:runtime_trading_controls:maga7")
    try:
        for pattern in (
            "live_ibkr_connector:maga7:*",
            "maga7:live_engine:*",
            "oms:live_positions:maga7:*",
            "oms:pending_orders:maga7:*",
        ):
            live_keys.extend(
                str(_decode(key))
                for key in client.scan_iter(match=pattern, count=100)
            )
    except Exception:
        pass
    hashes = []
    for key in list(dict.fromkeys(live_keys))[-100:]:
        try:
            typ = _decode(client.type(key))
            if typ == "hash":
                raw = client.hgetall(key)
                sample = {
                    str(_decode(k)): str(_decode(v))[:200]
                    for k, v in list(raw.items())[:8]
                }
                count = len(raw)
            elif typ == "string":
                sample = {"value": str(_decode(client.get(key)))[:500]}
                count = 1
            else:
                sample = {}
                count = 0
            hashes.append(
                {
                    "key": key,
                    "type": typ,
                    "entries": count,
                    "sample": json.dumps(sample, ensure_ascii=False),
                }
            )
        except Exception as exc:
            hashes.append({"key": key, "type": "error", "entries": 0, "sample": str(exc)})
    return {"streams": rows, "statuses": statuses, "live_hashes": hashes}


def job_snapshot(limit: int = 30) -> list[dict[str, Any]]:
    """Discover existing QQQ/Mag7 dashboard job-state files."""
    roots = (REPO / "maga7" / "results", REPO / "qqq_btc" / "results")
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(root.rglob("*dash*job*.json"))
    rows = []
    for path in sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        state = read_json(path)
        rows.append(
            {
                "product": "maga7" if "maga7" in path.parts else "qqq_btc",
                "state_file": str(path.relative_to(REPO)),
                "status": state.get("status", "unknown"),
                "pid": state.get("pid", ""),
                "kind": state.get("kind", state.get("tag", "")),
                "started_at": state.get("started_at", ""),
                "finished_at": state.get("finished_at", ""),
                "log": state.get("log", state.get("log_file", "")),
            }
        )
    return rows


def process_snapshot() -> list[dict[str, Any]]:
    """Read-only process topology probe (no start/stop actions)."""
    patterns = {
        "IBKR connector": ("ibkr_connector",),
        "FCS": ("feature_compute_service",),
        "Signal engine": ("engine_v8.py", "run_live_exec_qqq"),
        "OMS": ("execution_engine", "run_oms"),
        "Mag7 Redis": ("run_maga7_redis_sim",),
        "Mag7 Live": ("run_live_session",),
        "Dashboard": ("streamlit", "dash/app.py"),
    }
    try:
        proc = subprocess.run(
            ["ps", "-eo", "pid=,etimes=,args="],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return [{"component": "process probe", "status": "ERROR", "command": str(exc)}]
    rows = []
    lines = proc.stdout.splitlines()
    for component, needles in patterns.items():
        matches = []
        for line in lines:
            if any(needle in line for needle in needles):
                parts = line.strip().split(maxsplit=2)
                if len(parts) == 3:
                    matches.append(parts)
        if not matches:
            rows.append(
                {
                    "component": component,
                    "status": "OFF",
                    "pid": "",
                    "age_sec": "",
                    "command": "",
                }
            )
            continue
        for pid, age, command in matches:
            rows.append(
                {
                    "component": component,
                    "status": "RUNNING",
                    "pid": pid,
                    "age_sec": age,
                    "command": command,
                }
            )
    return rows


def load_csv(path: Path, limit: int = 2000) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    return df.tail(limit).reset_index(drop=True)


def run_frames(run: RunArtifact) -> dict[str, pd.DataFrame]:
    return {
        "trades": load_csv(run.path / "trades.csv"),
        "signals": load_csv(run.path / "signals.csv"),
        "audit": load_csv(run.path / "fill_audit_live.csv"),
        "compare": load_csv(run.path / "compare_offline.csv"),
        "daily": load_csv(run.path / "daily.csv"),
    }


def live_session_frames(session: LiveSessionArtifact) -> dict[str, pd.DataFrame]:
    locks = read_json(session.path / "locks.json").get("locks") or {}
    lock_rows = [
        {"symbol": symbol, **row}
        for symbol, rows in locks.items()
        for row in rows
        if isinstance(row, dict)
    ]
    events = []
    path = session.path / "order_events.jsonl"
    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(row, dict):
                        events.append(row)
        except OSError:
            pass
    spreads = load_csv(session.path / "trade_spreads.csv")
    if spreads is None or spreads.empty:
        spreads = trade_spreads_from_events(events)
    return {
        "locks": pd.DataFrame(lock_rows),
        "events": pd.DataFrame(events[-2000:]),
        "signals": load_csv(session.path / "signals.csv"),
        "trade_spreads": spreads,
    }


def trade_spreads_from_events(events: list[dict]) -> pd.DataFrame:
    """Fallback: derive OPEN/CLOSE spread rows from order_events.jsonl."""
    kind_map = {
        "POSITION_OPEN": "OPEN",
        "POSITION_CLOSE": "CLOSE",
        "POSITION_PARTIAL_CLOSE": "PARTIAL_CLOSE",
    }
    rows = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        action = kind_map.get(str(ev.get("kind") or ""))
        if not action:
            continue
        bid = ev.get("bid", ev.get("entry_bid") if action == "OPEN" else ev.get("last_bid"))
        ask = ev.get("ask", ev.get("entry_ask") if action == "OPEN" else ev.get("last_ask"))
        fill_px = ev.get("fill_px", ev.get("exit_price", ev.get("entry_price")))
        rows.append(
            {
                "ts": ev.get("ts"),
                "session_id": ev.get("session_id"),
                "mode": ev.get("mode"),
                "action": action,
                "symbol": ev.get("symbol"),
                "contract": ev.get("contract"),
                "side": "BUY" if action == "OPEN" else "SELL",
                "qty": ev.get("qty") or ev.get("filled_qty"),
                "fill_px": fill_px,
                "bid": bid,
                "ask": ask,
                "spread": ev.get("spread"),
                "spread_pct": ev.get("spread_pct"),
                "fill_spread_frac": ev.get("fill_spread_frac"),
                "reason": ev.get("reason") or ("ENTRY" if action == "OPEN" else ""),
                "ret": ev.get("ret"),
            }
        )
    return pd.DataFrame(rows)


def _decode_jsonish(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    if isinstance(raw, memoryview):
        raw = raw.tobytes()
    if isinstance(raw, bytes):
        try:
            from maga7.live.redis_fused import unpack_obj

            return unpack_obj(raw)
        except Exception:
            try:
                return json.loads(raw.decode("utf-8", errors="replace"))
            except Exception:
                return None
    text = str(_decode(raw))
    try:
        return json.loads(text)
    except Exception:
        return None


def list_maga7_session_ids(client: Any) -> list[str]:
    """Discover live Mag7 session ids from Redis position/scanner keys."""
    ids: set[str] = set()
    try:
        for key in client.scan_iter(match="oms:live_positions:maga7:*", count=100):
            s = str(_decode(key))
            ids.add(s.rsplit(":", 1)[-1])
        for key in client.scan_iter(match="maga7:scanner_state:*", count=100):
            s = str(_decode(key))
            ids.add(s.rsplit(":", 1)[-1])
        for key in client.scan_iter(match="maga7:live_engine:*", count=100):
            s = str(_decode(key))
            ids.add(s.rsplit(":", 1)[-1])
    except Exception:
        pass
    # Prefer freshest live session dirs as fallback ordering hints.
    sessions = discover_live_sessions(limit=20)
    ordered = []
    for session in sessions:
        sid = str((session.manifest or {}).get("session_id") or session.path.name)
        if sid in ids and sid not in ordered:
            ordered.append(sid)
    for sid in sorted(ids):
        if sid not in ordered:
            ordered.append(sid)
    return ordered


def fetch_oms_positions(
    client: Any | None,
    *,
    session_id: str | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    """Load Mag7 OMS positions from Redis hash or oms_state.json."""
    meta: dict[str, Any] = {
        "source": None,
        "session_id": session_id,
        "updated_at": None,
        "day_halted": False,
        "equity": None,
        "realized_pnl": None,
        "mode": None,
    }
    positions: dict[str, dict[str, Any]] = {}
    intents: dict[str, dict[str, Any]] = {}

    if session_dir is not None:
        state = read_json(Path(session_dir) / "oms_state.json")
        if state:
            meta.update(
                {
                    "source": "disk:oms_state.json",
                    "session_id": state.get("session_id") or session_id,
                    "updated_at": state.get("updated_at"),
                    "day_halted": bool(state.get("day_halted")),
                    "equity": state.get("equity"),
                    "realized_pnl": state.get("realized_pnl"),
                    "mode": state.get("mode"),
                    "trade_date": state.get("trade_date"),
                    "reconcile_ok": state.get("reconcile_ok"),
                }
            )
            positions = {
                str(k): v for k, v in (state.get("positions") or {}).items() if isinstance(v, dict)
            }
            intents = {
                str(k): v for k, v in (state.get("intents") or {}).items() if isinstance(v, dict)
            }
            return {"meta": meta, "positions": positions, "intents": intents}

    if client is not None and session_id:
        try:
            raw_map = client.hgetall(f"oms:live_positions:maga7:{session_id}") or {}
            for field, raw in raw_map.items():
                payload = _decode_jsonish(raw)
                if isinstance(payload, dict):
                    positions[str(_decode(field))] = payload
            intent_map = client.hgetall(f"oms:pending_orders:maga7:{session_id}") or {}
            for field, raw in intent_map.items():
                payload = _decode_jsonish(raw)
                if isinstance(payload, dict):
                    intents[str(_decode(field))] = payload
            if positions or intents:
                meta["source"] = "redis:oms:live_positions"
                meta["session_id"] = session_id
        except Exception as exc:
            meta["error"] = str(exc)
    return {"meta": meta, "positions": positions, "intents": intents}


def fetch_scanner_state(
    client: Any | None,
    *,
    session_id: str | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    """Load Mag7 sliding-window scanner snapshot (mf10 / streak / bars)."""
    if session_dir is not None:
        state = read_json(Path(session_dir) / "scanner_state.json")
        if state:
            state = dict(state)
            state["_source"] = "disk:scanner_state.json"
            return state
    if client is not None and session_id:
        try:
            raw = client.get(f"maga7:scanner_state:{session_id}")
            state = _decode_jsonish(raw)
            if isinstance(state, dict):
                state = dict(state)
                state["_source"] = "redis:maga7:scanner_state"
                return state
        except Exception as exc:
            return {"_source": "redis_error", "error": str(exc)}
    return {}


def positions_frame(positions: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    now = time.time()
    for symbol, pos in (positions or {}).items():
        if not isinstance(pos, dict):
            continue
        status = str(pos.get("status") or "")
        if status and status not in {"OPEN", "EXIT_PENDING"}:
            # still show closed briefly if present
            pass
        entry_ts = pos.get("entry_ts")
        held_min = None
        try:
            if entry_ts is not None:
                held_min = round((now - float(entry_ts)) / 60.0, 2)
        except Exception:
            held_min = None
        entry = float(pos.get("entry_price") or 0.0)
        bid = float(pos.get("last_bid") or 0.0)
        ask = float(pos.get("last_ask") or 0.0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else float("nan")
        mtm = (mid / entry - 1.0) if entry > 0 and mid == mid else float("nan")
        rows.append(
            {
                "symbol": symbol,
                "dir": pos.get("direction"),
                "status": status or "OPEN",
                "contract": pos.get("contract"),
                "qty": pos.get("qty"),
                "qty_frac": pos.get("qty_frac"),
                "entry": entry or None,
                "last_bid": bid or None,
                "last_ask": ask or None,
                "mtm_ret": mtm if mtm == mtm else None,
                "held_min": held_min,
                "rank": pos.get("rank"),
                "entry_ts": entry_ts,
            }
        )
    return pd.DataFrame(rows)


def mf_window_frame(scanner_state: dict[str, Any], *, mf_fast_n: int = 3) -> pd.DataFrame:
    """Cross-section of sliding MF windows from scanner snapshot."""
    rows = []
    states = (scanner_state or {}).get("states") or {}
    for symbol, st in states.items():
        if not isinstance(st, dict):
            continue
        bars = st.get("bars") or []
        nets = []
        for bar in bars:
            if isinstance(bar, dict) and bar.get("net$") is not None:
                try:
                    nets.append(float(bar["net$"]))
                except Exception:
                    continue
        mf_fast = float("nan")
        if len(nets) >= mf_fast_n:
            mf_fast = float(sum(nets[-mf_fast_n:]))
        close = None
        if bars and isinstance(bars[-1], dict):
            close = bars[-1].get("close")
        rows.append(
            {
                "symbol": symbol,
                "date": st.get("date"),
                "mf10": st.get("mf10"),
                "mf_fast": mf_fast if mf_fast == mf_fast else None,
                "cum": st.get("cum"),
                "streak_up": st.get("streak_up"),
                "streak_dn": st.get("streak_dn"),
                "fired_today": st.get("fired_today"),
                "close": close,
                "n_bars": len(bars),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "mf10" in df.columns:
        df = df.sort_values("mf10", key=lambda s: s.abs(), ascending=False)
    return df.reset_index(drop=True)


def mf_series_for_symbol(scanner_state: dict[str, Any], symbol: str, *, mf_window: int = 10, mf_fast_n: int = 3) -> pd.DataFrame:
    """Per-bar sliding window series for one symbol (traceable chart)."""
    st = ((scanner_state or {}).get("states") or {}).get(str(symbol).upper())
    if not isinstance(st, dict):
        # try case-insensitive
        for key, value in ((scanner_state or {}).get("states") or {}).items():
            if str(key).upper() == str(symbol).upper():
                st = value
                break
    if not isinstance(st, dict):
        return pd.DataFrame()
    bars = st.get("bars") or []
    rows = []
    nets: list[float] = []
    for bar in bars:
        if not isinstance(bar, dict):
            continue
        try:
            net = float(bar.get("net$", float("nan")))
        except Exception:
            net = float("nan")
        nets.append(net)
        mf10 = float(sum(x for x in nets[-mf_window:] if x == x)) if len(nets) >= mf_window else float("nan")
        mf_fast = float(sum(x for x in nets[-mf_fast_n:] if x == x)) if len(nets) >= mf_fast_n else float("nan")
        rows.append(
            {
                "timestamp": bar.get("timestamp"),
                "close": bar.get("close"),
                "volume": bar.get("volume"),
                "net$": net if net == net else None,
                "mf10": mf10 if mf10 == mf10 else None,
                "mf_fast": mf_fast if mf_fast == mf_fast else None,
            }
        )
    return pd.DataFrame(rows)


def day_fires_frame(scanner_state: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for item in (scanner_state or {}).get("day_fires") or []:
        if isinstance(item, dict):
            rows.append(item)
    for item in (scanner_state or {}).get("signals") or []:
        if isinstance(item, dict):
            rows.append({**item, "_kind": "signal"})
    return pd.DataFrame(rows)


def resolve_live_trace_bundle(
    client: Any | None,
    *,
    session_id: str | None = None,
    prefer_disk: bool = False,
) -> dict[str, Any]:
    """Bundle positions + scanner state for dashboard live-trace tab."""
    sessions = discover_live_sessions(limit=30)
    session_dirs = {
        str((s.manifest or {}).get("session_id") or s.path.name): s.path for s in sessions
    }
    redis_ids = list_maga7_session_ids(client) if client is not None else []
    candidates = []
    if session_id:
        candidates.append(session_id)
    candidates.extend(redis_ids)
    candidates.extend(session_dirs.keys())
    # unique preserve order
    seen = set()
    ordered = []
    for sid in candidates:
        if sid and sid not in seen:
            seen.add(sid)
            ordered.append(sid)
    chosen = ordered[0] if ordered else None
    session_dir = session_dirs.get(chosen) if chosen else None
    if prefer_disk and session_dir is None and sessions:
        session_dir = sessions[0].path
        chosen = str((sessions[0].manifest or {}).get("session_id") or sessions[0].path.name)

    oms = fetch_oms_positions(
        None if prefer_disk else client,
        session_id=chosen,
        session_dir=session_dir if (prefer_disk or client is None) else None,
    )
    # If redis empty, fall back to disk for the same session.
    if not oms["positions"] and session_dir is not None:
        oms = fetch_oms_positions(None, session_id=chosen, session_dir=session_dir)
    scanner = fetch_scanner_state(
        None if prefer_disk else client,
        session_id=chosen,
        session_dir=session_dir if (prefer_disk or client is None) else None,
    )
    if not scanner and session_dir is not None:
        scanner = fetch_scanner_state(None, session_id=chosen, session_dir=session_dir)

    pos_df = positions_frame(oms.get("positions") or {})
    open_n = int((pos_df["status"].isin(["OPEN", "EXIT_PENDING"])).sum()) if not pos_df.empty else 0
    return {
        "session_id": chosen,
        "session_ids": ordered,
        "session_dir": str(session_dir) if session_dir else None,
        "oms": oms,
        "scanner": scanner,
        "positions_df": pos_df,
        "mf_df": mf_window_frame(scanner),
        "fires_df": day_fires_frame(scanner),
        "open_positions": open_n,
        "concurrent": open_n,
    }


ALIGNMENT_GAPS = [
    {
        "component": "盘前锁约",
        "current": "LiveOpenLadderLockService 按真实 spot 锁 0/1/2DTE ladder 并固化 manifest",
        "live_target": "开盘前启动后，根据真实开盘价生成并冻结当日 ladder",
        "status": "IMPLEMENTED / NEEDS G4 EVIDENCE",
    },
    {
        "component": "期权订阅",
        "current": "IBKR tick 订阅、预算控制、按需 fallback 订阅、心跳重连",
        "live_target": "IBKR/行情源订阅 ladder 合约，维护订阅确认、延迟、断线重连",
        "status": "IMPLEMENTED / NEEDS G4 EVIDENCE",
    },
    {
        "component": "滑动窗口",
        "current": "入场、QQQ regime 与 mf_flip 共用实时分钟状态",
        "live_target": "入场与退出共用同一个实时窗口状态",
        "status": "IMPLEMENTED / NEEDS G4 EVIDENCE",
    },
    {
        "component": "OMS",
        "current": "Shadow/Paper/Live 单一 OMS，限价、部分成交、订单状态与 fill 审计",
        "live_target": "broker submit/orderStatus/execDetails/cancel/reconcile 状态机",
        "status": "IMPLEMENTED / NEEDS G5 EVIDENCE",
    },
    {
        "component": "恢复",
        "current": "session-scoped Redis + OMS 原子快照 + pending frame claim + broker reconcile",
        "live_target": "进程重启后恢复锁约、窗口、持仓、未完成订单和审计游标",
        "status": "IMPLEMENTED / NEEDS RESTART DRILL",
    },
    {
        "component": "可追溯",
        "current": "live manifest + locks + signals + order/fill JSONL + Redis health",
        "live_target": "每个 live session 固化 profile、锁约、frame、signal、order、fill lineage",
        "status": "IMPLEMENTED",
    },
]
