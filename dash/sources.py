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
# Legacy in-repo tree (research tags may still live here).
MAGA7_RESULTS = REPO / "maga7" / "results"
# Preferred outside-repo roots (see maga7.common.config).
MAGA7_RESULTS_EXTERNAL = Path("/mnt/s990/data/maga7/results")
MAGA7_LIVE_SESSIONS_EXTERNAL = Path("/mnt/s990/data/maga7/live_sessions")
PROD_PROFILE = (
    REPO
    / "maga7"
    / "CONFIG"
    / "strategy_profiles"
    / "single_qqq_open_ladder_atm5otm_extend_mtm_full_day_peer3_v1.json"
)


def maga7_results_roots() -> list[Path]:
    """Ordered unique results roots for offline/parity discovery."""
    import os

    roots: list[Path] = []
    env = os.environ.get("MAG7_RESULTS_DIR", "").strip()
    if env:
        roots.append(Path(os.path.expanduser(env)).resolve())
    try:
        from maga7.common.config import load_profile, resolve_results_dir

        roots.append(resolve_results_dir(load_profile(PROD_PROFILE).get("_paths")))
    except Exception:
        pass
    roots.extend([MAGA7_RESULTS_EXTERNAL, MAGA7_RESULTS])
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def maga7_live_session_roots() -> list[Path]:
    """Ordered unique live_sessions roots (new s990 path + legacy)."""
    import os

    roots: list[Path] = []
    env = os.environ.get("MAG7_LIVE_SESSIONS_DIR", "").strip()
    if env:
        roots.append(Path(os.path.expanduser(env)).resolve())
    try:
        from maga7.common.config import load_profile, resolve_live_sessions_dir

        roots.append(
            resolve_live_sessions_dir(load_profile(PROD_PROFILE).get("_paths"))
        )
    except Exception:
        pass
    roots.append(MAGA7_LIVE_SESSIONS_EXTERNAL)
    for results in maga7_results_roots():
        roots.append(results / "live_sessions")
    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out

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
        for root in maga7_live_session_roots():
            try:
                return str(self.path.relative_to(root))
            except ValueError:
                continue
        return str(self.path)

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
    rows: list[LiveSessionArtifact] = []
    seen_dirs: set[str] = set()
    for root in maga7_live_session_roots():
        if not root.is_dir():
            continue
        for path in root.rglob("manifest.json"):
            session_dir = path.parent
            key = str(session_dir.resolve())
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            manifest = read_json(path)
            if not manifest:
                continue
            counts: dict[str, int] = {}
            event_path = session_dir / "order_events.jsonl"
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
                    path=session_dir,
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


def load_tape_parity(session_dir: Path | None) -> dict[str, Any]:
    """Latest intraday tape↔Scanner parity report from session disk."""
    if session_dir is None:
        return {}
    path = Path(session_dir) / "tape_parity.json"
    if not path.is_file():
        return {"_missing": True, "path": str(path)}
    report = read_json(path)
    if not report:
        return {"_missing": True, "path": str(path), "_unreadable": True}
    try:
        mtime = path.stat().st_mtime
        report["_mtime"] = mtime
        report["_age_sec"] = round(time.time() - mtime, 1)
    except Exception:
        report["_age_sec"] = None
    report["path"] = str(path)
    return report


def load_watchdog_hunt(
    session_dir: Path | None,
    *,
    client: Any | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Watchdog state + Hunt counters for Dash (scanner_state / oms_meta)."""
    # Prefer live oms_meta (fresh counters), then scanner_state disk/redis.
    if client is not None and session_id:
        oms = fetch_oms_meta(client, session_id=session_id, session_dir=session_dir)
        wd = oms.get("watchdog") if isinstance(oms.get("watchdog"), dict) else None
        if wd:
            out = dict(wd)
            out["source"] = str(oms.get("source") or "oms_meta")
            return out
    if session_dir is not None:
        snap = read_json(Path(session_dir) / "scanner_state.json")
        if snap and isinstance(snap.get("watchdog"), dict):
            out = dict(snap["watchdog"])
            out["source"] = "disk:scanner_state.json"
            out["path"] = str(Path(session_dir) / "scanner_state.json")
            return out
        # Legacy: prevention blob carried watchdog_* fields only.
        prev = load_prevention(session_dir)
        if not prev.get("_missing"):
            return {
                "state": prev.get("state") or prev.get("watchdog_state") or "off",
                "reason": prev.get("reason") or prev.get("watchdog_reason") or "off",
                "route": prev.get("route_tag") or prev.get("watchdog_route") or "baseline",
                "source": "legacy:prevention",
                "_partial": True,
            }
    if client is not None and session_id:
        sc = fetch_scanner_state(client, session_id=session_id, session_dir=session_dir)
        if sc and isinstance(sc.get("watchdog"), dict):
            out = dict(sc["watchdog"])
            out["source"] = str(sc.get("_source") or "redis:scanner_state")
            return out
    return {"_missing": True}


def load_prevention(session_dir: Path | None) -> dict[str, Any]:
    """Predictive morning prevention snapshot (Watchdog prevention lane)."""
    if session_dir is None:
        return {"_missing": True}
    path = Path(session_dir) / "prevention.json"
    if not path.is_file():
        snap = read_json(Path(session_dir) / "scanner_state.json")
        if snap and isinstance(snap.get("prevention"), dict):
            blob = dict(snap["prevention"])
            blob["source"] = "disk:scanner_state.json"
            blob["path"] = str(Path(session_dir) / "scanner_state.json")
            return blob
        return {"_missing": True, "path": str(path)}
    report = read_json(path)
    if not report:
        return {"_missing": True, "path": str(path), "_unreadable": True}
    try:
        mtime = path.stat().st_mtime
        report["_mtime"] = mtime
        report["_age_sec"] = round(time.time() - mtime, 1)
    except Exception:
        report["_age_sec"] = None
    report["path"] = str(path)
    report.setdefault("source", "disk:prevention.json")
    return report


def load_exit_health(session_dir: Path | None) -> dict[str, Any]:
    """Exit-arm / health snapshot from session disk (written by OMS publish)."""
    if session_dir is None:
        return {"_missing": True}
    path = Path(session_dir) / "exit_health.json"
    if not path.is_file():
        # Fallback: embed from oms_state if present.
        oms = read_json(Path(session_dir) / "oms_state.json")
        if oms and (oms.get("exit_arms") or oms.get("exit_health")):
            return {
                "session_id": oms.get("session_id"),
                "trade_date": oms.get("trade_date"),
                "updated_at": oms.get("updated_at"),
                "exit_arms": oms.get("exit_arms") or {},
                "exit_health": oms.get("exit_health") or {},
                "source": "disk:oms_state.json",
                "path": str(Path(session_dir) / "oms_state.json"),
            }
        return {"_missing": True, "path": str(path)}
    report = read_json(path)
    if not report:
        return {"_missing": True, "path": str(path), "_unreadable": True}
    try:
        mtime = path.stat().st_mtime
        report["_mtime"] = mtime
        report["_age_sec"] = round(time.time() - mtime, 1)
    except Exception:
        report["_age_sec"] = None
    report["path"] = str(path)
    report.setdefault("source", "disk:exit_health.json")
    return report


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
    dirs: set[Path] = set()
    for root in maga7_results_roots():
        if not root.is_dir():
            continue
        for filename in ("summary.json", "parity_summary.json", "offline_summary.json"):
            for path in root.rglob(filename):
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
        name = str(path)
        for root in maga7_results_roots():
            try:
                name = str(path.relative_to(root))
                break
            except ValueError:
                continue
        rows.append(
            RunArtifact(
                path=path,
                name=name,
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
        "Mag7 Live": ("run_live_session",),
        "Mag7 Redis sim": ("run_maga7_redis_sim",),
        "Legacy IBKR connector": ("ibkr_connector",),
        "Legacy FCS": ("feature_compute_service",),
        "Legacy Signal engine": ("engine_v8.py", "run_live_exec_qqq"),
        "Legacy OMS": ("execution_engine", "run_oms"),
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


def _fmt_ny_ts(ts: Any) -> str:
    """Format unix seconds / ISO-ish timestamp for Dash tables."""
    if ts is None or ts == "" or (isinstance(ts, float) and pd.isna(ts)):
        return ""
    try:
        val = float(ts)
        # Heuristic: ns vs seconds
        if val > 1e14:
            val = val / 1e9
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(val))
    except Exception:
        return str(ts)


def tape_inventory(
    session_dir: Path | None,
    *,
    phase: str | None = None,
    sample_lines: int = 2,
) -> dict[str, Any]:
    """List session tape files under tape/{pre|rth|post}/ for Dash visibility."""
    if session_dir is None:
        return {
            "root": None,
            "phases": [],
            "files": pd.DataFrame(),
            "samples": {},
        }
    root = Path(session_dir) / "tape"
    phases = ["pre", "rth", "post"]
    if phase:
        phases = [str(phase).lower()]
    rows: list[dict[str, Any]] = []
    samples: dict[str, list[dict[str, Any]]] = {}
    for name in ("pre", "rth", "post"):
        if phase and name not in phases:
            continue
        d = root / name
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.jsonl")):
            try:
                st = path.stat()
                n_lines = 0
                with path.open("r", encoding="utf-8") as handle:
                    for n_lines, _ in enumerate(handle, 1):
                        pass
                mtime = st.st_mtime
                size = st.st_size
            except OSError:
                n_lines, mtime, size = 0, None, 0
            rows.append(
                {
                    "phase": name,
                    "symbol": path.name.split("_")[0],
                    "file": path.name,
                    "lines": n_lines,
                    "bytes": size,
                    "mtime": _fmt_ny_ts(mtime) if mtime else "",
                    "path": str(path),
                }
            )
            if sample_lines > 0 and n_lines > 0:
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        taken = []
                        for i, line in enumerate(handle):
                            if i >= sample_lines:
                                break
                            try:
                                taken.append(json.loads(line))
                            except Exception:
                                continue
                    if taken:
                        samples[f"{name}/{path.name}"] = taken
                except OSError:
                    pass
    return {
        "root": str(root),
        "phases": sorted({r["phase"] for r in rows}),
        "files": pd.DataFrame(rows),
        "samples": samples,
        "n_files": len(rows),
        "n_lines": int(sum(int(r.get("lines") or 0) for r in rows)),
    }


def fetch_locks_payload(
    session_dir: Path | None,
) -> dict[str, Any]:
    """Load session locks.json (open-ladder lock manifest)."""
    if session_dir is None:
        return {}
    path = Path(session_dir) / "locks.json"
    payload = read_json(path)
    if not payload:
        return {}
    out = dict(payload)
    out["_source"] = f"disk:{path}"
    return out


def locks_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    """Flatten locks.json → one row per locked contract (with trigger time)."""
    payload = payload or {}
    locks = payload.get("locks") or {}
    rows: list[dict[str, Any]] = []
    for symbol, contracts in locks.items():
        if not isinstance(contracts, list):
            continue
        for row in contracts:
            if not isinstance(row, dict):
                continue
            lock_ts = row.get("lock_ts")
            rows.append(
                {
                    "symbol": str(symbol).upper(),
                    "right": row.get("right"),
                    "strike": row.get("strike"),
                    "dte": row.get("front_dte"),
                    "rung": row.get("ladder_rung"),
                    "bucket": row.get("bucket_id"),
                    "localSymbol": row.get("local_symbol"),
                    "conId": row.get("con_id"),
                    "lock_spot": row.get("lock_spot"),
                    "lock_ts": lock_ts,
                    "lock_time": _fmt_ny_ts(lock_ts),
                    "expiry": row.get("expiry"),
                    "exchange": row.get("exchange"),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(
        by=["symbol", "dte", "right", "rung"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def locks_summary_frame(payload: dict[str, Any] | None) -> pd.DataFrame:
    """Per-symbol lock trigger summary (first/last lock_ts + spot)."""
    detail = locks_frame(payload)
    if detail.empty:
        return pd.DataFrame()
    rows = []
    for symbol, grp in detail.groupby("symbol", sort=True):
        ts_vals = pd.to_numeric(grp["lock_ts"], errors="coerce").dropna()
        first_ts = float(ts_vals.min()) if not ts_vals.empty else None
        last_ts = float(ts_vals.max()) if not ts_vals.empty else None
        spots = pd.to_numeric(grp["lock_spot"], errors="coerce").dropna()
        rows.append(
            {
                "symbol": symbol,
                "n_contracts": int(len(grp)),
                "lock_time": _fmt_ny_ts(first_ts),
                "lock_ts": first_ts,
                "lock_spot": float(spots.iloc[0]) if not spots.empty else None,
                "calls": int((grp["right"].astype(str).str.upper() == "C").sum()),
                "puts": int((grp["right"].astype(str).str.upper() == "P").sum()),
                "dtes": ",".join(
                    str(int(x))
                    for x in sorted(
                        {
                            int(v)
                            for v in pd.to_numeric(grp["dte"], errors="coerce").dropna()
                        }
                    )
                ),
                "last_lock_time": _fmt_ny_ts(last_ts) if last_ts != first_ts else "",
            }
        )
    return pd.DataFrame(rows)


def subscription_frame(
    connector: dict[str, Any] | None,
    *,
    profile: dict[str, Any] | None = None,
    locks_payload: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Subscribed underlyings + option quote / lock counts."""
    connector = connector or {}
    stock_feed = connector.get("stock_feed") or {}
    option_feed = connector.get("option_feed") or {}
    feed = connector.get("feed_health") or {}
    if isinstance(feed, dict):
        stock_feed = stock_feed or feed.get("stock_feed") or {}
        option_feed = option_feed or feed.get("option_feed") or {}

    symbols: list[str] = []
    for src in (
        list(stock_feed),
        list(option_feed),
        list(((locks_payload or {}).get("locks") or {})),
        list((((profile or {}).get("profile") or {}).get("symbols") or [])),
    ):
        for sym in src:
            u = str(sym).upper()
            if u and u not in symbols:
                symbols.append(u)

    lock_counts: dict[str, int] = {}
    for sym, rows in ((locks_payload or {}).get("locks") or {}).items():
        if isinstance(rows, list):
            lock_counts[str(sym).upper()] = len(rows)

    now = time.time()
    rows = []
    for symbol in symbols:
        sf = stock_feed.get(symbol) or stock_feed.get(symbol.lower()) or {}
        of = option_feed.get(symbol) or option_feed.get(symbol.lower()) or {}
        lag = sf.get("lag_sec")
        last_ts = sf.get("last_ts")
        if lag is None and last_ts:
            try:
                lag = round(now - float(last_ts), 3)
            except Exception:
                lag = None
        subscribed = sf.get("subscribed")
        if subscribed is None:
            subscribed = bool(last_ts) or symbol in stock_feed
        rows.append(
            {
                "symbol": symbol,
                "subscribed": bool(subscribed),
                "spot": sf.get("spot"),
                "stock_lag_sec": lag,
                "option_quotes": of.get("n_quotes"),
                "option_locked": of.get("n_locked")
                if of.get("n_locked") is not None
                else lock_counts.get(symbol),
                "option_lag_sec": of.get("lag_sec"),
                "role": (
                    "trade"
                    if symbol
                    in {
                        str(s).upper()
                        for s in (
                            ((profile or {}).get("profile") or {}).get("symbols") or []
                        )
                    }
                    else "ref"
                ),
            }
        )
    return pd.DataFrame(rows)


def spot_series_frame(
    scanner_state: dict[str, Any] | None,
    symbols: Iterable[str] | None = None,
    *,
    session_dir: Path | None = None,
    phase: str | None = None,
) -> pd.DataFrame:
    """Long-form close series for subscribed symbols (scanner 1m + optional tape)."""
    rows: list[dict[str, Any]] = []
    states = (scanner_state or {}).get("states") or {}
    wanted = [str(s).upper() for s in (symbols or list(states))]
    if not wanted:
        wanted = [str(k).upper() for k in states]

    for symbol in wanted:
        st_state = states.get(symbol)
        if st_state is None:
            for key, value in states.items():
                if str(key).upper() == symbol:
                    st_state = value
                    break
        bars = (st_state or {}).get("bars") if isinstance(st_state, dict) else None
        if isinstance(bars, list):
            for bar in bars:
                if not isinstance(bar, dict):
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": bar.get("timestamp"),
                        "close": bar.get("close"),
                        "source": "scanner_1m",
                    }
                )

        # Pre/post tape fills gaps before scanner RTH bars exist.
        if session_dir is not None:
            phase_name = (phase or "pre").lower()
            tape_dir = Path(session_dir) / "tape" / phase_name
            if tape_dir.is_dir():
                for path in sorted(tape_dir.glob(f"{symbol}_*.jsonl")):
                    try:
                        with path.open("r", encoding="utf-8") as handle:
                            for line in handle:
                                try:
                                    item = json.loads(line)
                                except Exception:
                                    continue
                                if not isinstance(item, dict):
                                    continue
                                ts = item.get("ts")
                                close = item.get("close")
                                if ts is None or close is None:
                                    continue
                                try:
                                    ts_label = time.strftime(
                                        "%Y-%m-%d %H:%M:%S",
                                        time.localtime(float(ts)),
                                    )
                                except Exception:
                                    ts_label = str(ts)
                                rows.append(
                                    {
                                        "symbol": symbol,
                                        "timestamp": ts_label,
                                        "close": close,
                                        "source": f"tape_{phase_name}",
                                    }
                                )
                    except OSError:
                        continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    return df.reset_index(drop=True)


def _to_unix_sec(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        if val > 1e14:
            val = val / 1e9
        return val if val > 0 else None
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/New_York")
        else:
            ts = ts.tz_convert("America/New_York")
        return float(ts.timestamp())
    except Exception:
        return None


def _aggregate_seconds_to_1m(sec_rows: list[dict[str, Any]]) -> dict[int, dict[str, float]]:
    """Aggregate second prints → left-labeled 1m OHLCV keyed by minute unix ts."""
    buckets: dict[int, dict[str, float]] = {}
    for row in sec_rows:
        ts = _to_unix_sec(row.get("ts"))
        if ts is None:
            continue
        try:
            o = float(row.get("open", row.get("close")))
            h = float(row.get("high", o))
            l = float(row.get("low", o))
            c = float(row.get("close", o))
            v = float(row.get("volume") or 0.0)
        except Exception:
            continue
        if not (o == o and h == h and l == l and c == c):
            continue
        minute = int(ts) - (int(ts) % 60)
        cur = buckets.get(minute)
        if cur is None:
            buckets[minute] = {
                "ts": float(minute),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        else:
            cur["high"] = max(cur["high"], h)
            cur["low"] = min(cur["low"], l)
            cur["close"] = c
            cur["volume"] = float(cur.get("volume") or 0.0) + v
    return buckets


def ohlcv_1m_for_symbol(
    symbol: str,
    *,
    scanner_state: dict[str, Any] | None = None,
    session_dir: Path | None = None,
    phases: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build TradingView-ready 1m OHLCV for one subscribed symbol.

    Sources (merged, scanner wins on same minute):
    - ``tape/{pre|rth|post}/{SYM}_*.jsonl`` second prints → 1m aggregate
    - scanner ``states[SYM].bars`` 1m OHLCV
    """
    symbol = str(symbol).upper()
    buckets: dict[int, dict[str, float]] = {}
    phase_list = [str(p).lower() for p in (phases or ("pre", "rth", "post"))]

    if session_dir is not None:
        for phase_name in phase_list:
            tape_dir = Path(session_dir) / "tape" / phase_name
            if not tape_dir.is_dir():
                continue
            sec_rows: list[dict[str, Any]] = []
            for path in sorted(tape_dir.glob(f"{symbol}_*.jsonl")):
                try:
                    with path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            try:
                                item = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(item, dict):
                                sec_rows.append(item)
                except OSError:
                    continue
            for minute, bar in _aggregate_seconds_to_1m(sec_rows).items():
                buckets[minute] = bar

    states = (scanner_state or {}).get("states") or {}
    st_state = states.get(symbol)
    if st_state is None:
        for key, value in states.items():
            if str(key).upper() == symbol:
                st_state = value
                break
    bars = (st_state or {}).get("bars") if isinstance(st_state, dict) else None
    if isinstance(bars, list):
        for bar in bars:
            if not isinstance(bar, dict):
                continue
            ts = _to_unix_sec(bar.get("timestamp") or bar.get("ts"))
            if ts is None:
                continue
            try:
                o = float(bar.get("open", bar.get("close")))
                h = float(bar.get("high", o))
                l = float(bar.get("low", o))
                c = float(bar.get("close", o))
                v = float(bar.get("volume") or 0.0)
            except Exception:
                continue
            minute = int(ts) - (int(ts) % 60)
            # Scanner is authority for RTH decision bars.
            buckets[minute] = {
                "ts": float(minute),
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }

    if not buckets:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
    df = pd.DataFrame([buckets[k] for k in sorted(buckets)])
    return df.reset_index(drop=True)


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
        # Mag7 oms_meta / many Dash keys are JSON; try that before msgpack to
        # avoid noisy "Comprehensive unpack failure" logs on every refresh.
        if raw[:1] in (b"{", b"["):
            try:
                return json.loads(raw.decode("utf-8"))
            except Exception:
                pass
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
        for key in client.scan_iter(match="maga7:oms_meta:*", count=100):
            s = str(_decode(key))
            ids.add(s.rsplit(":", 1)[-1])
        for key in client.scan_iter(match="maga7:feed_health:*", count=100):
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


def _oms_meta_from_state(state: dict[str, Any], *, source: str, session_id: str | None) -> dict[str, Any]:
    return {
        "source": source,
        "session_id": state.get("session_id") or session_id,
        "updated_at": state.get("updated_at"),
        "day_halted": bool(state.get("day_halted")),
        "equity": state.get("equity"),
        "day_start_equity": state.get("day_start_equity"),
        "realized_pnl": state.get("realized_pnl"),
        "available_funds": state.get("available_funds"),
        "account_ready": state.get("account_ready"),
        "mode": state.get("mode"),
        "trade_date": state.get("trade_date"),
        "reconcile_ok": state.get("reconcile_ok"),
        "last_reconcile": state.get("last_reconcile") or {},
        "profile_hash": state.get("profile_hash"),
        "n_positions": state.get("n_positions"),
        "n_intents": state.get("n_intents"),
        "exit_arms": state.get("exit_arms") or {},
        "exit_health": state.get("exit_health") or {},
        "exit_reason_counts": state.get("exit_reason_counts") or {},
    }


def fetch_oms_meta(
    client: Any | None,
    *,
    session_id: str | None = None,
    session_dir: Path | None = None,
) -> dict[str, Any]:
    """Load Mag7 OMS meta (equity / arm-relevant / last reconcile)."""
    if client is not None and session_id:
        try:
            raw = client.get(f"maga7:oms_meta:{session_id}")
            state = _decode_jsonish(raw)
            if isinstance(state, dict) and state:
                return _oms_meta_from_state(
                    state, source="redis:maga7:oms_meta", session_id=session_id
                )
        except Exception as exc:
            return {"source": "redis_error", "error": str(exc), "session_id": session_id}
    if session_dir is not None:
        state = read_json(Path(session_dir) / "oms_state.json")
        if state:
            return _oms_meta_from_state(
                state, source="disk:oms_state.json", session_id=session_id
            )
    return {"session_id": session_id, "source": None}


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
                _oms_meta_from_state(
                    state, source="disk:oms_state.json", session_id=session_id
                )
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
            oms_meta = fetch_oms_meta(client, session_id=session_id)
            if oms_meta.get("source"):
                meta.update(oms_meta)
            if positions or intents:
                meta["source"] = meta.get("source") or "redis:oms:live_positions"
                meta["session_id"] = session_id
            elif oms_meta.get("source"):
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


def fetch_connector_status(
    client: Any | None,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Load Mag7 IBKR connector status (+ optional feed_health projection)."""
    if client is None or not session_id:
        return {}
    try:
        raw = client.hget(f"live_ibkr_connector:maga7:{session_id}", "status")
        status = _decode_jsonish(raw)
        if not isinstance(status, dict):
            status = {}
        else:
            status = dict(status)
            status["_source"] = "redis:live_ibkr_connector"
        feed_raw = client.get(f"maga7:feed_health:{session_id}")
        if feed_raw is None:
            feed_raw = client.hget(
                f"live_ibkr_connector:maga7:{session_id}", "feed_health"
            )
        feed = _decode_jsonish(feed_raw)
        if isinstance(feed, dict):
            status["feed_health"] = feed
            # Prefer fresher per-second feed snapshot for lag panels.
            if feed.get("stock_feed"):
                status["stock_feed"] = feed["stock_feed"]
            if feed.get("option_feed"):
                status["option_feed"] = feed["option_feed"]
            if feed.get("ts") is not None:
                status["feed_ts"] = feed["ts"]
        return status
    except Exception as exc:
        return {"_source": "redis_error", "error": str(exc)}


def _health_label(
    lag_sec: float | None,
    *,
    ok_sec: float,
    warn_sec: float,
    missing: bool = False,
    warmup_bars: int | None = None,
    warmup_need: int = 10,
) -> str:
    if missing or lag_sec is None:
        if warmup_bars is not None and warmup_bars < warmup_need:
            return "🟡 Warmup"
        return "⚪ Missing"
    if warmup_bars is not None and warmup_bars < warmup_need:
        return "🟡 Warmup"
    if lag_sec <= ok_sec:
        return "🟢 OK"
    if lag_sec <= warn_sec:
        return "🟠 Warn"
    return "🔴 Stale"


def feed_health_frame(
    connector: dict[str, Any] | None,
    scanner_state: dict[str, Any] | None = None,
    *,
    profile: dict[str, Any] | None = None,
    stream_age_sec: float | None = None,
) -> pd.DataFrame:
    """
    Per-symbol subscription / freshness table (Mag7 Redis, not Postgres).

    Mirrors production monitor intent (OK / Warmup / Stale) using:
    - connector stock/option last tick ages
    - scanner 1m bar age (signal path)
    """
    connector = connector or {}
    risk = ((profile or {}).get("profile") or {}).get("risk") or {}
    stock_ok = float(
        connector.get("max_stock_staleness_sec")
        or risk.get("max_stock_staleness_sec")
        or 2.0
    )
    option_ok = float(
        connector.get("max_option_staleness_sec")
        or risk.get("max_option_staleness_sec")
        or 5.0
    )
    stock_warn = max(stock_ok * 5.0, 10.0)
    option_warn = max(option_ok * 3.0, 15.0)
    bar_ok = 90.0
    bar_warn = 180.0

    now = time.time()
    stock_feed = connector.get("stock_feed") or {}
    option_feed = connector.get("option_feed") or {}
    states = (scanner_state or {}).get("states") or {}

    symbols = sorted(
        set(stock_feed)
        | set(option_feed)
        | {str(k).upper() for k in states}
        | {
            str(s).upper()
            for s in (((profile or {}).get("profile") or {}).get("symbols") or [])
        }
    )
    data_mode = str(connector.get("data_mode") or "")
    connected = connector.get("connected")
    status_ts = connector.get("feed_ts") or connector.get("ts")
    try:
        status_age = round(now - float(status_ts), 1) if status_ts is not None else None
    except Exception:
        status_age = None

    rows = []
    for symbol in symbols:
        sf = stock_feed.get(symbol) or stock_feed.get(symbol.upper()) or {}
        of = option_feed.get(symbol) or option_feed.get(symbol.upper()) or {}
        st_state = states.get(symbol) or states.get(symbol.upper()) or {}
        bars = st_state.get("bars") if isinstance(st_state, dict) else None
        n_bars = len(bars) if isinstance(bars, list) else 0
        bar_lag = None
        if isinstance(bars, list) and bars:
            last_bar = bars[-1] if isinstance(bars[-1], dict) else {}
            bar_ts = last_bar.get("ts") or last_bar.get("timestamp")
            try:
                if bar_ts is not None:
                    bar_lag = round(now - float(bar_ts), 1)
            except Exception:
                bar_lag = None

        stock_last = float(sf.get("last_ts") or 0.0)
        stock_lag = None
        if stock_last > 0:
            stock_lag = round(now - stock_last, 2)
        elif sf.get("lag_sec") is not None:
            try:
                stock_lag = float(sf["lag_sec"])
            except Exception:
                stock_lag = None

        opt_last = float(of.get("last_ts") or 0.0)
        opt_lag = None
        if opt_last > 0:
            opt_lag = round(now - opt_last, 2)
        elif of.get("lag_sec") is not None:
            try:
                opt_lag = float(of["lag_sec"])
            except Exception:
                opt_lag = None

        stock_health = _health_label(
            stock_lag,
            ok_sec=stock_ok,
            warn_sec=stock_warn,
            missing=not bool(sf.get("subscribed", True)) and stock_lag is None,
            warmup_bars=n_bars if n_bars else None,
            warmup_need=10,
        )
        locked_n = int(of.get("n_locked") or 0)
        opt_health = _health_label(
            opt_lag,
            ok_sec=option_ok,
            warn_sec=option_warn,
            missing=locked_n > 0 and opt_lag is None,
        )
        # Locked ladder but zero quotes is an outage, not cold warmup.
        if locked_n > 0 and opt_lag is None and connected is not False:
            opt_health = "🔴 Stale"
        bar_health = _health_label(
            bar_lag,
            ok_sec=bar_ok,
            warn_sec=bar_warn,
            missing=bar_lag is None,
            warmup_bars=n_bars,
            warmup_need=10,
        )
        flags = (stock_health, opt_health, bar_health)
        if data_mode == "DELAYED_BLOCKED":
            overall = "🔴 DELAYED_BLOCKED"
        elif connected is False:
            overall = "🔴 DISCONNECTED"
        elif any("🔴" in x for x in (stock_health, opt_health)):
            overall = "🔴 Stale"
        elif any("🟠" in x for x in flags):
            overall = "🟠 Warn"
        elif any("🟡" in x or "⚪" in x for x in (stock_health, opt_health, bar_health)):
            overall = "🟡 Warmup"
        else:
            overall = "🟢 OK"

        rows.append(
            {
                "symbol": symbol,
                "overall": overall,
                "stock": stock_health,
                "stock_lag_s": stock_lag,
                "spot": sf.get("spot") or None,
                "option": opt_health,
                "option_lag_s": opt_lag,
                "n_opt_quotes": of.get("n_quotes"),
                "n_locked": of.get("n_locked"),
                "scanner_bar": bar_health,
                "bar_lag_s": bar_lag,
                "n_bars": n_bars or None,
                "stream_age_s": stream_age_sec,
                "data_mode": data_mode or None,
                "status_age_s": status_age,
            }
        )
    return pd.DataFrame(rows)


def fetch_runtime_controls(client: Any | None) -> dict[str, Any]:
    """Read Mag7 Arm/Disarm controls (read-only)."""
    out: dict[str, Any] = {
        "trading_enabled": None,
        "armed": False,
        "raw": {},
        "source": None,
    }
    if client is None:
        return out
    try:
        raw = client.hgetall("meta:runtime_trading_controls:maga7") or {}
        decoded = {str(_decode(k)): _decode(v) for k, v in raw.items()}
        out["raw"] = decoded
        out["source"] = "redis:meta:runtime_trading_controls:maga7"
        flag = decoded.get("trading_enabled")
        out["trading_enabled"] = flag
        out["armed"] = str(flag or "").strip().lower() in {"1", "true", "yes", "on"}
        for key in ("live_cap", "capital_cap", "max_notional", "note"):
            if key in decoded:
                out[key] = decoded[key]
    except Exception as exc:
        out["error"] = str(exc)
    return out


def fetch_engine_health(client: Any | None, *, session_id: str | None) -> dict[str, Any]:
    if client is None or not session_id:
        return {}
    try:
        raw = client.hgetall(f"maga7:live_engine:{session_id}") or {}
        out: dict[str, Any] = {}
        for key, value in raw.items():
            k = str(_decode(key))
            v = _decode(value)
            if isinstance(v, str):
                try:
                    out[k] = json.loads(v)
                    continue
                except Exception:
                    pass
            out[k] = v
        out["_source"] = "redis:maga7:live_engine"
        return out
    except Exception as exc:
        return {"_source": "redis_error", "error": str(exc)}


def fetch_legacy_account_info(client: Any | None) -> dict[str, Any]:
    """Fallback account snapshot from shared `live_account_info` hash."""
    if client is None:
        return {}
    try:
        raw = client.hget("live_account_info", "balance")
        if raw is None:
            return {}
        # Production stores msgpack/pickle; try jsonish first then pickle.
        payload = _decode_jsonish(raw)
        if isinstance(payload, dict):
            return {**payload, "_source": "redis:live_account_info"}
        try:
            payload = pickle.loads(raw if isinstance(raw, (bytes, bytearray)) else bytes(raw))
            if isinstance(payload, dict):
                return {**payload, "_source": "redis:live_account_info:pickle"}
        except Exception:
            pass
        return {"_source": "redis:live_account_info", "raw": str(_decode(raw))[:200]}
    except Exception as exc:
        return {"error": str(exc)}


def _lag_health(age: float | None, *, ok: float, warn: float) -> str:
    if age is None:
        return "⚪ N/A"
    if age <= ok:
        return "🟢 OK"
    if age <= warn:
        return "🟠 Warn"
    return "🔴 Crit"


def load_order_events(
    client: Any | None,
    *,
    session_id: str | None,
    session_dir: Path | None = None,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Load recent Mag7 order events from Redis stream or disk jsonl."""
    rows: list[dict[str, Any]] = []
    if client is not None and session_id:
        try:
            items = client.xrevrange(f"maga7:order_events:{session_id}", count=int(limit))
            for _eid, fields in items or []:
                raw = fields.get(b"data") if b"data" in fields else fields.get("data")
                payload = _decode_jsonish(raw)
                if isinstance(payload, dict):
                    rows.append(payload)
            if rows:
                rows.reverse()
                return rows
        except Exception:
            pass
    if session_dir is not None:
        path = Path(session_dir) / "order_events.jsonl"
        if path.is_file():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            rows.append(json.loads(line))
                        except Exception:
                            continue
            except OSError:
                pass
            if len(rows) > limit:
                rows = rows[-limit:]
    return rows


def gate_reject_frame(events: list[dict[str, Any]]) -> pd.DataFrame:
    """Aggregate ENTRY_REJECT / ENTRY_WAIT reasons for ops."""
    counts: dict[tuple[str, str], int] = {}
    last_ts: dict[tuple[str, str], float] = {}
    for row in events or []:
        kind = str(row.get("kind") or "")
        if kind not in {"ENTRY_REJECT", "ENTRY_WAIT"}:
            continue
        reason = str(row.get("reason") or "unknown")
        key = (kind, reason)
        counts[key] = counts.get(key, 0) + 1
        try:
            ts = float(row.get("ts") or 0.0)
        except Exception:
            ts = 0.0
        if ts >= last_ts.get(key, 0.0):
            last_ts[key] = ts
    rows = [
        {
            "kind": kind,
            "reason": reason,
            "count": count,
            "last_ts": last_ts.get((kind, reason)),
        }
        for (kind, reason), count in sorted(counts.items(), key=lambda x: (-x[1], x[0][0], x[0][1]))
    ]
    return pd.DataFrame(rows)


def reconcile_compare_frame(last_reconcile: dict[str, Any] | None) -> pd.DataFrame:
    """Broker vs OMS contract qty table from last_reconcile snapshot."""
    last = last_reconcile or {}
    broker = last.get("broker") or {}
    internal = last.get("internal") or {}
    if not isinstance(broker, dict):
        broker = {}
    if not isinstance(internal, dict):
        internal = {}
    contracts = sorted(set(map(str, broker)) | set(map(str, internal)))
    rows = []
    for contract in contracts:
        try:
            bq = int(broker.get(contract) or 0)
        except Exception:
            bq = 0
        try:
            iq = int(internal.get(contract) or 0)
        except Exception:
            iq = 0
        if bq == iq:
            status = "🟢 Match"
        elif bq and not iq:
            status = "🔴 Broker only"
        elif iq and not bq:
            status = "🔴 OMS only"
        else:
            status = "🔴 Qty mismatch"
        rows.append(
            {
                "contract": contract,
                "broker_qty": bq,
                "oms_qty": iq,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def live_ops_overview(
    client: Any | None,
    *,
    session_id: str | None = None,
    session_dir: Path | None = None,
    profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Sidebar / Live strip: env, account, arm, topology (IBKR→Fused→Scanner→OMS).
    Read-only; does not write Redis or send orders.
    """
    now = time.time()
    connector = fetch_connector_status(client, session_id=session_id) if session_id else {}
    if not connector and client is not None:
        # Fallback: any maga7 connector status / legacy projection.
        try:
            raw = client.hget("live_ibkr_connector", "maga7_status")
            connector = _decode_jsonish(raw) if raw is not None else {}
            if isinstance(connector, dict):
                connector = dict(connector)
                connector["_source"] = "redis:live_ibkr_connector:maga7_status"
                session_id = session_id or connector.get("session_id")
        except Exception:
            connector = {}

    oms_meta = fetch_oms_meta(client, session_id=session_id, session_dir=session_dir)
    if (not oms_meta.get("source")) and session_dir is not None:
        oms_meta = fetch_oms_meta(None, session_id=session_id, session_dir=session_dir)

    controls = fetch_runtime_controls(client)
    engine = fetch_engine_health(client, session_id=session_id)
    legacy_acct = fetch_legacy_account_info(client)
    order_events = load_order_events(
        client,
        session_id=session_id,
        session_dir=session_dir,
        limit=200,
    )
    planned_event_blackout = bool(
        oms_meta.get("day_halted")
        and oms_meta.get("reconcile_ok") is not False
        and any(
            str(row.get("kind") or "").upper() == "EVENT_BLACKOUT"
            and str(row.get("scope") or "").lower() == "full_day"
            for row in order_events
        )
    )

    port = None
    try:
        # Prefer connector payload; else profile live.port
        if connector.get("port") is not None:
            port = int(connector["port"])
        else:
            live_cfg = ((profile or {}).get("profile") or {}).get("live") or {}
            if live_cfg.get("port") is not None:
                port = int(live_cfg["port"])
    except Exception:
        port = None

    # Infer env from IB gateway port when known.
    if port == 4001:
        env_label = "🔴 REAL (4001)"
        env_kind = "real"
    elif port == 4002:
        env_label = "🟡 PAPER (4002)"
        env_kind = "paper"
    else:
        mode = str(oms_meta.get("mode") or connector.get("mode") or "").lower()
        if mode == "live":
            env_label = "🔴 LIVE mode"
            env_kind = "real"
        elif mode == "paper":
            env_label = "🟡 PAPER mode"
            env_kind = "paper"
        elif mode == "shadow":
            env_label = "🟢 SHADOW"
            env_kind = "shadow"
        else:
            env_label = "⚪ UNKNOWN"
            env_kind = "unknown"

    mode = str(oms_meta.get("mode") or "").lower()
    if mode == "shadow":
        arm_label = "SHADOW (sim fills)"
    elif mode == "paper":
        arm_label = "PAPER gate"
    elif controls.get("armed"):
        arm_label = "🟢 ARMED"
    else:
        arm_label = "🔴 DISARMED"

    equity = oms_meta.get("equity")
    available = oms_meta.get("available_funds")
    if equity is None:
        equity = legacy_acct.get("net_liquidation") or legacy_acct.get("NetLiquidation")
    if available is None:
        available = legacy_acct.get("available_funds") or legacy_acct.get("AvailableFunds")

    # Topology ages — prefer 1s feed_health over heartbeat status (~15s).
    feed_preview = connector.get("feed_health") if isinstance(connector, dict) else None
    if not isinstance(feed_preview, dict):
        feed_preview = {}
    conn_ts = (
        feed_preview.get("ts")
        or connector.get("feed_ts")
        or connector.get("ts")
    )
    try:
        ibkr_age = round(now - float(conn_ts), 1) if conn_ts is not None else None
    except Exception:
        ibkr_age = None

    # Prefer connector-reported phase; fall back to NY clock.
    try:
        from maga7.live.session_phase import session_phase as _session_phase
        from maga7.live.session_phase import tape_phase_dir as _tape_phase_dir

        phase = str(
            connector.get("session_phase")
            or (connector.get("feed_health") or {}).get("session_phase")
            or _session_phase(now)
        ).upper()
    except Exception:
        _tape_phase_dir = None
        phase = str(
            connector.get("session_phase")
            or (connector.get("feed_health") or {}).get("session_phase")
            or ""
        ).upper() or "RTH"

    auth_stream = connector.get("stream") or (
        f"fused_market_stream:maga7:{session_id}" if session_id else "fused_market_stream"
    )
    pre_stream = connector.get("stream_pre") or (
        f"{auth_stream}:pre" if auth_stream else None
    )
    post_stream = connector.get("stream_post") or (
        f"{auth_stream}:post" if auth_stream else None
    )
    if phase == "PRE":
        stream_key = pre_stream or auth_stream
    elif phase == "POST":
        stream_key = post_stream or auth_stream
    else:
        stream_key = auth_stream

    stream_age = None
    if client is not None and stream_key:
        try:
            stream_age = stream_probe(client, str(stream_key)).get("age_sec")
        except Exception:
            stream_age = None

    scanner = fetch_scanner_state(client, session_id=session_id, session_dir=session_dir)
    scanner_age = None
    try:
        # Prefer engine last_frame_ts; else newest scanner bar.
        if engine.get("last_frame_ts"):
            scanner_age = round(now - float(engine["last_frame_ts"]), 1)
        else:
            newest = None
            for st in (scanner.get("states") or {}).values():
                bars = (st or {}).get("bars") or []
                if bars and isinstance(bars[-1], dict):
                    ts = bars[-1].get("ts") or bars[-1].get("timestamp")
                    try:
                        newest = max(newest or 0.0, float(ts))
                    except Exception:
                        pass
            if newest:
                scanner_age = round(now - newest, 1)
    except Exception:
        scanner_age = None

    try:
        oms_age = (
            round(now - float(oms_meta["updated_at"]), 1)
            if oms_meta.get("updated_at") is not None
            else None
        )
    except Exception:
        oms_age = None

    data_mode = str(connector.get("data_mode") or "")
    connected = connector.get("connected")

    redis_ok: bool | None = None
    redis_rtt_ms: float | None = None
    if client is not None:
        try:
            t0 = time.time()
            client.ping()
            redis_ok = True
            redis_rtt_ms = round((time.time() - t0) * 1000.0, 1)
        except Exception:
            redis_ok = False

    stock_by = scanner.get("stock_by") if isinstance(scanner, dict) else None
    stock_syms = 0
    stock_bars = 0
    qqq_bars = 0
    if isinstance(stock_by, dict):
        stock_syms = len(stock_by)
        for sym, rows in stock_by.items():
            n = len(rows) if isinstance(rows, list) else 0
            stock_bars += n
            if str(sym).upper() == "QQQ":
                qqq_bars = n

    artifact_age = None
    artifact_detail = "-"
    tape_age = None
    tape_detail = "-"
    if session_dir is not None:
        for name in ("scanner_state.json", "oms_state.json", "manifest.json"):
            path = Path(session_dir) / name
            if not path.is_file():
                continue
            try:
                age = round(now - path.stat().st_mtime, 1)
            except Exception:
                continue
            if artifact_age is None or age < artifact_age:
                artifact_age = age
                artifact_detail = name
        # Phase-specific tape (pre/rth/post) — preferred Disk signal outside RTH.
        tape_dir = None
        if _tape_phase_dir is not None:
            try:
                tape_dir = _tape_phase_dir(session_dir, phase)
            except Exception:
                tape_dir = Path(session_dir) / "tape" / phase.lower()
        else:
            tape_dir = Path(session_dir) / "tape" / phase.lower()
        if tape_dir is not None and tape_dir.is_dir():
            newest_mtime = None
            newest_name = None
            for path in tape_dir.glob("*.jsonl"):
                try:
                    mtime = path.stat().st_mtime
                except Exception:
                    continue
                if newest_mtime is None or mtime > newest_mtime:
                    newest_mtime = mtime
                    newest_name = path.name
            if newest_mtime is not None:
                tape_age = round(now - newest_mtime, 1)
                tape_detail = f"tape/{phase.lower()}/{newest_name}"

    feed = {}
    if isinstance(connector, dict):
        feed = connector.get("feed_health") or {}
        if not isinstance(feed, dict):
            feed = {}
    feed_stale = feed.get("stale_symbols") or feed.get("stale_n")
    try:
        feed_stale_n = int(feed_stale) if feed_stale is not None else None
    except Exception:
        feed_stale_n = None
    try:
        stock_live = int(
            connector.get("stock_live_symbols")
            if connector.get("stock_live_symbols") is not None
            else feed.get("stock_live_symbols") or 0
        )
    except Exception:
        stock_live = 0
    validation_n = connector.get("validation_publishes")
    if validation_n is None:
        validation_n = feed.get("validation_publishes")

    pre_rth = phase in {"PRE", "POST"}
    # PRE/POST: validate IB→Redis validation stream→tape; Scanner/OMS expected idle.
    fused_ok = 30.0 if pre_rth else 3.0
    fused_warn = 120.0 if pre_rth else 10.0
    ibkr_ok = 60.0 if pre_rth else 20.0
    ibkr_warn = 180.0 if pre_rth else 45.0

    if data_mode == "DELAYED_BLOCKED":
        ibkr_health = "🔴 DELAYED"
    elif connected is False:
        ibkr_health = "🔴 Down"
    else:
        ibkr_health = _lag_health(ibkr_age, ok=ibkr_ok, warn=ibkr_warn)

    # Split MD roles (stock publisher vs options/OMS process).
    stock_md = feed.get("stock_md") if isinstance(feed.get("stock_md"), dict) else {}
    option_md = feed.get("option_md") if isinstance(feed.get("option_md"), dict) else {}
    stock_feed_map = feed.get("stock_feed") if isinstance(feed.get("stock_feed"), dict) else {}
    option_feed_map = feed.get("option_feed") if isinstance(feed.get("option_feed"), dict) else {}
    stock_lags = [
        float(row.get("lag_sec"))
        for row in stock_feed_map.values()
        if isinstance(row, dict) and row.get("lag_sec") is not None
    ]
    option_lags = [
        float(row.get("lag_sec"))
        for row in option_feed_map.values()
        if isinstance(row, dict) and row.get("lag_sec") is not None
    ]
    stock_md_age = min(stock_lags) if stock_lags else ibkr_age
    option_md_age = min(option_lags) if option_lags else ibkr_age
    stock_live_n = sum(
        1 for lag in stock_lags if lag <= (30.0 if pre_rth else 5.0)
    )
    option_quote_n = sum(1 for lag in option_lags if lag <= (60.0 if pre_rth else 10.0))
    if stock_md.get("connected") is False:
        stock_md_health = "🔴 Down"
    elif not stock_lags and str(feed.get("md_role") or "") == "split":
        stock_md_health = "🔴 No ticks"
    else:
        stock_md_health = _lag_health(stock_md_age, ok=ibkr_ok, warn=ibkr_warn)
    if option_md.get("connected") is False:
        option_md_health = "🔴 Down"
    elif phase == "RTH" and not option_lags and option_feed_map:
        option_md_health = "🔴 No quotes"
    elif phase != "RTH" and not option_lags:
        option_md_health = "⚪ Idle (pre-lock)"
    else:
        option_md_health = _lag_health(option_md_age, ok=ibkr_ok, warn=ibkr_warn)
    # Fallback for legacy combined process: mirror IBKR node.
    if str(feed.get("md_role") or connector.get("md_role") or "combined") == "combined":
        stock_md_health = ibkr_health
        option_md_health = ibkr_health
        stock_md_age = ibkr_age
        option_md_age = ibkr_age
    stock_md_detail = (
        f"live_sym={stock_live_n}/{len(stock_feed_map) or stock_live} | "
        f"mode={stock_md.get('data_mode') or data_mode or '-'}"
    )
    option_md_detail = (
        f"quoted={option_quote_n}/{len(option_feed_map) or '-'} | "
        f"mode={option_md.get('data_mode') or data_mode or '-'} | "
        f"port={port if port is not None else '-'}"
    )

    if redis_ok is False:
        fused_health = "🔴 Down"
    elif pre_rth and stream_age is None and stock_live > 0:
        # Ticks arriving but sparse partial seconds not yet published.
        fused_health = "🟠 Warm"
    else:
        fused_health = _lag_health(stream_age, ok=fused_ok, warn=fused_warn)

    if pre_rth:
        scanner_health = "⚪ Idle (pre/post)"
        oms_health = (
            "⚪ Event Blackout"
            if planned_event_blackout
            else (
                "🔴 Halted"
                if oms_meta.get("day_halted")
                else (
                    "🔴 Reconcile"
                    if oms_meta.get("reconcile_ok") is False
                    else "⚪ Idle (pre/post)"
                )
            )
        )
        disk_age = tape_age if tape_age is not None else artifact_age
        disk_health = (
            _lag_health(disk_age, ok=60.0, warn=300.0)
            if disk_age is not None
            else ("🟠 Warm" if stock_live > 0 else "⚪ N/A")
        )
        disk_detail = tape_detail if tape_age is not None else artifact_detail
    else:
        scanner_health = (
            "🟠 Warm"
            if stock_syms == 0 and scanner_age is not None and scanner_age <= 180
            else _lag_health(scanner_age, ok=90.0, warn=180.0)
        )
        if planned_event_blackout:
            oms_health = "⚪ Event Blackout"
        elif oms_meta.get("day_halted"):
            oms_health = "🔴 Halted"
        elif oms_meta.get("reconcile_ok") is False:
            oms_health = "🔴 Reconcile"
        else:
            # Flat shadow books only publish OMS state on events; do not Crit
            # when the engine is still consuming RTH frames.
            try:
                n_pos = int(oms_meta.get("n_positions") or 0)
            except Exception:
                n_pos = 0
            try:
                n_int = int(oms_meta.get("n_intents") or 0)
            except Exception:
                n_int = 0
            engine_fresh = scanner_age is not None and scanner_age <= 10.0
            if n_pos == 0 and n_int == 0 and engine_fresh:
                oms_health = "🟢 OK (flat)"
                oms_age = scanner_age
            else:
                oms_health = _lag_health(oms_age, ok=30.0, warn=90.0)
        # RTH Disk: tape is the authority data-flow signal (artifacts may be slower).
        if tape_age is not None:
            disk_age = tape_age
            disk_detail = tape_detail
            disk_health = _lag_health(disk_age, ok=5.0, warn=30.0)
        else:
            disk_health = _lag_health(artifact_age, ok=30.0, warn=120.0)
            disk_detail = artifact_detail
            disk_age = artifact_age

    # Mag7 live path: Redis → StockMD → OptionMD → Fused → Scanner → OMS → Disk
    topology = [
        {
            "node": "Redis",
            "health": (
                "⚪ N/A"
                if redis_ok is None
                else ("🟢 OK" if redis_ok else "🔴 Down")
            ),
            "age_sec": redis_rtt_ms,
            "detail": (
                f"rtt={redis_rtt_ms}ms"
                if redis_rtt_ms is not None
                else ("unreachable" if redis_ok is False else "-")
            ),
        },
        {
            "node": "StockMD",
            "health": stock_md_health,
            "age_sec": stock_md_age,
            "detail": stock_md_detail,
        },
        {
            "node": "OptionMD",
            "health": option_md_health,
            "age_sec": option_md_age,
            "detail": option_md_detail,
        },
        {
            "node": "Fused",
            "health": fused_health,
            "age_sec": stream_age,
            "detail": (
                f"{stream_key}"
                + (f" | stale_sym={feed_stale_n}" if feed_stale_n is not None else "")
                + (f" | val_pub={validation_n}" if validation_n is not None else "")
            ),
        },
        {
            "node": "Scanner",
            "health": scanner_health,
            "age_sec": scanner_age,
            "detail": (
                f"{engine.get('state') or scanner.get('_source') or '-'} | "
                f"stock_by={stock_syms}sym/{stock_bars}bars qqq={qqq_bars}"
            ),
        },
        {
            "node": "OMS",
            "health": oms_health,
            "age_sec": oms_age,
            "detail": (
                f"mode={oms_meta.get('mode') or '-'} | "
                f"reconcile={oms_meta.get('reconcile_ok')}"
                + (" | planned full-day gate" if planned_event_blackout else "")
            ),
        },
        {
            "node": "Disk",
            "health": disk_health,
            "age_sec": disk_age,
            "detail": (
                f"{disk_detail}"
                + (f" | {session_dir.name}" if session_dir is not None else "")
            ),
        },
    ]

    def _rank(health: str) -> int:
        text = str(health or "")
        if "🔴" in text:
            return 3
        if "🟠" in text:
            return 2
        if "🟢" in text:
            return 1
        return 0

    # PRE/POST overall: only Redis / StockMD / OptionMD / Fused / Disk count.
    if pre_rth:
        scored = [
            n
            for n in topology
            if n.get("node") in {"Redis", "StockMD", "OptionMD", "IBKR", "Fused", "Disk"}
        ]
    else:
        scored = topology
    worst = max((_rank(n.get("health")) for n in scored), default=0)
    overall = {0: "⚪ N/A", 1: "🟢 Healthy", 2: "🟠 Degraded", 3: "🔴 Unhealthy"}[
        worst
    ]
    if pre_rth and worst <= 1:
        overall = f"🟢 Healthy ({phase} validate)"
    elif pre_rth and worst == 2:
        overall = f"🟠 Degraded ({phase} validate)"
    elif pre_rth and worst >= 3:
        overall = f"🔴 Unhealthy ({phase} validate)"

    return {
        "session_id": session_id,
        "env_label": env_label,
        "env_kind": env_kind,
        "port": port,
        "arm_label": arm_label,
        "armed": bool(controls.get("armed")),
        "mode": mode or oms_meta.get("mode"),
        "equity": equity,
        "available_funds": available,
        "day_halted": bool(oms_meta.get("day_halted")),
        "reconcile_ok": oms_meta.get("reconcile_ok"),
        "data_mode": data_mode,
        "connected": connected,
        "connector_state": connector.get("state"),
        "session_phase": phase,
        "topology": topology,
        "topology_overall": overall,
        "stock_by_syms": stock_syms,
        "stock_by_bars": stock_bars,
        "qqq_bars": qqq_bars,
        "stream_key": stream_key,
        "controls": controls,
        "oms_meta": oms_meta,
        "connector": connector,
        "engine": engine,
        "account_source": oms_meta.get("source") or legacy_acct.get("_source"),
    }


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

    def _flatten(item: dict[str, Any], *, kind: str | None = None) -> dict[str, Any]:
        row = dict(item)
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        row.setdefault("event_source", meta.get("event_source", "baseline"))
        row.setdefault("watchdog_state", meta.get("watchdog_state"))
        row.setdefault("route", meta.get("route"))
        row.setdefault("hunt_detector", meta.get("hunt_detector"))
        if kind:
            row["_kind"] = kind
        return row

    for item in (scanner_state or {}).get("day_fires") or []:
        if isinstance(item, dict):
            rows.append(_flatten(item))
    for item in (scanner_state or {}).get("signals") or []:
        if isinstance(item, dict):
            rows.append(_flatten(item, kind="signal"))
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
