#!/usr/bin/env python3
"""Dashboard Offline Replay：扫描离线结果、日收益诊断、一键启动 live-aligned replay。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from qqq_btc.dashboard.backfill_board import resolve_python
from qqq_btc.dashboard.parity_board import (
    OFFLINE_ROOT,
    PROFILES_DIR,
    REPO,
    load_catalog,
    offline_headline,
)

REPLAY_SCRIPT = REPO / "qqq_btc" / "tools" / "replay_offline_live_aligned.py"
JOB_STATE_NAME = "dash_offline_job.json"


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except Exception:
        return str(path)


@dataclass
class OfflineRun:
    path: Path
    name: str
    mtime: float
    summary: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)

    @property
    def profile_id(self) -> str:
        return str(
            self.manifest.get("strategy_profile_id")
            or (self.summary.get("provenance") or {}).get("strategy_profile_id")
            or ""
        )

    @property
    def profile_sha(self) -> str:
        return str(
            self.manifest.get("strategy_profile_sha256")
            or (self.summary.get("provenance") or {}).get("strategy_profile_sha256")
            or ""
        )


def discover_offline_runs(*, root: Path | None = None, limit: int = 40) -> list[OfflineRun]:
    base = Path(root or OFFLINE_ROOT).expanduser()
    if not base.is_dir():
        return []
    candidates: list[Path] = []
    for child in base.iterdir():
        if not child.is_dir() or child.name in {"LATEST"}:
            continue
        if (child / "summary.json").is_file() or (child / "manifest.json").is_file():
            candidates.append(child)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out: list[OfflineRun] = []
    for path in candidates:
        summary = _read_json(path / "summary.json") or {}
        manifest = _read_json(path / "manifest.json") or {}
        if not summary and not manifest:
            continue
        out.append(
            OfflineRun(
                path=path,
                name=path.name,
                mtime=path.stat().st_mtime,
                summary=summary,
                manifest=manifest,
            )
        )
        if len(out) >= limit:
            break
    return out


def headline_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    headline = summary.get("headline")
    if isinstance(headline, dict):
        for ym, payload in headline.items():
            if not isinstance(payload, dict):
                continue
            rows.append(
                {
                    "month": ym,
                    "acct25_pct": payload.get("acct25_pct"),
                    "trades": payload.get("trades"),
                    "mdd_pct": payload.get("mdd_pct"),
                    "profiles": payload.get("profiles"),
                    "delta_vs_baseline_pp": payload.get("delta_vs_baseline_pp"),
                }
            )
    if rows:
        return rows
    months = summary.get("months")
    if isinstance(months, dict):
        for ym, payload in months.items():
            if not isinstance(payload, dict):
                continue
            regime = payload.get("regime") or {}
            rows.append(
                {
                    "month": ym,
                    "acct25_pct": regime.get("acct25_pct"),
                    "trades": regime.get("trades"),
                    "mdd_pct": regime.get("mdd_pct"),
                    "hit_pct": regime.get("hit_pct"),
                    "legs": regime.get("legs"),
                }
            )
    return rows


def daily_rows_from_summary(
    summary: dict[str, Any], *, month: str | None = None
) -> list[dict[str, Any]]:
    months = summary.get("months")
    if not isinstance(months, dict):
        return []
    keys = [month] if month and month in months else list(months.keys())
    rows: list[dict[str, Any]] = []
    for ym in keys:
        payload = months.get(ym) or {}
        if not isinstance(payload, dict):
            continue
        regime = payload.get("regime") or {}
        day_profiles = payload.get("day_profiles") or {}
        by_date: dict[str, dict[str, Any]] = {}
        for drow in regime.get("daily") or []:
            if not isinstance(drow, dict):
                continue
            day = str(drow.get("date") or "")[:10]
            legs = drow.get("legs") or {}
            by_date[day] = {
                "month": ym,
                "date": day,
                "n": drow.get("n"),
                "day_acct25_pct": (
                    float(drow["day_acct25"]) * 100.0
                    if drow.get("day_acct25") is not None
                    else None
                ),
                "cum_acct25_pct": (
                    float(drow["cum_acct25"]) * 100.0
                    if drow.get("cum_acct25") is not None
                    else None
                ),
                "hit": drow.get("hit"),
                "put": (legs.get("PUT") if isinstance(legs, dict) else None),
                "call": (legs.get("CALL") if isinstance(legs, dict) else None),
                "profile": (
                    day_profiles.get(day) if isinstance(day_profiles, dict) else None
                ),
            }
        # 补齐 day_profiles 有、但无成交的日（n=0），避免页面看起来像缺 10/13
        if isinstance(day_profiles, dict):
            for day in sorted(day_profiles):
                if day in by_date:
                    continue
                by_date[day] = {
                    "month": ym,
                    "date": day,
                    "n": 0,
                    "day_acct25_pct": 0.0,
                    "cum_acct25_pct": None,
                    "hit": None,
                    "put": 0,
                    "call": 0,
                    "profile": day_profiles.get(day),
                }
        rows.extend(by_date[d] for d in sorted(by_date))
    return rows


def diagnostic_bundle(summary: dict[str, Any], *, month: str) -> dict[str, Any]:
    months = summary.get("months") or {}
    payload = months.get(month) if isinstance(months, dict) else None
    if not isinstance(payload, dict):
        return {}
    regime = payload.get("regime") or {}
    baseline = payload.get("baseline_TREND_PUT_OK") or {}
    daily = daily_rows_from_summary(summary, month=month)
    winners = [r for r in daily if (r.get("day_acct25_pct") or 0) > 0]
    losers = [r for r in daily if (r.get("day_acct25_pct") or 0) < 0]
    worst = sorted(daily, key=lambda r: (r.get("day_acct25_pct") is None, r.get("day_acct25_pct") or 0))[
        :5
    ]
    best = sorted(
        daily,
        key=lambda r: (r.get("day_acct25_pct") is None, -(r.get("day_acct25_pct") or 0)),
    )[:5]
    return {
        "month": month,
        "acct25_pct": regime.get("acct25_pct"),
        "trades": regime.get("trades"),
        "hit_pct": regime.get("hit_pct"),
        "mdd_pct": regime.get("mdd_pct"),
        "legs": regime.get("legs"),
        "early4_min_cum": regime.get("early4_min_cum"),
        "segments": regime.get("segments") or [],
        "profile_day_counts": payload.get("profile_day_counts"),
        "delta_vs_baseline_pp": payload.get("delta_regime_vs_baseline_pp"),
        "baseline_acct25_pct": baseline.get("acct25_pct"),
        "baseline_trades": baseline.get("trades"),
        "n_win_days": len(winners),
        "n_loss_days": len(losers),
        "worst_days": worst,
        "best_days": best,
        "gates": summary.get("gates") or {},
    }


def recipe_offline_options() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for recipe in load_catalog():
        if not recipe.offline_cmd and not recipe.offline_result:
            continue
        hl = offline_headline(recipe.offline_result)
        rows.append(
            {
                "recipe_id": recipe.recipe_id,
                "title": recipe.title,
                "profile": _rel(recipe.strategy_profile) if recipe.strategy_profile else "",
                "offline_cmd": recipe.offline_cmd,
                "offline_result": _rel(recipe.offline_result) if recipe.offline_result else "",
                "baseline_acct25_pct": recipe.baseline_acct25_pct or hl.get("acct25_pct"),
                "baseline_trades": recipe.baseline_trades or hl.get("trades"),
                "notes": recipe.notes,
            }
        )
    return rows


def build_replay_cmd(
    *,
    strategy_profile: str | Path,
    months: str,
    out_name: str,
    python_bin: str | None = None,
) -> list[str]:
    py = resolve_python(python_bin)
    profile = Path(str(strategy_profile)).expanduser()
    if not profile.is_absolute():
        profile = REPO / profile
    return [
        py,
        str(REPLAY_SCRIPT),
        "--months",
        months,
        "--strategy-profile",
        str(profile),
        "--out-name",
        out_name,
    ]


def _job_state_path(root: Path | None = None) -> Path:
    base = Path(root or OFFLINE_ROOT).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    return base / JOB_STATE_NAME


def read_job_state(root: Path | None = None) -> dict[str, Any]:
    return _read_json(_job_state_path(root)) or {"status": "idle"}


def _write_job_state(state: dict[str, Any], root: Path | None = None) -> None:
    path = _job_state_path(root)
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def start_offline_replay_job(
    *,
    strategy_profile: str | Path,
    months: str,
    out_name: str,
    python_bin: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    """后台启动 offline live-aligned replay；同一 offline root 同时只允许一个 running job。"""
    import threading

    base = Path(root or OFFLINE_ROOT).expanduser()
    base.mkdir(parents=True, exist_ok=True)
    cur = read_job_state(base)
    if cur.get("status") == "running" and cur.get("pid"):
        try:
            os.kill(int(cur["pid"]), 0)
            raise RuntimeError(f"offline replay already running pid={cur['pid']}")
        except (ProcessLookupError, ValueError):
            pass

    cmd = build_replay_cmd(
        strategy_profile=strategy_profile,
        months=months,
        out_name=out_name,
        python_bin=python_bin,
    )
    logs = base / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs / f"offline_replay_{out_name}_{stamp}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )

    log_fh = log_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state = {
        "status": "running",
        "pid": proc.pid,
        "cmd": cmd,
        "months": months,
        "out_name": out_name,
        "strategy_profile": str(strategy_profile),
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_job_state(state, base)

    def _wait() -> None:
        code = proc.wait()
        log_fh.close()
        done = read_job_state(base)
        if done.get("pid") != proc.pid:
            return
        done["status"] = "done" if code == 0 else "failed"
        done["exit_code"] = code
        done["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_job_state(done, base)

    threading.Thread(target=_wait, daemon=True).start()
    return state


def stop_offline_replay_job(root: Path | None = None) -> dict[str, Any]:
    base = Path(root or OFFLINE_ROOT).expanduser()
    state = read_job_state(base)
    pid = state.get("pid")
    if state.get("status") == "running" and pid:
        try:
            os.killpg(int(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, ValueError):
            try:
                os.kill(int(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, ValueError):
                pass
        state["status"] = "stopped"
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_job_state(state, base)
    return state


def tail_log(path: str | Path | None, *, max_bytes: int = 24_000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    size = p.stat().st_size
    with p.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            data = fh.read()
        else:
            data = fh.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        return f"... truncated ({size} bytes) ...\n{text}"
    return text


def feature_norm_hint() -> dict[str, str]:
    """特征/归一化不在 Download 页；离线/流式前在数据侧单独做。"""
    return {
        "rolling": (
            "经典离线：对连续 quote_features_raw 跑 apply_rolling_norm_standalone "
            "(window=2000)，再喂 replay / 训练。"
        ),
        "frozen": (
            "流式对拍：用 export_frozen_norm_stats.py 从离线 raw 导出 .npz，"
            "FCS_FROZEN_NORM_PATH + 离线 quote_features_test 双边共用。"
        ),
        "note": (
            "Download 只负责锁约/quote/day_iv；完整连续特征与归一化属于 Offline Replay / "
            "Stream Parity 的前置数据准备，避免下载页逻辑分叉。"
        ),
        "profiles_dir": _rel(PROFILES_DIR),
    }
