#!/usr/bin/env python3
"""Stream Parity 作业：从离线 raw 导出 frozen → 触发流式三闸门。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from qqq_btc.dashboard.backfill_board import (
    DEFAULT_FEATURES_ROOT,
    resolve_python,
)
from qqq_btc.dashboard.parity_board import REPO, ParityRecipe, _read_json, _rel

EXPORT_SCRIPT = REPO / "qqq_btc/tools/export_frozen_norm_stats.py"
DEFAULT_FROZEN_OUT = REPO / "qqq_btc/CONFIG/frozen_norm_dash_stream.npz"
JOB_STATE_NAME = "dash_stream_parity_job.json"
JOBS_ROOT = REPO / "qqq_btc/results/_dash_stream_jobs"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def job_state_path() -> Path:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    return JOBS_ROOT / JOB_STATE_NAME


def read_job_state() -> dict[str, Any]:
    return _read_json(job_state_path()) or {"status": "idle"}


def _prev_month(ym: str) -> str:
    y, m = [int(x) for x in ym.split("-")]
    m -= 1
    if m <= 0:
        m = 12
        y -= 1
    return f"{y:04d}-{m:02d}"


def resolve_features_bundle(
    features_root: Path | str,
    *,
    symbol: str = "QQQ",
    month: str = "2026-07",
) -> dict[str, Any]:
    root = Path(features_root).expanduser()
    raw = root / "quote_features_raw" / symbol / "regular/09:30-16:00/1min" / f"{month}.parquet"
    norm = root / "quote_features_test" / symbol / "regular/09:30-16:00/1min" / f"{month}.parquet"
    raw_dir = root / "quote_features_raw"
    return {
        "features_root": str(root),
        "quote_features_raw_root": str(raw_dir),
        "offline_raw": str(raw),
        "offline_norm": str(norm),
        "raw_exists": raw.is_file(),
        "norm_exists": norm.is_file(),
        "raw_month_files": sorted(p.name for p in (raw_dir / symbol / "regular/09:30-16:00/1min").glob("*.parquet"))
        if (raw_dir / symbol / "regular/09:30-16:00/1min").is_dir()
        else [],
    }


def build_export_frozen_cmd(
    *,
    features_raw_root: Path | str,
    output: Path | str,
    upto_month: str,
    symbol: str = "QQQ",
    slow_config: Path | str | None = None,
    python_bin: str | None = None,
) -> list[str]:
    """features_raw_root: quote_features_raw 目录（或其父 features_root 下的 raw）。"""
    py = resolve_python(python_bin)
    raw_root = Path(features_raw_root).expanduser()
    if raw_root.name != "quote_features_raw" and (raw_root / "quote_features_raw").is_dir():
        raw_root = raw_root / "quote_features_raw"
    slow = Path(
        slow_config or (REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json")
    ).expanduser()
    if not slow.is_absolute():
        slow = REPO / slow
    return [
        py,
        str(EXPORT_SCRIPT),
        "--symbol",
        symbol,
        "--stage",
        "test",
        "--features-raw-root",
        str(raw_root),
        "--upto-month",
        upto_month,
        "--slow-config",
        str(slow),
        "--output",
        str(Path(output).expanduser()),
    ]


def build_stream_parity_cmd(
    *,
    recipe: ParityRecipe,
    frozen_norm: Path | str,
    features_root: Path | str,
    days: str,
    out_dir: Path | str | None = None,
    python_bin: str | None = None,
) -> tuple[list[str], dict[str, str]]:
    """返回 (cmd, extra_env)。"""
    if not recipe.stream_script and not recipe.stream_cmd:
        raise ValueError(f"recipe {recipe.recipe_id} 无 stream 入口")
    bundle = resolve_features_bundle(features_root, month=(days.strip().split()[0][:7] if days.strip() else "2026-07"))
    frozen = Path(frozen_norm).expanduser()
    if not frozen.is_absolute():
        frozen = REPO / frozen
    feat_root = Path(features_root).expanduser()
    env: dict[str, str] = {
        "FROZEN_NORM": str(frozen),
        "FCS_FROZEN_NORM_PATH": str(frozen),
        "HONEST_FEAT_ROOT": str(feat_root),
        "OFFLINE_RAW": bundle["offline_raw"],
        "OFFLINE_NORM": bundle["offline_norm"],
        "DAYS": days.strip(),
        "PYTHON": resolve_python(python_bin),
    }
    if recipe.strategy_profile:
        env["QQQ_BTC_STRATEGY_PROFILE"] = str(recipe.strategy_profile)
        # V0 wrapper
        env["V0_PROFILE"] = str(recipe.strategy_profile)
    if out_dir:
        out_p = Path(out_dir).expanduser()
        env["HONEST_OUT_DIR"] = str(out_p)
        env["V0_STREAM_OUT_DIR"] = str(out_p)
    env["V0_DAYS"] = days.strip()

    if recipe.stream_script and recipe.stream_script.is_file():
        cmd = ["bash", str(recipe.stream_script)]
    else:
        # stream_cmd 可能含 env 赋值；用 bash -lc
        cmd = ["bash", "-lc", str(recipe.stream_cmd)]
    return cmd, env


def _start_bg_job(
    *,
    kind: str,
    cmd: list[str],
    extra_env: dict[str, str] | None = None,
    log_prefix: str,
) -> dict[str, Any]:
    import threading

    cur = read_job_state()
    if cur.get("status") == "running" and cur.get("pid"):
        try:
            os.kill(int(cur["pid"]), 0)
            raise RuntimeError(f"已有 stream parity 任务在跑 pid={cur['pid']}")
        except (ProcessLookupError, ValueError):
            pass

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = JOBS_ROOT / f"{log_prefix}_{stamp}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items() if v is not None})

    log_fh = log_file.open("w", encoding="utf-8")
    log_fh.write(f"# started {stamp}\n# kind={kind}\n# cmd: {' '.join(cmd)}\n")
    if extra_env:
        log_fh.write("# env:\n")
        for k in sorted(extra_env):
            log_fh.write(f"#   {k}={extra_env[k]}\n")
    log_fh.write("\n")
    log_fh.flush()

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
        "kind": kind,
        "pid": proc.pid,
        "cmd": cmd,
        "env": extra_env or {},
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(job_state_path(), state)

    def _wait() -> None:
        code = proc.wait()
        log_fh.close()
        done = read_job_state()
        if done.get("pid") != proc.pid:
            return
        done["status"] = "done" if code == 0 else "failed"
        done["exit_code"] = code
        done["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(job_state_path(), done)

    threading.Thread(target=_wait, daemon=True).start()
    return state


def start_export_frozen_job(
    *,
    features_raw_root: Path | str,
    output: Path | str,
    upto_month: str,
    symbol: str = "QQQ",
    slow_config: Path | str | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    cmd = build_export_frozen_cmd(
        features_raw_root=features_raw_root,
        output=output,
        upto_month=upto_month,
        symbol=symbol,
        slow_config=slow_config,
        python_bin=python_bin,
    )
    return _start_bg_job(kind="export_frozen", cmd=cmd, log_prefix="export_frozen")


def start_stream_parity_job(
    *,
    recipe: ParityRecipe,
    frozen_norm: Path | str,
    features_root: Path | str,
    days: str,
    out_dir: Path | str | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    cmd, env = build_stream_parity_cmd(
        recipe=recipe,
        frozen_norm=frozen_norm,
        features_root=features_root,
        days=days,
        out_dir=out_dir,
        python_bin=python_bin,
    )
    return _start_bg_job(
        kind="stream_parity",
        cmd=cmd,
        extra_env=env,
        log_prefix=f"stream_{recipe.recipe_id}",
    )


def stop_job() -> dict[str, Any]:
    state = read_job_state()
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
        _write_json(job_state_path(), state)
    return state


def tail_log(path: str | Path | None, *, max_bytes: int = 32_000) -> str:
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


def frozen_npz_meta(path: Path | str) -> dict[str, Any]:
    p = Path(path).expanduser()
    if not p.is_file():
        return {"exists": False, "path": str(p)}
    try:
        import numpy as np

        z = np.load(p, allow_pickle=True)
        return {
            "exists": True,
            "path": str(p),
            "size": p.stat().st_size,
            "mtime": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
            "upto_month": str(z["upto_month"]) if "upto_month" in z.files else "",
            "upto_date": str(z["upto_date"]) if "upto_date" in z.files else "",
            "source_dir": str(z["source_dir"]) if "source_dir" in z.files else "",
            "dims": int(len(z["feature_names"])) if "feature_names" in z.files else None,
            "frames": int(z["count"]) if "count" in z.files else None,
        }
    except Exception as exc:
        return {"exists": True, "path": str(p), "error": str(exc)}


def default_paths() -> dict[str, str]:
    return {
        "features_root": str(DEFAULT_FEATURES_ROOT),
        "frozen_raw_root": str(Path.home() / "train_data/quote_features_raw"),
        "frozen_out": str(DEFAULT_FROZEN_OUT),
        "rel_frozen_out": _rel(DEFAULT_FROZEN_OUT),
    }
