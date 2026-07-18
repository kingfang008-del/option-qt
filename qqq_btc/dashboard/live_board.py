#!/usr/bin/env python3
"""Live Board：frozen 刷新 + shadow/dry/live 启动（对拍通过后的部署路径）。"""
from __future__ import annotations

import os
import signal
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from qqq_btc.dashboard.backfill_board import resolve_python
from qqq_btc.dashboard.parity_board import REPO, _read_json, _rel
from qqq_btc.dashboard.stream_parity_jobs import (
    DEFAULT_FROZEN_OUT,
    frozen_npz_meta,
)

DEPLOY_SCRIPT = REPO / "qqq_btc/tools/deploy_ft56_julw1_live.sh"
SHADOW_SCRIPT = REPO / "qqq_btc/tools/prepare_ft56_shadow_live.sh"
DEFAULT_LIVE_FROZEN = REPO / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"
JOB_STATE_NAME = "dash_live_job.json"
JOBS_ROOT = REPO / "qqq_btc/results/_dash_live_jobs"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        __import__("json").dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def job_state_path() -> Path:
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    return JOBS_ROOT / JOB_STATE_NAME


def read_job_state() -> dict[str, Any]:
    return _read_json(job_state_path()) or {"status": "idle"}


def suggested_live_upto_date(*, asof: str | None = None) -> str:
    """实盘默认：用昨天（NY 近似用本地昨天）作为 frozen 截止日——不含当日未走完的 bar。"""
    day = datetime.strptime(asof, "%Y-%m-%d").date() if asof else datetime.now().date()
    return (day - timedelta(days=1)).strftime("%Y-%m-%d")


def live_norm_policy() -> dict[str, str]:
    return {
        "same_as_parity": (
            "是：实盘与流式对拍共用同一套 frozen 提取逻辑——"
            "从过去的 quote_features_raw 冻 mean/std，FCS 用 FCS_FROZEN_NORM_PATH 加载。"
        ),
        "no_gold": (
            "差别：实盘没有 Offline raw/test 金标可对拍；只做因果归一化 + Deep Warmup + 推理。"
        ),
        "refresh": (
            "刷新节奏：开盘前用 --upto-date=昨收（或上月 upto-month）重导 .npz；"
            "对拍月评测用「前一月」；实盘日更常用「昨日」。"
        ),
        "warmup": (
            "预热：PG market_bars / spnq 历史喂 FCS Deep Warmup（TA/窗口指标）；"
            "与 frozen 是两件事——warmup=特征状态，frozen=归一化参数。"
        ),
    }


def build_live_export_cmd(
    *,
    features_raw_root: Path | str,
    output: Path | str,
    upto_date: str | None = None,
    upto_month: str | None = None,
    slow_config: Path | str | None = None,
    python_bin: str | None = None,
) -> list[str]:
    """实盘导出：优先 upto_date（日更）；也可 upto_month（月冻）。"""
    py = resolve_python(python_bin)
    raw_root = Path(features_raw_root).expanduser()
    if raw_root.name != "quote_features_raw" and (raw_root / "quote_features_raw").is_dir():
        raw_root = raw_root / "quote_features_raw"
    slow = Path(slow_config or (REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json")).expanduser()
    if not slow.is_absolute():
        slow = REPO / slow
    cmd = [
        py,
        str(REPO / "qqq_btc/tools/export_frozen_norm_stats.py"),
        "--symbol",
        "QQQ",
        "--stage",
        "test",
        "--features-raw-root",
        str(raw_root),
        "--slow-config",
        str(slow),
        "--output",
        str(Path(output).expanduser()),
    ]
    if upto_date:
        cmd.extend(["--upto-date", upto_date])
    elif upto_month:
        cmd.extend(["--upto-month", upto_month])
    return cmd


def start_refresh_live_frozen(
    *,
    features_raw_root: Path | str,
    output: Path | str,
    upto_date: str | None = None,
    upto_month: str | None = None,
    slow_config: Path | str | None = None,
    python_bin: str | None = None,
) -> dict[str, Any]:
    # 复用 stream 的 export job 通道会冲突；live 用独立 job state。
    import threading

    cur = read_job_state()
    if cur.get("status") == "running" and cur.get("pid"):
        try:
            os.kill(int(cur["pid"]), 0)
            raise RuntimeError(f"live job already running pid={cur['pid']}")
        except (ProcessLookupError, ValueError):
            pass

    cmd = build_live_export_cmd(
        features_raw_root=features_raw_root,
        output=output,
        upto_date=upto_date,
        upto_month=upto_month,
        slow_config=slow_config,
        python_bin=python_bin,
    )
    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = JOBS_ROOT / f"live_frozen_{stamp}.log"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    log_fh = log_file.open("w", encoding="utf-8")
    log_fh.write(f"# live frozen refresh {stamp}\n# {' '.join(cmd)}\n\n")
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
        "kind": "refresh_frozen",
        "pid": proc.pid,
        "cmd": cmd,
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_out": str(Path(output).expanduser()),
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


def start_deploy_job(
    *,
    mode: str,
    frozen_norm: Path | str,
    strategy_profile: Path | str | None = None,
    python_bin: str | None = None,
    live_trade: bool = False,
) -> dict[str, Any]:
    """mode: check | shadow | dry | live | stop"""
    import threading

    cur = read_job_state()
    if cur.get("status") == "running" and cur.get("pid") and mode != "stop":
        try:
            os.kill(int(cur["pid"]), 0)
            raise RuntimeError(f"live job already running pid={cur['pid']}")
        except (ProcessLookupError, ValueError):
            pass

    frozen = Path(frozen_norm).expanduser()
    if not frozen.is_absolute():
        frozen = REPO / frozen
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env["PYTHON"] = resolve_python(python_bin)
    env["FCS_FROZEN_NORM_PATH"] = str(frozen)
    env["FROZEN_NORM"] = str(frozen)
    if strategy_profile:
        sp = Path(strategy_profile).expanduser()
        if not sp.is_absolute():
            sp = REPO / sp
        env["QQQ_BTC_STRATEGY_PROFILE"] = str(sp)

    if mode == "stop":
        cmd = ["bash", str(DEPLOY_SCRIPT), "stop"]
    elif mode == "check":
        cmd = ["bash", str(SHADOW_SCRIPT), "check"]
    elif mode == "shadow":
        cmd = ["bash", str(SHADOW_SCRIPT), "start"]
        env["LIVE_TRADE"] = "0"
    elif mode == "dry":
        cmd = ["bash", str(DEPLOY_SCRIPT)]
        env["LIVE_TRADE"] = "0"
    elif mode == "live":
        cmd = ["bash", str(DEPLOY_SCRIPT)]
        env["LIVE_TRADE"] = "1"
    else:
        raise ValueError(f"unknown mode={mode}")

    JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = JOBS_ROOT / f"live_{mode}_{stamp}.log"
    log_fh = log_file.open("w", encoding="utf-8")
    log_fh.write(f"# live {mode} {stamp}\n# {' '.join(cmd)}\n# frozen={frozen}\n\n")
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
        "kind": f"deploy_{mode}",
        "pid": proc.pid,
        "cmd": cmd,
        "log_file": str(log_file),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "frozen_norm": str(frozen),
        "live_trade": bool(live_trade or mode == "live"),
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


def stop_job() -> dict[str, Any]:
    state = read_job_state()
    pid = state.get("pid")
    # 同时停实盘栈
    try:
        subprocess.run(
            ["bash", str(DEPLOY_SCRIPT), "stop"],
            cwd=str(REPO),
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        pass
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


def default_paths() -> dict[str, str]:
    return {
        "frozen_raw_root": str(Path.home() / "train_data/quote_features_raw"),
        "frozen_live": str(DEFAULT_LIVE_FROZEN),
        "frozen_dash": str(DEFAULT_FROZEN_OUT),
        "deploy_script": _rel(DEPLOY_SCRIPT),
        "shadow_script": _rel(SHADOW_SCRIPT),
    }
