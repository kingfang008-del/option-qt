#!/usr/bin/env python3
"""Dashboard Backfill：扫描产物 + 一键启动开盘价锁约补数流水线。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
# 与 step2_polygon_second_sniper_v1 默认一致：quote 落在 $EXP/{SYM}/{SYM}_{date}.parquet
DEFAULT_EXP = Path("/mnt/s990/data/raw_1s/dte1_options_old_lock")
DEFAULT_STOCK_RESAMP = Path.home() / "train_data/spnq_train_resampled"
DEFAULT_FEATURES_ROOT = Path.home() / "train_data/dte1_options_old_lock_feat"
DEFAULT_FEAT_HISTORY = Path.home() / "train_data/quote_features_raw"
DEFAULT_FROZEN_NORM = REPO / "qqq_btc/CONFIG/frozen_norm_qqq_daily.npz"
PIPELINE_SCRIPT = REPO / "preprocess/download/run_backfill_open_lock_pipeline.py"
LOCK_SCRIPT = REPO / "preprocess/download/step1_lock_4bucket_from_open.py"
ANCHOR_CONFIG = REPO / "preprocess/CONFIG/anchor_qqq_1dte_4bucket.json"
DEFAULT_PY = Path(
    os.environ.get("PYTHON")
    or (Path.home() / "anaconda3/envs/ibkr/bin/python")
)
JOB_STATE_NAME = "dash_backfill_job.json"


def _path_row(role: str, path: Path, *, note: str = "") -> dict[str, Any]:
    p = Path(path).expanduser()
    exists = p.exists()
    n_children = None
    if exists and p.is_dir():
        try:
            n_children = sum(1 for _ in p.iterdir())
        except OSError:
            n_children = None
    elif exists and p.is_file() and p.suffix == ".parquet":
        n_children = 1
    return {
        "role": role,
        "path": str(p),
        "exists": "yes" if exists else "no",
        "kind": "dir" if p.is_dir() else ("file" if p.is_file() else "-"),
        "entries": n_children,
        "note": note,
    }


def data_directory_rows(
    exp: Path | None = None,
    *,
    stock_resamp: Path | None = None,
    feat_history_root: Path | None = None,
    symbols: str = "QQQ",
) -> list[dict[str, Any]]:
    """Download 页展示的数据目录清单（quote exp + 股价依赖）。"""
    root = Path(exp or DEFAULT_EXP).expanduser()
    stock = Path(stock_resamp or DEFAULT_STOCK_RESAMP).expanduser()
    feat_hist = Path(feat_history_root or DEFAULT_FEAT_HISTORY).expanduser()
    sym = (symbols.split(",")[0] if symbols else "QQQ").strip().upper() or "QQQ"
    quote_dir = root / sym
    n_quotes = 0
    if quote_dir.is_dir():
        n_quotes = sum(1 for _ in quote_dir.glob(f"{sym}_*.parquet"))
    rows = [
        _path_row(
            "exp_root (quote)",
            root,
            note="1s quote 根目录；文件在 {SYM}/{SYM}_{date}.parquet",
        ),
        {
            **_path_row(
                f"quotes/{sym}",
                quote_dir,
                note=f"已有 {n_quotes} 个日 quote parquet",
            ),
            "entries": n_quotes if quote_dir.is_dir() else None,
        },
        _path_row(
            "by_date",
            root / "by_date",
            note="按日 staging（lock / 链接到 quote·1m·day_iv）",
        ),
        _path_row(
            "options_1m",
            root / "options_1m",
            note="1s→1m 聚合（派生，可写在 exp 下）",
        ),
        _path_row(
            "day_iv",
            root / "quote_options_day_iv",
            note="日 IV / volume（派生）",
        ),
        _path_row("logs", root / "logs", note="流水线与 dash job 日志"),
        _path_row(
            "lock_map",
            root / "locked_targets_map_open_4bucket.parquet",
            note="开盘价锁约 map",
        ),
        _path_row("lock_report", root / "lock_report.json", note="锁约报告"),
        _path_row("warmup_report", root / "warmup_report.json", note="缺数/预热报告"),
        _path_row(
            "pipeline_summary",
            root / "pipeline_summary.json",
            note="最近一次流水线摘要",
        ),
        _path_row(
            "stock_1min (缺数比对)",
            stock / sym / "regular/09:30-16:00/1min",
            note="Coverage/预热扫这里（股价），不是 quote exp",
        ),
        _path_row(
            "stock_root",
            stock,
            note="~/train_data/spnq_train_resampled",
        ),
        _path_row(
            "vixy_1min (缺数比对)",
            stock / "VIXY" / "regular/09:30-16:00/1min",
            note="VIXY 预热 / 历史月",
        ),
        _path_row(
            "anchor_config",
            ANCHOR_CONFIG,
            note="4-bucket 锁约配置",
        ),
        _path_row(
            "feat_history (later)",
            feat_hist,
            note="特征/rolling 前置，本页不写",
        ),
    ]
    return rows


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


@dataclass
class BackfillDay:
    date: str
    path: Path
    manifest: dict[str, Any]

    @property
    def n_contracts(self) -> int:
        try:
            return int(self.manifest.get("n_contracts") or 0)
        except Exception:
            return 0

    @property
    def stock_open(self) -> Optional[float]:
        val = self.manifest.get("stock_open")
        try:
            return float(val) if val is not None else None
        except Exception:
            return None


def discover_backfill_days(exp: Path | None = None, *, limit: int = 60) -> list[BackfillDay]:
    root = Path(exp or DEFAULT_EXP).expanduser()
    by_date = root / "by_date"
    if not by_date.is_dir():
        return []
    days: list[BackfillDay] = []
    for child in sorted(by_date.iterdir(), reverse=True):
        if not child.is_dir():
            continue
        man = _read_json(child / "manifest.json") or {}
        days.append(BackfillDay(date=child.name, path=child, manifest=man))
        if len(days) >= limit:
            break
    return days


def load_pipeline_summary(exp: Path | None = None) -> dict[str, Any]:
    root = Path(exp or DEFAULT_EXP).expanduser()
    return _read_json(root / "pipeline_summary.json") or {}


def load_lock_report(exp: Path | None = None) -> dict[str, Any]:
    root = Path(exp or DEFAULT_EXP).expanduser()
    return _read_json(root / "lock_report.json") or {}


def api_key_status() -> dict[str, Any]:
    for name in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "POLYGON_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return {"ok": True, "env": name, "hint": f"{name}=***{val[-4:]}"}
    return {
        "ok": False,
        "env": None,
        "hint": "未检测到 MASSIVE_API_KEY / POLYGON_API_KEY；启动前请在环境或下方输入框提供",
    }


def resolve_python(explicit: str | None = None) -> str:
    candidates = [
        explicit,
        os.environ.get("PYTHON"),
        str(DEFAULT_PY),
        "python3",
    ]
    for raw in candidates:
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return "python3"


def build_pipeline_cmd(
    *,
    start_date: str,
    end_date: str,
    exp: Path,
    mode: str,
    force: bool = False,
    symbols: str = "QQQ",
    python_bin: str | None = None,
    max_workers: int = 16,
    assume_iv: float = 0.22,
    strict_warmup: bool = True,
    warmup_trading_days: int = 10,
    vix_history_months: int = 7,
    norm_mode: str = "rolling",
    features_root: Path | str | None = None,
    feat_history_root: Path | str | None = None,
    frozen_norm: Path | str | None = None,
) -> list[str]:
    """mode: lock_only | download | full(features+norm)."""
    py = resolve_python(python_bin)
    cmd = [
        py,
        str(PIPELINE_SCRIPT),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--symbols",
        symbols,
        "--exp",
        str(exp),
        "--config",
        str(ANCHOR_CONFIG),
        "--python",
        py,
        "--assume-iv",
        str(assume_iv),
        "--max-workers",
        str(max_workers),
        "--warmup-trading-days",
        str(warmup_trading_days),
        "--vix-history-months",
        str(vix_history_months),
    ]
    if mode == "lock_only":
        cmd.append("--lock-only")
    elif mode == "download":
        pass
    elif mode == "full":
        feat_root = Path(features_root or DEFAULT_FEATURES_ROOT).expanduser()
        hist = Path(feat_history_root or DEFAULT_FEAT_HISTORY).expanduser()
        frozen = Path(frozen_norm or DEFAULT_FROZEN_NORM).expanduser()
        cmd.append("--features")
        cmd.extend(["--norm-mode", norm_mode])
        cmd.extend(["--features-root", str(feat_root)])
        cmd.extend(["--feat-history-root", str(hist)])
        cmd.extend(["--frozen-norm", str(frozen)])
        if strict_warmup:
            cmd.append("--strict-warmup")
    else:
        raise ValueError(f"unknown mode={mode!r}")
    if force:
        cmd.append("--force")
    return cmd


def suggested_commands(
    *,
    start_date: str,
    end_date: str,
    exp: Path | None = None,
    features: bool = False,
) -> dict[str, str]:
    root = Path(exp or DEFAULT_EXP).expanduser()
    mode = "full" if features else "download"
    cmd = build_pipeline_cmd(
        start_date=start_date,
        end_date=end_date,
        exp=root,
        mode=mode,
    )
    lock_cmd = build_pipeline_cmd(
        start_date=start_date,
        end_date=end_date,
        exp=root,
        mode="lock_only",
    )
    return {
        "lock_only": " ".join(lock_cmd),
        "full_pipeline": " ".join(cmd),
        "note": "需要 MASSIVE_API_KEY 或 POLYGON_API_KEY；不再先下全市场 trades。",
    }


def job_state_path(exp: Path) -> Path:
    return Path(exp).expanduser() / "logs" / JOB_STATE_NAME


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_job_state(exp: Path) -> dict[str, Any]:
    state = _read_json(job_state_path(exp)) or {}
    if not state:
        return {"status": "idle"}
    pid = int(state.get("pid") or 0)
    if state.get("status") == "running" and pid and not _pid_alive(pid):
        # process died; try read exit from marker
        exit_path = Path(str(state.get("exit_file") or ""))
        code = None
        if exit_path.is_file():
            try:
                code = int(exit_path.read_text().strip() or "1")
            except Exception:
                code = 1
        state["status"] = "done" if code == 0 else "failed"
        state["exit_code"] = code
        state["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _write_json(job_state_path(exp), state)
    return state


def tail_log(path: Path | str | None, *, max_bytes: int = 32_000) -> str:
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


def load_warmup_report(exp: Path | None = None) -> dict[str, Any]:
    root = Path(exp or DEFAULT_EXP).expanduser()
    return _read_json(root / "warmup_report.json") or {}


def run_warmup_check_job(
    *,
    start_date: str,
    end_date: str,
    exp: Path,
    symbols: str = "QQQ",
    python_bin: str | None = None,
    warmup_trading_days: int = 10,
    vix_history_months: int = 7,
) -> dict[str, Any]:
    """同步跑预热检查并写 warmup_report.json。"""
    root = Path(exp).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "warmup_report.json"
    py = resolve_python(python_bin)
    cmd = [
        py,
        str(REPO / "preprocess/download/backfill_warmup_check.py"),
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--symbols",
        symbols,
        "--warmup-trading-days",
        str(warmup_trading_days),
        "--vix-history-months",
        str(vix_history_months),
        "--report",
        str(report_path),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(cmd, cwd=str(REPO), env=env, capture_output=True, text=True)
    report = _read_json(report_path) or {}
    report["_exit_code"] = proc.returncode
    report["_stdout_tail"] = (proc.stdout or "")[-2000:]
    report["_stderr_tail"] = (proc.stderr or "")[-2000:]
    return report


def start_backfill_job(
    *,
    start_date: str,
    end_date: str,
    exp: Path,
    mode: str,
    force: bool = False,
    symbols: str = "QQQ",
    python_bin: str | None = None,
    api_key: str | None = None,
    max_workers: int = 16,
    strict_warmup: bool = True,
    warmup_trading_days: int = 10,
    vix_history_months: int = 7,
    norm_mode: str = "rolling",
    features_root: Path | str | None = None,
    feat_history_root: Path | str | None = None,
    frozen_norm: Path | str | None = None,
) -> dict[str, Any]:
    """后台启动流水线；返回 job state。同一 exp 同时只允许一个 running job。"""
    import threading

    root = Path(exp).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    current = read_job_state(root)
    if current.get("status") == "running":
        raise RuntimeError(f"已有任务在跑 pid={current.get('pid')} log={current.get('log_file')}")

    cmd = build_pipeline_cmd(
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        exp=root,
        mode=mode,
        force=force,
        symbols=symbols,
        python_bin=python_bin,
        max_workers=max_workers,
        strict_warmup=strict_warmup,
        warmup_trading_days=warmup_trading_days,
        vix_history_months=vix_history_months,
        norm_mode=norm_mode,
        features_root=features_root,
        feat_history_root=feat_history_root,
        frozen_norm=frozen_norm,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs / f"backfill_{mode}_{start_date}_{end_date}_{stamp}.log"
    exit_file = logs / f"backfill_{mode}_{stamp}.exit"
    if exit_file.exists():
        exit_file.unlink()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    key = (api_key or "").strip()
    if key:
        env["MASSIVE_API_KEY"] = key
        env["POLYGON_API_KEY"] = key
    elif not api_key_status()["ok"]:
        raise RuntimeError("缺少 API key：请设置环境变量或在页面输入 Massive/Polygon key")

    log_fh = open(log_file, "w", encoding="utf-8")
    log_fh.write(f"# started {stamp}\n# cmd: {' '.join(cmd)}\n\n")
    log_fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def _wait_and_mark() -> None:
        code = 1
        try:
            code = int(proc.wait())
        finally:
            try:
                log_fh.close()
            except Exception:
                pass
            try:
                exit_file.write_text(str(code), encoding="utf-8")
            except Exception:
                pass
            # Refresh persisted state if this job is still the active one.
            try:
                latest = _read_json(job_state_path(root)) or {}
                if int(latest.get("pid") or 0) == proc.pid:
                    latest["status"] = "done" if code == 0 else "failed"
                    latest["exit_code"] = code
                    latest["finished_at"] = datetime.now().isoformat(timespec="seconds")
                    _write_json(job_state_path(root), latest)
            except Exception:
                pass

    threading.Thread(target=_wait_and_mark, daemon=True).start()

    state = {
        "status": "running",
        "pid": proc.pid,
        "mode": mode,
        "start_date": start_date,
        "end_date": end_date,
        "symbols": symbols,
        "exp": str(root),
        "cmd": cmd,
        "log_file": str(log_file),
        "exit_file": str(exit_file),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "force": bool(force),
    }
    _write_json(job_state_path(root), state)
    return state


def stop_backfill_job(exp: Path) -> dict[str, Any]:
    state = read_job_state(exp)
    if state.get("status") != "running":
        return state
    pid = int(state.get("pid") or 0)
    if pid and _pid_alive(pid):
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        time.sleep(0.5)
        if _pid_alive(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    state["status"] = "stopped"
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(job_state_path(exp), state)
    return state


def scan_stock_bar_labels(
    *,
    start_date: str,
    end_date: str,
    symbols: str = "QQQ,VIXY",
    stock_resamp: Path | None = None,
) -> dict[str, Any]:
    """Download 页：扫描 resampled 股价是否符合 W1 右标签。"""
    from qqq_btc.common.bar_label_convention import scan_bar_labels

    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    # W1 对拍最少扫 QQQ；VIXY 一并查（put_gate / regime）
    if "QQQ" not in syms:
        syms = ["QQQ"] + syms
    if "VIXY" not in syms:
        syms.append("VIXY")
    return scan_bar_labels(
        stock_root=Path(stock_resamp or DEFAULT_STOCK_RESAMP),
        symbols=syms,
        start=start_date,
        end=end_date,
    )


def fix_stock_bar_labels(
    *,
    start_date: str,
    end_date: str,
    symbols: str = "QQQ,VIXY",
    stock_resamp: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """一键把区间内左标签 resampled 股价改为 W1 右标签。"""
    from qqq_btc.common.bar_label_convention import fix_bar_labels

    syms = [s.strip() for s in symbols.split(",") if s.strip()]
    if "QQQ" not in syms:
        syms = ["QQQ"] + syms
    if "VIXY" not in syms:
        syms.append("VIXY")
    return fix_bar_labels(
        stock_root=Path(stock_resamp or DEFAULT_STOCK_RESAMP),
        symbols=syms,
        start=start_date,
        end=end_date,
        dry_run=dry_run,
    )
