#!/usr/bin/env python3
"""Mag7 Download 板：缺数扫描 + 一键启停补数任务（对齐 qqq_btc backfill_board）。"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[1]
DEFAULT_STOCK_1S = Path("/mnt/s990/data/raw_1s/stocks")
DEFAULT_CALENDAR = REPO / "maga7" / "CONFIG" / "event_calendar_live.json"
DEFAULT_PY = Path(
    os.environ.get("PYTHON")
    or (Path.home() / "anaconda3/envs/ibkr/bin/python")
)
JOB_STATE_NAME = "dash_mag7_backfill_job.json"
COVERAGE_REPORT_NAME = "dash_mag7_coverage_report.json"
MIN_STOCK_1S_ROWS = 1000


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
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def resolve_python(explicit: str | None = None) -> str:
    for raw in (explicit, os.environ.get("PYTHON"), str(DEFAULT_PY), "python3"):
        if not raw:
            continue
        path = Path(raw).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path.resolve())
    return "python3"


def api_key_status() -> dict[str, Any]:
    for name in ("MASSIVE_API_KEY", "POLYGON_API_KEY", "POLYGON_KEY"):
        val = os.environ.get(name, "").strip()
        if val:
            return {"ok": True, "env": name, "hint": f"{name}=***{val[-4:]}"}
    return {
        "ok": False,
        "env": None,
        "hint": "未检测到 MASSIVE_API_KEY / POLYGON_API_KEY；股票 1s 下载前请在环境或下方输入",
    }


def default_symbols_from_profile(profile: dict[str, Any] | None) -> str:
    cfg = (profile or {}).get("profile") or profile or {}
    syms = list(cfg.get("symbols") or [])
    peer = ((cfg.get("signal") or {}).get("peer_symbols")) or []
    for s in peer:
        if s not in syms:
            syms.append(s)
    for extra in ("QQQ",):
        if extra not in syms:
            syms.append(extra)
    return ",".join(str(s).upper() for s in syms if s)


def path_from_profile(profile: dict[str, Any] | None, name: str, default: Path) -> Path:
    rows = (profile or {}).get("paths") or []
    for row in rows:
        if row.get("name") == name and row.get("path"):
            return Path(str(row["path"])).expanduser()
    cfg_paths = ((profile or {}).get("profile") or {}).get("paths") or {}
    if name in cfg_paths:
        return Path(str(cfg_paths[name])).expanduser()
    return Path(default).expanduser()


def logs_dir(stock_root: Path) -> Path:
    d = Path(stock_root).expanduser() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_state_path(stock_root: Path) -> Path:
    return logs_dir(stock_root) / JOB_STATE_NAME


def coverage_report_path(stock_root: Path) -> Path:
    return logs_dir(stock_root) / COVERAGE_REPORT_NAME


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


def read_job_state(stock_root: Path) -> dict[str, Any]:
    state = _read_json(job_state_path(stock_root)) or {}
    if not state:
        return {"status": "idle"}
    pid = int(state.get("pid") or 0)
    if state.get("status") == "running" and pid and not _pid_alive(pid):
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
        _write_json(job_state_path(stock_root), state)
    return state


def tail_log(path: Path | str | None, *, max_bytes: int = 48_000) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.is_file():
        return ""
    size = p.stat().st_size
    with p.open("rb") as fh:
        data = fh.read() if size <= max_bytes else (fh.seek(size - max_bytes) or fh.read())
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        return f"... truncated ({size} bytes) ...\n{text}"
    return text


def _weekday_dates(start: str, end: str) -> list[str]:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    out: list[str] = []
    cur = s
    while cur <= e:
        if cur.weekday() < 5:
            out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _stock_day_path(stock_root: Path, symbol: str, day: str) -> Path:
    sym = symbol.upper()
    return Path(stock_root) / sym / f"{sym}_{day}.parquet"


def _file_ok(path: Path, *, min_rows: int = MIN_STOCK_1S_ROWS) -> bool:
    if not path.is_file():
        return False
    try:
        import pandas as pd

        n = len(pd.read_parquet(path, columns=["ts"]))
    except Exception:
        try:
            import pandas as pd

            n = len(pd.read_parquet(path))
        except Exception:
            return False
    return n >= int(min_rows)


def scan_stock_1s_coverage(
    *,
    start_date: str,
    end_date: str,
    symbols: str,
    stock_root: Path,
    min_rows: int = MIN_STOCK_1S_ROWS,
    write_report: bool = True,
) -> dict[str, Any]:
    """扫描区间内股票 1s 缺日（周末跳过；节假日可能误报，下载时会空跑）。"""
    root = Path(stock_root).expanduser()
    syms = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    days = _weekday_dates(start_date.strip(), end_date.strip())
    per_symbol: list[dict[str, Any]] = []
    missing_pairs: list[dict[str, str]] = []
    for sym in syms:
        miss: list[str] = []
        ok_n = 0
        for d in days:
            p = _stock_day_path(root, sym, d)
            if _file_ok(p, min_rows=min_rows):
                ok_n += 1
            else:
                miss.append(d)
                missing_pairs.append({"symbol": sym, "date": d})
        per_symbol.append(
            {
                "symbol": sym,
                "ok_days": ok_n,
                "missing_days": len(miss),
                "missing_sample": miss[:12],
                "latest_ok": next((d for d in reversed(days) if _file_ok(_stock_day_path(root, sym, d), min_rows=min_rows)), None),
            }
        )
    report = {
        "ok": len(missing_pairs) == 0,
        "start_date": start_date.strip(),
        "end_date": end_date.strip(),
        "stock_root": str(root),
        "symbols": syms,
        "n_weekdays": len(days),
        "n_missing_pairs": len(missing_pairs),
        "per_symbol": per_symbol,
        "missing_pairs_sample": missing_pairs[:80],
        "scanned_at": datetime.now().isoformat(timespec="seconds"),
        "min_rows": int(min_rows),
        "note": "跳过周末；交易所休市日可能显示为缺日（下载会得到空/失败，可忽略）",
    }
    if write_report:
        _write_json(coverage_report_path(root), report)
    return report


def load_coverage_report(stock_root: Path) -> dict[str, Any]:
    return _read_json(coverage_report_path(stock_root)) or {}


def build_cmd(
    *,
    mode: str,
    start_date: str,
    end_date: str,
    symbols: str,
    stock_root: Path,
    python_bin: str | None = None,
    max_workers: int = 12,
    force: bool = False,
    calendar_out: Path | None = None,
) -> list[str]:
    py = resolve_python(python_bin)
    if mode == "sync_calendar":
        out = Path(calendar_out or DEFAULT_CALENDAR).expanduser()
        return [
            py,
            "-u",
            "-m",
            "maga7.tools.sync_event_calendar",
            "--start",
            start_date,
            "--end",
            end_date,
            "--symbols",
            symbols,
            "--out",
            str(out),
        ]
    if mode == "stock_1s":
        cmd = [
            py,
            "-u",
            "-m",
            "preprocess.download.download_stock_1s",
            "--symbols",
            symbols,
            "--start-date",
            start_date,
            "--end-date",
            end_date,
            "--stock-output-dir",
            str(Path(stock_root).expanduser()),
            "--max-workers",
            str(int(max_workers)),
        ]
        if force:
            cmd.append("--force")
        return cmd
    raise ValueError(f"unknown mode={mode!r}; use sync_calendar|stock_1s")


def suggested_commands(
    *,
    start_date: str,
    end_date: str,
    symbols: str,
    stock_root: Path,
) -> dict[str, str]:
    cal = build_cmd(
        mode="sync_calendar",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        stock_root=stock_root,
    )
    stk = build_cmd(
        mode="stock_1s",
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        stock_root=stock_root,
    )
    return {
        "sync_calendar": " ".join(cal),
        "stock_1s": " ".join(stk),
        "note": "股票 1s 需要 MASSIVE_API_KEY 或 POLYGON_API_KEY；日历可用 Finnhub/Polygon（earnings）。",
    }


def start_job(
    *,
    mode: str,
    start_date: str,
    end_date: str,
    symbols: str,
    stock_root: Path,
    python_bin: str | None = None,
    api_key: str | None = None,
    max_workers: int = 12,
    force: bool = False,
    calendar_out: Path | None = None,
) -> dict[str, Any]:
    """后台启动任务；同一 stock_root 同时只允许一个 running job。"""
    root = Path(stock_root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    logs = logs_dir(root)

    current = read_job_state(root)
    if current.get("status") == "running":
        raise RuntimeError(
            f"已有任务在跑 pid={current.get('pid')} log={current.get('log_file')}"
        )

    if mode == "stock_1s":
        key = (api_key or "").strip()
        if key:
            pass
        elif not api_key_status()["ok"]:
            raise RuntimeError("缺少 API key：请设置环境变量或在页面输入 Massive/Polygon key")

    cmd = build_cmd(
        mode=mode,
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        symbols=symbols.strip(),
        stock_root=root,
        python_bin=python_bin,
        max_workers=max_workers,
        force=force,
        calendar_out=calendar_out,
    )
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs / f"mag7_{mode}_{start_date}_{end_date}_{stamp}.log"
    exit_file = logs / f"mag7_{mode}_{stamp}.exit"
    if exit_file.exists():
        exit_file.unlink()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    key = (api_key or "").strip()
    if key:
        env["MASSIVE_API_KEY"] = key
        env["POLYGON_API_KEY"] = key

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
        "start_date": start_date.strip(),
        "end_date": end_date.strip(),
        "symbols": symbols.strip(),
        "stock_root": str(root),
        "cmd": cmd,
        "log_file": str(log_file),
        "exit_file": str(exit_file),
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "force": bool(force),
    }
    _write_json(job_state_path(root), state)
    return state


def stop_job(stock_root: Path) -> dict[str, Any]:
    state = read_job_state(stock_root)
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
    _write_json(job_state_path(stock_root), state)
    return state
