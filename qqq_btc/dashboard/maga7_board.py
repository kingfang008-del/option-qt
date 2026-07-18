#!/usr/bin/env python3
"""Legacy Mag7 Offline Replay + Stream Parity panel helpers.

Authoritative G0–G6 / live-session monitoring lives in `dash/run.py`.
This module remains for older Offline/Parity board embeds only.
"""
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

REPO = Path(__file__).resolve().parents[2]
MAGA7_RESULTS = REPO / "maga7" / "results"
MAGA7_PROFILE = REPO / "maga7" / "CONFIG" / "mf10_top2_v1.json"
JOB_STATE = MAGA7_RESULTS / "_dash_jobs" / "maga7_dash_job.json"


@dataclass
class Maga7Run:
    path: Path
    name: str
    mtime: float
    kind: str  # replay | parity
    summary: dict[str, Any] = field(default_factory=dict)
    parity: dict[str, Any] = field(default_factory=dict)


def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def discover_maga7_runs(*, limit: int = 40) -> list[Maga7Run]:
    if not MAGA7_RESULTS.is_dir():
        return []
    out: list[Maga7Run] = []
    for child in sorted(MAGA7_RESULTS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not child.is_dir() or child.name.startswith("_"):
            continue
        summary = _read_json(child / "summary.json") or _read_json(child / "offline_summary.json") or {}
        parity = _read_json(child / "parity_summary.json") or {}
        if not summary and not parity:
            # try nested offline/stream
            if not (child / "trades.csv").is_file() and not (child / "trades_offline.csv").is_file():
                continue
        kind = "parity" if parity or (child / "trades_stream.csv").is_file() else "replay"
        out.append(
            Maga7Run(
                path=child,
                name=child.name,
                mtime=child.stat().st_mtime,
                kind=kind,
                summary=summary,
                parity=parity,
            )
        )
        if len(out) >= limit:
            break
    return out


def read_job_state() -> dict[str, Any]:
    return _read_json(JOB_STATE) or {}


def _write_job_state(state: dict[str, Any]) -> None:
    JOB_STATE.parent.mkdir(parents=True, exist_ok=True)
    JOB_STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def stop_job() -> None:
    st = read_job_state()
    pid = st.get("pid")
    if pid:
        try:
            os.kill(int(pid), signal.SIGTERM)
        except Exception:
            pass
    st["status"] = "stopped"
    st["stopped_at"] = datetime.now().isoformat(timespec="seconds")
    _write_job_state(st)


def start_job(cmd: list[str], *, tag: str, log_name: str) -> dict[str, Any]:
    MAGA7_RESULTS.mkdir(parents=True, exist_ok=True)
    log_dir = MAGA7_RESULTS / "_dash_jobs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}:{env.get('PYTHONPATH', '')}"
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=str(REPO),
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    state = {
        "pid": proc.pid,
        "cmd": cmd,
        "tag": tag,
        "log": str(log_path),
        "status": "running",
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_job_state(state)
    return state


def build_replay_cmd(
    *,
    scheme: str,
    start: str,
    end: str,
    tag: str,
    profile: Path | None = None,
) -> list[str]:
    py = resolve_python()
    cmd = [
        py,
        "-m",
        "maga7.tools.run_replay_offline",
        "--scheme",
        scheme,
        "--start-date",
        start,
        "--end-date",
        end,
        "--tag",
        tag,
    ]
    if profile:
        cmd.extend(["--profile", str(profile)])
    return cmd


def build_parity_cmd(
    *,
    scheme: str,
    start: str,
    end: str,
    tag: str,
    profile: Path | None = None,
) -> list[str]:
    py = resolve_python()
    cmd = [
        py,
        "-m",
        "maga7.tools.run_stream_parity",
        "--scheme",
        scheme,
        "--start-date",
        start,
        "--end-date",
        end,
        "--tag",
        tag,
    ]
    if profile:
        cmd.extend(["--profile", str(profile)])
    return cmd


def build_prepare_cmd(*, step: str = "all", max_workers: int = 12) -> list[str]:
    py = resolve_python()
    return [
        py,
        "-m",
        "maga7.tools.prepare_jan_jul_data",
        "--step",
        step,
        "--max-workers",
        str(max_workers),
    ]


def tail_log(n: int = 40) -> str:
    st = read_job_state()
    log = st.get("log")
    if not log or not Path(log).is_file():
        return ""
    lines = Path(log).read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def headline_from_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    if not summary:
        return []
    rows = []
    mapping = [
        ("total_ret", "Total ret"),
        ("maxdd", "Max DD"),
        ("n_trades", "Trades"),
        ("n_days", "Days"),
        ("day_win", "Day win"),
        ("trade_win", "Trade win"),
        ("trade_exp", "Trade E[r]"),
        ("fill_frac", "Fill frac"),
        ("scheme", "Scheme"),
    ]
    for k, label in mapping:
        if k not in summary:
            continue
        v = summary[k]
        if isinstance(v, float) and k in {"total_ret", "maxdd", "day_win", "trade_win", "trade_exp"}:
            rows.append({"metric": label, "value": f"{v * 100:.1f}%"})
        else:
            rows.append({"metric": label, "value": v})
    return rows


def render_maga7_board() -> None:
    """Streamlit panel: Mag7 offline + parity (+ data prepare / scanner docs)."""
    import pandas as pd
    import streamlit as st

    st.markdown('<div class="qbd-title">Mag7 / maga7 Board</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="qbd-sub">规则 A · Top2 · ATM 短期权。'
        "Offline / 流式对拍走 <code>maga7</code>；"
        "实盘是多标的 Scanner → OMS（不经 QQQ TFT/FCS 主信号）。"
        "账户回撤 = 复利净值 DD（仓位 25%/TopK），非单笔权利金回撤。</div>",
        unsafe_allow_html=True,
    )

    tab_run, tab_results, tab_data, tab_live = st.tabs(
        ["▶ Run", "Results", "Data prepare", "Scanner → OMS"]
    )

    with tab_run:
        c1, c2, c3 = st.columns(3)
        with c1:
            scheme = st.selectbox("Scheme", ["single", "m5", "m5_circuit"], index=0)
        with c2:
            start = st.text_input("Start", value="2026-05-01")
        with c3:
            end = st.text_input("End", value="2026-07-13")
        tag = st.text_input("Result tag", value=f"dash_{scheme}_{datetime.now().strftime('%m%d_%H%M')}")
        profile = MAGA7_PROFILE if MAGA7_PROFILE.is_file() else None
        st.caption(f"profile: {profile or '(missing)'}")

        job = read_job_state()
        if job:
            st.info(
                f"Job status={job.get('status')} pid={job.get('pid')} "
                f"started={job.get('started_at')} tag={job.get('tag')}"
            )
            if job.get("log"):
                with st.expander("Job log tail", expanded=False):
                    st.code(tail_log(60) or "(empty)")

        b1, b2, b3 = st.columns(3)
        with b1:
            if st.button("▶ Offline Replay", type="primary", use_container_width=True):
                cmd = build_replay_cmd(scheme=scheme, start=start, end=end, tag=tag, profile=profile)
                start_job(cmd, tag=tag, log_name=f"replay_{tag}.log")
                st.success("started offline replay")
                st.rerun()
        with b2:
            if st.button("▶ Stream Parity", use_container_width=True):
                cmd = build_parity_cmd(scheme=scheme, start=start, end=end, tag=f"parity_{tag}", profile=profile)
                start_job(cmd, tag=f"parity_{tag}", log_name=f"parity_{tag}.log")
                st.success("started stream parity")
                st.rerun()
        with b3:
            if st.button("⏹ Stop job", use_container_width=True):
                stop_job()
                st.warning("stop signal sent")
                st.rerun()

        st.markdown("---")
        st.caption("Scanner / OMS（不下真单；S4=独立 stub）")
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            if st.button("▶ Scanner S1 (1m)", use_container_width=True):
                py = resolve_python()
                cmd = [
                    py, "-m", "maga7.tools.run_scanner_shadow",
                    "--start-date", start, "--end-date", end,
                ]
                if profile:
                    cmd += ["--profile", str(profile)]
                start_job(cmd, tag=f"scan1m_{tag}", log_name=f"scan1m_{tag}.log")
                st.success("started S1 scanner shadow")
                st.rerun()
        with sc2:
            if st.button("▶ Scanner S2 (1s→1m)", use_container_width=True):
                py = resolve_python()
                cmd = [
                    py, "-m", "maga7.tools.run_scanner_from_1s",
                    "--start-date", start, "--end-date", end,
                ]
                if profile:
                    cmd += ["--profile", str(profile)]
                start_job(cmd, tag=f"scan1s_{tag}", log_name=f"scan1s_{tag}.log")
                st.success("started S2 1s→1m scanner")
                st.rerun()
        with sc3:
            if st.button("▶ OMS dry (S3)", use_container_width=True):
                py = resolve_python()
                cmd = [
                    py, "-m", "maga7.tools.run_oms_dry_run",
                    "--start-date", start, "--end-date", end,
                    "--compare-offline",
                    "--tag", f"oms_dry_{tag}",
                ]
                if profile:
                    cmd += ["--profile", str(profile)]
                start_job(cmd, tag=f"oms_dry_{tag}", log_name=f"oms_dry_{tag}.log")
                st.success("started OMS dry-run")
                st.rerun()
        with sc4:
            if st.button("▶ OMS stub (S4)", use_container_width=True):
                py = resolve_python()
                cmd = [
                    py, "-m", "maga7.tools.run_oms_live_stub",
                    "--start-date", start, "--end-date", end,
                    "--compare-offline",
                    "--max-qty", "1",
                    "--tag", f"oms_stub_{tag}",
                ]
                if profile:
                    cmd += ["--profile", str(profile)]
                start_job(cmd, tag=f"oms_stub_{tag}", log_name=f"oms_stub_{tag}.log")
                st.success("started OMS stub")
                st.rerun()

    with tab_results:
        runs = discover_maga7_runs()
        if not runs:
            st.warning(f"No runs under {MAGA7_RESULTS}")
        else:
            labels = [f"{r.kind}: {r.name}" for r in runs]
            pick = st.selectbox("Run", labels, index=0)
            run = runs[labels.index(pick)]
            st.caption(str(run.path))
            if run.parity:
                st.subheader("Parity")
                st.json(run.parity)
                ok = run.parity.get("ok")
                st.metric("parity ok", str(ok))
            summary = run.summary or _read_json(run.path / "stream_summary.json") or {}
            if summary:
                st.subheader("Summary")
                st.dataframe(pd.DataFrame(headline_from_summary(summary)), hide_index=True, use_container_width=True)
            daily = run.path / "daily.csv"
            if daily.is_file():
                df = pd.read_csv(daily)
                st.subheader("Daily equity")
                st.line_chart(df.set_index("date")["equity"] if "equity" in df.columns else df)
            for name in ("trades.csv", "trades_offline.csv", "trades_stream.csv"):
                p = run.path / name
                if p.is_file():
                    st.subheader(name)
                    st.dataframe(pd.read_csv(p).head(50), use_container_width=True, hide_index=True)

    with tab_data:
        st.markdown(
            """
**流水线**（`maga7.tools.prepare_jan_jul_data`）:
1. report 缺口（正股 ~3/19–4/30 空洞）
2. step1 锁约 → `locked_targets_map_maga7_mf10_jan_jul.parquet`
3. step2 1s quote → `/mnt/s990/data/raw_1s/maga7_mf10_old_lock/`
"""
        )
        step = st.selectbox("Prepare step", ["report", "lock", "quotes", "all"], index=0)
        workers = st.number_input("max-workers", min_value=1, max_value=32, value=12)
        if st.button("▶ Prepare data", use_container_width=True):
            cmd = build_prepare_cmd(step=step, max_workers=int(workers))
            start_job(cmd, tag=f"prepare_{step}", log_name=f"prepare_{step}_{int(time.time())}.log")
            st.success("prepare started")
            st.rerun()

    with tab_live:
        st.markdown(
            """
### 架构（与 QQQ TFT 分流）

```text
正股 1s ──聚合──► Mag7 Scanner（Rule-A / TopK，决策=1m）
                        │  signal_audit / Redis（可选）
                        ▼
              OMS（合约 1s quote，fill_frac=0.8）
                        │
                        ▼
                   IBKR / mock
```

- **信号用 1m，成交用 1s**；不要把 Rule-A 改成秒级特征。
- **不**把 Mag7 塞进 QQQ 单标的 SE/TFT。
- S1：`python -m maga7.tools.run_scanner_shadow --start-date YYYY-MM-DD`
- S2：`python -m maga7.tools.run_scanner_from_1s --start-date YYYY-MM-DD`
- S3：`python -m maga7.tools.run_oms_dry_run --start-date YYYY-MM-DD --compare-offline`
- S4：`python -m maga7.tools.run_oms_live_stub --start-date YYYY-MM-DD --compare-offline`
- 设计文档：`maga7/docs/scanner_oms_integration.md`
"""
        )
        doc = REPO / "maga7" / "docs" / "scanner_oms_integration.md"
        if doc.is_file():
            with st.expander("scanner_oms_integration.md", expanded=False):
                st.markdown(doc.read_text(encoding="utf-8"))
