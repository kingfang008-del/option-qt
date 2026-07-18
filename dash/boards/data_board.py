"""Download / 补数据 board — 可配置日期区间、一键启停、页面看日志。"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from backfill_jobs import (
    DEFAULT_CALENDAR,
    api_key_status,
    default_symbols_from_profile,
    load_coverage_report,
    path_from_profile,
    read_job_state,
    resolve_python,
    scan_stock_1s_coverage,
    start_job,
    stop_job,
    suggested_commands,
    tail_log,
)
from sources import REPO


def _default_dates() -> tuple[str, str]:
    end = date.today()
    start = end - timedelta(days=10)
    return start.isoformat(), end.isoformat()


def render_data_board(profile: dict) -> None:
    st.markdown("### ① 补数据 / Download")
    st.caption(
        "对齐 qqq_btc Download：**页面可配日期 → 扫缺数 → 一键启停 → 看执行日志**。"
        "本板只做行情/日历准备，**不含**策略改参。"
        " Mag7 正股 1s 为左标签事实源（与 qqq_btc 右标签 resampled 约定不同）。"
    )

    paths = profile.get("paths") or []
    stock_default = path_from_profile(
        profile, "stock_1s_root", Path("/mnt/s990/data/raw_1s/stocks")
    )
    quote_default = path_from_profile(
        profile,
        "quote_1s_root",
        Path("/mnt/s990/data/raw_1s/maga7_mf10_open_ladder_otm5"),
    )
    lock = next(
        (r for r in paths if r.get("name") in {"open_locked_map", "locked_map"}),
        None,
    )
    d0, d1 = _default_dates()
    default_syms = default_symbols_from_profile(profile)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        start_date = st.text_input("Start date", value=d0, help="YYYY-MM-DD")
    with c2:
        end_date = st.text_input("End date", value=d1, help="YYYY-MM-DD")
    with c3:
        symbols = st.text_input(
            "Symbols",
            value=default_syms,
            help="默认 = profile universe + peer + QQQ（regime）",
        )
    with c4:
        force = st.toggle("Force overwrite", value=False)

    stock_root = Path(
        st.text_input("Stock 1s root", value=str(stock_default))
    ).expanduser()
    quote_root = Path(
        st.text_input("Quote 1s root（只读检查）", value=str(quote_default))
    ).expanduser()

    a, b, c, d = st.columns(4)
    a.metric("Universe", len([s for s in symbols.split(",") if s.strip()]))
    b.metric("Stock 1s", "OK" if stock_root.is_dir() else "MISSING")
    c.metric("Quote 1s", "OK" if quote_root.is_dir() else "MISSING")
    d.metric("Lock map", "OK" if lock and lock.get("exists") else "MISSING")

    st.markdown("**数据路径（profile）**")
    st.dataframe(pd.DataFrame(paths), use_container_width=True, hide_index=True)

    key_info = api_key_status()
    api_key_input = st.text_input(
        "Massive/Polygon API key（可空，优先用环境变量）",
        value="",
        type="password",
        help=key_info["hint"],
    )
    if key_info["ok"]:
        st.caption(f"环境已有 key：{key_info['hint']}")
    else:
        st.warning(key_info["hint"])

    python_bin = resolve_python()
    max_workers = 12
    calendar_out = str(DEFAULT_CALENDAR)
    auto_refresh = False
    with st.expander("高级选项", expanded=False):
        python_bin = st.text_input("Python", value=python_bin)
        max_workers = st.number_input("Max workers", min_value=1, max_value=64, value=int(max_workers))
        calendar_out = st.text_input("Event calendar out", value=calendar_out)
        auto_refresh = st.toggle("任务运行时自动刷新日志（5s）", value=auto_refresh)

    # ---- Coverage ----
    st.markdown("#### 缺数检查（股票 1s）")
    if st.button("检查缺数", use_container_width=False, type="secondary"):
        with st.spinner("扫描股票 1s parquet…"):
            rep = scan_stock_1s_coverage(
                start_date=start_date.strip(),
                end_date=end_date.strip(),
                symbols=symbols.strip(),
                stock_root=stock_root,
            )
        st.session_state["mag7_coverage_flash"] = rep
        st.rerun()

    cov = load_coverage_report(stock_root) or st.session_state.get("mag7_coverage_flash") or {}
    if cov:
        if cov.get("ok"):
            st.success(
                f"区间 {cov.get('start_date')}→{cov.get('end_date')}："
                f"{cov.get('n_weekdays')} 个工作日 × {len(cov.get('symbols') or [])} 标的均齐全。"
            )
        else:
            st.warning(
                f"缺 {cov.get('n_missing_pairs')} 个 (symbol, day) 对"
                f"（扫描于 {cov.get('scanned_at')}）"
            )
        if cov.get("per_symbol"):
            st.dataframe(
                pd.DataFrame(cov["per_symbol"]),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("coverage report", expanded=not bool(cov.get("ok"))):
            st.json(cov, expanded=False)
            st.caption(str(cov.get("note") or ""))
    else:
        st.info("建议先点「检查缺数」，确认区间内哪些标的缺 1s。")

    # ---- One-click ----
    st.markdown("#### 一键执行")
    b1, b2, b3 = st.columns(3)
    run_mode = None
    with b1:
        if st.button("① 同步事件日历", use_container_width=True):
            run_mode = "sync_calendar"
    with b2:
        if st.button("② 下载股票 1s", use_container_width=True, type="primary"):
            run_mode = "stock_1s"
    with b3:
        if st.button("⏹ 停止任务", use_container_width=True):
            stop_job(stock_root)
            st.rerun()

    if run_mode:
        try:
            state = start_job(
                mode=run_mode,
                start_date=start_date.strip(),
                end_date=end_date.strip(),
                symbols=symbols.strip(),
                stock_root=stock_root,
                python_bin=python_bin,
                api_key=api_key_input or None,
                max_workers=int(max_workers),
                force=bool(force),
                calendar_out=Path(calendar_out),
            )
            st.success(f"已启动 {run_mode} pid={state.get('pid')}")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))

    job = read_job_state(stock_root)
    status = str(job.get("status") or "idle")
    tone = {
        "done": "🟢",
        "running": "🟡",
        "failed": "🔴",
        "stopped": "🔴",
        "idle": "⚪",
    }.get(status, "⚪")
    st.markdown(
        f"{tone} **job={status}** · mode=`{job.get('mode', '-')}` · "
        f"pid=`{job.get('pid', '-')}` · "
        f"`{job.get('start_date', '-')} → {job.get('end_date', '-')}`"
    )
    if job.get("cmd"):
        st.code(" ".join(str(x) for x in job["cmd"]), language="bash")

    log_text = tail_log(job.get("log_file"))
    if log_text:
        st.markdown("#### 执行日志")
        st.code(log_text, language="text")
        if status == "running":
            st.caption("任务运行中：点「刷新日志」或侧栏「刷新」。")
            if st.button("刷新日志"):
                st.rerun()
            if auto_refresh:
                import time

                time.sleep(5)
                st.rerun()

    cmds = suggested_commands(
        start_date=start_date.strip(),
        end_date=end_date.strip(),
        symbols=symbols.strip(),
        stock_root=stock_root,
    )
    with st.expander("等价 CLI", expanded=False):
        st.caption(cmds.get("note", ""))
        st.code(cmds["sync_calendar"], language="bash")
        st.code(cmds["stock_1s"], language="bash")

    cal_path = Path(calendar_out).expanduser()
    st.markdown("**事件日历**")
    st.text(f"{cal_path}  exists={cal_path.is_file()}")

    st.markdown("**本页边界**")
    st.write(
        "- 做：缺数扫描、事件日历同步、股票 1s 下载、任务日志\n"
        "- 不做：期权 quote 全链路（仍看路径；锁约 map 用既有 open_ladder 产物）\n"
        "- 不做：改 TopK / 出场 / regime（Offline / Parity / Live 共用 profile）\n"
        f"- 当前 profile：`{profile.get('path')}` · repo=`{REPO}`"
    )
    if lock:
        st.caption(f"lock: {lock.get('path')} · {lock.get('detail') or ''}")
