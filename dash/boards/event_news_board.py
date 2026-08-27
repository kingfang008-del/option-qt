"""Event calendar + company-news audit board (prevent false blackouts)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from sources import REPO

CAL_PATH = REPO / "maga7" / "CONFIG" / "event_calendar_live.json"
AUDIT_PATH = REPO / "maga7" / "CONFIG" / "event_news_audit.json"
SUPPRESS_PATH = REPO / "maga7" / "CONFIG" / "event_news_suppress.json"
LLM_PATH = REPO / "maga7" / "CONFIG" / "event_news_llm.json"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _mtime(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


def render_event_news_board() -> None:
    from maga7.common.company_news import (
        apply_suppress_to_calendar_payload,
        load_news_suppress,
        normalize_news_tag,
        save_news_suppress,
        suppress_key,
    )

    st.markdown("### 事件 / 公司新闻审核")
    from maga7.common.event_news_policy import POLICY_SUMMARY_ZH

    st.info(POLICY_SUMMARY_ZH)
    st.caption(
        "可对重大标题跑 **LLM 利好/利空 + 二阶影响**（如涨价被砸）——"
        "仅人工参考，**不定方向、不写黑名单**。"
        "自动禁入仅：宏观 full-day / 财报 symbol / CEO hard_risk。"
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("日历文件", "OK" if CAL_PATH.is_file() else "MISSING")
    c2.metric("新闻审计", "OK" if AUDIT_PATH.is_file() else "MISSING")
    c3.metric("误杀驳回", "OK" if SUPPRESS_PATH.is_file() else "空")
    st.caption(
        f"cal mtime={_mtime(CAL_PATH)} · audit mtime={_mtime(AUDIT_PATH)} · "
        f"suppress mtime={_mtime(SUPPRESS_PATH)}"
    )

    if st.button("刷新文件", key="news_audit_refresh"):
        st.rerun()

    from maga7.common.news_llm import load_llm_cache, resolve_llm_api_key, resolve_llm_endpoint

    cal = _load_json(CAL_PATH)
    audit = _load_json(AUDIT_PATH)
    suppress_items = load_news_suppress(SUPPRESS_PATH)
    suppress_keys = {str(x.get("key") or suppress_key(x)) for x in suppress_items}
    llm_cache = load_llm_cache(LLM_PATH)
    llm_items: dict[str, Any] = dict(llm_cache.get("items") or {})

    # ---- Live blackout dates ----
    st.markdown("#### 当前生效黑名单（live calendar）")
    st.caption(
        "**full-day**（FOMC 等）→ 全日停手；"
        "**symbol**（财报/公司新闻）→ 只禁该标的，其它 Mag7 照常交易。"
    )
    events = list(cal.get("events") or [])
    dates = list(cal.get("dates") or [])
    sym_map = cal.get("symbol_blackout") or {}
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("full-day 日", len(dates))
    m2.metric("symbol 日", len(sym_map))
    news_ev = [e for e in events if str(e.get("tag") or "").startswith("news_")]
    m3.metric("news_* 行", len(news_ev))
    m4.metric("已驳回", len(suppress_items))
    if dates:
        st.write("full-day dates:", ", ".join(dates))
    if sym_map:
        st.write(
            "symbol_blackout:",
            {d: ",".join(v) if isinstance(v, list) else v for d, v in sym_map.items()},
        )
    if events:
        ev_df = pd.DataFrame(events)
        if "tag" in ev_df.columns:
            ev_df = ev_df.assign(
                scope=ev_df.apply(
                    lambda r: (
                        "symbol"
                        if str(r.get("tag") or "").startswith(("news_", "earnings"))
                        and r.get("symbol")
                        else "full"
                    ),
                    axis=1,
                )
            )
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
    else:
        st.info("尚无 live 日历事件。先 Download「同步事件日历」或 `sync-calendar`。")

    # ---- News audit ----
    st.markdown("#### 公司新闻打分（Finnhub + Investing RSS）")
    items = list(audit.get("items") or [])
    if not items:
        st.warning(
            f"未找到审计文件或为空：`{AUDIT_PATH}`。先跑一次日历同步。"
        )
        _render_suppress_table(suppress_items)
        return

    a1, a2, a3 = st.columns(3)
    a1.metric("scored", int(audit.get("n_total") or len(items)))
    a2.metric("hard", int(audit.get("n_hard") or 0))
    a3.metric("soft", int(audit.get("n_soft") or 0))
    st.caption(f"generated_at={audit.get('generated_at') or '-'}")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        tier = st.selectbox(
            "Tier",
            options=["soft", "hard", "flagged", "all", "unscored"],
            index=0,
            key="news_tier_filter",
            help="flagged=hard+soft 重大标题；默认看 soft（大单/合作等）。",
        )
    with f2:
        syms = sorted(
            {
                str(x.get("symbol") or "").upper()
                for x in items
                if x.get("symbol")
            }
            | {
                s
                for x in items
                for s in (x.get("symbols") or [])
                if s
            }
        )
        sym_pick = st.multiselect("Symbol", options=syms, default=[], key="news_sym_filter")
    with f3:
        srcs = sorted({str(x.get("source") or "") for x in items if x.get("source")})
        src_pick = st.multiselect("Source", options=srcs, default=[], key="news_src_filter")
    with f4:
        hide_rej = st.checkbox("隐藏已驳回", value=True, key="news_hide_rej")

    rows: list[dict[str, Any]] = []
    for it in items:
        t = it.get("tier")
        if tier == "hard" and t != "hard":
            continue
        if tier == "soft" and t != "soft":
            continue
        if tier == "flagged" and t not in {"hard", "soft"}:
            continue
        if tier == "unscored" and t is not None:
            continue
        if sym_pick:
            hit = str(it.get("symbol") or "").upper() in sym_pick
            hit = hit or any(str(s).upper() in sym_pick for s in (it.get("symbols") or []))
            if not hit:
                continue
        if src_pick and str(it.get("source") or "") not in src_pick:
            continue
        sk = suppress_key(
            {
                **it,
                "date": it.get("session_date"),
                "tag": normalize_news_tag(str(it.get("tag") or "")),
            }
        )
        if hide_rej and sk in suppress_keys:
            continue
        llm = llm_items.get(sk) or {}
        rows.append(
            {
                "analyze": False,
                "reject": False,
                "stance": llm.get("stance") or "",
                "impact": llm.get("impact") or "",
                "surprise": llm.get("surprise_risk") or "",
                "hint": llm.get("trade_hint") or "",
                "rationale_zh": llm.get("rationale_zh") or llm.get("error") or "",
                "tier": t or "",
                "tag": it.get("tag") or "",
                "symbol": it.get("symbol") or "",
                "session_date": it.get("session_date") or "",
                "source": it.get("source") or "",
                "title": it.get("title") or "",
                "summary": (it.get("summary") or "")[:240],
                "url": it.get("url") or "",
                "published_at": it.get("published_at") or "",
                "_key": sk,
            }
        )

    if not rows:
        st.info("当前过滤条件下无条目。")
        _render_suppress_table(suppress_items)
        return

    st.caption(
        f"显示 {len(rows)} 条。勾选 **analyze** 跑 LLM；**reject** 仅用于误杀 suppress。"
    )
    show_df = pd.DataFrame(rows)
    edited = st.data_editor(
        show_df,
        use_container_width=True,
        hide_index=True,
        disabled=[
            "stance",
            "impact",
            "surprise",
            "hint",
            "rationale_zh",
            "tier",
            "tag",
            "symbol",
            "session_date",
            "source",
            "title",
            "summary",
            "url",
            "published_at",
            "_key",
        ],
        column_config={
            "analyze": st.column_config.CheckboxColumn("LLM", default=False),
            "reject": st.column_config.CheckboxColumn("驳回", default=False),
            "stance": st.column_config.TextColumn("利好/空", width="small"),
            "impact": st.column_config.TextColumn("影响", width="small"),
            "surprise": st.column_config.TextColumn("二阶风险", width="small"),
            "hint": st.column_config.TextColumn("交易提示", width="small"),
            "rationale_zh": st.column_config.TextColumn("LLM 理由", width="large"),
            "url": st.column_config.LinkColumn("url", display_text="link"),
            "title": st.column_config.TextColumn("title", width="large"),
            "_key": st.column_config.TextColumn("_key", disabled=True, width="small"),
        },
        column_order=[
            "analyze",
            "reject",
            "stance",
            "impact",
            "surprise",
            "hint",
            "rationale_zh",
            "symbol",
            "session_date",
            "tier",
            "tag",
            "title",
            "url",
        ],
        key="news_audit_editor",
    )
    if "_key" not in edited.columns:
        edited = edited.copy()
        edited["_key"] = show_df["_key"].to_numpy()

    # ---- LLM analyze ----
    st.markdown("#### LLM 利好 / 利空（二阶影响）")
    has_key = bool(resolve_llm_api_key())
    base, model = resolve_llm_endpoint()
    if has_key:
        st.caption(f"endpoint=`{base}` model=`{model}` · cache=`{LLM_PATH.name}`")
    else:
        st.warning(
            "未检测到 LLM key。设置 `OPENAI_API_KEY` / `DEEPSEEK_API_KEY`，"
            "或写入 `~/openai.txt` / `~/deepseek.txt`；"
            "DeepSeek 可设 `MAG7_LLM_BASE_URL=https://api.deepseek.com/v1`。"
        )
    l1, l2, l3 = st.columns(3)
    with l1:
        max_n = st.number_input("最多分析条数", min_value=1, max_value=40, value=12)
    with l2:
        skip_cached = st.checkbox("跳过已分析", value=True, key="news_llm_skip")
    with l3:
        analyze_all_flagged = st.checkbox(
            "分析当前列表未缓存项",
            value=False,
            key="news_llm_all",
            help="不勾选则只分析表格里勾了 LLM 的行",
        )
    to_analyze_keys = set()
    if not edited.empty and "analyze" in edited.columns:
        to_analyze_keys = set(
            edited.loc[edited["analyze"] == True, "_key"].astype(str)  # noqa: E712
        )
    analyze_btn = st.button(
        "运行 LLM 分析",
        type="primary",
        disabled=not has_key,
        key="news_run_llm",
    )
    if analyze_btn and has_key:
        from maga7.common.news_llm import analyze_batch

        # Map keys back to full audit items (need summary)
        by_key = {}
        for it in items:
            sk = suppress_key(
                {
                    **it,
                    "date": it.get("session_date"),
                    "tag": normalize_news_tag(str(it.get("tag") or "")),
                }
            )
            by_key[sk] = it
        if analyze_all_flagged:
            pick_keys = [r["_key"] for r in rows]
        else:
            pick_keys = [k for k in to_analyze_keys if k in by_key]
            if not pick_keys:
                st.warning("请先勾选表格中的 LLM 列，或勾选「分析当前列表未缓存项」。")
                pick_keys = []
        batch = [by_key[k] for k in pick_keys if k in by_key]
        if batch:
            with st.spinner(f"LLM 分析中（最多 {int(max_n)} 条）…"):
                new_rows, _ = analyze_batch(
                    batch,
                    cache_path=LLM_PATH,
                    max_n=int(max_n),
                    skip_cached=bool(skip_cached),
                )
            st.success(f"完成本次 {len(new_rows)} 条 → `{LLM_PATH.name}`")
            st.rerun()

    # Highlight bearish / high surprise
    bear = [
        r
        for r in rows
        if (llm_items.get(r["_key"]) or {}).get("stance") == "bearish"
        or (llm_items.get(r["_key"]) or {}).get("surprise_risk") == "high"
    ]
    if bear:
        with st.expander(f"需警惕（bearish / 高二阶风险）{len(bear)}", expanded=True):
            for r in bear[:30]:
                llm = llm_items.get(r["_key"]) or {}
                st.markdown(
                    f"- **{llm.get('stance','?')}** / impact=`{llm.get('impact')}` "
                    f"/ surprise=`{llm.get('surprise_risk')}` · "
                    f"`{r.get('symbol')}` {r.get('session_date')} — {r.get('title')}\n\n"
                    f"  _{llm.get('rationale_zh') or ''}_"
                )

    rejected = edited[edited["reject"] == True] if not edited.empty else edited  # noqa: E712
    b1, b2 = st.columns(2)
    with b1:
        apply = st.button(
            f"确认驳回选中（{len(rejected)}）并更新 live 日历",
            disabled=rejected.empty,
            key="news_apply_reject",
        )
    with b2:
        st.caption("驳回仅影响误杀 suppress；LLM 结果不自动禁入。")

    if apply and not rejected.empty:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        new_items = list(suppress_items)
        for _, r in rejected.iterrows():
            new_items.append(
                {
                    "key": str(r["_key"]),
                    "date": str(r["session_date"])[:10],
                    "session_date": str(r["session_date"])[:10],
                    "symbol": str(r["symbol"] or "").upper() or None,
                    "tag": normalize_news_tag(str(r["tag"] or "")),
                    "title": str(r["title"] or "")[:200],
                    "url": str(r["url"] or ""),
                    "reason": "dash_false_positive",
                    "reviewed_at": now,
                }
            )
        save_news_suppress(new_items, SUPPRESS_PATH)
        if cal:
            updated = apply_suppress_to_calendar_payload(cal, suppress_path=SUPPRESS_PATH)
            CAL_PATH.parent.mkdir(parents=True, exist_ok=True)
            CAL_PATH.write_text(
                json.dumps(updated, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        st.success(
            f"已驳回 {len(rejected)} 条 → `{SUPPRESS_PATH.name}`，"
            f"并重写 `{CAL_PATH.name}`。"
        )
        st.rerun()

    # Quick links for hard titles
    hard_show = [r for r in rows if r.get("tier") == "hard"]
    if hard_show:
        with st.expander(f"Hard 标题速览（{len(hard_show)}）", expanded=True):
            for r in hard_show[:40]:
                link = f" [link]({r['url']})" if r.get("url") else ""
                st.markdown(
                    f"- **{r.get('session_date')}** `{r.get('symbol')}` "
                    f"`{r.get('tag')}` — {r.get('title')}{link}"
                )

    _render_suppress_table(suppress_items)


def _render_suppress_table(items: list[dict[str, Any]]) -> None:
    st.markdown("#### 已驳回清单（suppress）")
    if not items:
        st.caption("暂无驳回。")
        return
    st.dataframe(pd.DataFrame(items), use_container_width=True, hide_index=True)
    if st.button("清空全部驳回（危险）", key="news_clear_suppress"):
        from maga7.common.company_news import save_news_suppress

        save_news_suppress([], SUPPRESS_PATH)
        st.warning("已清空 suppress；请重新 sync-calendar 以恢复新闻 hard 命中。")
        st.rerun()
