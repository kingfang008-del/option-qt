#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Redis 实时流 vs strict replay 同日对拍。

流程:
  1. (可选) 从 quote_features 1min 推理 net_edge → parquet
  2. strict replay 收集 decision / SIGNAL 信号
  3. 跑 Redis 四进程 + 秒级发球机 (--sync 保证 FCS/SE 跟上)
  4. 对比 OMS signals audit CSV vs replay decision 层

用法:
  conda activate ibkr
  python qqq_btc/tools/compare_stream_replay_day.py --date 2026-02-02

  # 接老版桶级 + NPZ 阈值审计(verify_parity_raw / verify_parity_thresholds):
  python qqq_btc/tools/compare_stream_replay_day.py \\
    --date 2026-06-02 --legacy-parity --exit-layer bucket

  # 已有 infer parquet + 已跑过 redis:
  python qqq_btc/tools/compare_stream_replay_day.py \\
    --date 2026-02-02 \\
    --parquet /tmp/stream_parity/infer_2026-02.parquet \\
    --skip-redis
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_IBKR_PY = Path.home() / "anaconda3" / "envs" / "ibkr" / "bin" / "python"
_DEFAULT_CKPT = _REPO / "checkpoint" / "checkpoints_qqq_v4" / "best.pth"
_DEFAULT_FEATURES = (
    Path.home() / "train_data/quote_features_raw/QQQ/regular/09:30-16:00/1min/2026-02.parquet"
)
_WORK = Path("/tmp/qqq_btc_stream_parity")


def _resolve_python() -> str:
    if os.environ.get("PYTHON", "").strip() and Path(os.environ["PYTHON"]).is_file():
        return os.environ["PYTHON"]
    if _IBKR_PY.is_file():
        return str(_IBKR_PY)
    return sys.executable


def _date_yyyymmdd(date: str) -> str:
    return date.replace("-", "")


def _iso_from_yyyymmdd(ymd: str) -> str:
    ymd = _date_yyyymmdd(ymd)
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"


def resolve_warmup_config(
    warmup_from: str,
    *,
    parquet: Path,
    target_date: str,
) -> tuple[str, str]:
    """
    解析预热起点,与 strict replay 的 warmup_from_day 对齐。

    warmup_from:
      - auto: parquet 中 <= target 的最早交易日
      - same-day: 仅目标日(冷启动对拍)
      - YYYY-MM-DD / YYYYMMDD: 显式起点
    """
    target_iso = str(pd.Timestamp(target_date).date())
    mode = (warmup_from or "auto").strip().lower()
    if mode in ("same-day", "same_day", "cold"):
        ymd = _date_yyyymmdd(target_iso)
        return target_iso, ymd

    if mode == "auto":
        from qqq_btc.common.event_replay import prepare_minute_frame

        minute_df = prepare_minute_frame(pd.read_parquet(parquet))
        cutoff = pd.Timestamp(target_date).date()
        days = minute_df.loc[minute_df["_day"] <= cutoff, "_day"]
        if days.empty:
            raise ValueError(f"parquet has no rows on/before {target_date}: {parquet}")
        first = str(days.min())
        return first, _date_yyyymmdd(first)

    ymd = _date_yyyymmdd(mode)
    return _iso_from_yyyymmdd(ymd), ymd


def _infer_row_for_day_bar(df: pd.DataFrame, date: str, session_bar: int) -> pd.Series:
    """按月 parquet 取指定交易日 + session_bar 的行（避免误取月初同 sb）。"""
    work = df.copy()
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
    ny_dates = work["timestamp"].dt.tz_convert("America/New_York").dt.date
    mask = (ny_dates == pd.Timestamp(date).date()) & (work["session_bar"] == int(session_bar))
    hits = work.loc[mask]
    if hits.empty:
        raise ValueError(f"no infer row for date={date} session_bar={session_bar}")
    return hits.iloc[0]


def report_feature_parity_vs_pg(
    *,
    parquet: Path,
    date: str,
    session_bar: int = 15,
    frozen_norm: str | None = None,
) -> dict | None:
    """
    对比 stream PG debug_slow 与 infer parquet 在指定 bar 的归一化特征差。
    PG 表不存在时返回 None。

    trend_fit_* 由 SE 本地补算注入模型, FCS debug_slow 常为 0,不计入 normalized 门禁。
    """
    import os

    fcs_features = (
        "adx_smooth_10",
        "bb_width",
        "options_struc_skew",
        "options_vw_iv",
        "options_vw_imbalance",
        # 扩大门禁:此前仅抽检 5 列会漏掉导致分数分叉的大 GAP
        "vwap_log_return",
        "return_divergence",
        "poc_deviation",
        "options_vw_delta",
        "options_flow_skew",
        "close_log_return",
        "volume_log",
        "vwap_diff",
        "options_iv_momentum",
    )
    se_derived_features = ("trend_fit_r2_120m",)
    features = fcs_features + se_derived_features
    try:
        import psycopg2
    except ImportError:
        return None

    infer = pd.read_parquet(parquet)
    infer_row = _infer_row_for_day_bar(infer, date, session_bar)
    ts_unix = int(pd.Timestamp(infer_row["timestamp"]).timestamp())

    pg_url = os.environ.get("PG_DB_URL", "postgresql://postgres:postgres@localhost:5432/quant_trade")
    ymd = _date_yyyymmdd(date)
    table = f"debug_slow_{ymd}"
    try:
        conn = psycopg2.connect(pg_url)
        cols = ", ".join(features)
        pg = pd.read_sql(
            f"select ts, symbol, {cols} from {table} where ts=%s and symbol=%s limit 1",
            conn,
            params=(float(ts_unix), "QQQ"),
        )
        conn.close()
    except Exception as exc:
        print(f"[feature-parity] skip PG ({table}): {exc}")
        return None

    if pg.empty:
        print(f"[feature-parity] no PG row ts={ts_unix} in {table}")
        return None

    rows = []
    for feat in features:
        if feat not in infer_row.index or feat not in pg.columns:
            continue
        iv = float(infer_row[feat])
        pv = float(pg[feat].iloc[0])
        rows.append(
            {
                "feature": feat,
                "infer": iv,
                "stream_pg": pv,
                "diff": pv - iv,
                "layer": "se_derived" if feat in se_derived_features else "fcs",
            }
        )
    out = {
        "date": date,
        "session_bar": int(session_bar),
        "ts_unix": ts_unix,
        "features": rows,
    }
    print(f"\n=== Feature parity infer vs stream PG @ {date} sb={session_bar} ===")
    for r in rows:
        ok = abs(r["diff"]) < 0.05
        if r["layer"] == "se_derived":
            tag = "se_derived" if not ok else "ok"
        else:
            tag = "ok" if ok else "GAP"
        print(f"  {r['feature']:24s} diff={r['diff']:+.6f}  [{tag}]")
    return out


def _offline_raw_month_path(root: Path, symbol: str, date: str) -> Path | None:
    month = date[:7]
    for p in (
        root / symbol / "regular/09:30-16:00/1min" / f"{month}.parquet",
        root / "regular/09:30-16:00/1min" / f"{month}.parquet",
        root / f"{month}.parquet",
    ):
        if p.exists():
            return p
    return None


def report_raw_snapshot_parity(
    *,
    snapshot_path: Path,
    offline_root: Path,
    symbol: str,
    date: str,
    tolerance: float,
) -> dict | None:
    """Compare FCS final raw_vec snapshot with offline quote_features_raw."""
    try:
        from qqq_btc.common.feature_parity import PRICE_PANDAS_OPTIONAL, VIX_FEATURES
    except Exception:
        PRICE_PANDAS_OPTIONAL = ("poc_deviation",)
        VIX_FEATURES = ("vix_level",)

    optional_features = set(PRICE_PANDAS_OPTIONAL)
    diagnostic_features = set(VIX_FEATURES)
    if not snapshot_path.exists():
        print(f"[raw-parity] snapshot missing: {snapshot_path}")
        return None
    try:
        import numpy as np
    except ImportError:
        return None

    month_path = _offline_raw_month_path(offline_root.expanduser(), symbol, date)
    if month_path is None:
        print(f"[raw-parity] offline raw month missing under {offline_root}")
        return None

    snap = np.load(snapshot_path, allow_pickle=True)
    if "raw_feature_names" not in snap.files or "raw_feature_values" not in snap.files:
        print(f"[raw-parity] snapshot has no raw_feature_* arrays: {snapshot_path}")
        return None

    names = np.asarray(snap["raw_feature_names"]).astype(str)
    values = np.asarray(snap["raw_feature_values"], dtype=float)
    raw = dict(zip(names, values))
    alpha_ts = int(np.asarray(snap["alpha_label_ts"]).reshape(-1)[0])
    target_ts = pd.Timestamp(alpha_ts, unit="s", tz="UTC").tz_convert("America/New_York")

    offline = pd.read_parquet(month_path)
    offline["timestamp"] = pd.to_datetime(offline["timestamp"], utc=True).dt.tz_convert(
        "America/New_York"
    )
    hit = offline.loc[offline["timestamp"] == target_ts]
    if hit.empty:
        print(f"[raw-parity] no offline row timestamp={target_ts} in {month_path}")
        return None
    row = hit.iloc[0]

    rows = []
    for name, stream_val in raw.items():
        if name not in row.index:
            continue
        try:
            offline_val = float(row[name])
            stream_f = float(stream_val)
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(offline_val) and np.isfinite(stream_f)):
            continue
        diff = stream_f - offline_val
        if name in diagnostic_features:
            tier = "diagnostic_known_divergent"
            blocks_acceptance = False
        elif name in optional_features:
            tier = "optional"
            blocks_acceptance = False
        else:
            tier = "mandatory"
            blocks_acceptance = True
        rows.append(
            {
                "feature": str(name),
                "tier": tier,
                "offline_raw": offline_val,
                "stream_raw": stream_f,
                "diff": float(diff),
                "abs_diff": float(abs(diff)),
                "ok": bool(abs(diff) <= float(tolerance)),
                "blocks_acceptance": bool(blocks_acceptance),
            }
        )
    rows.sort(key=lambda r: r["abs_diff"], reverse=True)
    n_checked = len(rows)
    n_ok = sum(1 for r in rows if r["ok"])
    mandatory_rows = [r for r in rows if r["blocks_acceptance"]]
    mandatory_ok = sum(1 for r in mandatory_rows if r["ok"])
    diagnostic_rows = [r for r in rows if r["tier"] == "diagnostic_known_divergent"]
    optional_rows = [r for r in rows if r["tier"] == "optional"]
    out = {
        "symbol": symbol,
        "date": date,
        "snapshot": str(snapshot_path),
        "offline_raw": str(month_path),
        "ts_unix": alpha_ts,
        "timestamp": str(target_ts),
        "tolerance": float(tolerance),
        "n_checked": int(n_checked),
        "n_ok": int(n_ok),
        "n_mandatory": int(len(mandatory_rows)),
        "n_mandatory_ok": int(mandatory_ok),
        "n_optional": int(len(optional_rows)),
        "n_diagnostic_known_divergent": int(len(diagnostic_rows)),
        "ok": bool(mandatory_rows and mandatory_ok == len(mandatory_rows)),
        "max_abs_diff": float(rows[0]["abs_diff"]) if rows else None,
        "max_mandatory_abs_diff": (
            float(max(r["abs_diff"] for r in mandatory_rows)) if mandatory_rows else None
        ),
        "top_diffs": rows[:20],
    }
    print(f"\n=== Raw feature parity @ {target_ts} ===")
    print(
        f"  mandatory={mandatory_ok}/{len(mandatory_rows)} "
        f"optional={len(optional_rows)} diagnostic={len(diagnostic_rows)} "
        f"max_mandatory_abs={out['max_mandatory_abs_diff']} "
        f"→ {'PASS' if out['ok'] else 'FAIL'}"
    )
    for r in rows[:10]:
        tag = "ok" if r["ok"] else "GAP"
        print(f"  {r['feature']:28s} diff={r['diff']:+.6g} [{tag}] tier={r['tier']}")
    return out


def _month_feature_path(date: str, *, resolution: str = "1min") -> Path:
    y, m, _ = date.split("-")
    return (
        Path.home()
        / f"train_data/quote_features_raw/QQQ/regular/09:30-16:00/{resolution}/{y}-{m}.parquet"
    )


def _infer_parquet_path(date: str, *, frozen_norm: str | None = None) -> Path:
    month = date[:7]
    # 含 1m+5m merge,避免复用旧「仅 1min」缓存
    if frozen_norm:
        return _WORK / f"infer_{month}_frozen_1m5m.parquet"
    return _WORK / f"infer_{month}_1m5m.parquet"


def _feats_5m_from_config(cfg: dict) -> list[str]:
    return [
        str(f.get("name"))
        for f in cfg.get("features", [])
        if f.get("name") and str(f.get("resolution", "1min")).lower() in ("5min", "5m")
    ]


def ensure_infer_parquet(
    date: str,
    *,
    parquet: Path | None,
    checkpoint: Path,
    force: bool = False,
    frozen_norm: str | None = None,
) -> Path:
    out = parquet or _infer_parquet_path(date, frozen_norm=frozen_norm)
    if out.exists() and not force:
        print(f"[infer] reuse {out}")
        return out
    feat_1m = _month_feature_path(date, resolution="1min")
    if not feat_1m.exists():
        raise FileNotFoundError(f"feature parquet not found: {feat_1m}")

    import json
    import torch

    from qqq_btc.tools.eval_test_set import merge_1m_5m

    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg_path = _WORK / "config_from_checkpoint.json"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_cfg = bundle.get("config")
    if ckpt_cfg is None:
        cfg_path = _REPO / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"
        ckpt_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    else:
        cfg_path.write_text(json.dumps(ckpt_cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[infer] using config embedded in checkpoint → {cfg_path}")

    feats_5m = _feats_5m_from_config(ckpt_cfg if isinstance(ckpt_cfg, dict) else {})
    feat_5m = _month_feature_path(date, resolution="5min")
    merged = _WORK / f"features_{date[:7]}_1m5m.parquet"
    merged.parent.mkdir(parents=True, exist_ok=True)
    df_merged = merge_1m_5m(feat_1m, feat_5m, feats_5m)
    df_merged.to_parquet(merged, index=False)
    print(
        f"[infer] merged 1min+5min → {merged} "
        f"(5min_cols={len(feats_5m)}, rows={len(df_merged)})"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _resolve_python(),
        str(_REPO / "qqq_btc" / "tools" / "run_inference.py"),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(cfg_path),
        "--input",
        str(merged),
        "--output",
        str(out),
        "--symbol-map",
        str(_REPO / "qqq_btc" / "CONFIG" / "symbol_map.json"),
    ]
    if frozen_norm:
        cmd.extend(["--frozen-norm", str(Path(frozen_norm).expanduser())])
        print(f"[infer] frozen norm replay ← {frozen_norm} | features ← {merged}")
    print("[infer]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    return out


def truncate_debug_slow_day(date: str) -> None:
    """对拍前清空目标日 debug_slow 分区，避免旧 run 污染 PG。"""
    import psycopg2
    from datetime import datetime, timedelta

    sys.path.insert(0, str(_REPO / "New_Pro" / "baseline_qqq"))
    from config import PG_DB_URL, NY_TZ

    ymd = _date_yyyymmdd(date)
    day = datetime.strptime(ymd, "%Y%m%d")
    start_dt = NY_TZ.localize(day)
    end_dt = start_dt + timedelta(days=1)
    part = f"debug_slow_{ymd}"
    conn = psycopg2.connect(PG_DB_URL)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute(
        "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = %s)",
        (part,),
    )
    if cur.fetchone()[0]:
        cur.execute(f"TRUNCATE TABLE {part}")
        print(f"[pg] truncated {part}")
    conn.close()


def run_redis_stream(
    date: str,
    *,
    speed: float,
    fcs_wait: float,
    sync: bool,
    source: str = "raw",
    option_root: str | None = None,
    greek_parity: bool = False,
    greek_root: str | None = None,
    frozen_norm: str | None = None,
    warmup_from_date: str | None = None,
    deep_warmup: bool = False,
    max_session_bars: int | None = None,
    checkpoint: Path | None = None,
    fill_audit_path: Path | None = None,
) -> Path:
    ymd = _date_yyyymmdd(date)
    signals = Path.home() / "quant_project" / "shadow" / f"signals_{date}.csv"
    if signals.exists():
        signals.unlink()
    se_alpha = Path.home() / "quant_project" / "shadow" / f"se_alpha_{date}.csv"
    if se_alpha.exists():
        se_alpha.unlink()
    if fill_audit_path is not None:
        fill_audit_path = Path(fill_audit_path).expanduser()
        fill_audit_path.parent.mkdir(parents=True, exist_ok=True)
        # 按日隔离 + 截断,避免共享 fill_audit.csv 历史污染
        fill_audit_path.write_text("", encoding="utf-8")
        os.environ["QQQ_BTC_FILL_AUDIT_PATH"] = str(fill_audit_path)
        os.environ.setdefault("QQQ_BTC_FILL_AUDIT", "1")
        print(f"[redis] fresh fill_audit → {fill_audit_path}")
    cmd = [
        _resolve_python(),
        str(_REPO / "qqq_btc" / "tools" / "run_qqq_btc_redis_sim.py"),
        "--date",
        ymd,
        "--source",
        source,
        "--fcs-wait",
        str(fcs_wait),
        "--speed",
        str(speed),
    ]
    if warmup_from_date:
        cmd.extend(["--warmup-from-date", warmup_from_date])
    if deep_warmup:
        cmd.append("--deep-warmup")
    if option_root:
        cmd.extend(["--option-root", option_root])
    if greek_parity:
        cmd.append("--greek-parity")
    if greek_root:
        cmd.extend(["--greek-root", greek_root])
    if frozen_norm:
        cmd.extend(["--frozen-norm", frozen_norm])
    if max_session_bars:
        cmd.extend(["--max-session-bars", str(max_session_bars)])
    if checkpoint is not None:
        cmd.extend(["--checkpoint", str(Path(checkpoint).expanduser().resolve())])
    if not sync:
        cmd.append("--no-sync")
    print("[redis]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    if not signals.exists():
        # 目标日无 PASS 信号时 audit 文件不会创建，写空头占位以便后续 diff
        signals.parent.mkdir(parents=True, exist_ok=True)
        from qqq_btc.live.signal_audit_writer import _HEADER

        with signals.open("w", newline="") as f:
            import csv

            csv.writer(f).writerow(_HEADER)
        print(f"[redis] no PASS signals for {date}; created empty audit {signals}")
    return signals


def attach_exec_quotes_for_replay(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """为 strict replay 回补 exec_* 盘口(与 eval_test_set 同逻辑)。"""
    from qqq_btc.tools.eval_test_set import attach_exec_quotes
    from qqq_btc.qqq import config as qcfg

    candidates = [
        Path("/mnt/s990/data/raw_1m/options_databento"),
        Path.home() / "train_data/quote_options_day_iv",
        _WORK / "option_exec",
    ]
    for root in candidates:
        sym_dir = root / "QQQ"
        day_fp = sym_dir / f"QQQ_{date}.parquet"
        if day_fp.exists():
            return attach_exec_quotes(
                df,
                root,
                "QQQ",
                call_bucket=qcfg.TRADE_BUCKET_ID,
                put_bucket=0,
            )
        std = root / "QQQ" / "standard" / f"QQQ_{date}.parquet"
        if std.exists():
            link_root = _WORK / "option_exec"
            link_sym = link_root / "QQQ"
            link_sym.mkdir(parents=True, exist_ok=True)
            link_path = link_sym / f"QQQ_{date}.parquet"
            if not link_path.exists():
                link_path.symlink_to(std)
            return attach_exec_quotes(
                df,
                link_root,
                "QQQ",
                call_bucket=qcfg.TRADE_BUCKET_ID,
                put_bucket=0,
            )
    print(f"[warn] no option quotes for {date}; replay spread gate may block all entries")
    return df


def replay_baseline_signals(
    parquet: Path,
    date: str,
    *,
    warmup_from_day: str | None = None,
    warmup_through_day: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    import pandas as pd
    from qqq_btc.common.signal_collect import collect_decision_signals, collect_replay_signals
    from qqq_btc.qqq import config as qcfg

    through = warmup_through_day or date
    df = pd.read_parquet(parquet)
    df = attach_exec_quotes_for_replay(df, date)
    decision = collect_decision_signals(
        df,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        target_day=date,
        replay_cfg=qcfg.REPLAY,
    )
    operational = collect_replay_signals(
        df,
        replay_cfg=qcfg.REPLAY,
        warmup_from_day=warmup_from_day,
        warmup_through_day=through,
        target_day=date,
        signal_kinds=("SIGNAL",),
        source="strict_replay",
        signal_only=False,
    )
    return decision, operational


def run_diff(
    *,
    parquet: Path,
    date: str,
    dry_run_signals: Path,
    output: Path,
    tolerance_bars: int,
    se_alpha_signals: Path | None = None,
    warmup_from_day: str | None = None,
    warmup_through_day: str | None = None,
    max_session_bar: int | None = None,
    fill_audit_path: Path | None = None,
) -> dict:
    from qqq_btc.tools.signal_diff_day import run_day_diff

    enriched = _WORK / f"infer_{date}_exec.parquet"
    if not enriched.exists() or enriched.stat().st_mtime < parquet.stat().st_mtime:
        df = pd.read_parquet(parquet)
        df = attach_exec_quotes_for_replay(df, date)
        enriched.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(enriched, index=False)
    return run_day_diff(
        parquet=enriched,
        date=date,
        dry_run_signals=dry_run_signals,
        se_alpha_signals=se_alpha_signals,
        output=output,
        tolerance_bars=tolerance_bars,
        warmup_from_day=warmup_from_day,
        warmup_through_day=warmup_through_day or date,
        max_session_bar=max_session_bar,
        fill_audit_path=fill_audit_path,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Redis stream vs strict replay parity")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD (raw 期权 parquet 或 sqlite)")
    parser.add_argument(
        "--parquet",
        default=None,
        help="strict replay 基准 parquet(默认 /tmp/qqq_btc_test_eval_v4/test_infer.parquet)",
    )
    parser.add_argument("--source", choices=("raw", "sqlite", "auto"), default="raw")
    parser.add_argument("--option-root", default=str(Path("/mnt/s990/data/raw_1s/options_databento_v3")))
    parser.add_argument(
        "--greek-parity",
        action="store_true",
        help="[诊断] 发球机注入分钟 IV parquet Greeks;默认关闭(honest sim, FCS BSM 自算)",
    )
    parser.add_argument(
        "--no-greek-parity",
        action="store_true",
        help="[废弃] 已是默认行为;保留兼容",
    )
    parser.add_argument(
        "--greek-root",
        default=str(Path.home() / "train_data/quote_options_day_iv"),
    )
    parser.add_argument(
        "--frozen-norm",
        default=None,
        help="FCS/strict replay 共用冻结 normalizer .npz(默认 frozen_norm_qqq_daily.npz)",
    )
    parser.add_argument(
        "--rolling-norm-replay",
        action="store_true",
        help="strict replay 仍用滚动 norm infer(默认与 --frozen-norm 对齐 live)",
    )
    parser.add_argument("--checkpoint", default=str(_DEFAULT_CKPT))
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--skip-redis", action="store_true")
    parser.add_argument("--force-infer", action="store_true")
    parser.add_argument("--speed", type=float, default=float("inf"), help="发球速度 inf=全速+sync 背压")
    parser.add_argument("--fcs-wait", type=float, default=45.0)
    parser.add_argument("--no-sync", action="store_true")
    parser.add_argument("--tolerance-bars", type=int, default=0, help="decision 层建议 0")
    parser.add_argument(
        "--warmup-from",
        default="auto",
        help="预热起点(默认 auto=warm/carryover 口径,与实盘一致): "
        "same-day=无跨日预热(快速诊断) | auto=parquet最早日 | YYYY-MM-DD",
    )
    parser.add_argument(
        "--deep-warmup",
        action="store_true",
        help="FCS 从 PG 深预热(仅流目标日);默认靠多日流式预热",
    )
    parser.add_argument(
        "--max-session-bars",
        type=int,
        default=None,
        help="仅流目标日前 N 个 session bar(配合 warmup-from 做快速首小时对拍)",
    )
    parser.add_argument(
        "--feature-session-bar",
        type=int,
        default=15,
        help="对拍报告中的 session_bar(须与 --date 同日,默认 15)",
    )
    parser.add_argument(
        "--fresh-debug-pg",
        action="store_true",
        help="对拍前 TRUNCATE debug_slow_YYYYMMDD 分区",
    )
    parser.add_argument(
        "--raw-parity-root",
        default=None,
        help="离线 quote_features_raw 根目录；设置后捕获 FCS raw_vec snapshot 并做 raw 特征层对拍",
    )
    parser.add_argument("--raw-parity-symbol", default="QQQ")
    parser.add_argument("--raw-parity-tolerance", type=float, default=1e-3)
    parser.add_argument(
        "--raw-parity-output",
        default=None,
        help="FCS raw snapshot npz 输出路径；默认写到 /tmp",
    )
    parser.add_argument(
        "--legacy-parity",
        action="store_true",
        help="启用老版 verify_parity_raw(期权桶) + verify_parity_thresholds(NPZ) 分层审计",
    )
    parser.add_argument(
        "--bucket-parity-sqlite",
        default=None,
        help="可选:用 sqlite option_snapshots_1m 作桶参考(默认用 quote_options_*_iv)",
    )
    parser.add_argument(
        "--threshold-parity-right",
        default=None,
        help="可选:阈值审计 RIGHT NPZ;默认用离线分钟 IV 参考桶合成",
    )
    parser.add_argument(
        "--exit-layer",
        choices=(
            "raw",
            "normalized",
            "decision",
            "oms",
            "bucket",
            "threshold_npz",
            "exit_lifecycle",
        ),
        default="oms",
        help="决定脚本退出码的验收层。默认 oms；exit_lifecycle=首笔平仓+全量 EXIT 匹配。",
    )
    parser.add_argument("--output", default=None, help="JSON 报告")
    args = parser.parse_args()
    # honest sim 默认:不注入分钟 IV parquet,FCS 走 BSM 自算
    greek_parity = bool(args.greek_parity) and not args.no_greek_parity
    if greek_parity:
        print("[warn] --greek-parity 为诊断模式(读离线 IV 答案),不可作为 parity 验收")

    opt_probe = Path(args.option_root) / "QQQ" / f"QQQ_{args.date}.parquet"
    sqlite_db = Path.home() / "quant_project/data/history_sqlite_1s" / f"market_{_date_yyyymmdd(args.date)}.db"
    if args.source == "sqlite" and not sqlite_db.exists():
        print(f"ERROR: sqlite 1s 不存在: {sqlite_db}", file=sys.stderr)
        return 1
    if args.source in ("raw", "auto") and not opt_probe.exists() and not sqlite_db.exists():
        print(f"ERROR: raw option 与 sqlite 均不存在: {opt_probe}", file=sys.stderr)
        return 1

    frozen_norm = None if args.rolling_norm_replay else args.frozen_norm
    if frozen_norm is None and not args.rolling_norm_replay:
        for candidate in (
            _REPO / "qqq_btc" / "CONFIG" / "frozen_norm_qqq_daily.npz",
            _REPO / "qqq_btc" / "CONFIG" / "frozen_norm_qqq_test_upto202605.npz",
            _REPO / "qqq_btc" / "CONFIG" / "frozen_norm_qqq_test.npz",
        ):
            if candidate.exists():
                frozen_norm = str(candidate)
                break

    pq = Path(args.parquet).expanduser() if args.parquet else None
    if pq is None:
        candidates = []
        if frozen_norm:
            candidates.append(_infer_parquet_path(args.date, frozen_norm=frozen_norm))
        candidates.extend(
            [
                Path("/tmp/qqq_btc_test_eval_v4/test_infer.parquet"),
                _infer_parquet_path(args.date),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                pq = candidate
                break
    if not args.skip_infer:
        pq = ensure_infer_parquet(
            args.date,
            parquet=pq if args.parquet else None,
            checkpoint=Path(args.checkpoint),
            force=args.force_infer,
            frozen_norm=frozen_norm,
        )
    elif pq is None:
        pq = _infer_parquet_path(args.date, frozen_norm=frozen_norm)
    if not pq.exists():
        print(f"ERROR: infer parquet 不存在: {pq}", file=sys.stderr)
        return 1

    raw_snapshot_path: Path | None = None
    capture_parity_npz = bool(args.raw_parity_root) or bool(args.legacy_parity)
    if capture_parity_npz:
        try:
            raw_row = _infer_row_for_day_bar(
                pd.read_parquet(pq), args.date, int(args.feature_session_bar)
            )
            raw_ts = int(pd.Timestamp(raw_row["timestamp"]).timestamp())
            raw_snapshot_path = (
                Path(args.raw_parity_output).expanduser()
                if args.raw_parity_output
                else Path(
                    f"/tmp/fcs_{args.raw_parity_symbol.lower()}_"
                    f"{_date_yyyymmdd(args.date)}_sb{int(args.feature_session_bar)}_rawvec.npz"
                )
            )
            if raw_snapshot_path.exists() and not args.skip_redis:
                raw_snapshot_path.unlink()
            os.environ["FCS_FEATURE_PARITY_SYMBOL"] = str(args.raw_parity_symbol)
            os.environ["FCS_FEATURE_PARITY_TS"] = str(raw_ts)
            os.environ["FCS_FEATURE_PARITY_OUTPUT"] = str(raw_snapshot_path)
            print(
                f"[parity-npz] capture {args.raw_parity_symbol} "
                f"sb={args.feature_session_bar} ts={raw_ts} → {raw_snapshot_path}"
            )
        except Exception as exc:
            print(f"[parity-npz] setup skipped: {exc}")
            raw_snapshot_path = None

    warmup_from_iso, warmup_from_ymd = resolve_warmup_config(
        args.warmup_from, parquet=pq, target_date=args.date
    )
    print(
        f"[warmup] stream/replay aligned | from={warmup_from_iso} through={args.date} "
        f"| mode={args.warmup_from} deep_warmup={args.deep_warmup}"
    )
    if warmup_from_ymd != _date_yyyymmdd(args.date) and not args.skip_redis:
        n_days_hint = "多日"
        print(
            f"[warmup] 将流式预热 {n_days_hint} 数据 ({warmup_from_iso} → {args.date}),"
            " 耗时较长; 快速诊断可用 --warmup-from same-day"
        )

    signals = Path.home() / "quant_project" / "shadow" / f"signals_{args.date}.csv"
    se_alpha_signals = Path.home() / "quant_project" / "shadow" / f"se_alpha_{args.date}.csv"
    from qqq_btc.common.exit_lifecycle import date_scoped_fill_audit_path

    fill_audit_path = date_scoped_fill_audit_path(args.date)
    if not args.skip_redis:
        if args.fresh_debug_pg:
            truncate_debug_slow_day(args.date)
        signals = run_redis_stream(
            args.date,
            speed=args.speed,
            fcs_wait=args.fcs_wait,
            sync=not args.no_sync,
            source=args.source,
            option_root=args.option_root,
            greek_parity=greek_parity,
            greek_root=args.greek_root,
            frozen_norm=frozen_norm,
            warmup_from_date=warmup_from_ymd,
            deep_warmup=args.deep_warmup,
            max_session_bars=args.max_session_bars,
            checkpoint=Path(args.checkpoint),
            fill_audit_path=fill_audit_path,
        )
    elif not signals.exists():
        print(f"ERROR: --skip-redis 但无 signals CSV: {signals}", file=sys.stderr)
        return 1

    out = Path(args.output).expanduser() if args.output else (_WORK / f"diff_{args.date}.json")

    decision, operational = replay_baseline_signals(
        pq,
        args.date,
        warmup_from_day=warmup_from_iso,
        warmup_through_day=args.date,
    )
    print(
        f"[replay baseline] decision={len(decision)} operational SIGNAL={len(operational)}"
    )
    if len(decision) == 0 and len(operational) == 0:
        print(
            "[warn] strict replay 当日无入场信号;redis 流若也为 0 则一致,但无法验证有信号日 parity"
        )

    report = run_diff(
        parquet=pq,
        date=args.date,
        dry_run_signals=signals,
        output=out,
        tolerance_bars=int(args.tolerance_bars),
        se_alpha_signals=se_alpha_signals,
        warmup_from_day=warmup_from_iso,
        warmup_through_day=args.date,
        max_session_bar=args.max_session_bars,
        fill_audit_path=fill_audit_path if fill_audit_path.exists() else None,
    )

    dr = report.get("replay_vs_dry_run", {}).get("summary", {})
    fe = report.get("first_entry_vs_dry_run", {}).get("summary", {})
    ls = report.get("live_sim_vs_dry_run", {}).get("summary", {})
    dec = report.get("decision_replay_vs_live", {}).get("summary", {})
    se_dec = report.get("replay_vs_se_alpha_decision", {}).get("summary", {})
    if se_dec:
        ok_decision = se_dec.get("n_replay") == se_dec.get("n_live") == se_dec.get("n_matched")
    else:
        ok_decision = dec.get("n_replay") == dec.get("n_live") == dec.get("n_matched")
    # OMS 主门禁:占仓感知首笔(live_sim ENTER vs dry PASS);无信号日双方皆空也算通过
    if fe:
        ok_stream = (
            fe.get("n_matched", 0) == 1
            or (fe.get("n_replay", 0) == 0 and fe.get("n_live", 0) == 0)
        )
    elif ls:
        ok_stream = (
            ls.get("n_replay", 0) == ls.get("n_matched", 0) == ls.get("n_live", 0)
            if ls.get("n_replay", 0) or ls.get("n_live", 0)
            else True
        )
    else:
        ok_stream = (
            dr.get("n_replay", 0) == dr.get("n_matched", 0) == dr.get("n_live", 0)
            if dr.get("n_replay", 0) or dr.get("n_live", 0)
            else True
        )

    print("\n=== Parity verdict ===")
    decision_label = "SE alpha decision vs replay" if se_dec else "decision config parity on replay parquet"
    print(f"  {decision_label}: {'PASS' if ok_decision else 'FAIL'}")
    if fe:
        print(
            f"  OMS first-entry (live_sim vs stream): "
            f"offline sb={fe.get('session_bar_offline')} {fe.get('leg_offline')} | "
            f"stream sb={fe.get('session_bar_stream')} {fe.get('leg_stream')} | "
            f"delta={fe.get('bar_delta')} → {'PASS' if ok_stream else 'FAIL'}"
        )
    else:
        print(
            f"  strict replay decision vs redis stream: "
            f"replay={dr.get('n_replay', 0)} stream={dr.get('n_live', 0)} "
            f"matched={dr.get('n_matched', 0)} rate={dr.get('match_rate_replay', 0):.1%} "
            f"→ {'PASS' if ok_stream else 'FAIL'}"
        )
    print(f"  report: {out}")

    feat_report = None
    if not args.skip_redis:
        feat_report = report_feature_parity_vs_pg(
            parquet=pq,
            date=args.date,
            session_bar=int(args.feature_session_bar),
            frozen_norm=frozen_norm,
        )
    raw_report = None
    if args.raw_parity_root and raw_snapshot_path is not None:
        raw_report = report_raw_snapshot_parity(
            snapshot_path=raw_snapshot_path,
            offline_root=Path(args.raw_parity_root),
            symbol=str(args.raw_parity_symbol),
            date=args.date,
            tolerance=float(args.raw_parity_tolerance),
        )

    bucket_report = None
    threshold_report = None
    if args.legacy_parity:
        from qqq_btc.common.legacy_parity_audit import (
            build_ref_npz_from_option_buckets,
            load_offline_ref_buckets,
            report_redis_option_bucket_parity,
            report_threshold_npz_parity,
        )

        try:
            parity_row = _infer_row_for_day_bar(
                pd.read_parquet(pq), args.date, int(args.feature_session_bar)
            )
            parity_ts = int(pd.Timestamp(parity_row["timestamp"]).timestamp())
        except Exception as exc:
            print(f"[legacy-parity] resolve ts failed: {exc}")
            parity_ts = None

        rds = None
        try:
            import redis
            from config import REDIS_CFG, get_redis_db  # type: ignore

            rds = redis.Redis(
                host=REDIS_CFG.get("host", "localhost"),
                port=int(REDIS_CFG.get("port", 6379)),
                db=int(get_redis_db()),
            )
        except Exception:
            try:
                import redis

                rds = redis.Redis(host="localhost", port=6379, db=0)
            except Exception as exc:
                print(f"[legacy-parity] redis unavailable: {exc}")

        if parity_ts is not None:
            greek_root = Path(args.greek_root).expanduser() if getattr(args, "greek_root", None) else None
            bucket_report = report_redis_option_bucket_parity(
                symbol=str(args.raw_parity_symbol),
                date=args.date,
                ts_unix=int(parity_ts),
                greek_root=greek_root,
                redis_client=rds,
                sqlite_db=args.bucket_parity_sqlite,
            )

            if raw_snapshot_path is not None and raw_snapshot_path.exists():
                right_npz = (
                    Path(args.threshold_parity_right).expanduser()
                    if args.threshold_parity_right
                    else raw_snapshot_path.with_name(
                        raw_snapshot_path.stem + "_ref_from_minute_iv.npz"
                    )
                )
                if not args.threshold_parity_right:
                    ref_buckets = load_offline_ref_buckets(
                        str(args.raw_parity_symbol),
                        args.date,
                        int(parity_ts),
                        greek_root=greek_root,
                    )
                    build_ref_npz_from_option_buckets(
                        raw_snapshot_path, ref_buckets, right_npz
                    )
                threshold_report = report_threshold_npz_parity(
                    left_npz=raw_snapshot_path,
                    right_npz=right_npz,
                )
            else:
                print(
                    "[legacy-parity] FCS NPZ missing; threshold audit skipped "
                    "(need capture via --legacy-parity during redis run)"
                )

    norm_ok = None
    if feat_report and feat_report.get("features"):
        fcs_rows = [
            r for r in feat_report["features"] if r.get("layer", "fcs") != "se_derived"
        ]
        norm_ok = (
            all(abs(float(r.get("diff", 0.0))) < 0.05 for r in fcs_rows)
            if fcs_rows
            else None
        )
    raw_ok = raw_report.get("ok") if raw_report is not None else None
    bucket_ok = bucket_report.get("ok") if bucket_report is not None else None
    threshold_ok = threshold_report.get("ok") if threshold_report is not None else None
    fe_exit = report.get("first_exit", {}).get("summary", {})
    el_sum = report.get("exit_lifecycle", {}).get("summary", {})
    fm_sum = report.get("fill_model_audit", {}) or {}
    # exit 层主门禁:首笔平仓时机/原因 + model_frac 声明;
    # 全量 EXIT 匹配作为诊断(mock mid 可能导致后续单 HARD_STOP vs EARLY_STOP 分叉)
    ok_exit = bool(fe_exit.get("pass"))
    if fm_sum.get("pass") is False and int(fm_sum.get("n") or 0) > 0:
        ok_exit = False
    ok_exit_full = bool(el_sum.get("pass")) and ok_exit
    layer_ok = {
        "raw": bool(raw_ok) if raw_ok is not None else None,
        "bucket": bool(bucket_ok) if bucket_ok is not None else None,
        "threshold_npz": bool(threshold_ok) if threshold_ok is not None else None,
        "normalized": bool(norm_ok) if norm_ok is not None else None,
        "decision": bool(ok_decision),
        "oms": bool(ok_stream),
        "exit_lifecycle": bool(ok_exit),
        "exit_lifecycle_full": bool(ok_exit_full),
    }

    def _layer_status(value: bool | None) -> str:
        if value is None:
            return "SKIP"
        return "PASS" if value else "FAIL"

    print("\n=== Layered acceptance ===")
    print(f"  FCS raw snapshot / feature parity: {_layer_status(layer_ok['raw'])}")
    if raw_ok is None:
        print("    raw snapshot not checked; set --raw-parity-root to enable")
    print(f"  Option bucket parity (raw/greeks):  {_layer_status(layer_ok['bucket'])}")
    if bucket_ok is None:
        print("    bucket parity not checked; set --legacy-parity to enable")
    print(f"  FCS NPZ threshold parity:           {_layer_status(layer_ok['threshold_npz'])}")
    if threshold_ok is None:
        print("    threshold NPZ not checked; set --legacy-parity to enable")
    print(f"  FCS normalized feature parity:      {_layer_status(layer_ok['normalized'])}")
    if norm_ok is None:
        print("    normalized PG parity not checked")
    if se_dec:
        print(f"  SE decision-only parity:            {_layer_status(layer_ok['decision'])}")
    else:
        print(f"  decision-only config parity:        {_layer_status(layer_ok['decision'])}")
        print("    note: no SE alpha audit found; this is replay/live config-only fallback")
    print(f"  OMS audit timing/position parity:   {_layer_status(layer_ok['oms'])}")
    print(f"  Exit lifecycle (OPEN/CLOSE):        {_layer_status(layer_ok['exit_lifecycle'])}")
    if fe_exit:
        print(
            f"    first exit: offline sb={fe_exit.get('session_bar_offline')} "
            f"{fe_exit.get('leg_offline')} {fe_exit.get('reason_offline')} | "
            f"stream sb={fe_exit.get('session_bar_stream')} "
            f"{fe_exit.get('leg_stream')} {fe_exit.get('reason_stream')}"
        )
    if el_sum:
        print(
            f"    exits matched: offline={el_sum.get('n_replay')} "
            f"stream={el_sum.get('n_live')} matched={el_sum.get('n_matched')} "
            f"(full={_layer_status(layer_ok.get('exit_lifecycle_full'))})"
        )
    if fm_sum:
        print(
            f"    fill model_frac: median={fm_sum.get('model_frac_median')} "
            f"target={fm_sum.get('target')} realized={fm_sum.get('fill_spread_frac_median')} "
            f"→ {_layer_status(fm_sum.get('pass'))}"
        )
    print(f"  exit layer: {args.exit_layer}")

    summary = {
        "date": args.date,
        "warmup_from": warmup_from_iso,
        "warmup_through": args.date,
        "warmup_mode": args.warmup_from,
        "deep_warmup": args.deep_warmup,
        "frozen_norm": frozen_norm,
        "rolling_norm_replay": args.rolling_norm_replay,
        "infer_parquet": str(pq),
        "redis_signals": str(signals),
        "se_alpha_signals": str(se_alpha_signals),
        "fill_audit_path": str(report.get("fill_audit_path") or fill_audit_path),
        "report_json": str(out),
        "decision_replay_vs_live_ok": ok_decision,
        "decision_source": "se_alpha_audit" if se_dec else "replay_config_only",
        "replay_vs_redis_stream_ok": ok_stream,
        "exit_lifecycle_ok": ok_exit,
        "replay_vs_dry_run_summary": dr,
        "first_entry_vs_dry_run_summary": fe or None,
        "live_sim_vs_dry_run_summary": ls or None,
        "replay_vs_se_alpha_decision_summary": se_dec or None,
        "first_exit_summary": fe_exit or None,
        "exit_lifecycle_summary": el_sum or None,
        "fill_model_audit": fm_sum or None,
        "layer_ok": layer_ok,
        "feature_raw_parity": raw_report,
        "option_bucket_parity": bucket_report,
        "threshold_npz_parity": threshold_report,
        "feature_parity_sb": feat_report,
    }
    verdict_path = out.with_name(out.stem + "_verdict.json")
    verdict_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if layer_ok.get(args.exit_layer) is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
