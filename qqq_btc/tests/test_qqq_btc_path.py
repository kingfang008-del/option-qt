#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
qqq_btc 路径一致性测试。

核心不变量:标签、strict replay 使用同一个 FillModel 实例时,
对「固定持有 hold_bars、无 rails 干预」的同一笔交易必须给出相同净收益。
这是对上一代三层成交假设互相矛盾缺陷的回归防护。

运行: python -m pytest qqq_btc/tests/test_qqq_btc_path.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qqq_btc.common.exit_rails import ExitRailsConfig, PositionState, check_exit
from qqq_btc.common.fill_model import (
    OptionSpreadFillModel,
    PerpFillModel,
    spread_interpolate,
)
from qqq_btc.common.labels import (
    LabelHorizon,
    build_option_net_labels,
    build_perp_net_labels,
    label_quality_report,
)
from qqq_btc.common.replay_harness import ReplayConfig, run_strict_replay


# ---------------------------------------------------------------------------
# fill_model
# ---------------------------------------------------------------------------

def test_spread_interpolate_matches_legacy_formula():
    """插值公式与 production mock_ibkr 的 _spread_interpolate_fill 语义一致。"""
    bid, ask = 2.00, 2.20
    assert abs(spread_interpolate(bid, ask, 0.5, "BUY") - 2.10) < 1e-12   # mid
    assert abs(spread_interpolate(bid, ask, 0.775, "BUY") - 2.155) < 1e-12
    assert abs(spread_interpolate(bid, ask, 0.775, "SELL") - 2.045) < 1e-12
    assert abs(spread_interpolate(bid, ask, 0.0, "BUY") - 2.00) < 1e-12   # 己方最优
    assert abs(spread_interpolate(bid, ask, 1.0, "BUY") - 2.20) < 1e-12   # 对手价


def test_spread_interpolate_invalid_book_returns_nan():
    assert np.isnan(spread_interpolate(0.0, 2.2, 0.775, "BUY"))
    assert np.isnan(spread_interpolate(2.3, 2.2, 0.775, "BUY"))  # crossed book


def test_option_round_trip_drag_magnitude():
    """0.775 双边 → 往返点差摩擦 = 0.55 * spread_pct(审查报告的量级算术)。"""
    fm = OptionSpreadFillModel(entry_frac=0.775, exit_frac=0.775)
    assert abs(fm.round_trip_spread_drag(0.02) - 0.55 * 0.02) < 1e-12


def test_perp_round_trip_fee():
    fm = PerpFillModel(taker_fee_bps=5.0, slippage_bps=2.0)
    assert abs(fm.round_trip_fee_drag() - 14e-4) < 1e-12
    assert abs(fm.funding_drag(0.0001, holding_seconds=8 * 3600) - 0.0001) < 1e-12


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------

def _make_option_df(n=40, base=2.0, drift=0.01, spread=0.10):
    """构造单调上涨的期权盘口:mid 从 base 每 bar +drift,点差固定。"""
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    mid = base + drift * np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "exec_call_bid": mid - spread / 2,
            "exec_call_ask": mid + spread / 2,
            "exec_call_mid": mid,
        }
    )


def test_option_net_labels_exact_arithmetic():
    fm = OptionSpreadFillModel(entry_frac=0.775, exit_frac=0.775, commission_per_contract=0.65)
    hz = LabelHorizon(entry_delay_bars=1, hold_bars=5)
    df = build_option_net_labels(_make_option_df(), fm, hz)

    # 手工核对 t=0:入场 bar=1,出场 bar=6
    mid1, mid6 = 2.01, 2.06
    entry_px = (mid1 - 0.05) + 0.775 * 0.10
    exit_px = (mid6 + 0.05) - 0.775 * 0.10
    commission = 2 * 0.65 / (entry_px * 100)
    expected_net = exit_px / entry_px - 1.0 - commission
    expected_gross = mid6 / mid1 - 1.0

    assert abs(df.loc[0, "label_return_fwd_net"] - expected_net) < 1e-10
    assert abs(df.loc[0, "label_return_fwd_gross"] - expected_gross) < 1e-10
    # cost = gross - net > 0(点差 + 佣金)
    assert df.loc[0, "label_execution_cost"] > 0
    # net 不被截断为 0:上涨但涨幅小于成本时应为负,而非旧版的 0
    assert df.loc[0, "label_return_fwd_net"] < df.loc[0, "label_return_fwd_gross"]


def test_option_net_labels_no_zero_inflation():
    """旧版 sign(g)*max(|g|-c,0) 会产生大量零标签;新口径不应截断。"""
    rng = np.random.default_rng(7)
    n = 300
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    mid = 2.0 * np.exp(np.cumsum(rng.normal(0, 0.004, n)))
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "exec_call_bid": mid * 0.99,
            "exec_call_ask": mid * 1.01,
            "exec_call_mid": mid,
        }
    )
    df = build_option_net_labels(df, OptionSpreadFillModel(), LabelHorizon())
    rep = label_quality_report(df)
    assert rep["net_std"] > 0
    assert rep["net_nonzero_pct"] > 95.0
    assert rep["cost_mean_bps"] > 0


def test_perp_net_labels():
    n = 30
    ts = pd.date_range("2026-06-01", periods=n, freq="1min", tz="UTC")
    px = 100000.0 * (1 + 0.0005 * np.arange(n))
    df = pd.DataFrame({"timestamp": ts, "close": px, "funding_rate_8h": 0.0001})
    fm = PerpFillModel(taker_fee_bps=5.0, slippage_bps=2.0)
    df = build_perp_net_labels(df, fm, LabelHorizon(entry_delay_bars=1, hold_bars=5))

    mark1, mark6 = px[1], px[6]
    entry_px = mark1 * (1 + 7e-4)
    exit_px = mark6 * (1 - 7e-4)
    funding = 0.0001 * (5 * 60) / (8 * 3600)
    expected = exit_px / entry_px - 1.0 - funding
    assert abs(df.loc[0, "label_return_fwd_net"] - expected) < 1e-10


# ---------------------------------------------------------------------------
# exit rails
# ---------------------------------------------------------------------------

def test_exit_rails_hard_stop_and_trailing():
    cfg = ExitRailsConfig(hard_stop_roi=-0.12, trailing_trigger_roi=0.25, trailing_keep_ratio=0.6)
    pos = PositionState(entry_price=2.0, entry_bar=0)
    assert check_exit(cfg, pos, 2.0 * 0.87, current_bar=1) == "HARD_STOP"

    pos2 = PositionState(entry_price=2.0, entry_bar=0)
    assert check_exit(cfg, pos2, 2.0 * 1.30, current_bar=1) is None  # max_roi=30%
    # 回撤到 30%*0.6=18% 以下 → trailing
    assert check_exit(cfg, pos2, 2.0 * 1.15, current_bar=2) == "TRAILING"


def test_exit_rails_ladder_ratchet_cumulative():
    """peak 12% 应锁 8% floor(累积 ratchet),不是旧版单档 3%。"""
    from qqq_btc.common.exit_rails import ladder_floor

    cfg = ExitRailsConfig(
        ladder=((0.08, 0.05), (0.12, 0.08), (0.18, 0.12)),
        hard_stop_roi=-0.50,
        soft_stop_roi=-0.50,
        trailing_trigger_roi=9.0,
        flash_trigger_roi=9.0,
        max_hold_bars=999,
    )
    assert abs(ladder_floor(cfg, 0.12) - 0.08) < 1e-9
    pos = PositionState(entry_price=2.0, entry_bar=0)
    assert check_exit(cfg, pos, 2.0 * 1.12, current_bar=1) is None  # peak 12%
    assert check_exit(cfg, pos, 2.0 * 1.07, current_bar=2) == "STEP_PROTECT"  # 7% < floor 8%
    assert check_exit(cfg, pos, 2.0 * 1.09, current_bar=2) is None  # 9% >= floor 8%


def test_exit_rails_eod():
    cfg = ExitRailsConfig(eod_close_bar_index=380)
    pos = PositionState(entry_price=2.0, entry_bar=370)
    assert check_exit(cfg, pos, 2.0, current_bar=381, session_bar_index=381) == "EOD_CLOSE"


def test_replay_max_hold_forces_exit_without_current_quote():
    """MAX_HOLD 是时间硬上限，不能因当前分钟盘口缺失而继续持仓。"""
    from qqq_btc.common.replay_session import OpenPosition, ReplaySession, SessionQuotes

    rails = ExitRailsConfig(
        hard_stop_roi=-0.50,
        soft_stop_roi=-0.40,
        early_stop_bars=None,
        max_hold_bars=5,
        trailing_trigger_roi=9.0,
        flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),),
        eod_close_bar_index=None,
    )
    fill = OptionSpreadFillModel(
        entry_frac=0.775,
        exit_frac=0.775,
        commission_per_contract=0.65,
    )
    session = ReplaySession(ReplayConfig(), rails, fill)
    state = PositionState(entry_price=2.0, entry_bar=10)
    session.position = OpenPosition(
        leg="CALL",
        entry_price=2.0,
        entry_bar=10,
        entry_ts=pd.Timestamp("2026-07-01 10:00", tz="America/New_York"),
        signal_edge=0.10,
        state=state,
        rails=rails,
        last_valid_mtm=1.80,
    )
    no_quote = SessionQuotes(call_bid=float("nan"), call_ask=float("nan"))
    ev = session._try_minute_exit(
        15,
        pd.Timestamp("2026-07-01 10:05", tz="America/New_York"),
        35,
        no_quote,
    )
    assert ev is not None
    assert ev.reason == "MAX_HOLD|NO_QUOTE"
    assert ev.price == 1.80
    assert session.result.trades[0].bars_held == 5


# ---------------------------------------------------------------------------
# strict replay 与标签的一致性(核心不变量)
# ---------------------------------------------------------------------------

def test_replay_fill_consistent_with_labels():
    """
    关闭 rails 干预(极宽阈值)、强制 MAX_HOLD=hold_bars 退出时,
    replay 的单笔净收益必须等于同一 FillModel 下的标签 net(同一入场 bar)。
    """
    fm = OptionSpreadFillModel(entry_frac=0.775, exit_frac=0.775, commission_per_contract=0.65)
    hz = LabelHorizon(entry_delay_bars=1, hold_bars=5)
    df = _make_option_df(n=40, drift=0.02)
    df = build_option_net_labels(df.copy(), fm, hz)

    # 只在 bar0 给信号
    df["net_edge"] = 0.0
    df.loc[0, "net_edge"] = 0.05

    rails = ExitRailsConfig(
        hard_stop_roi=-9.0, soft_stop_roi=-9.0,
        time_stop_bars=9999, time_stop_min_roi=-9.0,
        max_hold_bars=5,                      # 与标签 hold_bars 相同
        trailing_trigger_roi=9.0, flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),),
        eod_close_bar_index=None,
    )
    result = run_strict_replay(
        df,
        fm,
        ReplayConfig(entry_threshold=0.015, entry_delay_bars=1, max_spread_pct=1.0, cooldown_bars=999),
        rails,
    )
    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.exit_reason == "MAX_HOLD"
    assert abs(trade.net_return - df.loc[0, "label_return_fwd_net"]) < 1e-10


def test_replay_spread_gate_blocks_wide_book():
    fm = OptionSpreadFillModel()
    df = _make_option_df(n=40, base=2.0, spread=0.30)  # spread_pct = 15% > 6%
    df["net_edge"] = 0.05
    result = run_strict_replay(df, fm, ReplayConfig(max_spread_pct=0.06), ExitRailsConfig())
    assert len(result.trades) == 0


def test_replay_summary_metrics():
    fm = OptionSpreadFillModel()
    df = _make_option_df(n=120, drift=0.02)
    df["net_edge"] = 0.0
    df.loc[0, "net_edge"] = 0.05
    df.loc[50, "net_edge"] = 0.05
    result = run_strict_replay(df, fm, ReplayConfig(cooldown_bars=5), ExitRailsConfig())
    s = result.summary()
    assert s["trades"] >= 1
    assert "max_drawdown_mtm" in s and "profit_factor" in s and "exit_reasons" in s


# ---------------------------------------------------------------------------
# 配置与锚点复用
# ---------------------------------------------------------------------------

def test_qqq_config_single_fill_source():
    from qqq_btc.qqq import config as qcfg
    assert qcfg.FILL_MODEL.entry_frac == 0.775
    assert qcfg.FILL_MODEL.exit_frac == 0.775
    assert qcfg.REPLAY.entry_delay_bars == qcfg.LABEL_HORIZON.entry_delay_bars


def test_btc_config_loads():
    from qqq_btc.btc import config as bcfg
    assert bcfg.FILL_MODEL.round_trip_fee_drag() == 14e-4
    assert bcfg.EXIT_RAILS.eod_close_bar_index is None


def test_anchor_internalized():
    """锚点已内化:配置与代码都在 qqq_btc 内,不再依赖 New_Pro。"""
    from qqq_btc.qqq import anchor

    assert "qqq_btc" in anchor.__file__ and "New_Pro" not in anchor.__file__
    cfg = anchor.load_qqq_anchor_config()
    assert "qqq_btc" in cfg["_config_path"]
    assert cfg["profile"] == "qqq_0dte"
    assert cfg.get("front_prefer_dte") == 0
    assert len(anchor.bucket_targets(cfg)) == 4  # 0DTE 4-bucket,无 NEXT
    # 0DTE profile 只订阅 front 4 档
    specs = anchor.active_bucket_specs(cfg)
    assert set(specs) == {"PUT_ATM", "PUT_OTM", "CALL_ATM", "CALL_OTM"}


def test_anchor_front_dte_selection():
    from qqq_btc.qqq import anchor

    cfg = anchor.load_qqq_anchor_config()
    assert anchor.select_front_dte([0, 1, 7], cfg) == 0     # 有 0DTE 选 0DTE
    assert anchor.select_front_dte([1, 2, 7], cfg) == 1     # 无 0DTE fallback 1DTE
    assert anchor.select_front_expiration([("20260703", 1), ("20260702", 0)], cfg) == ("20260702", 0)


# ---------------------------------------------------------------------------
# v2 优化:时间特征 / 分时段阈值 / q10 门控 / rails 标定 / 特征配置
# ---------------------------------------------------------------------------

def test_time_features_bounds_and_expiry():
    from qqq_btc.common.time_features import add_time_features, SESSION_MINUTES

    ts = pd.to_datetime(
        ["2026-06-01 09:30", "2026-06-01 12:45", "2026-06-01 16:00"]
    ).tz_localize("America/New_York")
    df = pd.DataFrame({"timestamp": ts})
    df = add_time_features(df)

    assert abs(df.loc[0, "time_session_progress"] - 0.0) < 1e-12
    assert abs(df.loc[2, "time_session_progress"] - 1.0) < 1e-12
    assert abs(df.loc[2, "time_to_expiry_norm"] - 0.0) < 1e-12   # 16:00 = 0DTE 到期
    assert abs(df.loc[1, "time_session_progress"] - (195 / SESSION_MINUTES)) < 1e-12
    assert (df["time_session_sin"].abs() <= 1).all()
    assert (df["time_session_cos"].abs() <= 1).all()


def test_replay_threshold_schedule():
    cfg = ReplayConfig(
        entry_threshold=0.015,
        entry_threshold_schedule=((0, 0.015), (270, 0.020), (330, 0.025)),
    )
    assert cfg.threshold_at(10) == 0.015
    assert cfg.threshold_at(269) == 0.015
    assert cfg.threshold_at(270) == 0.020
    assert cfg.threshold_at(345) == 0.025
    assert cfg.threshold_at(None) == 0.015  # 无 session_bar 回退基础阈值


def test_replay_afternoon_threshold_blocks_weak_edge():
    """同样 edge=0.017:上午可入场,下午(阈值 0.020)被拒。"""
    fm = OptionSpreadFillModel()
    cfg = ReplayConfig(
        entry_threshold=0.015,
        entry_threshold_schedule=((0, 0.015), (270, 0.020)),
        cooldown_bars=999,
    )
    df = _make_option_df(n=40, drift=0.02)
    df["net_edge"] = 0.0
    df.loc[0, "net_edge"] = 0.017

    df_am = df.copy()
    df_am["session_bar"] = np.arange(40)  # 上午
    assert len(run_strict_replay(df_am, fm, cfg, ExitRailsConfig()).trades) == 1

    df_pm = df.copy()
    df_pm["session_bar"] = np.arange(280, 320)  # 下午
    assert len(run_strict_replay(df_pm, fm, cfg, ExitRailsConfig()).trades) == 0


def test_replay_q10_gate():
    """q10 不高于 edge_q10_floor 时即使均值 edge 过阈值也不入场;轻度为负放行。"""
    fm = OptionSpreadFillModel()
    df = _make_option_df(n=40, drift=0.02)
    df["net_edge"] = 0.0
    df.loc[0, "net_edge"] = 0.05
    # ReplayConfig 默认 floor=-0.20: -0.25 拦截, -0.01 放行
    df["net_edge_q10"] = -0.25
    result = run_strict_replay(
        df, fm, ReplayConfig(), ExitRailsConfig(), edge_q10_col="net_edge_q10"
    )
    assert len(result.trades) == 0

    df["net_edge_q10"] = -0.01
    result2 = run_strict_replay(
        df, fm, ReplayConfig(), ExitRailsConfig(), edge_q10_col="net_edge_q10"
    )
    assert len(result2.trades) == 1


def test_calibrate_rails_on_synthetic_paths():
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
    from qqq_btc.tools.calibrate_rails import (
        compute_roi_paths,
        entry_bar_mask,
        merge_bucket_suggestions,
        suggest_rails,
    )

    rng = np.random.default_rng(11)
    n = 2000
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    mid = 2.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.006, n)))
    edge = rng.uniform(-0.01, 0.04, n)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "exec_call_bid": mid * 0.99,
            "exec_call_ask": mid * 1.01,
            "exec_call_mid": mid,
            "net_edge": edge,
        }
    )
    mask = entry_bar_mask(df, entry_threshold=0.015)
    paths = compute_roi_paths(
        df, OptionSpreadFillModel(), max_hold_bars=30,
        horizon_bars=(5, 15), entry_mask=mask,
    )
    assert len(paths) > 100
    assert "roi_h5" in paths.columns
    assert (paths["mae"] <= paths["final_roi"] + 1e-12).all()
    assert (paths["mfe"] >= paths["final_roi"] - 1e-12).all()

    report = suggest_rails(paths, tod_buckets={"all": (0, 391)})
    stats = report["all"]
    assert stats["samples"] > 100
    assert stats["suggested_hard_stop_roi"] <= stats["suggested_soft_stop_roi"] < 0
    assert "suggested_early_stop_roi" in stats
    assert stats["suggested_early_stop_bars"] == 5
    assert "_merged_conservative" in report
    merged = merge_bucket_suggestions(report)
    assert merged["soft_stop_roi"] <= stats["suggested_soft_stop_roi"] + 1e-9


def test_exit_rails_early_stop():
    cfg = ExitRailsConfig(
        early_stop_bars=5,
        early_stop_roi=-0.05,
        hard_stop_roi=-0.50,
        soft_stop_roi=-0.50,
        max_hold_bars=999,
        trailing_trigger_roi=9.0,
        flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),),
    )
    pos = PositionState(entry_price=2.0, entry_bar=0)
    assert check_exit(cfg, pos, 2.0 * 0.97, current_bar=4) is None   # 4 bar, -3%
    assert check_exit(cfg, pos, 2.0 * 0.94, current_bar=5) == "EARLY_STOP"  # 5 bar, -6%
    assert check_exit(cfg, pos, 2.0 * 0.97, current_bar=5) is None   # 5 bar, -3% ok

    cfg_off = ExitRailsConfig(
        early_stop_bars=None,
        early_stop_roi=-0.05,
        hard_stop_roi=-0.50,
        soft_stop_roi=-0.50,
        max_hold_bars=999,
        trailing_trigger_roi=9.0,
        flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),),
    )
    pos2 = PositionState(entry_price=2.0, entry_bar=0)
    assert check_exit(cfg_off, pos2, 2.0 * 0.94, current_bar=10) is None


def test_feature_config_v2_generated():
    import json

    cfg_path = _REPO_ROOT / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json"
    assert cfg_path.exists()
    with open(cfg_path) as f:
        cfg = json.load(f)
    names = {f["name"] for f in cfg["features"]}
    for feat in ("time_session_sin", "time_session_cos", "time_session_progress", "time_to_expiry_norm"):
        assert feat in names
    assert cfg["loss_weights"]["rank_net"] == 1.0
    assert cfg["loss_weights"]["net_edge_quantile"] == 1.0
    assert cfg["parameters"]["qqq_btc_v2"]["fill_model"]["entry_frac"] == 0.775


def test_model_modules_have_no_heavy_import_side_effects():
    """backbone/losses 只依赖 torch;dataset 的 lmdb/msgpack 延迟到实例化。"""
    import ast

    for rel in ("model/backbone.py", "model/losses.py", "model/dataset.py"):
        src = (_REPO_ROOT / "qqq_btc" / rel).read_text()
        tree = ast.parse(src)
        top_imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and node.col_offset == 0:
                top_imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.col_offset == 0 and node.module:
                top_imports.add(node.module.split(".")[0])
        forbidden = {"lmdb", "msgpack", "msgpack_numpy", "zstandard", "psycopg2", "argparse", "tqdm"}
        assert not (top_imports & forbidden), f"{rel} 顶层引入了重依赖: {top_imports & forbidden}"
        # 模块级不得有日志文件副作用
        assert "basicConfig" not in src or rel == "model/train.py"


def test_model_v2_quantile_heads():
    """torch 仅在训练机可用;本机缺依赖时跳过,不阻塞其余测试。"""
    import pytest

    torch = pytest.importorskip("torch")

    import json

    from qqq_btc.model.tft_qqq_v2 import (
        QQQAlphaNetV2,
        QQQNetEdgeLossV2,
        freeze_for_qqq_finetune,
    )

    with open(_REPO_ROOT / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json") as f:
        cfg = json.load(f)

    caps = {"stock": 2, "sector": 2}
    model = QQQAlphaNetV2(cfg, caps, hidden_dim=32)
    assert not hasattr(model, "head_rank")  # rank 流已结构性删除

    n_stock = len(model.feat_info["stock"]["real"]) + len(model.feat_info["stock"]["cat"])
    n_opt = len(model.feat_info["option"]["real"]) + len(model.feat_info["option"]["cat"])
    B, T = 4, 30
    out = model(
        torch.randn(B, T, n_stock),
        torch.randn(B, T, n_opt),
        {
            "stock_id": torch.zeros(B, dtype=torch.long),
            "sector_id": torch.zeros(B, dtype=torch.long),
            "day_of_week": torch.ones(B, dtype=torch.long),
        },
    )
    assert "rank_score" not in out
    # 分位数单调:q10 <= q50 <= q90
    assert (out["net_edge_q10"] <= out["net_edge_q50"]).all()
    assert (out["net_edge_q50"] <= out["net_edge_q90"]).all()

    loss_fn = QQQNetEdgeLossV2(cfg)
    target = {
        "direction": torch.randint(0, 3, (B,)),
        "return_fwd": torch.randn(B) * 0.02,
        "return_fwd_gross": torch.randn(B) * 0.02,
        "execution_cost": torch.rand(B) * 0.01,
    }
    loss, l_q = loss_fn(out, target)
    assert torch.isfinite(loss)
    assert l_q >= 0

    trainable, total = freeze_for_qqq_finetune(model)
    assert 0 < trainable < total
    for name, p in model.named_parameters():
        if name.startswith("tft_stock") or name.startswith("static_stock_embed"):
            assert not p.requires_grad
        if name.startswith("head_net_edge_quantile") or name.startswith("symbol_calibrator"):
            assert p.requires_grad


def test_trend_features_regular_trend():
    """单调趋势段:R2 接近 1、拟合收益接近真实段收益;并验证有界性。"""
    from qqq_btc.common.trend_features import add_trend_features

    n = 200
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    px = 500.0 * np.exp(0.0002 * np.arange(n))  # 完美指数趋势
    df = pd.DataFrame({"timestamp": ts, "close": px})
    df = add_trend_features(df)

    tail = df.iloc[150:]
    assert (tail["trend_fit_r2_30m"] > 0.99).all()
    assert (tail["trend_fit_r2_120m"] > 0.99).all()
    # 30 bar 拟合收益 ~= 29 * 0.0002
    assert np.allclose(tail["trend_fit_ret_30m"], 29 * 0.0002, atol=1e-6)
    # 单调上涨:始终贴着当日高点
    assert (tail["day_range_pos"] > 0.99).all()
    assert (tail["drawdown_from_day_high"] > -1e-9).all()
    assert (df["trend_fit_r2_30m"].between(0, 1)).all()
    assert (df["day_range_pos"].between(0, 1)).all()


def test_trend_features_causality():
    """因果性:修改未来数据不得改变当前 bar 的特征值。"""
    from qqq_btc.common.trend_features import add_trend_features

    rng = np.random.default_rng(3)
    n = 200
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    px = 500.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))

    df_a = add_trend_features(pd.DataFrame({"timestamp": ts, "close": px.copy()}))
    px_b = px.copy()
    px_b[150:] *= 1.05  # 篡改未来
    df_b = add_trend_features(pd.DataFrame({"timestamp": ts, "close": px_b}))

    from qqq_btc.common.trend_features import TREND_FEATURE_NAMES

    for col in TREND_FEATURE_NAMES:
        assert np.allclose(df_a[col].iloc[:150], df_b[col].iloc[:150], atol=1e-12), col


def test_trend_features_choppy_low_r2():
    """无趋势锯齿行情:R2 应显著低于趋势段。"""
    from qqq_btc.common.trend_features import add_trend_features

    n = 200
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    px = 500.0 + 0.5 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)  # 纯锯齿
    df = add_trend_features(pd.DataFrame({"timestamp": ts, "close": px}))
    assert (df["trend_fit_r2_30m"].iloc[50:] < 0.05).all()


def test_open30_features_inverted_v_snapshot():
    """开盘冲高后回落:bar29 快照应体现倒 V(open30_reversal>0, peak_dd<0)。"""
    from qqq_btc.common.trend_features import add_open30_features

    n = 390
    ts = pd.date_range("2026-06-10 09:30", periods=n, freq="1min", tz="America/New_York")
    px = np.full(n, 500.0)
    px[:15] = np.linspace(500.0, 504.0, 15)   # 前 15min 拉升
    px[15:30] = np.linspace(504.0, 501.0, 15)  # 后 15min 回落
    px[30:] = 501.0
    df = add_open30_features(pd.DataFrame({"timestamp": ts, "close": px}))
    snap = df.iloc[29]
    assert snap["open30_max_ret"] > 0.005
    assert snap["open30_peak_dd"] < -0.003
    assert snap["open30_reversal"] > 0.003
    # 10:00 后冻结
    assert np.isclose(df.iloc[100]["open30_max_ret"], snap["open30_max_ret"])
    assert np.isclose(df.iloc[100]["open30_reversal"], snap["open30_reversal"])


def test_open30_features_non_contiguous_index():
    """concat 后非连续 index 也应正确写入。"""
    from qqq_btc.common.trend_features import add_open30_features

    ts = pd.date_range("2026-06-10 09:30", periods=60, freq="1min", tz="America/New_York")
    df = pd.DataFrame({"timestamp": ts, "close": np.linspace(500, 502, 60)})
    df.index = list(range(10, 40)) + list(range(100, 130))
    out = add_open30_features(df.copy())
    assert out.loc[39, "open30_max_ret"] > 0
    assert out.loc[129, "open30_max_ret"] == out.loc[39, "open30_max_ret"]


def test_feature_config_v2_contains_trend_features():
    import json

    with open(_REPO_ROOT / "qqq_btc" / "CONFIG" / "slow_feature_qqq_v2.json") as f:
        cfg = json.load(f)
    names = {f["name"] for f in cfg["features"]}
    from qqq_btc.common.trend_features import TREND_FEATURE_NAMES, OPEN30_FEATURE_NAMES

    for feat in TREND_FEATURE_NAMES:
        assert feat in names
    for feat in OPEN30_FEATURE_NAMES:
        assert feat in names
    assert set(cfg["parameters"]["qqq_btc_v2"]["trend_features"]) == set(TREND_FEATURE_NAMES)
    assert set(cfg["parameters"]["qqq_btc_v2"]["open30_features"]) == set(OPEN30_FEATURE_NAMES)


def test_qqq_config_v2_wiring():
    from qqq_btc.qqq import config as qcfg

    assert qcfg.REPLAY.entry_threshold_schedule is not None
    assert qcfg.REPLAY.threshold_at(280) > qcfg.REPLAY.threshold_at(10)
    assert qcfg.EDGE_Q10_COL == "net_edge_q10"
    assert qcfg.FEATURE_CONFIG_PATH.exists()


# ---------------------------------------------------------------------------
# 双腿(CALL/PUT)标签与方向决策
# ---------------------------------------------------------------------------

def _make_dual_leg_df(n=40, call_drift=-0.02, put_drift=0.02, spread=0.10):
    """下跌行情:CALL 腿 mid 走低、PUT 腿 mid 走高,各有独立盘口。"""
    ts = pd.date_range("2026-06-01 09:30", periods=n, freq="1min", tz="America/New_York")
    call_mid = 3.0 + call_drift * np.arange(n)
    put_mid = 3.0 + put_drift * np.arange(n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "exec_call_bid": call_mid - spread / 2,
            "exec_call_ask": call_mid + spread / 2,
            "exec_call_mid": call_mid,
            "exec_put_bid": put_mid - spread / 2,
            "exec_put_ask": put_mid + spread / 2,
            "exec_put_mid": put_mid,
        }
    )


def test_dual_leg_labels_independent_arithmetic():
    """PUT 腿标签 = 用 PUT 自己盘口算的净收益,不是负的 CALL 收益。"""
    from qqq_btc.common.labels import build_dual_leg_net_labels

    fm = OptionSpreadFillModel(entry_frac=0.775, exit_frac=0.775, commission_per_contract=0.65)
    hz = LabelHorizon(entry_delay_bars=1, hold_bars=5)
    df = build_dual_leg_net_labels(_make_dual_leg_df(), fm, hz)

    # PUT 腿与单腿构建器在 PUT 盘口上的结果一致(同一内核)
    df_put_only = _make_dual_leg_df().rename(
        columns={
            "exec_put_bid": "exec_call_bid_x", "exec_call_bid": "drop1",
            "exec_put_ask": "exec_call_ask_x", "exec_call_ask": "drop2",
            "exec_put_mid": "exec_call_mid_x", "exec_call_mid": "drop3",
        }
    )
    ref = build_option_net_labels(
        df_put_only, fm, hz,
        bid_col="exec_call_bid_x", ask_col="exec_call_ask_x", mid_col="exec_call_mid_x",
    )
    assert np.allclose(df["label_put_return_fwd_net"], ref["label_return_fwd_net"], atol=1e-12)

    # 下跌行情:CALL 腿净收益为负,PUT 腿为正,方向 = 0(PUT 有利)
    assert df.loc[0, "label_call_return_fwd_net"] < 0
    assert df.loc[0, "label_put_return_fwd_net"] > 0
    assert df.loc[0, "label_direction_net"] == 0
    # 主标签(向后兼容)仍是 CALL 腿口径
    assert df.loc[0, "label_return_fwd_net"] == df.loc[0, "label_call_return_fwd_net"]


def test_replay_dual_leg_picks_put():
    """双腿模式:PUT 头更强且过阈值 → 买 PUT,用 PUT 盘口成交并盈利。"""
    fm = OptionSpreadFillModel()
    df = _make_dual_leg_df(n=60)
    df["call_net_edge"] = 0.001
    df["put_net_edge"] = 0.0
    df.loc[0, "put_net_edge"] = 0.05

    result = run_strict_replay(
        df, fm,
        ReplayConfig(entry_threshold=0.015, long_only=False, cooldown_bars=999),
        ExitRailsConfig(),
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
    )
    assert len(result.trades) == 1
    assert result.trades[0].leg == "PUT"
    assert result.trades[0].net_return > 0

    # long_only=True 时 PUT 腿被禁用,同样信号不产生交易
    result_lo = run_strict_replay(
        df, fm,
        ReplayConfig(entry_threshold=0.015, long_only=True, cooldown_bars=999),
        ExitRailsConfig(),
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
    )
    assert len(result_lo.trades) == 0


def test_replay_single_edge_negative_buys_put():
    """过渡模式:有符号 net_edge <= -阈值且有 PUT 盘口 → 买 PUT。"""
    fm = OptionSpreadFillModel()
    df = _make_dual_leg_df(n=60)
    df["net_edge"] = 0.0
    df.loc[0, "net_edge"] = -0.05

    result = run_strict_replay(
        df, fm, ReplayConfig(long_only=False, cooldown_bars=999), ExitRailsConfig()
    )
    assert len(result.trades) == 1
    assert result.trades[0].leg == "PUT"

    # 无 PUT 盘口(列缺失)时负 edge 不可执行,跳过
    df_no_put = df.drop(columns=["exec_put_bid", "exec_put_ask", "exec_put_mid"])
    result_np = run_strict_replay(
        df_no_put, fm, ReplayConfig(long_only=False, cooldown_bars=999), ExitRailsConfig()
    )
    assert len(result_np.trades) == 0


# ---------------------------------------------------------------------------
# 频率治理:日内笔数上限 / 日亏损熔断 / 连亏冷却
# ---------------------------------------------------------------------------

_FAST_EXIT_RAILS = ExitRailsConfig(
    hard_stop_roi=-9.0, soft_stop_roi=-9.0,
    time_stop_bars=9999, time_stop_min_roi=-9.0,
    max_hold_bars=3,
    trailing_trigger_roi=9.0, flash_trigger_roi=9.0,
    ladder=((9.0, -9.0),),
    eod_close_bar_index=None,
)


def test_replay_max_trades_per_day():
    fm = OptionSpreadFillModel()
    df = _make_option_df(n=200, drift=0.02)
    df["net_edge"] = 0.05  # 每 bar 都有信号

    unlimited = run_strict_replay(
        df, fm, ReplayConfig(cooldown_bars=0), _FAST_EXIT_RAILS
    )
    assert len(unlimited.trades) > 2

    capped = run_strict_replay(
        df, fm, ReplayConfig(cooldown_bars=0, max_trades_per_day=2), _FAST_EXIT_RAILS
    )
    assert len(capped.trades) == 2


def test_replay_daily_loss_stop():
    """持续亏损行情:当日累计亏损触发熔断后不再开新仓。"""
    fm = OptionSpreadFillModel()
    df = _make_option_df(n=200, base=10.0, drift=-0.05)  # 单调下跌 → 每笔都亏
    df["net_edge"] = 0.05

    result = run_strict_replay(
        df, fm, ReplayConfig(cooldown_bars=0, daily_loss_stop=-0.005), _FAST_EXIT_RAILS
    )
    assert len(result.trades) == 1
    assert result.trades[0].net_return < 0


def test_replay_loss_streak_cooldown():
    fm = OptionSpreadFillModel()
    df = _make_option_df(n=200, base=10.0, drift=-0.05)
    df["net_edge"] = 0.05

    result = run_strict_replay(
        df, fm,
        ReplayConfig(cooldown_bars=0, loss_streak_n=2, loss_streak_cooldown_bars=999),
        _FAST_EXIT_RAILS,
    )
    assert len(result.trades) == 2  # 连亏 2 笔后进入冷却,不再交易


def test_replay_governance_resets_per_day():
    """日界切换:熔断/笔数限制按自然日重置。"""
    fm = OptionSpreadFillModel()
    day1 = _make_option_df(n=100, drift=0.02)
    day2 = _make_option_df(n=100, drift=0.02)
    day2["timestamp"] = day2["timestamp"] + pd.Timedelta(days=1)
    df = pd.concat([day1, day2], ignore_index=True)
    df["net_edge"] = 0.05

    result = run_strict_replay(
        df, fm, ReplayConfig(cooldown_bars=0, max_trades_per_day=1), _FAST_EXIT_RAILS
    )
    assert len(result.trades) == 2  # 每天 1 笔
    days = {pd.Timestamp(t.entry_ts).date() for t in result.trades}
    assert len(days) == 2


def test_replay_next_day_put_quarantine_then_unlocks():
    """前日 PUT sleeve 触阈后仅锁下一交易日，第三日自动恢复。"""
    from dataclasses import replace

    fm = OptionSpreadFillModel()
    day1 = _make_dual_leg_df(n=20, put_drift=-0.10)
    day2 = _make_dual_leg_df(n=20, put_drift=0.05)
    day3 = _make_dual_leg_df(n=20, put_drift=0.05)
    day2["timestamp"] = day2["timestamp"] + pd.Timedelta(days=1)
    day3["timestamp"] = day3["timestamp"] + pd.Timedelta(days=2)
    df = pd.concat([day1, day2, day3], ignore_index=True)
    df["call_net_edge"] = 0.0
    df["put_net_edge"] = 0.0
    df.loc[[0, 20, 40], "put_net_edge"] = 0.05

    base = ReplayConfig(
        entry_threshold=0.015,
        long_only=False,
        cooldown_bars=0,
        position_frac=0.25,
    )
    unlocked = run_strict_replay(
        df,
        fm,
        base,
        _FAST_EXIT_RAILS,
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
    )
    assert len(unlocked.trades) == 3
    assert unlocked.trades[0].net_return < 0

    quarantined = run_strict_replay(
        df,
        fm,
        replace(base, next_day_put_quarantine_loss=-0.02),
        _FAST_EXIT_RAILS,
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
    )
    assert len(quarantined.trades) == 2
    trade_days = [pd.Timestamp(t.entry_ts).date().isoformat() for t in quarantined.trades]
    assert trade_days == ["2026-06-01", "2026-06-03"]

    # 当前日不满足低波 regime 条件时，不得仅凭前日亏损误锁。
    df["regime_vix_z"] = 0.0
    regime_filtered = run_strict_replay(
        df,
        fm,
        replace(
            base,
            next_day_put_quarantine_loss=-0.02,
            next_day_put_quarantine_vix_z_max=-0.75,
        ),
        _FAST_EXIT_RAILS,
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
    )
    assert len(regime_filtered.trades) == 3

    # VX 曲线不够陡时同样不锁；期限结构输入必须来自当日前已完成数据。
    df["vx_curve_slope"] = 0.01
    vx_filtered = run_strict_replay(
        df,
        fm,
        replace(
            base,
            next_day_put_quarantine_loss=-0.02,
            next_day_put_quarantine_vx_slope_min=0.05,
        ),
        _FAST_EXIT_RAILS,
        call_edge_col="call_net_edge",
        put_edge_col="put_net_edge",
    )
    assert len(vx_filtered.trades) == 3


def test_vx_rule_profile_selector():
    from qqq_btc.common.rule_profiles import select_profile_name_vx

    assert select_profile_name_vx(-0.01, qqq_up_frac=0.8, qqq_range_mean=0.02) == (
        "OPEN_DEFENSE"
    )
    assert select_profile_name_vx(0.06, qqq_up_frac=0.4, qqq_range_mean=0.016) == (
        "CHOP_NO_TRADE"
    )
    assert select_profile_name_vx(0.06, qqq_up_frac=0.8, qqq_range_mean=0.016) == (
        "TREND_PUT_OK"
    )
    assert select_profile_name_vx(float("nan")) == "TREND_PUT_OK"


def test_loss_call_put_term():
    """call/put 双腿损失:有符号标签 + rank 项参与总损失。"""
    import pytest

    torch = pytest.importorskip("torch")
    from qqq_btc.model.losses import NetEdgeLoss

    B = 8
    out = {
        "logits_dir": torch.randn(B, 3),
        "net_edge": torch.randn(B, 1) * 0.02,
        "net_edge_raw": torch.randn(B, 1) * 0.02,
        "gross_return": torch.randn(B, 1) * 0.02,
        "execution_cost": torch.rand(B, 1) * 0.01,
        "net_edge_q10": torch.randn(B, 1) * 0.02,
        "net_edge_q50": torch.randn(B, 1) * 0.02,
        "net_edge_q90": torch.randn(B, 1) * 0.02,
        "call_net_edge": torch.randn(B, 1) * 0.03,
        "put_net_edge": torch.randn(B, 1) * 0.03,
    }
    target = {
        "direction": torch.randint(0, 3, (B,)),
        "return_fwd": torch.randn(B) * 0.02,
        "return_fwd_gross": torch.randn(B) * 0.02,
        "execution_cost": torch.rand(B) * 0.01,
        "call_return_fwd": torch.randn(B) * 0.02,
        "put_return_fwd": torch.randn(B) * 0.02,
    }
    cfg_on = {"loss_weights": {"call_put_edge": 0.5, "rank_net": 1.0}}
    cfg_off = {"loss_weights": {"call_put_edge": 0.0, "rank_net": 0.0}}
    loss_on, _ = NetEdgeLoss(cfg_on)(out, target)
    loss_off, _ = NetEdgeLoss(cfg_off)(out, target)
    assert torch.isfinite(loss_on) and torch.isfinite(loss_off)
    assert loss_on.item() >= loss_off.item()

    # 负 label 不再被截断为 0:高预测 + 负 label 的回归损失应大于贴近 label 的预测
    cfg_cp_only = {
        "loss_weights": {
            "direction_net": 0.0,
            "return_fwd_net": 0.0,
            "return_fwd_gross": 0.0,
            "execution_cost": 0.0,
            "net_edge_quantile": 0.0,
            "rank_net": 0.0,
            "call_put_edge": 1.0,
        }
    }
    out_neg = dict(out)
    target_neg = dict(target)
    target_neg["call_return_fwd"] = torch.full((B,), -0.10)
    target_neg["put_return_fwd"] = torch.zeros(B)
    out_neg["call_net_edge"] = torch.full((B, 1), 0.05)
    loss_high, _ = NetEdgeLoss(cfg_cp_only)(out_neg, target_neg)
    out_neg["call_net_edge"] = torch.full((B, 1), -0.08)
    loss_low, _ = NetEdgeLoss(cfg_cp_only)(out_neg, target_neg)
    assert loss_high.item() > loss_low.item()

    # 缺双腿标签时自动跳过,不报错
    target_no_legs = {k: v for k, v in target.items() if k not in ("call_return_fwd", "put_return_fwd")}
    loss_skip, _ = NetEdgeLoss(cfg_on)(out, target_no_legs)
    assert torch.isfinite(loss_skip)


# ---------------------------------------------------------------------------
# MTM 节奏契约:分钟收盘 rails + tick 级灾难止损
# ---------------------------------------------------------------------------

def test_disaster_stop_tick_level():
    from qqq_btc.common.exit_rails import check_disaster_stop, check_tick_stops

    cfg = ExitRailsConfig(hard_stop_roi=-0.12, disaster_stop_roi=-0.25)
    pos = PositionState(entry_price=2.0, entry_bar=0)

    # 分钟内 -15% 影线:超过 hard_stop 水位,但灾难止损不响 —— 交给分钟收盘裁决
    assert check_disaster_stop(cfg, pos, 2.0 * 0.85) is None
    # -30% 闪崩:tick 级立即离场
    assert check_disaster_stop(cfg, pos, 2.0 * 0.70) == "DISASTER_STOP"
    # 未启用时永远 None
    cfg_off = ExitRailsConfig(disaster_stop_roi=None)
    assert check_disaster_stop(cfg_off, pos, 2.0 * 0.50) is None


def test_tick_fast_hard_between_hard_and_disaster():
    from qqq_btc.common.exit_rails import check_tick_stops

    cfg = ExitRailsConfig(
        hard_stop_roi=-0.12,
        tick_fast_hard_roi=-0.15,
        disaster_stop_roi=-0.25,
    )
    pos = PositionState(entry_price=2.0, entry_bar=0)
    # -13%: 深于分钟 hard 水位,但未到 fast_hard
    assert check_tick_stops(cfg, pos, 2.0 * 0.87) is None
    # -16%: tick_fast_hard
    assert check_tick_stops(cfg, pos, 2.0 * 0.84) == "TICK_FAST_HARD"
    # -30%: disaster 优先于 fast_hard
    assert check_tick_stops(cfg, pos, 2.0 * 0.70) == "DISASTER_STOP"
    assert pos.max_roi == 0.0  # 不污染分钟棘轮


def test_tick_fast_hard_applies_reentry_lockout_and_freezes_quantile():
    from qqq_btc.common.replay_session import (
        OpenPosition,
        ReplaySession,
        SessionQuotes,
        SessionSignal,
    )

    replay_cfg = ReplayConfig(
        cooldown_bars=10,
        tick_stop_cooldown_bars=30,
        tick_stop_lock_leg_for_day=True,
        entry_quantile=0.80,
        entry_quantile_min_obs=1,
        max_spread_pct=1.0,
    )
    rails = ExitRailsConfig(
        tick_fast_hard_roi=-0.18,
        disaster_stop_roi=-0.35,
        tick_profit_trigger_roi=None,
        tick_profit_ladder=(),
    )
    session = ReplaySession(replay_cfg, rails, OptionSpreadFillModel())
    session.position = OpenPosition(
        leg="CALL",
        entry_price=2.0,
        entry_bar=0,
        entry_ts=pd.Timestamp("2026-07-01 09:30", tz="America/New_York"),
        signal_edge=0.05,
        state=PositionState(entry_price=2.0, entry_bar=0),
        rails=rails,
        last_valid_mtm=2.0,
    )
    events = session.on_tick(
        5,
        pd.Timestamp("2026-07-01 09:35:05", tz="America/New_York"),
        SessionQuotes(call_bid=1.60, call_ask=1.64),
        smoothed_mtm=1.62,
    )
    assert events and events[0].reason == "TICK_FAST_HARD"
    assert session.tick_stop_cooldown_until == 35
    assert session.tick_stopped_legs == {"CALL"}

    session.on_minute_bar(
        6,
        pd.Timestamp("2026-07-01 09:36", tz="America/New_York"),
        6,
        SessionQuotes(call_bid=2.0, call_ask=2.02),
        SessionSignal(edge=0.10),
    )
    assert list(session._edge_buf or []) == []
    assert session.position is None


def test_tick_profit_trail_independent_of_minute_max_roi():
    """冲高回落:tick_peak 独立,不写入 max_roi。"""
    from qqq_btc.common.exit_rails import check_tick_stops

    cfg = ExitRailsConfig(
        tick_profit_trigger_roi=0.20,
        tick_profit_keep_ratio=0.50,
        disaster_stop_roi=-0.25,
    )
    pos = PositionState(entry_price=2.0, entry_bar=0)
    # 冲到 +100%
    assert check_tick_stops(cfg, pos, 2.0 * 2.0) is None
    assert pos.tick_peak_roi == 1.0
    assert pos.max_roi == 0.0
    # 回落到 +40%(< peak*0.5=50%) → 锁利
    assert check_tick_stops(cfg, pos, 2.0 * 1.40) == "TICK_PROFIT_TRAIL"
    assert pos.max_roi == 0.0


def test_tick_profit_step_ladder():
    from qqq_btc.common.exit_rails import check_tick_stops

    cfg = ExitRailsConfig(
        tick_profit_ladder=((0.30, 0.18), (0.50, 0.30)),
        disaster_stop_roi=None,
    )
    pos = PositionState(entry_price=2.0, entry_bar=0)
    assert check_tick_stops(cfg, pos, 2.0 * 1.40) is None  # peak 40%, floor 18%
    assert check_tick_stops(cfg, pos, 2.0 * 1.20) is None  # 20% > 18%
    assert check_tick_stops(cfg, pos, 2.0 * 1.10) == "TICK_PROFIT_STEP"  # 10% < 18%


def test_disaster_stop_stateless_no_max_roi_pollution():
    """tick 影线不得棘轮 max_roi,否则 trailing/flash 相对影线提前引爆。"""
    from qqq_btc.common.exit_rails import check_disaster_stop

    cfg = ExitRailsConfig(
        trailing_trigger_roi=0.25, trailing_keep_ratio=0.6, disaster_stop_roi=-0.25
    )
    pos = PositionState(entry_price=2.0, entry_bar=0)

    # 分钟内影线冲到 +40% 又回落:tick 路径不更新 max_roi
    check_disaster_stop(cfg, pos, 2.0 * 1.40)
    assert pos.max_roi == 0.0

    # 分钟收盘 +10%:trailing 未触发(若 max_roi 被影线污染到 0.40,
    # 此处 10% < 40%*0.6=24% 会被误杀)
    assert check_exit(cfg, pos, 2.0 * 1.10, current_bar=1) is None
    assert abs(pos.max_roi - 0.10) < 1e-12


# ---------------------------------------------------------------------------
# 跨式(straddle)标签与回放
# ---------------------------------------------------------------------------

def test_straddle_label_premium_weighted():
    """跨式标签 = 权利金加权的两腿净收益;含双份佣金后与"分腿各算"精确一致。"""
    from qqq_btc.common.labels import build_dual_leg_net_labels

    fm = OptionSpreadFillModel(entry_frac=0.775, exit_frac=0.775, commission_per_contract=0.65)
    hz = LabelHorizon(entry_delay_bars=1, hold_bars=5)
    df = build_dual_leg_net_labels(_make_dual_leg_df(), fm, hz)

    # 手工核对 t=0:两腿入场 fill 价加权
    row = df.iloc[0]
    call_mid1, put_mid1 = 3.0 - 0.02, 3.0 + 0.02
    e_c = (call_mid1 - 0.05) + 0.775 * 0.10
    e_p = (put_mid1 - 0.05) + 0.775 * 0.10
    expected = (
        e_c * row["label_call_return_fwd_net"] + e_p * row["label_put_return_fwd_net"]
    ) / (e_c + e_p)
    assert abs(row["label_straddle_return_fwd_net"] - expected) < 1e-10
    assert bool(row["label_straddle_valid"])


def test_straddle_label_flat_market_negative():
    """盘整日(两腿 mid 都不动):跨式净收益必为负 —— 双份点差 + 佣金。"""
    from qqq_btc.common.labels import build_dual_leg_net_labels

    df = build_dual_leg_net_labels(
        _make_dual_leg_df(call_drift=0.0, put_drift=0.0),
        OptionSpreadFillModel(), LabelHorizon(),
    )
    valid = df["label_straddle_valid"]
    assert (df.loc[valid, "label_straddle_return_fwd_net"] < 0).all()


def test_replay_straddle_entry_and_fill_equivalence():
    """跨式入场:合成盘口 fill = 两腿 fill 之和;佣金按 2 张合约扣。"""
    fm = OptionSpreadFillModel(entry_frac=0.775, exit_frac=0.775, commission_per_contract=0.65)
    df = _make_dual_leg_df(n=60, call_drift=-0.02, put_drift=0.06)  # 大波动日
    df["call_net_edge"] = 0.0
    df["put_net_edge"] = 0.0
    df["straddle_net_edge"] = 0.0
    df.loc[0, "straddle_net_edge"] = 0.08

    rails = ExitRailsConfig(
        hard_stop_roi=-9.0, soft_stop_roi=-9.0,
        time_stop_bars=9999, time_stop_min_roi=-9.0,
        max_hold_bars=5,
        trailing_trigger_roi=9.0, flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),),
        eod_close_bar_index=None,
    )
    result = run_strict_replay(
        df, fm,
        ReplayConfig(entry_threshold=0.015, long_only=False, cooldown_bars=999,
                     straddle_entry_threshold=0.03),
        rails,
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
        straddle_edge_col="straddle_net_edge",
    )
    assert len(result.trades) == 1
    t = result.trades[0]
    assert t.leg == "STRADDLE"

    # 手工核对:入场 bar=1,出场 bar=6(MAX_HOLD=5)
    def fills(mid_entry, mid_exit):
        e = (mid_entry - 0.05) + 0.775 * 0.10
        x = (mid_exit + 0.05) - 0.775 * 0.10
        return e, x

    ec, xc = fills(3.0 - 0.02 * 1, 3.0 - 0.02 * 6)
    ep, xp = fills(3.0 + 0.06 * 1, 3.0 + 0.06 * 6)
    entry, exit_ = ec + ep, xc + xp
    commission_2x = 2 * (2 * 0.65 / (entry * 100))
    expected_net = exit_ / entry - 1.0 - commission_2x
    assert abs(t.entry_price - entry) < 1e-10
    assert abs(t.net_return - expected_net) < 1e-10
    # PUT 腿涨幅覆盖 CALL 腿损失(用户场景):整体为正
    assert t.net_return > 0


def test_replay_straddle_beats_weaker_directional_leg():
    """跨式与方向腿竞争:跨式 edge 更强时选跨式,反之选方向腿。"""
    fm = OptionSpreadFillModel()
    df = _make_dual_leg_df(n=60, call_drift=0.02, put_drift=0.0)
    df["call_net_edge"] = 0.0
    df["put_net_edge"] = 0.0
    df["straddle_net_edge"] = 0.0
    df.loc[0, "call_net_edge"] = 0.06
    df.loc[0, "straddle_net_edge"] = 0.04   # 过阈值但弱于 CALL

    cfg = ReplayConfig(entry_threshold=0.015, long_only=False, cooldown_bars=999,
                       straddle_entry_threshold=0.03)
    r1 = run_strict_replay(
        df, fm, cfg, ExitRailsConfig(),
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
        straddle_edge_col="straddle_net_edge",
    )
    assert r1.trades[0].leg == "CALL"

    df.loc[0, "straddle_net_edge"] = 0.10   # 强于 CALL
    r2 = run_strict_replay(
        df, fm, cfg, ExitRailsConfig(),
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
        straddle_edge_col="straddle_net_edge",
    )
    assert r2.trades[0].leg == "STRADDLE"


def test_replay_straddle_gates():
    """跨式门槛:long_only 禁用;低于跨式阈值不进;日内笔数上限生效。"""
    fm = OptionSpreadFillModel()
    df = _make_dual_leg_df(n=200, call_drift=-0.005, put_drift=0.01)
    df["call_net_edge"] = 0.0
    df["put_net_edge"] = 0.0
    df["straddle_net_edge"] = 0.05  # 每 bar 都有跨式信号

    common = dict(
        call_edge_col="call_net_edge", put_edge_col="put_net_edge",
        straddle_edge_col="straddle_net_edge",
    )
    # long_only=True:跨式含 PUT 腿,禁用
    r_lo = run_strict_replay(
        df, fm, ReplayConfig(long_only=True, cooldown_bars=0), _FAST_EXIT_RAILS, **common
    )
    assert len(r_lo.trades) == 0

    # 阈值 0.06 > 信号 0.05:不进
    r_th = run_strict_replay(
        df, fm,
        ReplayConfig(long_only=False, cooldown_bars=0, straddle_entry_threshold=0.06),
        _FAST_EXIT_RAILS, **common,
    )
    assert len(r_th.trades) == 0

    # 日内跨式上限 2 笔
    r_cap = run_strict_replay(
        df, fm,
        ReplayConfig(long_only=False, cooldown_bars=0,
                     straddle_entry_threshold=0.03, max_straddles_per_day=2),
        _FAST_EXIT_RAILS, **common,
    )
    assert len([t for t in r_cap.trades if t.leg == "STRADDLE"]) == 2


# ---------------------------------------------------------------------------
# 新工具: entry_decision / OMS 适配 / FCS 适配
# ---------------------------------------------------------------------------

def test_choose_entry_dual_mode_put():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(entry_threshold=0.015, long_only=False, max_spread_pct=0.06)
    d = choose_entry(
        rc, session_bar=10, dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
    )
    assert d is not None and d.leg == "PUT"


def test_choose_entry_q10_blocks_call():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(entry_threshold=0.01)
    # floor 语义:q10 须严格高于 edge_q10_floor(默认 -0.20)才放行 CALL
    d = choose_entry(rc, session_bar=0, edge=0.05, spread_pct=0.01, edge_q10=-0.25)
    assert d is None
    d2 = choose_entry(rc, session_bar=0, edge=0.05, spread_pct=0.01, edge_q10=-0.01)
    assert d2 is not None


def test_choose_entry_session_window():
    from qqq_btc.common.entry_decision import choose_entry

    rc_open = ReplayConfig(entry_threshold=0.01, session_entry_start_bar=0, session_entry_end_bar=360)
    assert choose_entry(rc_open, session_bar=0, edge=0.05, spread_pct=0.01) is not None

    rc_late = ReplayConfig(entry_threshold=0.01, session_entry_start_bar=15, session_entry_end_bar=360)
    assert choose_entry(rc_late, session_bar=10, edge=0.05, spread_pct=0.01) is None
    assert choose_entry(rc_late, session_bar=15, edge=0.05, spread_pct=0.01) is not None

    rc_end = ReplayConfig(entry_threshold=0.01, session_entry_start_bar=0, session_entry_end_bar=360)
    assert choose_entry(rc_end, session_bar=361, edge=0.05, spread_pct=0.01) is None


def test_choose_entry_morning_fade_put_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        morning_fade_min_ret=0.004,
        morning_fade_max_peak_dd=-0.003,
        morning_fade_session_end_bar=60,
    )
    # vix 不足,无 fade 形态 → PUT 被拒
    d = choose_entry(
        rc, session_bar=20, dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.10,
        open30_max_ret=0.002,
        open30_peak_dd=-0.001,
    )
    assert d is None or d.leg != "PUT"
    # fade 形态满足 → PUT 允许
    d2 = choose_entry(
        rc, session_bar=20, dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.10,
        open30_max_ret=0.005,
        open30_peak_dd=-0.004,
    )
    assert d2 is not None and d2.leg == "PUT"
    # fade 路径仅早盘:bar>60 不再豁免 vix
    d3 = choose_entry(
        rc, session_bar=90, dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.10,
        open30_max_ret=0.005,
        open30_peak_dd=-0.004,
    )
    assert d3 is None or d3.leg != "PUT"


def test_choose_entry_early_put_vix_gate():
    """早盘 PUT 要求更高 vix;午盘仍用 put_gate_min;fade 可豁免早盘加强。"""
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        put_early_session_bar=30,
        put_early_vix_min=0.6,
        morning_fade_min_ret=0.004,
        morning_fade_max_peak_dd=-0.003,
        morning_fade_session_end_bar=60,
    )
    common = dict(
        dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        trend_ret_30m=-0.001,
    )
    # July1 型:早盘 vix=0.55 介于 put_gate_min 与 early_vix 之间 → 拒 PUT
    d = choose_entry(rc, session_bar=17, put_gate=0.55, **common)
    assert d is None or d.leg != "PUT"
    # July7 型:早盘 vix=0.74 → 允许
    d2 = choose_entry(rc, session_bar=17, put_gate=0.74, **common)
    assert d2 is not None and d2.leg == "PUT"
    # 午盘:vix=0.55 仍过普通 put_gate
    d3 = choose_entry(rc, session_bar=90, put_gate=0.55, **common)
    assert d3 is not None and d3.leg == "PUT"
    # 早盘 fade 豁免 early_vix
    d4 = choose_entry(
        rc, session_bar=17, put_gate=0.40, **common,
        open30_max_ret=0.005, open30_peak_dd=-0.004,
    )
    assert d4 is not None and d4.leg == "PUT"


def test_choose_entry_early_put_mid_vix_ban():
    """中段 vix 禁令:挡 July1≈0.84,放行高 vix / 低 vix+fade。"""
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        put_early_session_bar=30,
        put_early_vix_min=0.6,
        put_early_vix_ban_lo=0.8,
        put_early_vix_ban_hi=1.0,
        morning_fade_min_ret=0.004,
        morning_fade_max_peak_dd=-0.003,
        morning_fade_session_end_bar=60,
    )
    common = dict(
        dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        trend_ret_30m=-0.001,
    )
    # July1 型中段 vix → 拒
    d = choose_entry(rc, session_bar=16, put_gate=0.84, **common)
    assert d is None or d.leg != "PUT"
    # July8 型高 vix → 允许
    d2 = choose_entry(rc, session_bar=15, put_gate=1.06, **common)
    assert d2 is not None and d2.leg == "PUT"
    # 中段但 fade → 允许
    d3 = choose_entry(
        rc, session_bar=16, put_gate=0.84, **common,
        open30_max_ret=0.005, open30_peak_dd=-0.004,
    )
    assert d3 is not None and d3.leg == "PUT"


def test_choose_entry_early_put_open30_range_blocks():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        put_early_session_bar=30,
        put_early_open30_max_min=1e-12,
        put_early_range30_min=0.003,
    )
    common = dict(
        dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.70, trend_ret_30m=-0.001,
    )
    # 空洞开盘:open30_max=0 / range=0 → 拒
    d = choose_entry(
        rc, session_bar=17, open30_max_ret=0.0, spot_range_30m=0.0, **common,
    )
    assert d is None or d.leg != "PUT"
    # 有结构 → 允许
    d2 = choose_entry(
        rc, session_bar=17, open30_max_ret=0.001, spot_range_30m=0.01, **common,
    )
    assert d2 is not None and d2.leg == "PUT"
    # 午盘不受 early open30/range 约束
    d3 = choose_entry(
        rc, session_bar=90, open30_max_ret=0.0, spot_range_30m=0.0, **common,
    )
    assert d3 is not None and d3.leg == "PUT"


def test_choose_entry_rapid_drop_blocks_call():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        rapid_drop_ret=-0.004,
        block_call_on_rapid_drop=True,
    )
    d = choose_entry(
        rc, session_bar=20, dual_mode=True,
        call_edge=0.05, put_edge=0.01,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        spot_ret_5bar=-0.005,
    )
    assert d is None or d.leg == "PUT"
    d2 = choose_entry(
        rc, session_bar=20, edge=0.05, spread_pct=0.02,
        spot_ret_5bar=-0.005,
    )
    assert d2 is None


def test_choose_entry_put_trend_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        put_trend_max_ret=0.0,
    )
    common = dict(
        session_bar=20, dual_mode=True,
        call_edge=0.01, put_edge=0.05,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40,
    )
    # 趋势向上 → PUT 被拦截(即使 vix 门控已通过)
    d = choose_entry(rc, **common, trend_ret_30m=0.002)
    assert d is None or d.leg != "PUT"
    # 趋势向下 → PUT 允许
    d2 = choose_entry(rc, **common, trend_ret_30m=-0.001)
    assert d2 is not None and d2.leg == "PUT"
    # 趋势缺失 → 不拦截(减法保护,缺数据不误杀)
    d3 = choose_entry(rc, **common, trend_ret_30m=None)
    assert d3 is not None and d3.leg == "PUT"


def test_choose_entry_call_trend_r2_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        call_trend_r2_min=0.15,
        put_gate_min=0.25,
    )
    common = dict(
        session_bar=20, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40,
    )
    # 低 r2 震荡 → CALL 被拦截,仍可开 PUT
    d = choose_entry(
        rc, **common, trend_r2_30m=0.05, call_edge=0.05, put_edge=0.05,
    )
    assert d is not None and d.leg == "PUT"
    # 高 r2 → CALL 允许(且分数更高时选 CALL)
    d2 = choose_entry(
        rc, **common, trend_r2_30m=0.30, call_edge=0.06, put_edge=0.05,
    )
    assert d2 is not None and d2.leg == "CALL"
    # r2 缺失 → 不拦截
    d3 = choose_entry(
        rc, **common, trend_r2_30m=None, call_edge=0.06, put_edge=0.05,
    )
    assert d3 is not None and d3.leg == "CALL"


def test_choose_entry_vix_regime_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        regime_vix_reversal_max=6,
    )
    common = dict(
        session_bar=20, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        call_edge=0.05, put_edge=0.05,
    )
    assert choose_entry(rc, **common, vix_reversal_count_30m=6) is not None
    assert choose_entry(rc, **common, vix_reversal_count_30m=7) is None
    assert choose_entry(rc, **common, vix_reversal_count_30m=None) is not None


def test_choose_entry_put_late_session_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        put_late_session_bar=240,
    )
    common = dict(
        session_bar=241, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.04, put_edge=0.05,
    )
    d = choose_entry(rc, **common)
    assert d is not None and d.leg == "CALL"
    only_put = {**common, "call_edge": 0.01}
    assert choose_entry(rc, **only_put) is None


def test_choose_entry_call_chase_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02,
        long_only=False,
        max_spread_pct=0.06,
        put_gate_min=0.25,
        call_chase_vix_rev_min=6,
        call_chase_spot_day_ret_min=0.0,
    )
    common = dict(
        session_bar=80, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.06, put_edge=0.05,
        spot_day_ret=0.01, vix_reversal_count_30m=6,
    )
    d = choose_entry(rc, **common)
    assert d is not None and d.leg == "PUT"
    d2 = choose_entry(
        rc, session_bar=80, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.06, put_edge=0.05,
        spot_day_ret=0.01, vix_reversal_count_30m=5,
    )
    assert d2 is not None and d2.leg == "CALL"


def test_choose_entry_call_spike_range_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02, long_only=False, max_spread_pct=0.06,
        put_gate_min=0.25, call_spike_range30_min=0.020,
    )
    common = dict(
        session_bar=20, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.06, put_edge=0.05,
    )
    blocked = {**common, "spot_range_30m": 0.025, "put_edge": 0.01}
    assert choose_entry(rc, **blocked) is None
    allowed = {**common, "spot_range_30m": 0.015}
    d = choose_entry(rc, **allowed)
    assert d is not None and d.leg == "CALL"


def test_choose_entry_call_spent_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02, long_only=False, max_spread_pct=0.06,
        put_gate_min=0.25,
        call_spent_day_range_pos_min=0.85,
        call_spent_bb_width_max=0.0,
        call_spent_min_session_bar=210,
    )
    common = dict(
        session_bar=238, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.06, put_edge=0.05,
        day_range_pos=0.98, bb_width=-0.07,
    )
    # spent → CALL 被挡,可转 PUT
    d = choose_entry(rc, **common)
    assert d is not None and d.leg == "PUT"
    only_call = {**common, "put_edge": 0.01}
    assert choose_entry(rc, **only_call) is None
    # 早于 min_bar 不拦
    early = {**only_call, "session_bar": 180}
    d2 = choose_entry(rc, **early)
    assert d2 is not None and d2.leg == "CALL"
    # bb 未压缩不拦
    wide = {**only_call, "bb_width": 0.05}
    d3 = choose_entry(rc, **wide)
    assert d3 is not None and d3.leg == "CALL"
    # 缺字段不拦
    miss = {**only_call}
    miss.pop("bb_width")
    d4 = choose_entry(rc, **miss)
    assert d4 is not None and d4.leg == "CALL"


def test_choose_entry_call_timing_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02, long_only=False, max_spread_pct=0.06,
        put_gate_min=0.25,
        call_timing_spot_min=0.003,
        call_timing_max_bar=200,
        call_timing_vix_min=5,
    )
    common = dict(
        session_bar=160, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.06, put_edge=0.05,
        spot_day_ret=0.0048, vix_reversal_count_30m=7,
    )
    blocked = {**common, "put_edge": 0.01}
    d = choose_entry(rc, **blocked)
    assert d is None
    late = {**blocked, "session_bar": 210}
    d2 = choose_entry(rc, **late)
    assert d2 is not None and d2.leg == "CALL"
    low_vix = {**blocked, "vix_reversal_count_30m": 4}
    d3 = choose_entry(rc, **low_vix)
    assert d3 is not None and d3.leg == "CALL"


def test_choose_entry_put_spot_day_ret_gate():
    from qqq_btc.common.entry_decision import choose_entry

    rc = ReplayConfig(
        entry_threshold=0.02, long_only=False, max_spread_pct=0.06,
        put_gate_min=0.25, put_spot_day_ret_min=0.008,
    )
    common = dict(
        session_bar=200, dual_mode=True,
        spread_pct=0.02, put_spread_pct=0.02, has_put=True,
        put_gate=0.40, call_edge=0.01, put_edge=0.05,
    )
    assert choose_entry(rc, **common, spot_day_ret=0.01) is None
    d = choose_entry(rc, **common, spot_day_ret=0.005)
    assert d is not None and d.leg == "PUT"


def test_regime_features_reversal_count():
    import pandas as pd
    from qqq_btc.common.regime_features import add_vix_regime_features, count_reversals

    prices = [100.0, 100.2, 100.0, 100.2, 100.0]
    assert count_reversals(np.array(prices), threshold=0.0015) == 3

    ts = pd.date_range("2026-06-10 09:30", periods=5, freq="1min", tz="America/New_York")
    df = pd.DataFrame({"timestamp": ts, "vix_proxy_close": prices})
    out = add_vix_regime_features(df)
    assert "vix_reversal_count_30m" in out.columns
    assert out["vix_reversal_count_30m"].iloc[-1] == 3.0


def test_replay_session_immediate_entry():
    """immediate_entry=True 时信号当根 bar 即 ENTER,不 pending 下一 bar。"""
    from dataclasses import replace
    from qqq_btc.common.replay_session import ReplaySession, SessionQuotes, SessionSignal
    from qqq_btc.common.exit_rails import ExitRailsConfig
    from qqq_btc.common.fill_model import OptionSpreadFillModel
    from qqq_btc.common.replay_types import ReplayConfig

    cfg = replace(
        ReplayConfig(entry_threshold=0.01, max_spread_pct=1.0, cooldown_bars=999),
        entry_delay_bars=0,
        immediate_entry=True,
    )
    sess = ReplaySession(cfg, ExitRailsConfig(), OptionSpreadFillModel(), is_option=True)
    sq = SessionQuotes(call_bid=1.98, call_ask=2.02, call_spread_pct=0.02)
    sig = SessionSignal(edge=0.05)
    evs = sess.on_minute_bar(0, "2026-06-02 10:00:00", 30, sq, sig)
    assert len(evs) == 1 and evs[0].kind == "ENTER"
    assert sess.position is not None
    assert sess.pending_entry_bar is None


def test_replay_session_bar_zero_can_signal():
    """首根 session bar(09:30)可发信号,无固定 seq_len 预热。"""
    from qqq_btc.common.replay_session import ReplaySession, SessionQuotes, SessionSignal

    fm = OptionSpreadFillModel()
    session = ReplaySession(
        ReplayConfig(entry_threshold=0.01, entry_delay_bars=1, max_spread_pct=1.0),
        ExitRailsConfig(),
        fm,
    )
    q = SessionQuotes(call_bid=2.0, call_ask=2.2, call_spread_pct=0.05)
    evs = session.on_minute_bar(0, "2026-06-01 09:30", 0, q, SessionSignal(edge=0.05))
    assert len(evs) == 1 and evs[0].kind == "SIGNAL"


def test_session_carryover_augment_multiday():
    from qqq_btc.common.inference_tensors import SEQ_LEN, build_feature_maps, row_to_tensors
    from qqq_btc.common.session_history import augment_with_session_carryover

    d1 = pd.date_range("2026-06-01 09:30", periods=50, freq="1min", tz="America/New_York")
    d2 = pd.date_range("2026-06-02 09:30", periods=10, freq="1min", tz="America/New_York")
    raw = pd.DataFrame({
        "timestamp": list(d1) + list(d2),
        "close": np.linspace(100, 101, 60),
        "feat_a": np.arange(60, dtype=float),
    })
    aug = augment_with_session_carryover(raw, carryover_bars=29)
    assert len(aug) == 60 + 29
    assert aug["_carryover"].sum() == 29

    # 第二日首 bar 在 aug 中的 index: 50 + 29 = 79, tensor 应有 30 非零步
    idx = 79
    stock_map, option_map, ns, no = build_feature_maps({"features": [{"name": "feat_a", "resolution": "1min"}]})
    x_s, x_o = row_to_tensors(aug, idx, stock_map, option_map, ns, no)
    assert np.count_nonzero(x_s[:, 0]) == SEQ_LEN


def test_apply_5m_stair_step_matches_fill_column():
    """FCS payload 用的 stair-step 须与 row_to_tensors 锚点一致。"""
    from qqq_btc.common.inference_tensors import (
        FIVE_MIN_STRIDE,
        SEQ_LEN,
        WINDOW_5M_BARS,
        apply_5m_stair_step,
        build_feature_maps,
        row_to_tensors,
    )

    n = 60
    ts = pd.date_range("2026-06-10 09:30", periods=n, freq="1min", tz="America/New_York")
    f5 = np.full(n, np.nan)
    f5[::FIVE_MIN_STRIDE] = np.arange(n // FIVE_MIN_STRIDE, dtype=float)
    f5 = pd.Series(f5).ffill().to_numpy()
    df = pd.DataFrame({"timestamp": ts, "feat_5m": f5})
    cfg = {"features": [{"name": "feat_5m", "resolution": "5min"}]}
    stock_map, option_map, ns, no = build_feature_maps(cfg)
    x_s, _ = row_to_tensors(df, 59, stock_map, option_map, ns, no)
    stepped = apply_5m_stair_step(f5[-SEQ_LEN:])
    assert stepped.shape == (SEQ_LEN,)
    assert np.allclose(stepped, x_s[:, 0])
    for block in range(WINDOW_5M_BARS):
        seg = stepped[block * FIVE_MIN_STRIDE : (block + 1) * FIVE_MIN_STRIDE]
        assert np.allclose(seg, seg[0])


def test_row_to_tensors_5min_repeat_matches_dataset():
    """5min 特征须 repeat×5 铺满 30 窗,与 LMDBAlphaDataset 一致。"""
    from qqq_btc.common.inference_tensors import (
        FIVE_MIN_STRIDE,
        SEQ_LEN,
        WINDOW_5M_BARS,
        build_feature_maps,
        row_to_tensors,
    )

    n = 60
    ts = pd.date_range("2026-06-10 09:30", periods=n, freq="1min", tz="America/New_York")
    f5 = np.full(n, np.nan)
    f5[::FIVE_MIN_STRIDE] = np.arange(n // FIVE_MIN_STRIDE, dtype=float)
    f5 = pd.Series(f5).ffill().to_numpy()  # 模拟 merge_asof 前向填充
    df = pd.DataFrame({"timestamp": ts, "feat_1m": np.arange(n, dtype=float), "feat_5m": f5})
    cfg = {
        "features": [
            {"name": "feat_1m", "resolution": "1min"},
            {"name": "feat_5m", "resolution": "5min"},
        ]
    }
    stock_map, option_map, ns, no = build_feature_maps(cfg)
    x_s, _ = row_to_tensors(df, 59, stock_map, option_map, ns, no)
    col5 = x_s[:, 1]
    # 6 锚点 × repeat 5 = 30;同一 5min bucket 内 5 个值相同
    tail = col5[-30:]
    for block in range(WINDOW_5M_BARS):
        seg = tail[block * FIVE_MIN_STRIDE : (block + 1) * FIVE_MIN_STRIDE]
        assert seg.size == FIVE_MIN_STRIDE
        assert np.allclose(seg, seg[0])
    assert col5[-30] < col5[-1]
    assert list(x_s[-30:, 0]) == list(range(30, 60))


def test_session_prepend_carryover():
    from qqq_btc.common.session_history import prepend_carryover, session_tail

    prior = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-01 14:00", periods=30, freq="1min", tz="America/New_York"),
        "close": np.ones(30),
    })
    today = pd.DataFrame({
        "timestamp": pd.date_range("2026-06-02 09:30", periods=3, freq="1min", tz="America/New_York"),
        "close": np.ones(3) * 2,
    })
    out = prepend_carryover(today, session_tail(prior, 29))
    assert len(out) == 32
    assert out["_carryover"].sum() == 29


def test_oms_limit_and_audit():
    from qqq_btc.live.oms_adapter import audit_fill, limit_price_from_quote

    bid, ask = 2.0, 2.2
    lp = limit_price_from_quote(bid, ask, "BUY")
    assert abs(lp - 2.155) < 1e-9
    rec = audit_fill(bid, ask, lp, "BUY")
    assert abs(rec.fill_spread_frac - 0.775) < 1e-6


def test_fcs_adapter_enrich():
    from qqq_btc.live.fcs_adapter import enrich_fcs_bars

    ts = pd.date_range("2026-06-01 09:30", periods=50, freq="1min", tz="America/New_York")
    df = pd.DataFrame({"timestamp": ts, "close": 500.0 + np.arange(50) * 0.1})
    out = enrich_fcs_bars(df)
    assert "time_session_progress" in out.columns
    assert "trend_fit_r2_30m" in out.columns


def test_qqq_config_governance_wiring():
    from qqq_btc.qqq import config as qcfg

    assert qcfg.REPLAY.max_trades_per_day is not None
    assert qcfg.REPLAY.daily_loss_stop < 0
    assert qcfg.REPLAY.loss_streak_n >= 2
    assert qcfg.CALL_EDGE_COL == "call_net_edge"
    assert qcfg.PUT_EDGE_COL == "put_net_edge"
    assert qcfg.STRADDLE_EDGE_COL == "straddle_net_edge"
    # 跨式门槛不得低于单腿(双份权利金 + 双份 theta;单腿基础阈值已提至 0.03)
    assert qcfg.REPLAY.straddle_entry_threshold >= qcfg.REPLAY.entry_threshold
    assert qcfg.REPLAY.max_straddles_per_day is not None
    # 双腿已开启(2026-07 三时期验证);PUT 受 vix 门控 + 趋势对齐双重保护
    assert qcfg.REPLAY.long_only is False
    assert qcfg.REPLAY.put_gate_min is not None
    assert qcfg.REPLAY.put_trend_max_ret is not None
    assert qcfg.REPLAY.call_trend_r2_min is not None
    # 尾盘禁入:14:30 后不新开仓(theta 尾段衰减 + Q2 时段审计)
    assert qcfg.REPLAY.session_entry_end_bar <= 300
    # live 立即入场 vs replay 延迟 1 bar(标签 60s)
    assert qcfg.LIVE_REPLAY.immediate_entry is True
    assert qcfg.LIVE_REPLAY.entry_delay_bars == 0
    assert qcfg.REPLAY.entry_delay_bars == 1


# ---------------------------------------------------------------------------
# event replay (L1/L2)
# ---------------------------------------------------------------------------

def test_event_replay_l1_matches_strict_replay():
    """L1 event replay 与 run_strict_replay 必须 bit-identical。"""
    from qqq_btc.common.event_replay import run_event_replay
    from qqq_btc.common.replay_harness import run_strict_replay

    fm = OptionSpreadFillModel()
    df = _make_option_df(n=80, drift=0.015)
    df["net_edge"] = 0.0
    df.loc[5, "net_edge"] = 0.05
    df.loc[40, "net_edge"] = 0.05
    cfg = ReplayConfig(entry_threshold=0.015, cooldown_bars=5)
    rails = ExitRailsConfig()

    r_strict = run_strict_replay(df, fm, cfg, rails)
    r_event = run_event_replay(df, fm, cfg, rails, tick_df=None)
    assert len(r_strict.trades) == len(r_event.trades)
    if r_strict.trades:
        assert abs(r_strict.trades[0].net_return - r_event.trades[0].net_return) < 1e-12


def test_event_replay_first_tick_changes_fill():
    from qqq_btc.common.event_replay import EventReplayConfig, FillTiming, run_event_replay

    fm = OptionSpreadFillModel()
    n = 20
    minute = _make_option_df(n=n, base=2.0, drift=0.01, spread=0.10)
    minute["net_edge"] = 0.0
    minute.loc[0, "net_edge"] = 0.05

    ticks = []
    for i, ts in enumerate(minute["timestamp"]):
        base_mid = 2.0 + 0.01 * i
        # 分钟首 tick 比 minute close 更贵 → first_tick 入场价更高
        first_mid = base_mid + 0.05
        for sec in range(3):
            t = ts + pd.Timedelta(seconds=sec)
            mid = first_mid if sec == 0 else base_mid
            ticks.append(
                {
                    "timestamp": t,
                    "exec_call_bid": mid - 0.05,
                    "exec_call_ask": mid + 0.05,
                }
            )
    tick_df = pd.DataFrame(ticks)

    rails = ExitRailsConfig(
        hard_stop_roi=-9.0, soft_stop_roi=-9.0, max_hold_bars=5,
        time_stop_bars=999, trailing_trigger_roi=9.0, flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),), eod_close_bar_index=None,
    )
    cfg = ReplayConfig(entry_threshold=0.015, entry_delay_bars=1, cooldown_bars=99, max_spread_pct=1.0)

    r_close = run_event_replay(
        minute, fm, cfg, rails, tick_df=tick_df,
        event_cfg=EventReplayConfig(fill_timing=FillTiming.MINUTE_CLOSE),
    )
    r_first = run_event_replay(
        minute, fm, cfg, rails, tick_df=tick_df,
        event_cfg=EventReplayConfig(fill_timing=FillTiming.FIRST_TICK),
    )
    assert len(r_close.trades) == 1 and len(r_first.trades) == 1
    assert r_first.trades[0].entry_price > r_close.trades[0].entry_price


def test_event_replay_disaster_stop_on_tick():
    from qqq_btc.common.event_replay import EventReplayConfig, run_event_replay

    fm = OptionSpreadFillModel()
    minute = _make_option_df(n=15, base=2.0, drift=0.0, spread=0.10)
    minute["net_edge"] = 0.0
    minute.loc[0, "net_edge"] = 0.05
    minute.loc[1, "net_edge"] = 0.0
    # 分钟收盘 mid 同步暴跌,否则持仓无法在 n=15 内平仓 → trades 为空
    crash_mid = 1.4
    for col, off in (("exec_call_mid", 0.0), ("exec_call_bid", -0.05), ("exec_call_ask", 0.05)):
        minute.loc[2:, col] = crash_mid + off

    crash_ts = minute["timestamp"].iloc[2]
    ticks = []
    for i, ts in enumerate(minute["timestamp"]):
        mid = 2.0 if i < 2 else (1.4 if ts == crash_ts else 2.0)
        ticks.append({"timestamp": ts, "exec_call_bid": mid - 0.05, "exec_call_ask": mid + 0.05})
        if ts == crash_ts:
            ticks.append(
                {
                    "timestamp": ts + pd.Timedelta(seconds=15),
                    "exec_call_bid": 1.4 - 0.05,
                    "exec_call_ask": 1.4 + 0.05,
                }
            )
    tick_df = pd.DataFrame(ticks)

    rails = ExitRailsConfig(
        hard_stop_roi=-0.12, disaster_stop_roi=-0.25, disaster_smooth_n=1, max_hold_bars=99,
        eod_close_bar_index=None,
    )
    cfg = ReplayConfig(entry_threshold=0.015, entry_delay_bars=1, cooldown_bars=99, max_spread_pct=1.0)

    r_no_tick = run_event_replay(minute, fm, cfg, rails, tick_df=None)
    r_tick = run_event_replay(
        minute, fm, cfg, rails, tick_df=tick_df,
        event_cfg=EventReplayConfig(tick_disaster_stop=True, tick_smooth_n=1),
    )
    assert len(r_no_tick.trades) >= 1
    assert len(r_tick.trades) >= 1
    assert "DISASTER_STOP" in r_tick.trades[0].exit_reason


def test_alpha_frame_bridge_payload():
    from qqq_btc.live.alpha_frame_bridge import build_alpha_frame, build_opt_data_from_quotes

    quotes = {
        "exec_call_bid": 1.98,
        "exec_call_ask": 2.02,
        "exec_call_mid": 2.0,
        "exec_put_bid": 1.48,
        "exec_put_ask": 1.52,
    }
    opt = build_opt_data_from_quotes(quotes, leg="CALL")
    assert opt["has_feed"] is True
    assert abs(opt["price"] - 2.0) < 1e-9

    frame = build_alpha_frame(
        curr_ts=1_700_000_000.0,
        frame_id="f1",
        symbol="QQQ",
        preds={"net_edge": 0.02, "call_net_edge": 0.02, "net_edge_q10": 0.005},
        quotes=quotes,
        stock_price=450.0,
    )
    assert frame["action"] == "ALPHA_FRAME"
    assert frame["source"] == "qqq_btc_live"
    assert frame["items"][0]["alpha"] == 0.02
    assert frame["items"][0]["opt_data"]["bid"] == 1.98


def test_oms_integration_entry_limit_uses_fill_model():
    from qqq_btc.live.oms_integration import entry_limit_price_qqq_btc

    sig = {"meta": {"bid": 1.98, "ask": 2.02}}
    px = entry_limit_price_qqq_btc(sig, base_price=2.0, attempt_no=0)
    # 0.775 fill on [1.98, 2.02] → 2.011, capped below ask-0.01
    assert 1.98 <= px < 2.02


def test_oms_tick_stop_reads_position_leg_quote():
    from qqq_btc.live.oms_integration import _quote_option_mid

    quote = {
        "call_bid": 8.0,
        "call_ask": 8.2,
        "put_bid": 3.65,
        "put_ask": 3.69,
    }
    assert abs(_quote_option_mid(quote, 1) - 8.1) < 1e-12
    assert abs(_quote_option_mid(quote, -1) - 3.67) < 1e-12


def test_oms_stale_guard_allows_qqq_btc_forced_time_exits():
    from qqq_btc.live.oms_integration import allow_qqq_btc_forced_exits_on_stale_quotes

    class _Engine:
        STALE_GUARD_RISK_EXIT_TOKENS = ("TIME_STOP",)

    tokens = allow_qqq_btc_forced_exits_on_stale_quotes(_Engine)
    assert "QQQ_BTC_MAX_HOLD" in tokens
    assert "QQQ_BTC_EOD_CLOSE" in tokens
    # 重复 bootstrap 不得不断追加相同 token。
    assert allow_qqq_btc_forced_exits_on_stale_quotes(_Engine) == tokens


def test_live_clock_converts_fcs_start_label_to_end_label(monkeypatch):
    from qqq_btc.live.live_clock import live_session_bar, live_end_label_ts

    ts = pd.Timestamp("2026-07-02 09:48:00", tz="America/New_York").timestamp()
    monkeypatch.delenv("QQQ_BTC_LIVE_LABEL_SHIFT_SEC", raising=False)
    assert live_session_bar(ts) == 19
    end = live_end_label_ts(ts)
    assert end.tz_convert("America/New_York").strftime("%H:%M") == "09:49"
    monkeypatch.setenv("QQQ_BTC_LIVE_LABEL_SHIFT_SEC", "0")
    assert live_session_bar(ts) == 18


def test_live_start_minute_allows_fcs_0944_as_session_bar_15():
    """START_MINUTE=45 不得挡掉 FCS 09:44 start-label（end-label=09:45=sb15）。"""
    import sys
    from datetime import datetime
    from pathlib import Path

    from pytz import timezone

    from qqq_btc.common.replay_types import ReplayConfig
    from qqq_btc.live.session_governor import get_session_governor, _GOVERNORS
    import qqq_btc.qqq.config as qcfg

    _GOVERNORS.clear()
    cfg = ReplayConfig(
        long_only=False,
        entry_quantile=None,
        entry_threshold_schedule=((0, 0.03),),
        put_gate_min=None,
        put_early_session_bar=None,
        put_early_open30_max_min=None,
        put_trend_max_ret=None,
        call_trend_r2_min=None,
        regime_vix_reversal_max=None,
        session_entry_start_bar=15,
        session_entry_end_bar=240,
    )
    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import baseline_paths  # noqa: E402,F401

    from strategy.core_v0 import StrategyCoreV0
    from strategy.config0 import StrategyConfig
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

    old_replay = qcfg.REPLAY
    qcfg.REPLAY = cfg
    try:
        apply_strategy_entry_patch(StrategyCoreV0)
        scfg = StrategyConfig()
        scfg.START_MINUTE = 45
        core = StrategyCoreV0(scfg)
        ny = timezone("America/New_York")
        # FCS start-label for end-label 09:45 / sb15
        ctx = {
            "is_ready": True,
            "is_banned": False,
            "position": 0,
            "cooldown_until": 0.0,
            "curr_ts": pd.Timestamp("2026-07-08 09:44:00", tz="America/New_York").timestamp(),
            "time": datetime(2026, 7, 8, 9, 44, tzinfo=ny),
            "net_edge_raw": 0.05,
            "call_edge": 0.02,
            "put_edge": 0.05,
            "alpha_z": 0.05,
            "vol_z": 0.5,
            "bid": 1.98,
            "ask": 2.02,
            "curr_price": 2.0,
            "put_bid": 1.98,
            "put_ask": 2.02,
            "put_spread_pct": 0.02,
            "options_vw_spread": 0.02,
            "options_iv_momentum": 0.0,
            "symbol": "QQQ",
            "spy_roc": 0.001,
            "qqq_roc": 0.001,
            "spread_divergence": 0.0,
            "vix_level": 1.4,
            "trend_fit_ret_30m": -0.0001,
            "open30_max_ret": 0.001,
        }
        get_session_governor(cfg)
        sig = core.decide_entry(dict(ctx))
        assert sig is not None, core.get_last_reject_reason()
    finally:
        qcfg.REPLAY = old_replay
        _GOVERNORS.clear()


def test_bootstrap_tick_exits_mode():
    import os
    from qqq_btc.live.bootstrap import tick_exits_mode

    os.environ.pop("QQQ_BTC_LIVE", None)
    assert tick_exits_mode() == "legacy"
    os.environ["QQQ_BTC_LIVE"] = "1"
    os.environ["QQQ_BTC_TICK_EXITS"] = "disaster_only"
    assert tick_exits_mode() == "disaster_only"
    os.environ.pop("QQQ_BTC_LIVE", None)
    os.environ.pop("QQQ_BTC_TICK_EXITS", None)


def test_strategy_exit_bridge_hard_stop():
    from qqq_btc.live.strategy_exit_bridge import check_exit_via_rails
    from qqq_btc.common.exit_rails import ExitRailsConfig
    from datetime import datetime
    from pytz import timezone

    rails = ExitRailsConfig(
        hard_stop_roi=-0.12,
        soft_stop_roi=-0.08,
        early_stop_bars=None,
        max_hold_bars=99,
        trailing_trigger_roi=9.0,
        flash_trigger_roi=9.0,
        ladder=((9.0, -9.0),),
        eod_close_bar_index=None,
    )
    ctx = {
        "holding": {"entry_price": 2.0, "entry_ts": 0, "dir": 1, "max_roi": 0.05, "entry_bar": 0},
        "curr_price": 1.70,
        "held_mins": 3.0,
        "time": datetime(2026, 6, 2, 10, 30, tzinfo=timezone("America/New_York")),
    }
    sig = check_exit_via_rails(ctx, rails)
    assert sig is not None
    assert "HARD_STOP" in sig["reason"]


def test_strategy_exit_bridge_max_hold_without_current_quote():
    from datetime import datetime

    from pytz import timezone

    from qqq_btc.common.exit_rails import ExitRailsConfig
    from qqq_btc.live.strategy_exit_bridge import check_exit_via_rails

    rails = ExitRailsConfig(max_hold_bars=55, eod_close_bar_index=None)
    ctx = {
        "holding": {
            "entry_price": 2.0,
            "dir": 1,
            "max_roi": 0.10,
            "entry_bar": 30,
        },
        "curr_price": 0.0,
        "held_mins": 55.0,
        "time": datetime(2026, 7, 1, 10, 55, tzinfo=timezone("America/New_York")),
    }
    sig = check_exit_via_rails(ctx, rails)
    assert sig is not None
    assert sig["reason"] == "QQQ_BTC_MAX_HOLD|NO_QUOTE"


def test_strategy_entry_bridge_call_with_q10(monkeypatch):
    import sys
    from datetime import datetime
    from pathlib import Path

    from pytz import timezone

    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import baseline_paths  # noqa: E402,F401

    from strategy.core_v0 import StrategyCoreV0
    from strategy.config0 import StrategyConfig
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

    apply_strategy_entry_patch(StrategyCoreV0)
    core = StrategyCoreV0(StrategyConfig())

    ny = timezone("America/New_York")
    base_ctx = {
        "is_ready": True,
        "is_banned": False,
        "position": 0,
        "cooldown_until": 0.0,
        "curr_ts": 1_700_000_000.0,
        "time": datetime(2026, 6, 2, 10, 0, tzinfo=ny),
        # bar30 静态阈值 0.03(entry_threshold_schedule),需高于阈值才触发
        "net_edge_raw": 0.035,
        "net_edge_q10": 0.005,
        "alpha_z": 0.035,
        "vol_z": 0.5,
        "bid": 1.98,
        "ask": 2.02,
        "curr_price": 2.0,
        "options_vw_spread": 0.02,
        "options_iv_momentum": 0.0,
        "symbol": "QQQ",
        "spy_roc": 0.001,
        "qqq_roc": 0.001,
        "spread_divergence": 0.0,
    }

    sig = core.decide_entry(dict(base_ctx))
    assert sig is not None
    assert sig["action"] == "BUY"
    assert sig["dir"] == 1
    assert "QQQ_BTC_ENTRY" in sig.get("reason", "")

    blocked = dict(base_ctx)
    # QQQ LIVE_REPLAY floor=-0.25:q10 须严格高于 floor;相等或更低拦截
    blocked["net_edge_q10"] = -0.25
    assert core.decide_entry(blocked) is None
    # frozen 重标定后轻度悲观(>-0.25)应放行
    mild = dict(base_ctx)
    mild["net_edge_q10"] = -0.24
    assert core.decide_entry(mild) is not None


def test_minimal_stack_session_config():
    import os
    from pathlib import Path

    env_path = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq" / "config" / "minimal_stack.env"
    text = env_path.read_text(encoding="utf-8")
    assert "BIDIRECTIONAL_ENABLED=1" in text
    assert "BIDIRECTIONAL_DUAL_EDGE_ENABLED=1" in text
    assert "BIDIRECTIONAL_DISLOCATION_ENTRY_ENABLED=0" in text
    assert "START_MINUTE=45" in text
    assert "FAST_GATE_ENABLED=0" in text
    assert "COOLDOWN_MINUTES=5" in text

    os.environ["START_MINUTE"] = "45"
    os.environ["BIDIRECTIONAL_DISLOCATION_ENTRY_ENABLED"] = "0"
    # re-import would be heavy; spot-check config module if on path
    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    import sys
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import importlib
    import config as np_config

    importlib.reload(np_config)
    assert np_config.START_MINUTE == 45
    assert np_config.BIDIRECTIONAL_DISLOCATION_ENTRY_ENABLED is False


def test_regime_ctx_extract_and_merge():
    import numpy as np
    from qqq_btc.live.regime_ctx import extract_regime_ctx, merge_regime_into_ctx
    from qqq_btc.live.se_feature_bridge import _SymbolBarHistory

    hist = _SymbolBarHistory()
    base_ts = 1_700_000_000.0
    for i, px in enumerate(np.linspace(400, 402, 40)):
        hist.append(base_ts + i * 60, float(px))

    batch = {
        "features_dict": {
            "vix_level": np.full((1, 30), 0.35, dtype=np.float32),
        }
    }
    out = extract_regime_ctx(batch, ["QQQ"], history_store={"QQQ": hist})
    assert "QQQ" in out
    assert out["QQQ"].get("spot_day_ret") is not None
    assert abs(out["QQQ"].get("vix_level", 0) - 0.35) < 1e-5
    assert out["QQQ"].get("trend_fit_r2_30m", 0) > 0.5

    ctx = {"symbol": "QQQ"}
    item = {"spot_day_ret": 0.01, "trend_fit_r2_30m": 0.2, "vix_reversal_count_30m": 3.0}
    merge_regime_into_ctx(ctx, item)
    assert ctx["spot_day_ret"] == 0.01
    assert ctx["trend_fit_r2_30m"] == 0.2
    assert ctx["vix_reversal_count_30m"] == 3.0


def test_extract_regime_ctx_short_history_falls_back_to_features_dict():
    """开盘不足 30 根时,本地 enrich 的 trend=0 不得盖住 FCS features_dict。"""
    import numpy as np
    from qqq_btc.live.regime_ctx import extract_regime_ctx
    from qqq_btc.live.se_feature_bridge import _SymbolBarHistory

    hist = _SymbolBarHistory()
    base_ts = 1_700_000_000.0
    for i in range(15):
        hist.append(base_ts + i * 60, 400.0 + 0.01 * i)

    batch = {
        "features_dict": {
            "trend_fit_r2_30m": np.full((1, 30), 0.196, dtype=np.float32),
            "trend_fit_ret_30m": np.full((1, 30), 0.002, dtype=np.float32),
        }
    }
    out = extract_regime_ctx(batch, ["QQQ"], history_store={"QQQ": hist})
    assert abs(out["QQQ"]["trend_fit_r2_30m"] - 0.196) < 1e-5
    assert abs(out["QQQ"]["trend_fit_ret_30m"] - 0.002) < 1e-5


def test_symbol_bar_history_dedupes_same_ts():
    from qqq_btc.live.se_feature_bridge import _SymbolBarHistory

    h = _SymbolBarHistory()
    h.append(1000.0, 400.0)
    h.append(1000.0, 401.0)  # same ts → update
    h.append(999.0, 399.0)   # older → ignore
    h.append(1060.0, 402.0)
    assert len(h) == 2
    assert list(h._close) == [401.0, 402.0]


def test_seed_qqq_feature_history_enables_early_trend():
    import os
    from qqq_btc.live import signal_integration as si
    from qqq_btc.live.fcs_adapter import enrich_fcs_bars

    os.environ["QQQ_BTC_SPOT_SEED_BEFORE"] = "20260702"
    store: dict = {}
    n = si._seed_qqq_feature_history(store, max_bars=390)
    assert n >= 30
    hist = store["QQQ"]
    base = list(hist._ts)[-1] + 60
    for i in range(19):
        hist.append(base + i * 60, 736.0 + 0.05 * i)
    enriched = enrich_fcs_bars(hist.to_frame())
    r2 = float(enriched.iloc[-1]["trend_fit_r2_30m"])
    assert r2 > 0.05, f"expected non-zero early r2 after seed, got {r2}"


def test_strategy_entry_bridge_put_dual(monkeypatch):
    import sys
    from datetime import datetime
    from pathlib import Path

    from pytz import timezone

    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import baseline_paths  # noqa: E402,F401

    from strategy.core_v0 import StrategyCoreV0
    from strategy.config0 import StrategyConfig
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

    apply_strategy_entry_patch(StrategyCoreV0)
    core = StrategyCoreV0(StrategyConfig())

    ny = timezone("America/New_York")
    base_ctx = {
        "is_ready": True,
        "is_banned": False,
        "position": 0,
        "cooldown_until": 0.0,
        "curr_ts": 1_700_000_000.0,
        "time": datetime(2026, 6, 2, 10, 0, tzinfo=ny),
        "net_edge_raw": 0.01,
        "call_edge": 0.01,
        "put_edge": 0.04,
        "alpha_z": 0.01,
        "vol_z": 0.5,
        "bid": 1.98,
        "ask": 2.02,
        "curr_price": 2.0,
        "call_bid": 1.98,
        "call_ask": 2.02,
        "put_bid": 1.98,
        "put_ask": 2.02,
        "options_vw_spread": 0.02,
        "options_iv_momentum": 0.0,
        "symbol": "QQQ",
        "spy_roc": 0.001,
        "qqq_roc": 0.001,
        "spread_divergence": 0.0,
        "vix_level": 0.30,
    }

    sig = core.decide_entry(dict(base_ctx))
    assert sig is not None
    assert sig["dir"] == -1
    assert "PUT" in sig.get("reason", "")


def test_strategy_entry_bridge_put_blocked_by_put_spread(monkeypatch):
    """空仓 ctx.bid/ask 为紧 CALL 时,宽 PUT spread 仍应挡 PUT(对齐 offline)。"""
    import sys
    from datetime import datetime
    from pathlib import Path

    from pytz import timezone

    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import baseline_paths  # noqa: E402,F401

    from strategy.core_v0 import StrategyCoreV0
    from strategy.config0 import StrategyConfig
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

    apply_strategy_entry_patch(StrategyCoreV0)
    core = StrategyCoreV0(StrategyConfig())

    ny = timezone("America/New_York")
    # CALL spread ~1.2%; PUT spread ~6.6% > max_spread 0.06
    ctx = {
        "is_ready": True,
        "is_banned": False,
        "position": 0,
        "cooldown_until": 0.0,
        "curr_ts": 1_700_000_000.0,
        "time": datetime(2026, 7, 7, 10, 40, tzinfo=ny),
        "net_edge_raw": 0.06,
        "call_edge": 0.05,
        "put_edge": 0.06,
        "alpha_z": 0.06,
        "vol_z": 0.5,
        "bid": 1.65,
        "ask": 1.67,
        "curr_price": 1.66,
        "call_bid": 1.65,
        "call_ask": 1.67,
        "put_bid": 8.21,
        "put_ask": 8.77,
        "options_vw_spread": 0.02,
        "options_iv_momentum": 0.0,
        "symbol": "QQQ",
        "spy_roc": -0.001,
        "qqq_roc": -0.001,
        "spread_divergence": 0.0,
        "vix_level": 0.76,
        "trend_fit_ret_30m": -0.001,
        "trend_fit_r2_30m": 0.5,
    }
    assert core.decide_entry(dict(ctx)) is None
    # 收窄 PUT 后应放行
    ctx["put_ask"] = 8.50
    sig = core.decide_entry(dict(ctx))
    assert sig is not None
    assert sig["dir"] == -1


def test_entry_quantile_threshold():
    from collections import deque

    from qqq_btc.common.entry_quantile import quantile_threshold
    from qqq_btc.common.replay_types import ReplayConfig

    cfg = ReplayConfig(entry_quantile=0.80, entry_quantile_min_obs=3)
    buf = deque([0.01, 0.02, 0.03, 0.04, 0.05])
    th = quantile_threshold(buf, cfg)
    assert th is not None
    assert th >= 0.04
    assert quantile_threshold(deque([0.01, 0.02]), cfg) is None


def test_live_session_governor_frequency():
    from qqq_btc.common.replay_types import ReplayConfig
    from qqq_btc.live.session_governor import LiveSessionGovernor

    cfg = ReplayConfig(
        max_trades_per_day=2,
        daily_loss_stop=-0.10,
        loss_streak_n=2,
        loss_streak_cooldown_bars=60,
        entry_quantile=None,
    )
    gov = LiveSessionGovernor(cfg)
    ts = 1_700_000_000.0
    blocked, reason = gov.blocked_for_entry("QQQ", curr_ts=ts)
    assert not blocked

    gov.record_trade_close("QQQ", net_ret=-0.05, curr_ts=ts, leg="CALL")
    gov.record_trade_close("QQQ", net_ret=-0.06, curr_ts=ts + 60, leg="CALL")
    blocked, reason = gov.blocked_for_entry("QQQ", curr_ts=ts + 120)
    assert blocked
    assert reason in ("daily_loss_stop", "loss_streak_cooldown")
    assert gov._state("QQQ").day_halted

    gov2 = LiveSessionGovernor(ReplayConfig(max_trades_per_day=1, entry_quantile=None))
    gov2.record_trade_close("QQQ", net_ret=0.01, curr_ts=ts, leg="CALL")
    blocked, reason = gov2.blocked_for_entry("QQQ", curr_ts=ts + 60)
    assert blocked and reason == "max_trades_per_day"


def test_live_cooldown_still_feeds_entry_quantile_edges():
    """V0 pre_conditions 冷却只挡开仓，不挡 edge 缓冲（对齐 offline CLOSE）。"""
    import sys
    from datetime import datetime
    from pathlib import Path

    from pytz import timezone

    from qqq_btc.common.replay_types import ReplayConfig
    from qqq_btc.live.session_governor import get_session_governor, _GOVERNORS
    import qqq_btc.qqq.config as qcfg

    _GOVERNORS.clear()
    cfg = ReplayConfig(
        long_only=False,
        entry_quantile=0.80,
        entry_quantile_window=100,
        entry_quantile_min_obs=5,
        entry_threshold_schedule=((0, 0.99),),  # 禁止实际入场
        put_gate_min=None,
        call_trend_r2_min=None,
        regime_vix_reversal_max=None,
        cooldown_bars=10,
    )
    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import baseline_paths  # noqa: E402,F401

    from strategy.core_v0 import StrategyCoreV0
    from strategy.config0 import StrategyConfig
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch

    old_replay = qcfg.REPLAY
    qcfg.REPLAY = cfg
    try:
        apply_strategy_entry_patch(StrategyCoreV0)
        core = StrategyCoreV0(StrategyConfig())
        ny = timezone("America/New_York")
        ts = 1_700_000_000.0
        cool_until = ts + 10 * 60.0
        ctx = {
            "is_ready": True,
            "is_banned": False,
            "position": 0,
            "cooldown_until": cool_until,
            "curr_ts": ts + 60.0,  # 仍在冷却内
            "time": datetime(2026, 6, 2, 10, 30, tzinfo=ny),
            "net_edge_raw": 0.05,
            "call_edge": 0.05,
            "put_edge": 0.02,
            "alpha_z": 0.05,
            "vol_z": 0.5,
            "bid": 1.98,
            "ask": 2.02,
            "curr_price": 2.0,
            "options_vw_spread": 0.02,
            "options_iv_momentum": 0.0,
            "symbol": "QQQ",
            "spy_roc": 0.001,
            "qqq_roc": 0.001,
            "spread_divergence": 0.0,
        }
        gov = get_session_governor(cfg)
        before = len(gov._state("QQQ").edge_buf or [])
        assert core.decide_entry(dict(ctx)) is None
        after = len(gov._state("QQQ").edge_buf or [])
        assert after == before + 1
        # 同 bar 去重
        assert core.decide_entry(dict(ctx)) is None
        assert len(gov._state("QQQ").edge_buf or []) == after
    finally:
        qcfg.REPLAY = old_replay
        _GOVERNORS.clear()


def test_live_session_governor_tick_stop_lockout_freezes_edges():
    from qqq_btc.common.replay_types import ReplayConfig
    from qqq_btc.live.session_governor import LiveSessionGovernor

    cfg = ReplayConfig(
        cooldown_bars=10,
        tick_stop_cooldown_bars=30,
        tick_stop_lock_leg_for_day=True,
        entry_quantile=0.80,
        entry_quantile_min_obs=1,
    )
    gov = LiveSessionGovernor(cfg)
    ts = 1_700_000_000.0
    gov.record_edges(
        "QQQ",
        session_bar=10,
        call_edge=0.05,
        put_edge=0.04,
        dual_mode=True,
        curr_ts=ts,
    )
    before = len(gov._state("QQQ").edge_buf or [])
    cool_until = gov.record_trade_close(
        "QQQ",
        net_ret=-0.19,
        curr_ts=ts,
        leg="CALL",
        reason="QQQ_BTC_TICK_FAST_HARD",
    )
    assert cool_until == ts + 30 * 60
    blocked, reason = gov.blocked_for_entry("QQQ", curr_ts=ts + 60)
    assert blocked and reason == "tick_stop_cooldown"
    blocked, reason = gov.blocked_for_entry("QQQ", curr_ts=cool_until)
    assert blocked and reason == "tick_stop_cooldown"
    blocked, _ = gov.blocked_for_entry("QQQ", curr_ts=cool_until + 1)
    assert not blocked
    assert gov.leg_blocked_for_entry("QQQ", leg="CALL", curr_ts=ts + 31 * 60)
    assert not gov.leg_blocked_for_entry("QQQ", leg="PUT", curr_ts=ts + 31 * 60)

    gov.record_edges(
        "QQQ",
        session_bar=11,
        call_edge=0.10,
        put_edge=0.09,
        dual_mode=True,
        curr_ts=ts + 60,
    )
    assert len(gov._state("QQQ").edge_buf or []) == before


def test_live_vixy_put_gate_uses_previous_minute():
    from qqq_btc.common.vix_level import VixyCloseBuffer
    from qqq_btc.live.signal_integration import append_vixy_and_get_lagged_gate

    buf = VixyCloseBuffer()
    buf.extend([100.0 + i * 0.1 for i in range(25)])
    expected = buf.gate_level()
    got = append_vixy_and_get_lagged_gate(buf, 110.0)
    assert got == expected
    assert len(buf) == 26
    assert buf.gate_level() != expected


def test_live_entry_quantile_raises_threshold():
    import sys
    from datetime import datetime
    from pathlib import Path

    from pytz import timezone

    from qqq_btc.common.replay_types import ReplayConfig
    from qqq_btc.live.session_governor import get_session_governor, _GOVERNORS

    _GOVERNORS.clear()
    cfg = ReplayConfig(
        long_only=False,
        entry_quantile=0.80,
        entry_quantile_window=100,
        entry_quantile_min_obs=5,
        entry_threshold_schedule=((0, 0.03),),
        put_gate_min=None,
        call_trend_r2_min=None,
        regime_vix_reversal_max=None,
    )

    baseline = Path(__file__).resolve().parents[2] / "New_Pro" / "baseline_qqq"
    if str(baseline) not in sys.path:
        sys.path.insert(0, str(baseline))
    import baseline_paths  # noqa: E402,F401

    from strategy.core_v0 import StrategyCoreV0
    from strategy.config0 import StrategyConfig
    from qqq_btc.live.strategy_entry_bridge import apply_strategy_entry_patch
    import qqq_btc.qqq.config as qcfg

    monkeypatch_cfg = cfg
    old_replay = qcfg.REPLAY
    qcfg.REPLAY = monkeypatch_cfg
    try:
        apply_strategy_entry_patch(StrategyCoreV0)
        core = StrategyCoreV0(StrategyConfig())
        ny = timezone("America/New_York")
        base = {
            "is_ready": True,
            "is_banned": False,
            "position": 0,
            "cooldown_until": 0.0,
            "curr_ts": 1_700_000_000.0,
            "time": datetime(2026, 6, 2, 10, 30, tzinfo=ny),
            "net_edge_raw": 0.035,
            "call_edge": 0.035,
            "put_edge": 0.01,
            "alpha_z": 0.035,
            "vol_z": 0.5,
            "bid": 1.98,
            "ask": 2.02,
            "curr_price": 2.0,
            "options_vw_spread": 0.02,
            "options_iv_momentum": 0.0,
            "symbol": "QQQ",
            "spy_roc": 0.001,
            "qqq_roc": 0.001,
            "spread_divergence": 0.0,
        }
        gov = get_session_governor(monkeypatch_cfg)
        for _ in range(6):
            gov.record_edges("QQQ", session_bar=60, call_edge=0.05, put_edge=0.01, dual_mode=True)
        assert gov.dynamic_thresholds("QQQ")[0] is not None

        assert core.decide_entry(dict(base)) is None
        base["call_edge"] = 0.06
        base["net_edge_raw"] = 0.06
        sig = core.decide_entry(dict(base))
        assert sig is not None
    finally:
        qcfg.REPLAY = old_replay
        _GOVERNORS.clear()


def test_live_session_governor_vol_scaled_rails():
    import numpy as np

    from qqq_btc.common.replay_types import ReplayConfig
    from qqq_btc.live.session_governor import LiveSessionGovernor
    from qqq_btc.qqq import config as qcfg

    gov = LiveSessionGovernor(ReplayConfig(entry_quantile=None))
    ts = 1_700_000_000.0
    base = qcfg.EXIT_RAILS
    rails0, s0 = gov.scaled_exit_rails("QQQ", base)
    assert s0 == 1.0
    assert rails0.ladder == base.ladder

    # 高波动 minute returns → scale > 1(vol_scale_min=1.0 下只有 std/ref>1 才放大)
    px = 2.0
    for i in range(40):
        px *= 1.10 if i % 2 == 0 else 0.90
        gov.record_minute_mid("QQQ", float(px), ts + i * 60)
    rails1, s1 = gov.scaled_exit_rails("QQQ", base)
    assert s1 > 1.0
    assert rails1.ladder[0][0] > base.ladder[0][0]
    assert rails1.hard_stop_roi == base.hard_stop_roi  # profit_only


def test_signal_audit_writer(tmp_path, monkeypatch):
    import os
    from datetime import datetime

    from pytz import timezone

    monkeypatch.setenv("QQQ_BTC_LIVE", "1")
    monkeypatch.setenv("QQQ_BTC_SIGNAL_AUDIT_DIR", str(tmp_path))

    from qqq_btc.common.entry_decision import EntryDecision
    from qqq_btc.common.signal_collect import load_dry_run_signals
    from qqq_btc.live.signal_audit_writer import record_entry_signal_audit

    ny = timezone("America/New_York")
    ts = datetime(2026, 6, 2, 10, 30, tzinfo=ny).timestamp()
    ctx = {
        "symbol": "QQQ",
        "curr_ts": ts,
        "call_edge": 0.04,
        "put_edge": 0.01,
        "net_edge_raw": 0.04,
        "net_edge_q10": 0.005,
        "bid": 1.98,
        "ask": 2.02,
        "curr_price": 2.0,
    }
    row = record_entry_signal_audit(
        ctx=ctx,
        decision=EntryDecision(leg="CALL", edge=0.04, threshold=0.03),
        session_bar=60,
    )
    assert row is not None
    assert row["decision"] == "PASS"

    path = tmp_path / "signals_2026-06-02.csv"
    assert path.exists()
    loaded = load_dry_run_signals(path)
    assert len(loaded) == 1
    assert loaded.iloc[0]["leg"] == "CALL"

    os.environ.pop("QQQ_BTC_LIVE", None)


def test_signal_decision_replay_vs_live_match():
    from qqq_btc.common.signal_collect import collect_decision_signals, diff_signal_frames
    from qqq_btc.common.replay_harness import run_strict_replay
    from qqq_btc.qqq import config as qcfg

    fm = qcfg.FILL_MODEL
    df = _make_option_df(n=120, drift=0.02)
    df["net_edge"] = 0.05
    df["call_net_edge"] = 0.05
    df["put_net_edge"] = 0.02
    df["net_edge_q10"] = 0.01
    df["vix_level"] = 0.3

    day = str(pd.Timestamp(df.iloc[60]["timestamp"]).date())
    r = collect_decision_signals(df.iloc[:80], warmup_through_day=day, target_day=day, replay_cfg=qcfg.REPLAY)
    l = collect_decision_signals(df.iloc[:80], warmup_through_day=day, target_day=day, replay_cfg=qcfg.LIVE_REPLAY)
    d = diff_signal_frames(r, l, time_tolerance_bars=0)
    assert d["summary"]["n_matched"] == d["summary"]["n_replay"]
    assert d["summary"]["n_matched"] == d["summary"]["n_live"]

    from qqq_btc.common.signal_collect import first_entry_diff

    off = pd.DataFrame([{"session_bar": 17, "leg": "CALL", "edge": 0.14, "ts": "a"}])
    st_ok = pd.DataFrame([{"session_bar": 17, "leg": "CALL", "edge": 0.15, "ts": "b"}])
    st_miss = pd.DataFrame([{"session_bar": 39, "leg": "CALL", "edge": 0.16, "ts": "b"}])
    assert first_entry_diff(off, st_ok, time_tolerance_bars=0)["summary"]["n_matched"] == 1
    assert first_entry_diff(off, st_miss, time_tolerance_bars=0)["summary"]["n_matched"] == 0
    assert first_entry_diff(off, st_miss, time_tolerance_bars=30)["summary"]["n_matched"] == 1

    # event replay 仍返回 events
    result = run_strict_replay(df.iloc[:80], fm, qcfg.REPLAY, qcfg.EXIT_RAILS)
    assert isinstance(result.events, list)


def test_bootstrap_gate_convergence_env():
    import os
    os.environ["QQQ_BTC_LIVE"] = "1"
    for k in ("FAST_GATE_ENABLED", "COOLDOWN_MINUTES", "BIDIRECTIONAL_ENABLED", "BIDIRECTIONAL_DUAL_EDGE_ENABLED"):
        os.environ.pop(k, None)
    from qqq_btc.live.bootstrap import bootstrap_qqq_btc_live

    bootstrap_qqq_btc_live(patch_oms=False)
    assert os.environ.get("FAST_GATE_ENABLED") == "0"
    assert os.environ.get("COOLDOWN_MINUTES") == "5"
    assert os.environ.get("EXECUTION_DELAY_BARS") == "0"
    assert os.environ.get("OMS_SIGNAL_DELAY_BARS") == "0"
    assert os.environ.get("BIDIRECTIONAL_ENABLED") == "1"
    assert os.environ.get("BIDIRECTIONAL_DUAL_EDGE_ENABLED") == "1"
    os.environ.pop("QQQ_BTC_LIVE", None)


def test_fill_audit_writer(tmp_path, monkeypatch):
    import os
    from qqq_btc.tools.parity_audit import audit_fill, audit_exit_reasons

    monkeypatch.setenv("QQQ_BTC_LIVE", "1")
    monkeypatch.setenv("QQQ_BTC_FILL_AUDIT_PATH", str(tmp_path / "fill_audit.csv"))

    from qqq_btc.live.fill_audit_writer import record_fill_audit

    record_fill_audit(
        symbol="QQQ",
        side="BUY",
        fill_px=2.011,
        bid=1.98,
        ask=2.02,
        qty=1,
        action="OPEN",
        ts=1_700_000_000.0,
        reason="CH_TREND",
    )
    record_fill_audit(
        symbol="QQQ",
        side="SELL",
        fill_px=1.989,
        bid=1.98,
        ask=2.02,
        qty=1,
        action="CLOSE",
        ts=1_700_000_300.0,
        exit_reason="QQQ_BTC_STEP_PROTECT",
    )
    log = tmp_path / "fill_audit.csv"
    assert log.exists()
    rep = audit_fill(log, target_frac=0.775, tol=0.05)
    assert rep["n"] == 2
    assert rep["pass"] is True
    ex = audit_exit_reasons(log)
    assert ex["n_close"] == 1
    assert "QQQ_BTC_STEP_PROTECT" in ex["live_distribution"]


def test_exit_lifecycle_reason_normalize_and_first_exit():
    from qqq_btc.common.exit_lifecycle import (
        diff_exit_lifecycle,
        first_exit_diff,
        normalize_exit_reason,
    )

    assert normalize_exit_reason("QQQ_BTC_STEP_PROTECT|NO_QUOTE") == "STEP_PROTECT"
    assert normalize_exit_reason("STEP_PROTECT") == "STEP_PROTECT"
    assert normalize_exit_reason(None) == ""

    off = pd.DataFrame(
        [
            {
                "session_bar": 42,
                "leg": "CALL",
                "reason": "QQQ_BTC_STEP_PROTECT",
                "reason_family": "STEP_PROTECT",
                "net_return": 0.01,
            }
        ]
    )
    live = pd.DataFrame(
        [
            {
                "session_bar": 43,
                "leg": "CALL",
                "reason": "STEP_PROTECT",
                "reason_family": "STEP_PROTECT",
                "net_return": 0.011,
            }
        ]
    )
    fe = first_exit_diff(off, live, time_tolerance_bars=1)
    assert fe["summary"]["pass"] is True
    assert fe["summary"]["bar_delta"] == 1.0
    el = diff_exit_lifecycle(off, live, time_tolerance_bars=1)
    assert el["summary"]["pass"] is True
    assert el["summary"]["n_matched"] == 1
    # both empty → pass
    empty = first_exit_diff(pd.DataFrame(), pd.DataFrame())
    assert empty["summary"]["pass"] is True
    # mismatch reason → fail
    bad = live.copy()
    bad["reason_family"] = "TIME_STOP"
    assert first_exit_diff(off, bad, time_tolerance_bars=1)["summary"]["pass"] is False


def test_load_fill_audit_exits_parses_leg_and_session_bar(tmp_path):
    from qqq_btc.common.exit_lifecycle import load_fill_audit_exits

    # 2026-06-02 10:00 ET ≈ 1748872800? use explicit NY session ts
    # 2026-06-02 09:45 ET = session_bar 15
    ts_close = pd.Timestamp("2026-06-02 10:12:00", tz="America/New_York").timestamp()
    path = tmp_path / "fill_audit_20260602.csv"
    path.write_text(
        "ts,symbol,action,side,qty,fill_px,bid,ask,spread_pct,fill_spread_frac,"
        "model_frac,delta_frac,reason,exit_reason,mode,leg,session_bar,net_return\n"
        f"{ts_close},QQQ,CLOSE,SELL,1,2.0,1.98,2.02,0.02,0.5,0.775,-0.275,,"
        "QQQ_BTC_STEP_PROTECT,REALTIME_DRY,CALL,42,0.01\n",
        encoding="utf-8",
    )
    out = load_fill_audit_exits(path, "2026-06-02")
    assert len(out) == 1
    assert int(out.iloc[0]["session_bar"]) == 42
    assert out.iloc[0]["leg"] == "CALL"
    assert out.iloc[0]["reason_family"] == "STEP_PROTECT"


def test_se_feature_bridge_injects_time_features():
    from qqq_btc.live.se_feature_bridge import inject_qqq_btc_features

    slow_cfg = {
        "features": [
            {"name": "time_session_sin"},
            {"name": "time_session_progress"},
            {"name": "trend_fit_ret_30m"},
        ]
    }
    store = {}
    base_ts = float(pd.Timestamp("2024-06-18 09:30", tz="America/New_York").timestamp())
    batch = {"ts": base_ts, "stock_price": [450.0], "features_dict": {}}
    for k in range(35):
        batch["ts"] = base_ts + k * 60
        batch["stock_price"] = [450.0 + 0.01 * k]
        inject_qqq_btc_features(batch, ["QQQ"], slow_cfg=slow_cfg, history_store=store)

    fd = batch["features_dict"]
    assert "time_session_sin" in fd
    assert fd["time_session_sin"].shape == (1, 30)
    assert np.count_nonzero(fd["time_session_sin"]) > 0


def test_oms_tick_max_roi_revert_pattern():
    """验证 tick cache 后还原 max_roi 的模式(与 oms_integration patch 同语义)。"""
    class _St:
        position = 1
        max_roi = 0.05

    class _Eng:
        states = {"QQQ": _St()}

        def cache(self, pkt):
            self.states["QQQ"].max_roi = 0.25

    eng = _Eng()
    saved = {s: float(st.max_roi) for s, st in eng.states.items()}
    eng.cache({})
    assert eng.states["QQQ"].max_roi == 0.25
    for sym, prev in saved.items():
        eng.states[sym].max_roi = prev
    assert eng.states["QQQ"].max_roi == 0.05

