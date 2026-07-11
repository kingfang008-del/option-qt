#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V8 fixed-8 A/B/C/D 隔离对比实验:

  A: V4 ckpt + V4 dynamic 特征 (基线)
  B: V4 ckpt + V8 fixed-8 特征 (数据影响)
  C: V8 finetune ckpt + V8 fixed-8 特征 (推荐路径)
  D: V8 scratch ckpt + V8 fixed-8 特征 (已知失败对照)

用法:
  python qqq_btc/tools/compare_v8_abcd.py
  python qqq_btc/tools/compare_v8_abcd.py --skip-infer   # 仅汇总已有 infer
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qqq_btc.common.event_replay import prepare_minute_frame
from qqq_btc.common.replay_harness import run_strict_replay
from qqq_btc.qqq import config as qcfg

logger = logging.getLogger("qqq_btc.compare_abcd")

SCENARIOS = {
    "A_v4_on_v4": {
        "checkpoint": "checkpoints_qqq_v4/best.pth",
        "config": "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
        "feature_root": "~/train_data/quote_features_test",
        "option_1m_root": "/mnt/s990/data/raw_1m/options_databento",
        "output_dir": "/tmp/qqq_btc_abcd_A_v4_on_v4",
        "label": "A: V4 ckpt + V4 features",
    },
    "B_v4_on_v8": {
        "checkpoint": "checkpoints_qqq_v4/best.pth",
        "config": "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
        "feature_root": "~/train_data/quote_features_test_fixed8_v8",
        "option_1m_root": "/mnt/s990/data/raw_1m/options_databento_fixed8_corrected",
        "output_dir": "/tmp/qqq_btc_abcd_B_v4_on_v8",
        "label": "B: V4 ckpt + V8 fixed-8 features",
    },
    "C_v8ft_on_v8": {
        "checkpoint": "checkpoints_qqq_v8_fixed8_finetune/best.pth",
        "config": "qqq_btc/CONFIG/slow_feature_qqq_v4.json",
        "feature_root": "~/train_data/quote_features_test_fixed8_v8",
        "option_1m_root": "/mnt/s990/data/raw_1m/options_databento_fixed8_corrected",
        "output_dir": "/tmp/qqq_btc_abcd_C_v8ft_on_v8",
        "label": "C: V8 finetune + V8 features",
    },
    "D_v8scr_on_v8": {
        "checkpoint": "checkpoints_qqq_v8_fixed8/best.pth",
        "config": "qqq_btc/CONFIG/slow_feature_qqq_v2.json",
        "feature_root": "~/train_data/quote_features_test_fixed8_v8",
        "option_1m_root": "/mnt/s990/data/raw_1m/options_databento_fixed8_corrected",
        "output_dir": "/tmp/qqq_btc_abcd_D_v8scr_on_v8",
        "label": "D: V8 scratch + V8 features",
    },
}


def _replay_metrics(infer_path: Path) -> dict:
    df = prepare_minute_frame(pd.read_parquet(infer_path))
    f = qcfg.REPLAY.position_frac
    kw = dict(
        edge_col="net_edge",
        edge_q10_col=qcfg.EDGE_Q10_COL,
        call_edge_col=qcfg.CALL_EDGE_COL,
        put_edge_col=qcfg.PUT_EDGE_COL,
        put_gate_col=qcfg.PUT_GATE_COL,
    )
    r = run_strict_replay(df, qcfg.FILL_MODEL, qcfg.REPLAY, qcfg.EXIT_RAILS, **kw)
    monthly: dict[int, dict] = {}
    for t in r.trades:
        m = pd.to_datetime(t.entry_ts).month
        monthly.setdefault(m, []).append(t.net_return)
    monthly_out = {}
    for m, rets in monthly.items():
        eq = 1.0
        for nr in rets:
            eq *= 1 + f * nr
        monthly_out[int(m)] = {"return_pct": (eq - 1) * 100, "n_trades": len(rets)}
    ne = pd.to_numeric(df["net_edge"], errors="coerce").dropna()
    return {
        "total_return_pct": r.summary()["total_net_return"] * 100,
        "n_trades": len(r.trades),
        "monthly": monthly_out,
        "net_edge_mean": float(ne.mean()) if len(ne) else 0.0,
        "net_edge_std": float(ne.std()) if len(ne) else 0.0,
        "net_edge_ge_003_pct": float((ne >= 0.03).mean() * 100) if len(ne) else 0.0,
    }


def _run_infer(scenario: dict, device: str = "auto") -> Path:
    import subprocess

    ckpt = _REPO / scenario["checkpoint"]
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")

    cmd = [
        sys.executable,
        str(_REPO / "qqq_btc/tools/eval_test_set.py"),
        "--checkpoint", str(ckpt),
        "--config", str(_REPO / scenario["config"]),
        "--feature-root", str(Path(scenario["feature_root"]).expanduser()),
        "--option-1m-root", scenario["option_1m_root"],
        "--output-dir", scenario["output_dir"],
        "--device", device,
    ]
    logger.info("running infer: %s", scenario["label"])
    subprocess.run(cmd, check=True, cwd=str(_REPO))
    return Path(scenario["output_dir"]) / "test_infer.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description="V8 fixed-8 A/B/C/D comparison")
    parser.add_argument("--skip-infer", action="store_true", help="只汇总已有 infer parquet")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = {}
    monthly_rows = []

    for key, sc in SCENARIOS.items():
        infer_path = Path(sc["output_dir"]) / "test_infer.parquet"
        if not args.skip_infer:
            try:
                infer_path = _run_infer(sc, device=args.device)
            except FileNotFoundError as e:
                logger.warning("skip %s: %s", key, e)
                continue
        if not infer_path.exists():
            logger.warning("skip %s: no infer at %s", key, infer_path)
            continue
        m = _replay_metrics(infer_path)
        m["label"] = sc["label"]
        m["infer_path"] = str(infer_path)
        m["checkpoint"] = sc["checkpoint"]
        results[key] = m
        for month, md in m["monthly"].items():
            monthly_rows.append({
                "scenario": key,
                "label": sc["label"],
                "month": month,
                "return_pct": md["return_pct"],
                "n_trades": md["n_trades"],
            })
        logger.info(
            "%s: return=%.2f%% trades=%d edge_mean=%.4f edge>=0.03=%.1f%%",
            key, m["total_return_pct"], m["n_trades"],
            m["net_edge_mean"], m["net_edge_ge_003_pct"],
        )

    out_dir = _REPO / "qqq_btc/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "v8_fixed8_abcd_summary.json"
    monthly_path = out_dir / "v8_fixed8_abcd_monthly.csv"
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(monthly_rows).to_csv(monthly_path, index=False)
    print(f"\nwrote {summary_path}")
    print(f"wrote {monthly_path}")

    if results:
        print("\n=== Q2 Summary ===")
        for key, m in results.items():
            print(
                f"{m['label']}: {m['total_return_pct']:.2f}% | "
                f"trades={m['n_trades']} | edge>=0.03={m['net_edge_ge_003_pct']:.1f}%"
            )


if __name__ == "__main__":
    main()
