#!/usr/bin/env python3
"""生产周更微调一键入口。

规范（见 CONFIG/weekly_finetune_policy.json）:
  - 底座: V4（或 policy.base_checkpoint）
  - 近月微调: 最近 N 月 train / 最近 M 月 val
  - OOS 门禁: 最近一周上候选不得明显差于 baseline，否则不晋升
  - 归档: run 目录写 manifest / status / infer / summary，便于实时查看

实时反馈:
  - 控制台分阶段打印
  - <run_dir>/status.json 每阶段覆盖写入（可另开终端 watch）
  - <run_dir>/pipeline.log 完整日志

用法:
  python qqq_btc/tools/weekly_finetune.py
  python qqq_btc/tools/weekly_finetune.py --train-months 2026-05,2026-06 --val-months 2026-06
  python qqq_btc/tools/weekly_finetune.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = REPO / "qqq_btc/CONFIG/weekly_finetune_policy.json"

logger = logging.getLogger("weekly_finetune")


def _expand(p: str | Path) -> Path:
    return Path(os.path.expanduser(str(p))).resolve()


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_policy(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


class StatusWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.path = run_dir / "status.json"
        self.state: dict[str, Any] = {
            "run_id": run_dir.name,
            "started_at": _now(),
            "updated_at": _now(),
            "stage": "init",
            "pct": 0,
            "ok": True,
            "message": "starting",
            "gate": None,
            "paths": {},
        }

    def update(self, *, stage: str, pct: int, message: str, ok: bool = True, **extra: Any) -> None:
        self.state["updated_at"] = _now()
        self.state["stage"] = stage
        self.state["pct"] = int(pct)
        self.state["message"] = message
        self.state["ok"] = bool(ok)
        for k, v in extra.items():
            self.state[k] = v
        self.path.write_text(json.dumps(self.state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        flag = "OK" if ok else "FAIL"
        logger.info("[%s %3d%%] %s | %s", flag, pct, stage, message)


def list_months(feat_1min_dir: Path) -> list[str]:
    months = sorted(p.stem for p in feat_1min_dir.glob("????-??.parquet"))
    return months


def resolve_windows(
    policy: dict[str, Any],
    *,
    train_months: list[str] | None,
    val_months: list[str] | None,
) -> tuple[list[str], list[str]]:
    feat_root = _expand(policy["paths"]["feature_train_root"])
    available = list_months(feat_root / "1min")
    if not available:
        raise SystemExit(f"no monthly features under {feat_root / '1min'}")

    if train_months:
        train = train_months
    else:
        n = int(policy["windows"]["train_lookback_months"])
        train = available[-n:]
    if val_months:
        val = val_months
    else:
        m = int(policy["windows"]["val_lookback_months"])
        val = available[-m:]

    missing = [ym for ym in train + val if ym not in available]
    if missing:
        raise SystemExit(f"missing feature months {missing}; available={available}")
    return train, val


def setup_feature_links(dest: Path, src_root: Path, months: list[str], symbol: str) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for res in ("1min", "5min"):
        out = dest / symbol / "regular" / "09:30-16:00" / res
        out.mkdir(parents=True, exist_ok=True)
        for ym in months:
            src = src_root / res / f"{ym}.parquet"
            if not src.exists():
                if res == "5min":
                    continue
                raise FileNotFoundError(src)
            out.joinpath(f"{ym}.parquet").symlink_to(src)


def run_cmd(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    logger.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def summarize_replay(summary_path: Path) -> dict[str, Any]:
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "trades": s.get("trades"),
        "total_net_return": s.get("total_net_return"),
        "hit_rate": s.get("hit_rate"),
        "max_drawdown_mtm": s.get("max_drawdown_mtm"),
        "trades_by_leg": s.get("trades_by_leg"),
        "ic": (s.get("label_metrics") or {}).get("ic"),
        "n_rows": s.get("n_rows"),
        "checkpoint": s.get("checkpoint"),
    }


def decide_gate(baseline: dict[str, Any], candidate: dict[str, Any], gate: dict[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    passed = True

    b_ret = float(baseline.get("total_net_return") or 0.0)
    c_ret = float(candidate.get("total_net_return") or 0.0)
    b_dd = float(baseline.get("max_drawdown_mtm") or 0.0)
    c_dd = float(candidate.get("max_drawdown_mtm") or 0.0)
    c_trades = int(candidate.get("trades") or 0)
    c_hit = candidate.get("hit_rate")

    min_trades = int(gate.get("min_trades", 3))
    if c_trades < min_trades:
        passed = False
        reasons.append(f"trades {c_trades} < min_trades {min_trades}")

    if gate.get("require_not_worse_return", True):
        min_delta = float(gate.get("min_return_delta", 0.0))
        if c_ret + 1e-12 < b_ret + min_delta:
            passed = False
            reasons.append(
                f"return {c_ret:+.4f} < baseline {b_ret:+.4f} + delta {min_delta:+.4f}"
            )

    tol = float(gate.get("max_drawdown_worse_tol", 0.05))
    # drawdown is negative; worse means more negative
    if c_dd < b_dd - tol:
        passed = False
        reasons.append(f"drawdown {c_dd:.4f} worse than baseline {b_dd:.4f} by > {tol}")

    hit_floor = float(gate.get("require_hit_rate_floor", 0.0))
    if c_hit is not None and float(c_hit) < hit_floor:
        passed = False
        reasons.append(f"hit_rate {c_hit:.3f} < floor {hit_floor:.3f}")

    if passed:
        reasons.append("candidate passes OOS gate vs baseline")

    return {
        "passed": passed,
        "reasons": reasons,
        "baseline": baseline,
        "candidate": candidate,
        "delta_return": c_ret - b_ret,
        "delta_drawdown": c_dd - b_dd,
    }


def promote_checkpoint(src: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = link_path.with_suffix(".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(src.resolve())
    tmp.replace(link_path)


def main() -> int:
    ap = argparse.ArgumentParser(description="Weekly production finetune with OOS gate")
    ap.add_argument("--policy", default=str(DEFAULT_POLICY))
    ap.add_argument("--train-months", default=None, help="comma months, e.g. 2026-05,2026-06")
    ap.add_argument("--val-months", default=None, help="comma months, e.g. 2026-06")
    ap.add_argument("--base-checkpoint", default=None)
    ap.add_argument("--feature-train-root", default=None)
    ap.add_argument("--feature-oos-root", default=None)
    ap.add_argument("--option-1m-oos-root", default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--skip-train", action="store_true", help="reuse existing ckpt in run_dir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-promote", action="store_true")
    args = ap.parse_args()

    policy = load_policy(Path(args.policy))
    if args.feature_train_root:
        policy["paths"]["feature_train_root"] = args.feature_train_root
    if args.feature_oos_root:
        policy["paths"]["feature_oos_root"] = args.feature_oos_root
    if args.option_1m_oos_root:
        policy["paths"]["option_1m_oos_root"] = args.option_1m_oos_root

    seed = int(policy.get("seed", 42))
    os.environ["QQQ_BTC_SEED"] = str(seed)
    os.environ["PYTHONPATH"] = f"{REPO}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

    train_months = [x.strip() for x in args.train_months.split(",") if x.strip()] if args.train_months else None
    val_months = [x.strip() for x in args.val_months.split(",") if x.strip()] if args.val_months else None
    train_yms, val_yms = resolve_windows(policy, train_months=train_months, val_months=val_months)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    runs_root = _expand(policy["paths"]["runs_root"])
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "pipeline.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )

    status = StatusWriter(run_dir)
    py = os.environ.get("PYTHON", sys.executable)
    symbol = policy.get("symbol", "QQQ")
    config = str(_expand(policy["config"]) if not Path(policy["config"]).is_absolute() else policy["config"])
    if not Path(config).is_absolute():
        config = str(REPO / policy["config"])
    sym_map = str(REPO / policy["symbol_map"])
    base_ckpt = _expand(args.base_checkpoint or policy["base_checkpoint"])
    if not base_ckpt.is_absolute():
        base_ckpt = (REPO / (args.base_checkpoint or policy["base_checkpoint"])).resolve()

    feat_train_root = _expand(policy["paths"]["feature_train_root"])
    feat_oos_root = _expand(policy["paths"]["feature_oos_root"])
    opt_oos_root = _expand(policy["paths"]["option_1m_oos_root"])
    lmdb_root = _expand(policy["paths"]["lmdb_root"])
    results_root = REPO / policy["paths"]["results_root"] / run_id
    results_root.mkdir(parents=True, exist_ok=True)

    ckpt_out = run_dir / "checkpoint"
    eval_base = results_root / "baseline"
    eval_ft = results_root / "candidate"
    train_feat = run_dir / "features_train"
    val_feat = run_dir / "features_val"

    status.update(
        stage="resolve",
        pct=5,
        message=f"train={train_yms} val={val_yms} seed={seed}",
        paths={
            "run_dir": str(run_dir),
            "status": str(status.path),
            "log": str(log_path),
            "base_checkpoint": str(base_ckpt),
            "feature_train_root": str(feat_train_root),
            "feature_oos_root": str(feat_oos_root),
            "results_root": str(results_root),
        },
        windows={"train_months": train_yms, "val_months": val_yms},
    )

    # preflight
    oos_1min = feat_oos_root / symbol / "regular" / "09:30-16:00" / "1min"
    if not any(oos_1min.glob("*.parquet")):
        status.update(stage="preflight", pct=5, message=f"missing OOS features under {oos_1min}", ok=False)
        return 2
    if not (opt_oos_root / symbol).exists():
        status.update(stage="preflight", pct=5, message=f"missing OOS option 1m {opt_oos_root / symbol}", ok=False)
        return 2
    if not base_ckpt.exists():
        status.update(stage="preflight", pct=5, message=f"missing base checkpoint {base_ckpt}", ok=False)
        return 2

    manifest = {
        "policy": policy.get("name"),
        "run_id": run_id,
        "started_at": _now(),
        "seed": seed,
        "code_head": subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True).strip(),
        "base_checkpoint": str(base_ckpt),
        "train_months": train_yms,
        "val_months": val_yms,
        "feature_train_root": str(feat_train_root),
        "feature_oos_root": str(feat_oos_root),
        "option_1m_oos_root": str(opt_oos_root),
        "note": "底座保留；仅 OOS 门禁通过才晋升 production link",
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if args.dry_run:
        status.update(stage="dry_run", pct=100, message="dry-run ok; no train/eval", manifest=manifest)
        print(json.dumps(manifest, indent=2, ensure_ascii=False))
        return 0

    status.update(stage="link_features", pct=10, message="link train/val monthly features")
    setup_feature_links(train_feat, feat_train_root, train_yms, symbol)
    setup_feature_links(val_feat, feat_train_root, val_yms, symbol)

    train_lmdb_name = f"train_qqq_weekly_{run_id}.lmdb"
    val_lmdb_name = f"val_qqq_weekly_{run_id}.lmdb"
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["QQQ_BTC_SEED"] = str(seed)

    status.update(stage="build_lmdb", pct=20, message="build train/val LMDB")
    run_cmd(
        [
            py, "qqq_btc/tools/build_lmdb.py",
            "--feature-root", str(train_feat),
            "--config", config,
            "--symbol-map", sym_map,
            "--output", str(lmdb_root / train_lmdb_name),
            "--symbols", symbol,
            "--window-step", str(policy["train"]["window_step"]),
        ],
        cwd=REPO,
        env=env,
    )
    run_cmd(
        [
            py, "qqq_btc/tools/build_lmdb.py",
            "--feature-root", str(val_feat),
            "--config", config,
            "--symbol-map", sym_map,
            "--output", str(lmdb_root / val_lmdb_name),
            "--symbols", symbol,
            "--window-step", str(policy["train"]["window_step"]),
        ],
        cwd=REPO,
        env=env,
    )

    cand_ckpt = ckpt_out / "best.pth"
    if args.skip_train and cand_ckpt.exists():
        status.update(stage="finetune", pct=55, message=f"skip-train reuse {cand_ckpt}")
    else:
        status.update(stage="finetune", pct=35, message=f"finetune from {base_ckpt.name}")
        if ckpt_out.exists():
            shutil.rmtree(ckpt_out)
        ckpt_out.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [
                py, "-m", "qqq_btc.model.train",
                "--mode", "finetune",
                "--config", config,
                "--data-root", str(lmdb_root),
                "--train-lmdb", train_lmdb_name,
                "--val-lmdbs", val_lmdb_name,
                "--checkpoint-dir", str(ckpt_out),
                "--init-checkpoint", str(base_ckpt),
                "--epochs", str(policy["train"]["epochs"]),
                "--batch-size", str(policy["train"]["batch_size"]),
                "--num-workers", str(policy["train"]["num_workers"]),
                "--seed", str(seed),
                "--device", str(policy.get("device", "cuda")),
            ],
            cwd=REPO,
            env=env,
        )
        if not cand_ckpt.exists():
            status.update(stage="finetune", pct=55, message="best.pth missing after train", ok=False)
            return 3

    status.update(stage="eval_baseline", pct=65, message="OOS replay baseline")
    run_cmd(
        [
            py, "qqq_btc/tools/eval_test_set.py",
            "--checkpoint", str(base_ckpt),
            "--config", config,
            "--feature-root", str(feat_oos_root),
            "--option-1m-root", str(opt_oos_root),
            "--output-dir", str(eval_base),
            "--seed", str(seed),
            "--device", str(policy.get("device", "cuda")),
        ],
        cwd=REPO,
        env=env,
    )

    status.update(stage="eval_candidate", pct=80, message="OOS replay candidate")
    run_cmd(
        [
            py, "qqq_btc/tools/eval_test_set.py",
            "--checkpoint", str(cand_ckpt),
            "--config", config,
            "--feature-root", str(feat_oos_root),
            "--option-1m-root", str(opt_oos_root),
            "--output-dir", str(eval_ft),
            "--seed", str(seed),
            "--device", str(policy.get("device", "cuda")),
        ],
        cwd=REPO,
        env=env,
    )

    baseline = summarize_replay(eval_base / "replay_summary.json")
    candidate = summarize_replay(eval_ft / "replay_summary.json")
    gate = decide_gate(baseline, candidate, policy.get("gate", {}))

    promoted = False
    prod_link = REPO / policy.get("production_checkpoint_link", "checkpoint/checkpoints_qqq_prod/best.pth")
    if gate["passed"] and policy.get("gate", {}).get("promote_on_pass", True) and not args.no_promote:
        promote_checkpoint(cand_ckpt, prod_link)
        promoted = True

    summary = {
        "run_id": run_id,
        "finished_at": _now(),
        "seed": seed,
        "windows": {"train_months": train_yms, "val_months": val_yms},
        "base_checkpoint": str(base_ckpt),
        "candidate_checkpoint": str(cand_ckpt),
        "production_link": str(prod_link),
        "promoted": promoted,
        "gate": gate,
        "baseline": baseline,
        "candidate": candidate,
        "paths": {
            "run_dir": str(run_dir),
            "status": str(status.path),
            "log": str(log_path),
            "results": str(results_root),
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (results_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # archive key artifacts into run_dir for one-stop feedback
    if policy.get("archive", {}).get("copy_replay_summary", True):
        shutil.copy2(eval_base / "replay_summary.json", run_dir / "baseline_replay_summary.json")
        shutil.copy2(eval_ft / "replay_summary.json", run_dir / "candidate_replay_summary.json")

    msg = (
        f"gate={'PASS' if gate['passed'] else 'REJECT'} "
        f"base_ret={baseline['total_net_return']:+.2%} "
        f"cand_ret={candidate['total_net_return']:+.2%} "
        f"promoted={promoted}"
    )
    status.update(
        stage="done",
        pct=100,
        message=msg,
        ok=True,  # 流水线跑完即为成功；是否晋升看 gate.passed
        gate=gate,
        summary=summary,
        promoted=promoted,
    )

    print("\n=== WEEKLY FINETUNE RESULT ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nstatus: {status.path}")
    print(f"log:    {log_path}")
    print(f"summary:{run_dir / 'summary.json'}")
    return 0 if gate["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
