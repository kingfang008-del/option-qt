#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""现管线 prefer_primary 多 seed 重训：关闭 cudnn deterministic，IC 达标则停。"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.environ.pop("QQQ_BTC_SEED", None)


def patch_nondeterministic() -> None:
    from qqq_btc.common import seed_utils
    from qqq_btc.model import train as train_mod

    def set_global_seed(seed: int = 42, deterministic: bool = True) -> int:
        import numpy as np
        import torch

        seed = int(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        print(f"[multiseed] seed={seed} deterministic=False cudnn.benchmark=True", flush=True)
        return seed

    seed_utils.set_global_seed = set_global_seed  # type: ignore[assignment]
    train_mod.set_seed = lambda seed=42: set_global_seed(seed, False)  # type: ignore[assignment]


def parse_val_ics(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"\[Val:net_edge->return_fwd\] IC=([-\d.]+)", text)]


def run_one_trial(args: argparse.Namespace, trial: int, seed: int, trial_dir: Path) -> dict:
    from qqq_btc.model.train import run as train_run

    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"

    ns = argparse.Namespace(
        mode=args.mode,
        config=args.config,
        data_root=args.data_root,
        train_lmdb=args.train_lmdb,
        val_lmdbs=args.val_lmdb,
        checkpoint_dir=str(trial_dir),
        init_checkpoint=args.init_checkpoint if args.mode == "finetune" else None,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=1e-4,
        max_lr=5e-4,
        device=args.device,
        seed=seed,
    )

    print(f"\n===== trial {trial}/{args.max_trials} seed={seed} =====", flush=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    train_run(ns)

    text = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    curve = parse_val_ics(text)
    best = max(curve) if curve else float("nan")
    meta = {
        "trial": trial,
        "seed": seed,
        "best_val_ic": best,
        "curve": curve,
        "ckpt": str(trial_dir / "best.pth"),
    }
    (trial_dir / "trial_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pretrain", "finetune"], default="pretrain")
    ap.add_argument("--target-ic", type=float, default=0.20)
    ap.add_argument("--max-trials", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--config", default=str(REPO / "qqq_btc/CONFIG/slow_feature_qqq_v4.json"))
    ap.add_argument("--data-root", default=str(Path.home() / "train_data/lmdb"))
    ap.add_argument("--train-lmdb", default="train_qqq_v4_prefer_primary.lmdb")
    ap.add_argument("--val-lmdb", default="val_qqq_v4_prefer_primary.lmdb")
    ap.add_argument(
        "--init-checkpoint",
        default=str(REPO / "checkpoint/checkpoints_qqq_v4/best.pth"),
    )
    ap.add_argument(
        "--out-root",
        default=str(REPO / "qqq_btc/results/prefer_primary_multiseed"),
    )
    ap.add_argument(
        "--ckpt-root",
        default=str(REPO / "checkpoint/checkpoints_qqq_v4_prefer_primary_multiseed"),
    )
    ap.add_argument(
        "--feat-root",
        default=str(Path.home() / "train_data/builds/0dte_prefer_primary"),
    )
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ckpt_root = Path(args.ckpt_root)
    out_root.mkdir(parents=True, exist_ok=True)
    ckpt_root.mkdir(parents=True, exist_ok=True)

    patch_nondeterministic()
    rng = random.SystemRandom()
    trials: list[dict] = []
    best_meta: dict | None = None

    print(
        f"=== prefer_primary multi-seed {args.mode} "
        f"target_ic={args.target_ic} max_trials={args.max_trials} deterministic=OFF ===",
        flush=True,
    )

    for i in range(1, args.max_trials + 1):
        seed = rng.randint(1, 2_000_000_000)
        trial_dir = ckpt_root / f"trial_{i}_seed_{seed}"
        meta = run_one_trial(args, i, seed, trial_dir)
        trials.append(meta)
        print(f"trial {i} best_val_ic={meta['best_val_ic']:.4f}", flush=True)
        if best_meta is None or meta["best_val_ic"] > best_meta["best_val_ic"]:
            best_meta = meta
            print(f"NEW GLOBAL BEST ic={best_meta['best_val_ic']:.4f} trial={i}", flush=True)
        if meta["best_val_ic"] >= args.target_ic:
            print(f"TARGET HIT trial={i} ic={meta['best_val_ic']:.4f}", flush=True)
            break

    report = {
        "mode": args.mode,
        "target_ic": args.target_ic,
        "deterministic": False,
        "features": "builds/0dte_prefer_primary",
        "train": "2023-03..2025-12",
        "val": "2026-01..03",
        "test": "2026-04..06",
        "trials": trials,
        "best": best_meta,
    }

    if best_meta and Path(best_meta["ckpt"]).exists():
        eval_dir = out_root / "best_test_eval"
        eval_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(REPO / "qqq_btc/tools/eval_test_set.py"),
            "--checkpoint",
            best_meta["ckpt"],
            "--config",
            args.config,
            "--feature-root",
            str(Path(args.feat_root) / "quote_features_test"),
            "--option-1m-root",
            str(Path(args.feat_root) / "raw_1m_prefer_primary"),
            "--output-dir",
            str(eval_dir),
            "--device",
            args.device,
        ]
        print("=== eval best on test Apr-Jun ===", flush=True)
        subprocess.check_call(cmd, cwd=str(REPO), env={**os.environ, "PYTHONPATH": str(REPO)})
        summ = eval_dir / "replay_summary.json"
        if summ.exists():
            s = json.loads(summ.read_text())
            report["best_test_eval"] = {
                "ic": (s.get("label_metrics") or {}).get("ic"),
                "trades": s.get("trades"),
                "acct25": s.get("total_net_return"),
                "hit": s.get("hit_rate"),
            }

    (out_root / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
