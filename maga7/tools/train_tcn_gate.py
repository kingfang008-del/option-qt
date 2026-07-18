#!/usr/bin/env python3
"""Train TinyTCN gate on ``build_tcn_gate_dataset.py`` parquet.

Walk-forward friendly: pass --train-end / --valid-start to hold out regimes.
Writes a checkpoint loadable by ``maga7.common.tcn_gate.load_tcn_gate``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maga7.common.tcn_gate import TinyTCN, save_tcn_checkpoint


def _load_xy(path: Path, meta: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_parquet(path)
    y = df["label"].to_numpy(dtype=np.float32)
    feat_cols = [c for c in df.columns if c.startswith("f")]
    X_flat = df[feat_cols].to_numpy(dtype=np.float32)
    T, C = int(meta["feature_shape"][0]), int(meta["feature_shape"][1])
    X = X_flat.reshape(-1, T, C)
    return X, y, df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="maga7/results/tcn_gate/dataset_rule_a.parquet")
    ap.add_argument("--out", default="maga7/results/tcn_gate/tcn_gate_v1.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument(
        "--pos-weight",
        default="auto",
        help="BCE pos_weight: auto = n_neg/n_pos on train, or a float, or off",
    )
    ap.add_argument("--train-end", default=None, help="inclusive date for train split")
    ap.add_argument("--valid-start", default=None, help="inclusive date for valid split")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds_path = Path(args.dataset)
    meta = json.loads(ds_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    X, y, df = _load_xy(ds_path, meta)
    dates = df["date"].astype(str)

    if args.train_end and args.valid_start:
        tr = dates <= str(args.train_end)
        va = dates >= str(args.valid_start)
    else:
        # default: last 20% by date as valid
        uniq = sorted(dates.unique())
        cut = uniq[max(0, int(len(uniq) * 0.8) - 1)]
        tr = dates <= cut
        va = dates > cut

    if int(tr.sum()) < 50 or int(va.sum()) < 20:
        raise SystemExit(f"split too small train={tr.sum()} valid={va.sum()}")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyTCN(
        n_channels=int(meta["n_channels"]),
        hidden=args.hidden,
        n_layers=3,
    )
    # move modules
    model.net.to(device)
    model.head.to(device)

    def _loader(mask: pd.Series, shuffle: bool) -> DataLoader:
        xt = torch.as_tensor(X[mask.to_numpy()], dtype=torch.float32)
        yt = torch.as_tensor(y[mask.to_numpy()], dtype=torch.float32)
        return DataLoader(TensorDataset(xt, yt), batch_size=args.batch_size, shuffle=shuffle)

    # channel-wise train norm (saved into checkpoint for inference parity)
    X_tr = X[tr.to_numpy()]
    x_mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0).astype(np.float32)
    x_std = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0).astype(np.float32)
    x_std = np.maximum(x_std, 1e-6)
    model.set_norm(x_mean, x_std)

    opt = torch.optim.Adam(
        list(model.net.parameters()) + list(model.head.parameters()), lr=args.lr
    )
    y_tr = y[tr.to_numpy()]
    n_pos = float((y_tr > 0.5).sum())
    n_neg = float((y_tr <= 0.5).sum())
    pw_arg = str(args.pos_weight).strip().lower()
    if pw_arg in {"", "off", "none", "1", "1.0"}:
        pos_weight = None
    elif pw_arg == "auto":
        pos_weight = torch.tensor([max(n_neg / max(n_pos, 1.0), 1.0)], device=device)
    else:
        pos_weight = torch.tensor([float(args.pos_weight)], device=device)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(
        {
            "train_n": int(tr.sum()),
            "valid_n": int(va.sum()),
            "train_pos_rate": float(n_pos / max(n_pos + n_neg, 1)),
            "pos_weight": None if pos_weight is None else float(pos_weight.item()),
            "channels": meta.get("channels"),
            "label_mode": meta.get("label_mode"),
        },
        flush=True,
    )
    tr_loader = _loader(tr, True)
    va_loader = _loader(va, False)

    def _auc(y_true: np.ndarray, scores: np.ndarray) -> float:
        y_true = y_true.astype(np.float64)
        scores = scores.astype(np.float64)
        pos = scores[y_true > 0.5]
        neg = scores[y_true <= 0.5]
        if len(pos) == 0 or len(neg) == 0:
            return float("nan")
        # Mann–Whitney AUC
        return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())

    best_auc = -1.0
    best_state = None
    history = []
    for ep in range(1, args.epochs + 1):
        model.net.train()
        model.head.train()
        tr_loss = 0.0
        n_tr = 0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model.forward(xb)
            loss = bce(logit, yb)
            loss.backward()
            opt.step()
            tr_loss += float(loss.item()) * len(yb)
            n_tr += len(yb)
        model.net.eval()
        model.head.eval()
        va_loss = 0.0
        n_va = 0
        correct = 0
        logits_all = []
        y_all = []
        with torch.no_grad():
            for xb, yb in va_loader:
                xb, yb = xb.to(device), yb.to(device)
                logit = model.forward(xb)
                loss = bce(logit, yb)
                va_loss += float(loss.item()) * len(yb)
                n_va += len(yb)
                prob = torch.sigmoid(logit)
                pred = (prob >= 0.5).float()
                correct += int((pred == yb).sum().item())
                logits_all.append(prob.detach().cpu().numpy())
                y_all.append(yb.detach().cpu().numpy())
        p_va = np.concatenate(logits_all) if logits_all else np.array([])
        y_va = np.concatenate(y_all) if y_all else np.array([])
        va_auc = _auc(y_va, p_va)
        row = {
            "epoch": ep,
            "train_loss": tr_loss / max(n_tr, 1),
            "valid_loss": va_loss / max(n_va, 1),
            "valid_acc": correct / max(n_va, 1),
            "valid_auc": va_auc,
            "p_mean": float(p_va.mean()) if len(p_va) else None,
            "p_std": float(p_va.std()) if len(p_va) else None,
        }
        history.append(row)
        print(row, flush=True)
        score = va_auc if np.isfinite(va_auc) else -1.0
        if score > best_auc:
            best_auc = score
            best_state = {
                "net": {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in model.head.state_dict().items()},
            }

    if best_state is not None:
        model.net.load_state_dict(best_state["net"])
        model.head.load_state_dict(best_state["head"])

    out = save_tcn_checkpoint(
        model,
        args.out,
        channels=meta["channels"],
        window=int(meta["window"]),
        meta={
            "dataset": str(ds_path),
            "train_end": args.train_end,
            "valid_start": args.valid_start,
            "history": history,
            "best_valid_auc": best_auc,
            "label_mode": meta.get("label_mode"),
        },
    )
    print(f"wrote {out} best_valid_auc={best_auc:.4f}", flush=True)


if __name__ == "__main__":
    main()
