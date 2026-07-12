"""离线 strict replay 与 FCS live 共用的日冻结归一化。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FAT_TAIL_FEATURES = frozenset(
    {"options_iv_momentum", "options_gamma_accel", "options_iv_divergence"}
)


@dataclass(frozen=True)
class FrozenNormState:
    feature_names: tuple[str, ...]
    mean: np.ndarray
    std: np.ndarray
    categorical_mask: np.ndarray
    fat_tail_mask: np.ndarray
    use_tanh: bool

    @classmethod
    def from_npz(cls, path: Path | str) -> "FrozenNormState":
        blob = np.load(Path(path).expanduser(), allow_pickle=True)
        names = tuple(str(x) for x in blob["feature_names"].tolist())
        mean = np.asarray(blob["mean"], dtype=np.float32)
        std = np.asarray(blob["std"], dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
        if "categorical_mask" in blob.files:
            cat = np.asarray(blob["categorical_mask"], dtype=bool)
        else:
            cat = np.zeros(len(names), dtype=bool)
        fat_tail_mask = np.array([n in FAT_TAIL_FEATURES for n in names], dtype=bool)
        use_tanh = bool(int(blob["use_tanh"])) if "use_tanh" in blob.files else True
        return cls(
            feature_names=names,
            mean=mean,
            std=std,
            categorical_mask=cat,
            fat_tail_mask=fat_tail_mask,
            use_tanh=use_tanh,
        )


def normalize_raw_vector(raw: np.ndarray, state: FrozenNormState) -> np.ndarray:
    """与 FCS RollingWindowNormalizer.normalize_only 同口径(冻结 mean/std)。"""
    x = np.asarray(raw, dtype=np.float32).copy()
    if not np.isfinite(x).all():
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if state.fat_tail_mask.any():
        target = x[state.fat_tail_mask]
        x[state.fat_tail_mask] = np.sign(target) * np.log1p(np.abs(target))
    x_norm = (x - state.mean) / (state.std + 1e-6)
    x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=10.0, neginf=-10.0)
    if state.use_tanh:
        real_mask = ~state.categorical_mask
        x_norm[real_mask] = np.tanh(x_norm[real_mask] / 3.0)
    else:
        x_norm = np.clip(x_norm, -10.0, 10.0)
    return x_norm.astype(np.float32, copy=False)


def _normalize_column(raw: np.ndarray, state: FrozenNormState, j: int) -> np.ndarray:
    col = np.asarray(raw, dtype=np.float32)
    if state.categorical_mask[j]:
        return col
    if state.fat_tail_mask[j]:
        col = np.sign(col) * np.log1p(np.abs(col))
    normed = (col - state.mean[j]) / (state.std[j] + 1e-6)
    normed = np.nan_to_num(normed, nan=0.0, posinf=10.0, neginf=-10.0)
    if state.use_tanh:
        normed = np.tanh(normed / 3.0)
    else:
        normed = np.clip(normed, -10.0, 10.0)
    return normed.astype(np.float32, copy=False)


def apply_frozen_norm_df(
    df: pd.DataFrame,
    frozen_norm: Path | str,
    *,
    feature_names: tuple[str, ...] | list[str] | None = None,
) -> pd.DataFrame:
    """将 raw 特征列变换为 frozen z-score(+tanh),供 run_inference / strict replay。"""
    state = FrozenNormState.from_npz(frozen_norm)
    names = tuple(feature_names) if feature_names else state.feature_names
    name_to_idx = {n: i for i, n in enumerate(state.feature_names)}
    out = df.copy()
    for name in names:
        if name not in out.columns:
            continue
        j = name_to_idx.get(name)
        if j is None:
            continue
        raw = pd.to_numeric(out[name], errors="coerce").fillna(0.0).values
        out[name] = _normalize_column(raw, state, j)
    return out
