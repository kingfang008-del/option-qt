"""Raw SVI slice calibration (Gatheral) for offline L1 verification.

w(k) = a + b * (rho*(k-m) + sqrt((k-m)^2 + sigma^2))
k = log(K/F), w = iv^2 * T
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def as_dict(self) -> dict[str, float]:
        return {
            "a": float(self.a),
            "b": float(self.b),
            "rho": float(self.rho),
            "m": float(self.m),
            "sigma": float(self.sigma),
        }


def svi_total_var(k: np.ndarray, p: SVIParams) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    x = k - p.m
    return p.a + p.b * (p.rho * x + np.sqrt(x * x + p.sigma * p.sigma))


def svi_iv(k: np.ndarray, p: SVIParams, T: float) -> np.ndarray:
    w = np.maximum(svi_total_var(k, p), 1e-12)
    return np.sqrt(w / max(float(T), 1e-8))


def _pack(theta: np.ndarray) -> SVIParams:
    # unconstrained -> constrained
    a = float(theta[0])
    b = float(np.exp(theta[1]))
    rho = float(np.tanh(theta[2]))
    m = float(theta[3])
    sigma = float(np.exp(theta[4]))
    return SVIParams(a=a, b=b, rho=rho, m=m, sigma=sigma)


def fit_svi_raw(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    *,
    weights: np.ndarray | None = None,
) -> tuple[SVIParams, dict[str, float]]:
    """Weighted least-squares fit of raw SVI on one expiry slice."""
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    m = np.isfinite(k) & np.isfinite(iv) & (iv > 0.01) & (iv < 3.0)
    k, iv = k[m], iv[m]
    if len(k) < 5:
        raise ValueError(f"need >=5 points, got {len(k)}")
    w_mkt = np.maximum(iv * iv * float(T), 1e-12)
    if weights is None:
        wt = np.ones_like(k)
    else:
        wt = np.asarray(weights, dtype=float)[m]
        wt = np.where(np.isfinite(wt) & (wt > 0), wt, 1.0)
    wt = wt / np.mean(wt)

    # init near ATM
    atm = np.argmin(np.abs(k))
    a0 = float(w_mkt[atm])
    x0 = np.array([a0 * 0.5, np.log(0.1), 0.0, 0.0, np.log(0.1)], dtype=float)

    def resid(theta: np.ndarray) -> np.ndarray:
        p = _pack(theta)
        w_model = svi_total_var(k, p)
        # soft no-arb: w>0 and butterfly proxy dw2>=0 via penalty on neg w
        pen = np.maximum(0.0, -w_model) * 10.0
        return np.sqrt(wt) * ((w_model - w_mkt) + pen)

    res = least_squares(resid, x0, method="trf", max_nfev=400)
    p = _pack(res.x)
    iv_hat = svi_iv(k, p, T)
    rmse = float(np.sqrt(np.mean((iv_hat - iv) ** 2)))
    mae = float(np.mean(np.abs(iv_hat - iv)))
    w_hat = svi_total_var(k, p)
    frac_neg_w = float(np.mean(w_hat < 0))
    # dense butterfly check on grid
    kg = np.linspace(float(k.min()) - 0.05, float(k.max()) + 0.05, 80)
    wg = svi_total_var(kg, p)
    # numerical second derivative
    d2 = np.gradient(np.gradient(wg, kg), kg)
    frac_neg_d2 = float(np.mean(d2 < -1e-6))
    metrics = {
        "n": float(len(k)),
        "rmse_iv": rmse,
        "mae_iv": mae,
        "cost": float(res.cost),
        "success": float(res.success),
        "frac_neg_w": frac_neg_w,
        "frac_neg_butterfly": frac_neg_d2,
        "iv_atm_mkt": float(iv[atm]),
        "iv_atm_svi": float(iv_hat[atm]),
    }
    return p, metrics
