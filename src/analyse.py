#!/usr/bin/env python3
"""
analyse.py — signatures, contrasts, and the two figures.

Networks:
  U : vertices = the 896 residual-stream units, correlations measured across items
  F : vertices = the items, correlations measured across the 896 dimensions
      (per-dimension centring across items is MANDATORY: without it every pair of
       items correlates > 0.95 at every layer and the curve is flat at 0.95)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pmt import beta1, edges, signature

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
TAUS = np.round(np.linspace(0.0, 0.95, 20), 3)


def corr(X: np.ndarray) -> np.ndarray:
    """Correlation between rows of X."""
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return np.clip((X / n) @ (X / n).T, -1.0, 1.0)


def area_curve(S: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Normalised area, tau_star, and the normalised curve."""
    n = S.shape[0]
    c = np.array([beta1(n, edges(S, t)) for t in TAUS], dtype=float)
    cn = c / (c[0] if c[0] > 0 else 1.0)
    peak = float(c.max())
    if peak <= 0:
        return 0.0, float(TAUS[-1]), cn
    below = np.where(c <= 0.10 * peak)[0]
    return (float(np.trapezoid(cn, TAUS)), float(TAUS[below[0]] if len(below) else TAUS[-1]), cn)


def cell_areas(states: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Area per layer, for each cell. states[cell] has shape (n, layers, hidden)."""
    out = {}
    for cell, X in states.items():
        out[cell] = np.array([area_curve(corr(X[:, l, :].T))[0] for l in range(X.shape[1])])
    return out


def layer_bootstrap(d: np.ndarray, n_boot: int = 4000, seed: int = 20261027):
    """Interval for a per-layer contrast, resampling LAYERS."""
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))


if __name__ == "__main__":
    z = np.load(RES / "pmt_v2_etats.npz", allow_pickle=False)
    states = {c: z[c].astype(np.float32) for c in ("V", "M", "I", "A")}
    A = cell_areas(states)
    for a, b in (("I", "V"), ("M", "V"), ("A", "I")):
        d = A[a] - A[b]
        print(f"{a}-{b}: {layer_bootstrap(d)}")
