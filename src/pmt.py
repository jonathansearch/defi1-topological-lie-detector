"""
pmt.py — the PMT instrument.

beta1 = E - V + C   (cyclomatic rank, exact in dimension 1, O(E) time)

No ripser, no simplex tree, no persistence library. Sweeping tau gives the
whole barcode for free: beta1(tau) *is* the barcode.

Why this works: for a graph (a 1-dimensional simplicial complex), the first
Betti number equals the number of independent cycles, and Euler's formula
gives it exactly as  E - V + components.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 1. graph
def similarities(A: np.ndarray) -> np.ndarray:
    """Correlation matrix between rows of A, shape (n_units, n_observations)."""
    A = np.asarray(A, dtype=float)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    C = (A / n) @ (A / n).T
    return np.clip(C, -1.0, 1.0)


def edges(S: np.ndarray, tau: float) -> np.ndarray:
    """Edges {u,v} with sim(u,v) >= tau (strict upper triangle)."""
    iu = np.triu_indices_from(S, k=1)
    return np.column_stack([iu[0][S[iu] >= tau], iu[1][S[iu] >= tau]])


# ---------------------------------------------------------------- 2. beta1
def components(n: int, E: np.ndarray) -> int:
    """Number of connected components (union-find)."""
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    c = n
    for u, v in E:
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
            c -= 1
    return c


def beta1(n_vertices: int, E: np.ndarray) -> int:
    """beta1 = E - V + C. The number of independent loops. That's all."""
    if len(E) == 0:
        return 0
    return len(E) - n_vertices + components(n_vertices, E)


def beta0(n_vertices: int, E: np.ndarray) -> int:
    return n_vertices if len(E) == 0 else components(n_vertices, E)


# ---------------------------------------------------------------- 3. signatures
def beta1_curve(S: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """beta1(tau): the whole barcode as one curve, for free."""
    n = S.shape[0]
    return np.array([beta1(n, edges(S, t)) for t in taus], dtype=float)


def signature(S: np.ndarray, taus: np.ndarray) -> dict:
    """Four numbers per graph: area, peak, tau_star, beta1(0)."""
    c = beta1_curve(S, taus)
    imax = float(c.max()) if c.size else 0.0
    area = float(np.trapezoid(c, taus) if hasattr(np, "trapezoid") else np.trapz(c, taus))
    if imax <= 0:
        return {"beta1": c, "beta1_max": 0.0, "area": 0.0, "tau_star": float(taus[-1])}
    below = np.where(c <= 0.10 * imax)[0]
    return {"beta1": c, "beta1_max": imax, "area": area,
            "tau_star": float(taus[below[0]] if len(below) else taus[-1])}
