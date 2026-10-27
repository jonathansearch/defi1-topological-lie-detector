#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMT v1 — Protocole du Mensonge Topologique.
Instrument de mesure des signatures. Tourne sur un telephone.

IDEE CLE (la liberation) :
    sur un GRAPHE, beta1 = E - V + C   (rang du cycle)
    -> pas besoin de persistance, pas besoin de 2-simplexes, pas besoin de ripser
    -> O(E), instantane, frugal

    et bonus : beta1(tau) quand tau monte EST le barcode, gratuitement.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------- 1. graphe
def similarites(A: np.ndarray) -> np.ndarray:
    """Matrice de similarite (correlation) entre unites. A : (n_unites, n_tokens)."""
    A = np.asarray(A, dtype=float)
    A = A - A.mean(axis=1, keepdims=True)
    n = np.linalg.norm(A, axis=1, keepdims=True)
    n[n == 0] = 1.0
    C = (A / n) @ (A / n).T
    return np.clip(C, -1.0, 1.0)


def aretes(S: np.ndarray, tau: float) -> np.ndarray:
    """Aretes {u,v} telles que sim(u,v) >= tau (triangle superieur strict)."""
    iu = np.triu_indices_from(S, k=1)
    return np.column_stack([iu[0][S[iu] >= tau], iu[1][S[iu] >= tau]])


# ---------------------------------------------------------------- 2. beta1
def composantes(n: int, E: np.ndarray) -> int:
    """Nombre de composantes connexes (union-find)."""
    parent = list(range(n))

    def trouve(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    c = n
    for u, v in E:
        ru, rv = trouve(int(u)), trouve(int(v))
        if ru != rv:
            parent[ru] = rv
            c -= 1
    return c


def beta1(n_sommets: int, E: np.ndarray) -> int:
    """beta1 = E - V + C. Le nombre de boucles independantes. C'est tout."""
    if len(E) == 0:
        return 0
    return len(E) - n_sommets + composantes(n_sommets, E)


def beta0(n_sommets: int, E: np.ndarray) -> int:
    return n_sommets if len(E) == 0 else composantes(n_sommets, E)


# ---------------------------------------------------------------- 3. signatures
def courbe_beta1(S: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """beta1(tau) : la courbe qui contient TOUT le barcode, pour pas un rond."""
    n = S.shape[0]
    return np.array([beta1(n, aretes(S, t)) for t in taus])


def signature(A: np.ndarray, taus: np.ndarray) -> dict:
    """Les 3 invariants du PMT : beta0, beta1, energie -- sous forme de courbes."""
    S = similarites(A)
    c1 = courbe_beta1(S, taus)
    n = S.shape[0]
    c0 = np.array([beta0(n, aretes(S, t)) for t in taus])
    energie = float(np.var(A))
    return {
        "beta0": c0,
        "beta1": c1,
        "energie": energie,
        "aire_beta1": float(np.trapezoid(c1, taus)) if hasattr(np, "trapezoid") else float(np.trapz(c1, taus)),
        "beta1_max": int(c1.max()),
        "tau_pic": float(taus[int(np.argmax(c1))]),
    }
