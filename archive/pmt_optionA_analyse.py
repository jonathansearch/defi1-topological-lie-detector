#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION A — analyse topologique du mensonge force.

V : le modele lit la VRAIE capitale apres "La capitale de <pays> est"
F : le modele lit une FAUSSE capitale (derangement -> meme multiset)

Capture au dernier token de la reponse = le pas t-1 avant l'emission de la suite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "resultats" / "pmt_optionA.npz"
PNG = ROOT / "resultats" / "pmt_optionA_signatures.png"
RNG = np.random.default_rng(20261027)


def aire(y, x):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def corr(X):
    X = X - X.mean(axis=1, keepdims=True)
    nrm = np.linalg.norm(X, axis=1, keepdims=True); nrm[nrm == 0] = 1.0
    return np.clip((X / nrm) @ (X / nrm).T, -1.0, 1.0)


def beta1_courbe(S, taus):
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    w = S[iu]; o = np.argsort(-w)
    ws, us, vs = w[o], iu[0][o], iu[1][o]
    out = np.empty(len(taus))
    for k, t in enumerate(taus):
        m = int(np.searchsorted(-ws, -t, side="right"))
        if m == 0:
            out[k] = 0.0; continue
        g = csr_matrix((np.ones(m, np.int8), (us[:m], vs[:m])), shape=(n, n))
        c, _ = connected_components(g, directed=False)
        out[k] = m - n + c
    return out


def signature(S, taus):
    c = beta1_courbe(S, taus)
    cn = c / (c[0] if c[0] > 0 else 1.0)
    imax = float(c.max())
    if imax <= 0:
        return 0.0, float(taus[-1]), cn
    ok = np.where(c <= 0.10 * imax)[0]
    return aire(cn, taus), float(taus[ok[0]] if len(ok) else taus[-1]), cn


def main():
    taus = np.round(np.linspace(0.0, 0.95, 20), 3)
    z = np.load(NPZ, allow_pickle=False)
    et = {"V": z["V"].astype(np.float32), "F": z["F"].astype(np.float32)}
    pays = [str(p) for p in z["pays"]]
    vraies, fausses = z["vraies"], z["fausses"]
    n, n_l, _ = et["V"].shape

    print("=" * 92)
    print(f"OPTION A — MENSONGE PAR PREFIXE FORCE · {n} capitales sues · {n_l} couches")
    print("=" * 92)

    reseaux = {}
    for nom, f in (("U · 896 unités", lambda X, l: corr(X[:, l, :].T)),
                   ("F · 34 faits", lambda X, l: corr(X[:, l, :]))):
        S = {c: [f(et[c], l) for l in range(n_l)] for c in ("V", "F")}
        A = {c: np.zeros(n_l) for c in ("V", "F")}
        T = {c: np.zeros(n_l) for c in ("V", "F")}
        C = {c: [] for c in ("V", "F")}
        for c in ("V", "F"):
            for l in range(n_l):
                A[c][l], T[c][l], cn = signature(S[c][l], taus)
                C[c].append(cn)
        print(f"\n--- RESEAU {nom} ---")
        print(f"{'couche':>7}{'V':>10}{'F':>10}{'F-V':>10}{'tauV':>8}{'tauF':>8}")
        print("-" * 92)
        for l in range(n_l):
            print(f"{l:>7}{A['V'][l]:>10.3f}{A['F'][l]:>10.3f}{A['F'][l]-A['V'][l]:>10.3f}"
                  f"{T['V'][l]:>8.2f}{T['F'][l]:>8.2f}")
        print("-" * 92)
        print(f"{'MOY':>7}{A['V'].mean():>10.3f}{A['F'].mean():>10.3f}"
              f"{A['F'].mean()-A['V'].mean():>10.3f}{T['V'].mean():>8.2f}{T['F'].mean():>8.2f}")
        d = A["F"] - A["V"]
        print(f"  F>V dans {int((d>0).sum())}/{n_l} couches · "
              f"tau* : V={T['V'].mean():.2f} F={T['F'].mean():.2f} "
              f"(+{(T['F']-T['V']).mean():.2f})")
        reseaux[nom] = (A, T, C, S)

    # ---- incertitude
    print("\n" + "=" * 92)
    print("CONTRASTE F − V  (aire normalisée, moyenne sur les couches)")
    print("=" * 92)
    for nom, (A, T, C, S) in reseaux.items():
        if nom.startswith("U"):
            v = np.array([A["F"][p].mean() - A["V"][p].mean()
                          for p in (RNG.integers(0, n_l, n_l) for _ in range(2000))])
            src = "bootstrap couches, 2 000"
        else:
            v = []
            for _ in range(500):
                p = RNG.integers(0, n, n)
                aF = np.mean([signature(S["F"][l][np.ix_(p, p)], taus)[0] for l in range(n_l)])
                aV = np.mean([signature(S["V"][l][np.ix_(p, p)], taus)[0] for l in range(n_l)])
                v.append(aF - aV)
            v = np.array(v)
            src = "bootstrap faits, 500"
        lo, hi = np.percentile(v, [2.5, 97.5])
        pv = 2 * min((v <= 0).mean(), (v >= 0).mean())
        print(f"  {nom:<16} Δ = {v.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]   p={pv:.4f}   ({src})")
        if nom.startswith("F"):
            tV = [signature(S["V"][l][np.ix_(p, p)], taus)[1] for l in range(n_l)]
            tF = [signature(S["F"][l][np.ix_(p, p)], taus)[1] for l in range(n_l)]
            print(f"  {'':16} tau* : V={np.mean(tV):.2f}  F={np.mean(tF):.2f}  "
                  f"écart {np.mean(tF)-np.mean(tV):+.2f}")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib indisponible :", e); return

    COUL = {"V": "#1f77b4", "F": "#d62728"}
    NOM = {"V": "V  lit la vérité", "F": "F  lit le mensonge (préfixe forcé)"}
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

    for j, (nom, (A, T, C, S)) in enumerate(reseaux.items()):
        for c in ("V", "F"):
            M_ = np.stack(C[c]); m, s = M_.mean(0), M_.std(0)
            ax[j].plot(taus, m, color=COUL[c], lw=2.3, label=NOM[c])
            ax[j].fill_between(taus, m - s, m + s, color=COUL[c], alpha=0.14)
        ax[j].axhline(0.10, color="k", ls=":", lw=1)
        ax[j].set_xlabel(r"seuil $\tau$"); ax[j].set_ylabel(r"$\beta_1(\tau)/\beta_1(0)$")
        ax[j].set_title(f"Réseau {nom}\nQwen2.5-0.5B-Instruct · 34 capitales")
        ax[j].legend(fontsize=9); ax[j].grid(alpha=0.25); ax[j].set_ylim(-0.02, 1.05)

    A, T, C, S = reseaux["U · 896 unités"]
    x = np.arange(n_l)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].bar(x, A["F"] - A["V"], color=["#d62728" if v > 0 else "#4c72b0"
                                         for v in (A["F"] - A["V"])], width=0.75)
    ax[2].set_xlabel("couche"); ax[2].set_ylabel(r"$\Delta$ aire $(F-V)$")
    ax[2].set_title("Le mensonge forcé, couche par couche\n(réseau U)")
    ax[2].grid(alpha=0.25, axis="y")

    fig.tight_layout(); fig.savefig(PNG, dpi=130)
    print(f"\nfigure : {PNG} ({PNG.stat().st_size/1024:.0f} Ko)")

    out = {"n": n, "taus": taus.tolist(),
           "aire": {k: {c: list(map(float, v[c])) for c in ("V", "F")}
                    for k, (v, _, _, _) in reseaux.items()},
           "tau": {k: {c: list(map(float, v[c])) for c in ("V", "F")}
                   for k, (_, v, _, _) in reseaux.items()}}
    (ROOT / "resultats" / "pmt_optionA_analyse.json").write_text(json.dumps(out, indent=1))
    print("export : resultats/pmt_optionA_analyse.json")


if __name__ == "__main__":
    main()
