#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMT v2 — analyse finale consolidee.

Reseau U : nœuds = 896 unites, corr. mesuree sur les 18 faits
           (centrage : on retire a chaque unite sa moyenne sur les faits)
Reseau F : nœuds = 18 faits, corr. mesuree sur les 896 dimensions
           (centrage : on retire a chaque dimension sa moyenne sur les faits
            -> tue le mode commun, sinon la matrice est saturee a 0,95)

Figure : resultats/pmt_v2_signatures.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "resultats" / "pmt_v2_etats.npz"
OUT = ROOT / "resultats" / "pmt_v2_signatures.png"
CELLULES = ["V", "M", "I", "A"]
RNG = np.random.default_rng(20261027)
NC = 2000          # tirages pour l'incertitude sur les couches
NF = 500           # tirages pour l'incertitude sur les faits


def aire(y, x):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def corr(X):
    """correlation entre les LIGNES de X (n, d)."""
    X = X - X.mean(axis=1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1.0
    return np.clip((X / n) @ (X / n).T, -1.0, 1.0)


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


def tableau(nom, sims, taus, n_l):
    A = {c: np.zeros(n_l) for c in CELLULES}
    T = {c: np.zeros(n_l) for c in CELLULES}
    C = {c: [] for c in CELLULES}
    for c in CELLULES:
        for l in range(n_l):
            A[c][l], T[c][l], cn = signature(sims[c][l], taus)
            C[c].append(cn)
    print("\n" + "=" * 96)
    print(f"RESEAU {nom}")
    print("=" * 96)
    print(f"{'couche':>7}{'V':>9}{'M':>9}{'I':>9}{'A':>9}{'I-V':>9}{'M-V':>9}")
    print("-" * 96)
    for l in range(n_l):
        print(f"{l:>7}{A['V'][l]:>9.3f}{A['M'][l]:>9.3f}{A['I'][l]:>9.3f}{A['A'][l]:>9.3f}"
              f"{A['I'][l]-A['V'][l]:>9.3f}{A['M'][l]-A['V'][l]:>9.3f}")
    print("-" * 96)
    print(f"{'MOY':>7}{A['V'].mean():>9.3f}{A['M'].mean():>9.3f}{A['I'].mean():>9.3f}{A['A'].mean():>9.3f}"
          f"{A['I'].mean()-A['V'].mean():>9.3f}{A['M'].mean()-A['V'].mean():>9.3f}")
    print(f"  tau* : V={T['V'].mean():.2f}  M={T['M'].mean():.2f}  I={T['I'].mean():.2f}  A={T['A'].mean():.2f}")
    print(f"  I>V dans {int((A['I']>A['V']).sum())}/{n_l} couches · "
          f"M<V dans {int((A['M']<A['V']).sum())}/{n_l} couches · "
          f"couches ou I>V : {[l for l in range(n_l) if A['I'][l]>A['V'][l]]}")
    return A, T, C


def ci_couches(A, n_l):
    out = {}
    for k, (x, y) in {"I−V": ("I", "V"), "M−V": ("M", "V"), "A−I": ("A", "I")}.items():
        v = np.array([A[y][p].mean() - A[x][p].mean() if False else A[x][p].mean() - A[y][p].mean()
                      for p in (RNG.integers(0, n_l, n_l) for _ in range(NC))])
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = (v.mean(), lo, hi, 2 * min((v <= 0).mean(), (v >= 0).mean()))
    return out


def main():
    d = np.load(NPZ, allow_pickle=False)
    taus = d["taus"].astype(float)
    et = {c: d[c].astype(np.float32) for c in CELLULES}
    n, n_l, _ = et["V"].shape

    # ---------- reseau U
    SU = {c: [corr(et[c][:, l, :].T) for l in range(n_l)] for c in CELLULES}
    AU, TU, CU = tableau("U — 896 unites, correlation a travers les 18 faits", SU, taus, n_l)
    icU = ci_couches(AU, n_l)

    # ---------- reseau F  (centrage sur les faits)
    SF = {c: [corr(et[c][:, l, :] - et[c][:, l, :].mean(axis=0, keepdims=True))
              for l in range(n_l)] for c in CELLULES}
    AF, TF, CF = tableau("F — 18 faits, correlation a travers les 896 dimensions (mode commun retire)",
                         SF, taus, n_l)

    # ---------- reseau F : incertitude par bootstrap sur les faits
    boot = {c: [] for c in CELLULES}
    for _ in range(NF):
        p = RNG.integers(0, n, n)
        for c in CELLULES:
            boot[c].append(float(np.mean([signature(SF[c][l][np.ix_(p, p)], taus)[0]
                                          for l in range(n_l)])))
    boot = {c: np.array(v) for c, v in boot.items()}

    print("\n" + "=" * 96)
    print("CONTRASTES — IC 95 %")
    print("=" * 96)
    for nom, ic in (("U (bootstrap couches)", icU),):
        print(f"\n  Reseau {nom}")
        for k, (m, lo, hi, p) in ic.items():
            print(f"    {k:<6} = {m:+.4f}   [{lo:+.4f}, {hi:+.4f}]   p={p:.4f}")
    print("\n  Reseau F (bootstrap faits, 500 tirages)")
    for k, (x, y) in {"I−V": ("I", "V"), "M−V": ("M", "V"), "A−I": ("A", "I")}.items():
        v = boot[x] - boot[y]
        lo, hi = np.percentile(v, [2.5, 97.5])
        p = 2 * min((v <= 0).mean(), (v >= 0).mean())
        print(f"    {k:<6} = {v.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]   p={p:.4f}")

    # ---------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib indisponible :", e); return

    COUL = {"V": "#1f77b4", "M": "#d62728", "I": "#7f7f7f", "A": "#2ca02c"}
    NOM = {"V": "V  sait · vrai", "M": "M  sait · consigne «mens»",
           "I": "I  ne sait pas · vrai", "A": "A  ne sait pas · consigne «mens»"}
    fig, ax = plt.subplots(2, 2, figsize=(13.6, 9.0))

    for col, (lab, C, A, T) in enumerate((
            ("U · 896 unités", CU, AU, TU), ("F · 18 faits", CF, AF, TF))):
        for c in CELLULES:
            M_ = np.stack(C[c]); m, s = M_.mean(0), M_.std(0)
            ax[0, col].plot(taus, m, color=COUL[c], lw=2.2, label=NOM[c])
            ax[0, col].fill_between(taus, m - s, m + s, color=COUL[c], alpha=0.13)
        ax[0, col].axhline(0.10, color="k", ls=":", lw=1)
        ax[0, col].set_xlabel(r"seuil $\tau$"); ax[0, col].set_ylabel(r"$\beta_1(\tau)\,/\,\beta_1(0)$")
        ax[0, col].set_title(f"Réseau {lab} — signature du flux résiduel\nQwen2.5-0.5B-Instruct, 18 faits/cellule")
        ax[0, col].legend(fontsize=8); ax[0, col].grid(alpha=0.25); ax[0, col].set_ylim(-0.02, 1.05)

    x = np.arange(n_l)
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 0].plot(x, AU["I"] - AU["V"], "o-", color="#7b4fbf", lw=2, ms=3.5, label="I − V  (savoir)")
    ax[1, 0].plot(x, AU["M"] - AU["V"], "s-", color="#d62728", lw=1.8, ms=3.5, label="M − V  (consigne)")
    ax[1, 0].set_xlabel("couche"); ax[1, 0].set_ylabel(r"$\Delta$ aire sur le réseau U")
    ax[1, 0].set_title("Le savoir sépare, la consigne ne fait presque rien\n(bascule de signe aux couches 1–4)")
    ax[1, 0].legend(fontsize=9); ax[1, 0].grid(alpha=0.25)

    for c in CELLULES:
        ax[1, 1].plot(x, TU[c], color=COUL[c], lw=1.8, marker="o", ms=2.8, label=f"{c}  (moy {TU[c].mean():.2f})")
    ax[1, 1].set_xlabel("couche"); ax[1, 1].set_ylabel(r"$\tau^\ast$  (effondrement à 10 %)")
    ax[1, 1].set_title(r"$\tau^\ast$ par couche — l'ignorance tient plus haut que le savoir")
    ax[1, 1].legend(fontsize=9); ax[1, 1].grid(alpha=0.25)

    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print(f"\nfigure : {OUT} ({OUT.stat().st_size/1024:.0f} Ko)")

    # ---------- export
    res = {"modele": "Qwen2.5-0.5B-Instruct", "n_faits_cellule": n, "n_couches": n_l,
           "taus": taus.tolist(),
           "aire_U": {c: AU[c].tolist() for c in CELLULES},
           "tau_U": {c: TU[c].tolist() for c in CELLULES},
           "aire_F": {c: AF[c].tolist() for c in CELLULES},
           "tau_F": {c: TF[c].tolist() for c in CELLULES}}
    (ROOT / "resultats" / "pmt_v2_analyse.json").write_text(json.dumps(res, indent=1))
    print("export : resultats/pmt_v2_analyse.json")


if __name__ == "__main__":
    main()
