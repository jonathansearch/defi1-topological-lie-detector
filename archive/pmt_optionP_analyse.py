#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION P — analyse.

Question : a NLL egal, la topologie distingue-t-elle
  (K) un fait CONNU contredit   de   (U) une assertion improbable sur un fait INCONNU ?

Si aire = f(NLL) seulement, les deux regimes doivent tomber sur la meme droite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "resultats" / "pmt_optionP.npz"
PNG = ROOT / "resultats" / "pmt_optionP_signatures.png"
RNG = np.random.default_rng(20261027)
TAUS = np.round(np.linspace(0.0, 0.95, 20), 3)
NOM_NIV = {0: "vraie réponse", 1: "NLL≈4", 2: "NLL≈9", 3: "NLL≈13"}


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


def signature(S, taus=TAUS):
    c = beta1_courbe(S, taus)
    cn = c / (c[0] if c[0] > 0 else 1.0)
    imax = float(c.max())
    if imax <= 0:
        return 0.0, float(taus[-1]), cn
    ok = np.where(c <= 0.10 * imax)[0]
    return aire(cn, taus), float(taus[ok[0]] if len(ok) else taus[-1]), cn


def main():
    z = np.load(NPZ, allow_pickle=False)
    E = z["etats"].astype(np.float32)          # (N, 25, 896)
    nll, regime = z["nll"], z["regime"]
    niveau, fait = z["niveau"], z["fait"]
    pays, reponse = z["pays"], z["reponse"]
    N, n_l, _ = E.shape
    regimes = ["K", "U"]
    n_f = {r: int((regime == r).max() and len(set(fait[regime == r]))) for r in regimes}

    print("=" * 100)
    print("OPTION P — LE CONFLIT AU-DELA DE LA SURPRISE")
    print(f"{N} états · {n_l} couches · régimes {regimes} · niveau 0 = vraie capitale")
    print("=" * 100)

    # ---------- index
    def sel(r, niv):
        return np.where((regime == r) & (niveau == niv))[0]

    # ---------- signatures par (regime, niveau)
    res = {}
    for r in regimes:
        for niv in (0, 1, 2, 3):
            ix = sel(r, niv)
            A, T, C = np.zeros(n_l), np.zeros(n_l), []
            for l in range(n_l):
                S = corr(E[ix][:, l, :].T)         # reseau U : 896 unites x les faits
                a, t, cn = signature(S)
                A[l], T[l] = a, t
                C.append(cn)
            res[(r, niv)] = {"aire": A, "tau": T, "cn": np.stack(C),
                             "nll": float(np.nanmean(nll[ix]))}
            print(f"  {r}{niv}  NLL={res[(r,niv)]['nll']:6.2f}  "
                  f"aire={A.mean():.4f}  tau*={T.mean():.3f}")

    print("\n" + "=" * 100)
    print("NLL REELLEMENT OBTENU (le controle d'appariement)")
    print("=" * 100)
    print(f"{'niveau':<16}{'NLL cible':>11}{'NLL connu':>12}{'NLL inconnu':>13}{'écart':>9}")
    print("-" * 100)
    for niv, cible in zip((0, 1, 2, 3), (0.5, 4.0, 9.0, 13.0)):
        kk, uu = res[("K", niv)]["nll"], res[("U", niv)]["nll"]
        print(f"{NOM_NIV[niv]:<16}{cible:>11.1f}{kk:>12.2f}{uu:>13.2f}{uu-kk:>9.2f}")

    # ---------- droite aire = a + b*NLL, par regime
    print("\n" + "=" * 100)
    print("AIRE EN FONCTION DU NLL")
    print("=" * 100)
    droites = {}
    for r in regimes:
        x = np.array([res[(r, n)]["nll"] for n in (0, 1, 2, 3)])
        y = np.array([res[(r, n)]["aire"].mean() for n in (0, 1, 2, 3)])
        b, a = np.polyfit(x, y, 1)
        droites[r] = (a, b, x, y)
        print(f"  régime {r} : aire = {a:+.4f} {b:+.5f} × NLL    "
              f"(points : {', '.join(f'{v:.3f}' for v in y)})")

    # ---------- LE TEST : meme NLL, regimes differents
    print("\n" + "=" * 100)
    print("TEST DECISIF — à NLL apparié, la topologie distingue-t-elle connu et inconnu ?")
    print("=" * 100)

    def contraste(niv, lbl):
        ixK, ixU = sel("K", niv), sel("U", niv)
        aK = np.array([signature(corr(E[ixK][:, l, :].T))[0] for l in range(n_l)])
        aU = np.array([signature(corr(E[ixU][:, l, :].T))[0] for l in range(n_l)])
        d = aK - aU
        # bootstrap sur les faits
        nk = len(set(fait[ixK]))
        bs = []
        for _ in range(500):
            pK = RNG.integers(0, nk, nk); pU = RNG.integers(0, nk, nk)
            fK = np.unique(fait[ixK]); fU = np.unique(fait[ixU])
            iK = np.concatenate([np.where(fait[ixK] == fK[j])[0] for j in pK])
            iU = np.concatenate([np.where(fait[ixU] == fU[j])[0] for j in pU])
            bK = np.mean([signature(corr(E[ixK][iK][:, l, :].T))[0] for l in range(n_l)])
            bU = np.mean([signature(corr(E[ixU][iU][:, l, :].T))[0] for l in range(n_l)])
            bs.append(bK - bU)
        bs = np.array(bs)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
        print(f"  {lbl:<28} Δ(K−U) = {d.mean():+.4f}   [{lo:+.4f}, {hi:+.4f}]   p={p:.3f}"
              f"   K>U sur {int((d>0).sum())}/{n_l} couches")
        return d.mean(), lo, hi, p, d

    c0 = contraste(0, "vraie réponse (NLL bas)")
    c1 = contraste(1, "NLL ≈ 4")
    c2 = contraste(2, "NLL ≈ 9  ← LE TEST")
    c3 = contraste(3, "NLL ≈ 13")

    # ---------- pente : la surprise agit-elle pareil ?
    print("\n  pente aire/NLL : "
          f"connu {droites['K'][1]:+.5f} · inconnu {droites['U'][1]:+.5f} · "
          f"écart {droites['K'][1]-droites['U'][1]:+.5f}")

    # ---------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib indisponible :", e); return

    COUL = {("K", 0): "#1f77b4", ("K", 2): "#d62728", ("U", 0): "#7f7f7f", ("U", 2): "#2ca02c"}
    fig, ax = plt.subplots(1, 3, figsize=(16.6, 5.0))

    for r, mk in (("K", "o"), ("U", "s")):
        a, b, x, y = droites[r]
        ax[0].plot(x, y, mk + "-", color="#d62728" if r == "K" else "#2ca02c",
                   lw=2, ms=8, label=f"{'CONNU (conflit)' if r=='K' else 'INCONNU (pas de conflit)'}")
        xx = np.linspace(0, 14, 50)
        ax[0].plot(xx, a + b * xx, ls=":", lw=1.4,
                   color="#d62728" if r == "K" else "#2ca02c", alpha=0.7)
    ax[0].set_xlabel("NLL de la réponse lue (nats)")
    ax[0].set_ylabel(r"aire $\int \beta_1(\tau)/\beta_1(0)d\tau$, moy. couches")
    ax[0].set_title("L'aire suit-elle la surprise, ou le régime ?\n4 niveaux de surprise × 2 régimes de savoir")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

    for (r, niv) in (("K", 0), ("K", 2), ("U", 0), ("U", 2)):
        C = res[(r, niv)]["cn"]
        m, s = C.mean(0), C.std(0)
        lab = ("connu" if r == "K" else "inconnu") + " · " + NOM_NIV[niv]
        ax[1].plot(TAUS, m, color=COUL[(r, niv)], lw=2.2, label=lab)
        ax[1].fill_between(TAUS, m - s, m + s, color=COUL[(r, niv)], alpha=0.12)
    ax[1].axhline(0.10, color="k", ls=":", lw=1)
    ax[1].set_xlabel(r"seuil $\tau$"); ax[1].set_ylabel(r"$\beta_1(\tau)/\beta_1(0)$")
    ax[1].set_title("Signatures : le cas apparié en surprise\n(rouge vs vert = même imprévu, savoir différent)")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=0.25); ax[1].set_ylim(-0.02, 1.05)

    x = np.arange(n_l)
    ax[2].axhline(0, color="k", lw=1)
    ax[2].bar(x, c2[4], color=["#d62728" if v > 0 else "#4c72b0" for v in c2[4]], width=0.75)
    ax[2].set_xlabel("couche"); ax[2].set_ylabel(r"$\Delta$ aire $K_2-U_2$")
    ax[2].set_title("Test apparié au niveau NLL≈9, couche par couche")
    ax[2].grid(alpha=0.25, axis="y")

    fig.tight_layout(); fig.savefig(PNG, dpi=130)
    print(f"\nfigure : {PNG} ({PNG.stat().st_size/1024:.0f} Ko)")

    out = {"n_par_regime": {r: n_f[r] for r in regimes},
           "nll": {f"{r}{n}": res[(r, n)]["nll"] for r in regimes for n in (0, 1, 2, 3)},
           "aire": {f"{r}{n}": res[(r, n)]["aire"].tolist() for r in regimes for n in (0, 1, 2, 3)},
           "tau": {f"{r}{n}": res[(r, n)]["tau"].tolist() for r in regimes for n in (0, 1, 2, 3)},
           "droites": {r: [float(droites[r][0]), float(droites[r][1])] for r in regimes},
           "contrastes": {"K0-U0": c0[:4], "K1-U1": c1[:4], "K2-U2": c2[:4], "K3-U3": c3[:4]}}
    (ROOT / "resultats" / "pmt_optionP_analyse.json").write_text(json.dumps(out, indent=1))
    print("export : resultats/pmt_optionP_analyse.json")


if __name__ == "__main__":
    main()
