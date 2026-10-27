#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION S — H2-SYNC : le conflit est-il dans la VERTICALITE du reseau ?

PREDICTION ENREGISTREE AVANT CALCUL (issue de l'ordre du chef) :
   le mensonge force cree une DESYNCHRONISATION entre les couches de
   connaissance (5-15) et les couches de generation (16-24), absente dans
   le regime sans conflit.
   -> on attend  sync(K) < sync(U)  sur le bloc 5-15 x 16-24.

Aucun nouveau run de modele : tout est calcule sur les etats deja captures.

Metriques par etat (25 couches x 896 dims) :
   sync_adj[l]  = cos(h_l, h_(l+1))                 profil de 24 valeurs
   sync_bloc    = cos moyen entre couches 5-15 et 16-24
   sync_totale  = cos moyen sur toutes les paires hors diagonale
   derive[l]    = 1 - sync_adj[l]                   perte d'information verticale

Incertitude : bootstrap sur les FAITS (chaque fait fournit un profil complet).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resultats"
RNG = np.random.default_rng(20261027)
BLOC_A = list(range(5, 16))      # connaissance
BLOC_B = list(range(16, 25))     # generation


def profils(X: np.ndarray) -> dict:
    """X : (n, 25, 896). Retourne les profils par item."""
    H = X / np.linalg.norm(X, axis=2, keepdims=True)
    # matrice de similarite inter-couches, par item : (n, 25, 25)
    M = np.einsum("ild,ikd->ilk", H, H)
    adj = np.einsum("ild,ild->il", H[:, :-1], H[:, 1:])   # (n, 24)
    blocs = M[:, BLOC_A][:, :, BLOC_B].mean(axis=(1, 2))
    hors = ~np.eye(25, dtype=bool)
    totale = M[:, hors].mean(axis=1)
    return {"adj": adj, "bloc": blocs, "totale": totale, "M": M}


def bootstrap_faits(a, b=None, n_boot: int = 5000) -> tuple:
    """b=None : IC de la moyenne de a. b fourni : IC de la difference (tirages independants)."""
    if b is None:
        bs = np.array([a[RNG.integers(0, len(a), len(a))].mean() for _ in range(n_boot)])
        obs = float(a.mean())
    else:
        bs = np.array([a[RNG.integers(0, len(a), len(a))].mean()
                       - b[RNG.integers(0, len(b), len(b))].mean() for _ in range(n_boot)])
        obs = float(a.mean() - b.mean())
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return obs, float(lo), float(hi), float(2 * min((bs <= 0).mean(), (bs >= 0).mean()))


def main():
    print("=" * 92)
    print("OPTION S — H2-SYNC : la verticalite du reseau voit-elle le conflit ?")
    print("=" * 92)
    print(f"prédiction pré-enregistrée : sync_bloc(K) < sync_bloc(U)   "
          f"[bloc {BLOC_A[0]}-{BLOC_A[-1]} × {BLOC_B[0]}-{BLOC_B[-1]}]")
    print("contrastes à NLL ET longueur appariés (run 4)\n")

    z = np.load(RES / "pmt_optionP_len.npz", allow_pickle=False)
    X, reg, T = z["etats"].astype(np.float32), z["regime"], z["T"]
    P = {r: profils(X[reg == r]) for r in "KU"}

    print(f"{'mesure':<26}{'K (conflit)':>14}{'U (sans)':>12}{'Δ = K−U':>12}{'IC 95 %':>24}{'p':>8}")
    print("-" * 92)
    sortie = {}
    for nom, cle in (("sync bloc 5-15 × 16-24", "bloc"),
                     ("sync totale (toutes paires)", "totale")):
        m, lo, hi, p = bootstrap_faits(P["K"][cle], P["U"][cle])
        print(f"{nom:<26}{P['K'][cle].mean():>14.4f}{P['U'][cle].mean():>12.4f}"
              f"{m:>+12.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>24}{p:>8.3f}")
        sortie[cle] = (m, lo, hi, p)

    # profil couche par couche (24 adjacences)
    print(f"\n{'adjacence':<14}{'K':>9}{'U':>9}{'Δ':>9}{'p':>8}")
    print("-" * 92)
    prof = []
    for l in range(24):
        m, lo, hi, p = bootstrap_faits(P["K"]["adj"][:, l], P["U"]["adj"][:, l])
        prof.append({"l": l, "K": float(P["K"]["adj"][:, l].mean()),
                     "U": float(P["U"]["adj"][:, l].mean()), "d": m, "p": p})
        flag = " ***" if p < 0.05 else ""
        print(f"{l:>2}→{l+1:<11}{prof[-1]['K']:>9.4f}{prof[-1]['U']:>9.4f}{m:>+9.4f}{p:>8.3f}{flag}")

    n_pos = sum(1 for x in prof if x["d"] > 0)
    n_sig = sum(1 for x in prof if x["p"] < 0.05)
    print(f"\n  adjacences où K > U : {n_pos}/24 · significatives (p<0,05) : {n_sig}/24")

    # ---------- controle : la sync suit-elle le NLL ? (4 niveaux)
    print("\n" + "=" * 92)
    print("CONTROLE — la sync suit-elle simplement la surprise ? (run 3, 4 niveaux)")
    print("=" * 92)
    z2 = np.load(RES / "pmt_optionP.npz", allow_pickle=False)
    X2, reg2, niv2 = z2["etats"].astype(np.float32), z2["regime"], z2["niveau"]
    print(f"{'niveau':<10}{'NLL moyen':>11}{'sync_bloc K':>14}{'sync_bloc U':>14}{'Δ':>10}")
    print("-" * 92)
    for n in (0, 1, 2, 3):
        sl = {}
        for r in "KU":
            ix = (reg2 == r) & (niv2 == n)
            sl[r] = profils(X2[ix])["bloc"].mean()
        nll = float(np.nanmean(z2["nll"][(niv2 == n)]))
        print(f"{n:<10}{nll:>11.2f}{sl['K']:>14.4f}{sl['U']:>14.4f}{sl['K']-sl['U']:>+10.4f}")

    # ---------- controle : axe du savoir (run 1)
    print("\n" + "=" * 92)
    print("REPLICATION — axe du SAVOIR sur le run 1 (V sait / I ne sait pas)")
    print("=" * 92)
    z3 = np.load(RES / "pmt_v2_etats.npz", allow_pickle=False)
    for a, b in (("V", "I"), ("M", "A")):
        Xa, Xb = z3[a].astype(np.float32), z3[b].astype(np.float32)
        pa, pb = profils(Xa), profils(Xb)
        m, lo, hi, p = bootstrap_faits(pa["bloc"], pb["bloc"])
        print(f"  sync_bloc {a} − {b} = {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p={p:.3f}"
              f"   ({a}={pa['bloc'].mean():.4f} · {b}={pb['bloc'].mean():.4f})")

    # ---------- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 4.9))
    x = np.arange(24)
    for r, c in (("K", "#d62728"), ("U", "#2ca02c")):
        m = P[r]["adj"].mean(0)
        s = P[r]["adj"].std(0)
        ax[0].plot(x + 1, m, "o-", color=c, lw=2, ms=4,
                   label=f"{'CONNU (conflit)' if r=='K' else 'INCONNU (sans)'}")
        ax[0].fill_between(x + 1, m - s, m + s, color=c, alpha=0.13)
    ax[0].set_xlabel("couche $\\ell \\to \\ell+1$")
    ax[0].set_ylabel(r"$\cos(h_\ell, h_{\ell+1})$")
    ax[0].set_title("H₂-Sync : synchronisation verticale\nNLL et longueur appariés")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

    pv = [x["p"] for x in prof]
    ax[1].axhline(0, color="k", lw=1)
    ax[1].bar(x + 1, [x["d"] for x in prof],
              color=["#d62728" if x["d"] > 0 else "#4c72b0" for x in prof], width=0.75)
    ax[1].set_xlabel("adjacence $\\ell \\to \\ell+1$")
    ax[1].set_ylabel(r"$\Delta\cos$ $(K-U)$")
    ax[1].set_title("Différence couche par couche\n(rien ne franchit le seuil de façon stable)")
    ax[1].grid(alpha=0.25, axis="y")

    fig.tight_layout()
    out = RES / "pmt_optionS_sync.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure : {out} ({out.stat().st_size/1024:.0f} Ko)")

    (RES / "pmt_optionS.json").write_text(json.dumps(
        {"prediction": "sync_bloc(K) < sync_bloc(U)",
         "contrastes": {k: list(v) for k, v in sortie.items()},
         "profil_adjacence": prof,
         "adj_K_sup_U": n_pos, "adj_significatives": n_sig}, indent=1))


if __name__ == "__main__":
    main()
