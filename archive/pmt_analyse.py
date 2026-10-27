#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMT v2 — analyse du run reel, deux graphes declares, calcul rapide.

  Reseau U : nœuds = 896 unites du flux residuel, arêtes = corr. entre unites
             mesuree sur les 18 faits  -> topologie interne de la representation
  Reseau F : nœuds = 18 faits, arêtes = corr. entre les flux residuels
             -> l'objet de la formalisation (Path_G(u, v_abs))

Incertitude : jackknife (suppression d'un fait) sur U, bootstrap sur F.
beta1 calcule par Kruskal trie + connected_components (scipy, C).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmt import similarites

ROOT = Path(__file__).resolve().parents[1]
NPZ = ROOT / "resultats" / "pmt_v2_etats.npz"
OUT = ROOT / "resultats" / "pmt_v2_signatures.png"
CELLULES = ["V", "M", "I", "A"]
NB_F = 500
RNG = np.random.default_rng(20261027)


def aire(y, x):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def courbe_beta1(S, taus):
    """beta1(tau) = E - V + C, via aretes triees + composantes connexes (C)."""
    n = S.shape[0]
    iu = np.triu_indices(n, k=1)
    w = S[iu]
    o = np.argsort(-w)
    ws, us, vs = w[o], iu[0][o], iu[1][o]
    out = np.empty(len(taus))
    for k, t in enumerate(taus):
        m = int(np.searchsorted(-ws, -t, side="right"))
        if m == 0:
            out[k] = 0.0
            continue
        g = csr_matrix((np.ones(m, dtype=np.int8), (us[:m], vs[:m])), shape=(n, n))
        c, _ = connected_components(g, directed=False)
        out[k] = m - n + c
    return out


def signature(S, taus):
    c = courbe_beta1(S, taus)
    c0 = c[0] if c[0] > 0 else 1.0
    cn = c / c0
    imax = float(c.max())
    if imax <= 0:
        return 0.0, float(taus[-1]), 0.0, cn
    ok = np.where(c <= 0.10 * imax)[0]
    return aire(cn, taus), float(taus[ok[0]] if len(ok) else taus[-1]), imax, cn


def analyse(nom, sims, taus, n_l, n, mode):
    print("\n" + "=" * 104)
    print(f"RESEAU {nom}   ({n} nœuds · {n_l} couches · incertitude = {mode})")
    print("=" * 104)
    res = {c: {"aire": np.zeros(n_l), "tau": np.zeros(n_l), "bmax": np.zeros(n_l),
               "cn": []} for c in CELLULES}
    for c in CELLULES:
        for l in range(n_l):
            a, t, m, cn = signature(sims[c][l], taus)
            res[c]["aire"][l] = a; res[c]["tau"][l] = t; res[c]["bmax"][l] = m
            res[c]["cn"].append(cn)

    print(f"{'couche':>7}{'V':>9}{'M':>9}{'I':>9}{'A':>9}{'M-V':>9}{'A-I':>9}{'H1':>9}")
    print("-" * 104)
    for l in range(n_l):
        v, m, i_, a = (res[c]["aire"][l] for c in CELLULES)
        print(f"{l:>7}{v:>9.3f}{m:>9.3f}{i_:>9.3f}{a:>9.3f}{m-v:>9.3f}{a-i_:>9.3f}{(m-v)-(a-i_):>9.3f}")
    mo = {c: res[c]["aire"].mean() for c in CELLULES}
    print("-" * 104)
    print(f"{'MOY':>7}{mo['V']:>9.3f}{mo['M']:>9.3f}{mo['I']:>9.3f}{mo['A']:>9.3f}"
          f"{mo['M']-mo['V']:>9.3f}{mo['A']-mo['I']:>9.3f}"
          f"{(mo['M']-mo['V'])-(mo['A']-mo['I']):>9.3f}")
    h1l = (res["M"]["aire"] - res["V"]["aire"]) - (res["A"]["aire"] - res["I"]["aire"])
    mv = res["M"]["aire"] - res["V"]["aire"]
    print(f"   signe de H1 par couche : {int((h1l>0).sum())}/{n_l} positives   "
          f"| M−V : {int((mv<0).sum())}/{n_l} negatives")
    ts = {c: res[c]["tau"].mean() for c in CELLULES}
    print(f"tau* MOY :  V={ts['V']:.2f}   M={ts['M']:.2f}   I={ts['I']:.2f}   A={ts['A']:.2f}")

    # ---- incertitude
    boot = {c: [] for c in CELLULES}
    if mode == "couches":
        # on reechantillonne les COUCHES : pas de recalcul de graphe
        for _ in range(2000):
            p = RNG.integers(0, n_l, n_l)
            for c in CELLULES:
                boot[c].append(float(res[c]["aire"][p].mean()))
    else:
        # on reechantillonne les FAITS : recalcul du graphe a chaque tirage
        for _ in range(NB_F):
            p = RNG.integers(0, n, n)
            for c in CELLULES:
                acc = [signature(sims[c][l][np.ix_(p, p)], taus)[0] for l in range(n_l)]
                boot[c].append(float(np.mean(acc)))
    boot = {c: np.array(v) for c, v in boot.items()}
    dMV, dAI = boot["M"] - boot["V"], boot["A"] - boot["I"]
    it = dMV - dAI

    print(f"\n{'contraste':<32}{'moy':>10}{'IC 95 %':>22}{'p':>8}{'>0':>8}")
    print("-" * 104)
    for nm, x in (("M − V  (consigne | sait)", dMV),
                  ("A − I  (consigne | ne sait pas)", dAI),
                  ("H1 = (M−V) − (A−I)", it)):
        lo, hi = np.percentile(x, [2.5, 97.5])
        p = 2 * min((x <= 0).mean(), (x >= 0).mean())
        print(f"{nm:<32}{x.mean():>10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>22}{p:>8.3f}{100*(x>0).mean():>7.1f}%")
    return res, boot, mo, ts


def main():
    d = np.load(NPZ, allow_pickle=False)
    taus = d["taus"].astype(float)
    etats = {c: d[c].astype(np.float32) for c in CELLULES}
    n, n_l, _ = etats["V"].shape
    idsS, idsN = d["ids_sait"], d["ids_sait_pas"]
    print(f"PMT v2 — Qwen2.5-0.5B-Instruct · {n} faits/cellule · {n_l} couches · {len(taus)} tau")

    S_U = {c: [similarites(etats[c][:, l, :].T) for l in range(n_l)] for c in CELLULES}
    resU, bootU, moU, tsU = analyse("U  (896 unites)", S_U, taus, n_l, 896, "couches")
    del S_U

    S_F = {c: [similarites(etats[c][:, l, :]) for l in range(n_l)] for c in CELLULES}
    resF, bootF, moF, tsF = analyse("F  (18 faits)", S_F, taus, n_l, n, "faits")

    # ---- controle de type : capitales seules
    capS = np.array([i for i, s in enumerate(idsS) if str(s).startswith("cap:")])
    capN = np.array([i for i, s in enumerate(idsN) if str(s).startswith("cap:")])
    k = min(len(capS), len(capN))
    print("\n" + "=" * 104)
    print(f"CONTROLE DE TYPE — capitales seules ({k} contre {k}) — l'ecart est-il un artefact du melange de types ?")
    print("=" * 104)
    for nom, sl in (("F", slice(None)),):
        aV = np.mean([signature(similarites(etats['V'][capS[:k], l, :]), taus)[0] for l in range(n_l)])
        aI = np.mean([signature(similarites(etats['I'][capN[:k], l, :]), taus)[0] for l in range(n_l)])
        print(f"  Reseau F :  V (capitales sues) = {aV:.3f}   I (capitales inconnues) = {aI:.3f}"
              f"   Δ(I−V) = {aI-aV:+.3f}")
    aV = np.mean([signature(similarites(etats['V'][:, l, :].T), taus)[0] for l in range(n_l)])
    print(f"  repere Reseau U pleine largeur : V = {aV:.3f}  (aire normalisee, couches moyennées)")

    # ---- figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib indisponible :", e); return

    COUL = {"V": "#1f77b4", "M": "#d62728", "I": "#7f7f7f", "A": "#2ca02c"}
    NOM = {"V": "V  sait · vrai", "M": "M  sait · doit mentir",
           "I": "I  ne sait pas · vrai", "A": "A  ne sait pas · doit mentir"}
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.0))
    for col, (lab, res, bootx) in enumerate((("U · 896 unités", resU, bootU),
                                             ("F · 18 faits", resF, bootF))):
        for c in CELLULES:
            C = np.stack(res[c]["cn"]); m, s = C.mean(0), C.std(0)
            ax[0, col].plot(taus, m, color=COUL[c], lw=2.2, label=NOM[c])
            ax[0, col].fill_between(taus, m - s, m + s, color=COUL[c], alpha=0.13)
        ax[0, col].axhline(0.10, color="k", ls=":", lw=1)
        ax[0, col].set_xlabel(r"seuil $\tau$"); ax[0, col].set_ylabel(r"$\beta_1(\tau)/\beta_1(0)$")
        ax[0, col].set_title(f"Réseau {lab} — signature du flux résiduel")
        ax[0, col].legend(fontsize=8.5, loc="upper right")
        ax[0, col].grid(alpha=0.25); ax[0, col].set_ylim(-0.02, 1.05)

        it = (bootx["M"] - bootx["V"]) - (bootx["A"] - bootx["I"])
        ax[1, col].axhline(0, color="k", lw=1)
        ax[1, col].hist(it, bins=min(32, len(it)), color="#7b4fbf", alpha=0.78, edgecolor="white")
        ax[1, col].axvline(it.mean(), color="#d62728", lw=2, label=f"moy = {it.mean():+.4f}")
        ax[1, col].set_xlabel(r"$H_1=(M-V)-(A-I)$"); ax[1, col].set_ylabel("tirages")
        ax[1, col].set_title(f"Réseau {lab} — H1 ({(bootx['V']).size} rééchantillonnages)")
        ax[1, col].legend(fontsize=9); ax[1, col].grid(alpha=0.25, axis="y")
    fig.tight_layout(); fig.savefig(OUT, dpi=130)
    print(f"\nfigure : {OUT} ({OUT.stat().st_size/1024:.0f} Ko)")


if __name__ == "__main__":
    main()
