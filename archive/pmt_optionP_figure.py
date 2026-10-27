#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figure finale de l'option P : le test, l'artefact, et le test corrige."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmt import aretes, beta1

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resultats"
RNG = np.random.default_rng(7)
TAUS = np.round(np.linspace(0, 0.95, 20), 3)


def corr(X):
    X = X - X.mean(1, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True); n[n == 0] = 1
    return np.clip((X / n) @ (X / n).T, -1, 1)


def sig(S):
    n = S.shape[0]
    c = np.array([beta1(n, aretes(S, t)) for t in TAUS], float)
    cn = c / (c[0] if c[0] > 0 else 1)
    imax = c.max()
    if imax <= 0:
        return 0.0, float(TAUS[-1])
    idx = np.where(c <= 0.10 * imax)[0]
    return float(np.trapezoid(cn, TAUS)), float(TAUS[idx[0]] if len(idx) else TAUS[-1])


def aires(E):
    return np.array([sig(corr(E[:, l, :].T)) for l in range(E.shape[1])])


def main():
    # --- niveau 4 niveaux (contamine au niveau 2)
    z = np.load(RES / "pmt_optionP.npz", allow_pickle=False)
    E, reg, niv = z["etats"].astype(np.float32), z["regime"], z["niveau"]
    A, NLL = {}, {}
    for r in "KU":
        for n in (0, 1, 2, 3):
            ix = np.where((reg == r) & (niv == n))[0]
            a = aires(E[ix])
            A[(r, n)] = a[:, 0]
            NLL[(r, n)] = float(np.nanmean(z["nll"][ix]))

    # --- test corrige (longueur appariee)
    z2 = np.load(RES / "pmt_optionP_len.npz", allow_pickle=False)
    E2, reg2, T2 = z2["etats"].astype(np.float32), z2["regime"], z2["T"]
    C = {}
    for t in (1, 2):
        for r in "KU":
            ix = np.where((T2 == t) & (reg2 == r))[0]
            C[(t, r)] = aires(E2[ix])[:, 0]

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib indisponible :", e); return

    COUL = {"K": "#d62728", "U": "#2ca02c"}
    fig, ax = plt.subplots(1, 3, figsize=(17.0, 5.1))

    # 1. aire vs NLL
    for r in "KU":
        x = [NLL[(r, n)] for n in (0, 1, 2, 3)]
        y = [A[(r, n)].mean() for n in (0, 2, 3)] if False else [A[(r, n)].mean() for n in (0, 1, 2, 3)]
        ax[0].plot(x, y, "o-", color=COUL[r], lw=2, ms=8,
                   label="CONNU · conflit" if r == "K" else "INCONNU · pas de conflit")
        for n in (0, 1, 2, 3):
            ax[0].annotate(f"NLL {x[n]:.1f}", (x[n], y[n]), fontsize=7,
                           xytext=(4, -10), textcoords="offset points", color=COUL[r])
    ax[0].plot(NLL[("K", 2)], A[("K", 2)].mean(), "o", ms=19, mfc="none", mec="k", mew=2)
    ax[0].annotate("point contaminé\n(réponses plus longues)",
                   (NLL[("K", 2)], A[("K", 2)].mean()), fontsize=8,
                   xytext=(-105, 6), textcoords="offset points",
                   arrowprops=dict(arrowstyle="->", lw=1))
    ax[0].set_xlabel("NLL de la réponse lue (nats)")
    ax[0].set_ylabel(r"aire $\int\beta_1(\tau)/\beta_1(0)d\tau$")
    ax[0].set_title("1. L'aire monte avec la surprise\nmais les deux régimes se suivent")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.25)

    # 2. longueur des réponses : l'artefact
    import os
    os.environ.setdefault("HF_HOME", "/home/user/.cache/hf")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    lab, kL, uL = [], [], []
    for n in (0, 1, 2, 3):
        for r, dest in (("K", kL), ("U", uL)):
            reps = [x for x, g, v in zip(z["reponse"], reg, niv) if g == r and v == n]
            dest.append(np.mean([len(tok(" " + x)["input_ids"]) for x in reps]))
        lab.append(f"NLL {NLL[('K', n)]:.0f}")
    xx = np.arange(4); w = 0.36
    ax[1].bar(xx - w / 2, kL, w, color=COUL["K"], label="CONNU")
    ax[1].bar(xx + w / 2, uL, w, color=COUL["U"], label="INCONNU")
    ax[1].set_xticks(xx); ax[1].set_xticklabels(lab)
    ax[1].set_ylabel("tokens de la réponse lue")
    ax[1].set_title("2. L'artefact : au niveau NLL≈9 les réponses\nCONNUES sont 1,6× plus longues")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.25, axis="y")

    # 3. test corrige
    x = np.arange(25)
    ax[2].axhline(0, color="k", lw=1)
    for t, c in ((1, "#7b4fbf"), (2, "#ff7f0e")):
        d = C[(t, "K")] - C[(t, "U")]
        ax[2].plot(x, d, "o-", color=c, lw=1.9, ms=3.6,
                   label=f"réponses de {t} token(s) · Δ moy = {d.mean():+.4f}")
    ax[2].set_xlabel("couche"); ax[2].set_ylabel(r"$\Delta$ aire $(K-U)$")
    ax[2].set_title("3. Test corrigé : surprise ET longueur appariées\nles deux courbes sont sous zéro")
    ax[2].legend(fontsize=8.5); ax[2].grid(alpha=0.25)

    fig.tight_layout()
    out = RES / "pmt_optionP_final.png"
    fig.savefig(out, dpi=130)
    print(f"figure : {out} ({out.stat().st_size/1024:.0f} Ko)")

    for t in (1, 2):
        d = C[(t, "K")] - C[(t, "U")]
        bs = np.array([d[RNG.integers(0, 25, 25)].mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  T={t} : Δ = {d.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  K>U sur {int((d>0).sum())}/25")


if __name__ == "__main__":
    main()
