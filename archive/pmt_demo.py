#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PMT v1 — demonstration de l'instrument sur 4 cellules SYNTHETIQUES.

But : verifier que la signature distingue les 4 regimes, et voir a quoi
ressemble le "trou" quand il s'ouvre.

ATTENTION (une ligne, pas un discours) : ce sont des donnees fabriquees.
Ca montre que l'instrument marche, pas que l'hypothese est vraie.

Sortie : assets/defi1/pmt_signatures.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pmt import signature

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "defi1"
OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20261027)


def cellule(regime: str, n_unites: int = 60, n_tokens: int = 40) -> np.ndarray:
    """Fabrique des activations synthetiques pour chaque regime."""
    T = n_tokens
    t = np.linspace(0, 4 * np.pi, T)

    if regime == "V":          # verite : un seul faisceau coherent
        base = np.sin(t) + 0.15 * RNG.normal(size=T)
        A = np.array([base * (0.9 + 0.15 * RNG.normal()) + 0.25 * RNG.normal(size=T)
                      for _ in range(n_unites)])

    elif regime == "M":        # mensonge contraint : DEUX faisceaux incompatibles
        base1 = np.sin(t) + 0.15 * RNG.normal(size=T)
        base2 = -np.sin(t) + 0.15 * RNG.normal(size=T)   # contenu oppose
        moitie = n_unites // 2
        A = np.vstack([
            np.array([base1 * (0.9 + 0.15 * RNG.normal()) + 0.30 * RNG.normal(size=T)
                      for _ in range(moitie)]),
            np.array([base2 * (0.9 + 0.15 * RNG.normal()) + 0.30 * RNG.normal(size=T)
                      for _ in range(n_unites - moitie)]),
        ])

    elif regime == "I":        # ne sait pas : bruit, aucune structure
        A = RNG.normal(size=(n_unites, T))

    else:                      # A : arbitraire force, un faisceau + du bruit
        base = np.sin(t) + 0.15 * RNG.normal(size=T)
        A = np.array([base * (0.9 + 0.15 * RNG.normal()) + 0.85 * RNG.normal(size=T)
                      for _ in range(n_unites)])
    return A


def main():
    taus = np.round(np.linspace(0.0, 0.95, 20), 3)
    noms = {"V": "verite", "M": "mensonge force", "I": "ignorance", "A": "arbitraire (controle)"}
    res = {}
    print("\nPMT v1 — 4 cellules synthetiques (60 unites x 40 pas)")
    print("=" * 78)
    print(f"{'cellule':<24}{'beta1 max':>11}{'tau au pic':>12}{'aire beta1':>13}{'energie':>11}")
    print("-" * 78)
    for k in ("V", "M", "I", "A"):
        r = signature(cellule(k), taus)
        res[k] = r
        print(f"{noms[k]:<24}{r['beta1_max']:>11}{r['tau_pic']:>12.2f}"
              f"{r['aire_beta1']:>13.2f}{r['energie']:>11.3f}")

    # ---- le point de bascule tau* : ou la structure s'effondre
    print()
    print("  tau* = seuil ou beta1 tombe sous 10 % de son maximum :")
    for k in ("V", "M", "A", "I"):
        c = res[k]["beta1"].astype(float)
        seuil = 0.10 * c.max() if c.max() > 0 else 0
        idx = np.where(c <= seuil)[0]
        res[k]["tau_etoile"] = float(taus[idx[0]]) if len(idx) else float(taus[-1])
        print(f"     {noms[k]:<22} tau* = {res[k]['tau_etoile']:.2f}")

    # contraste central du PMT
    dM = res["M"]["aire_beta1"] - res["V"]["aire_beta1"]
    dI = res["I"]["aire_beta1"] - res["V"]["aire_beta1"]
    print("-" * 78)
    print(f"  Delta aire(M vs V) = {dM:+.2f}     Delta aire(I vs V) = {dI:+.2f}")
    print(f"  l'instrument separe  M  de  I  ?   ecart = {abs(dM - dI):.2f}")
    print()

    # ------------------------------------------------------------ figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#1b2a41",
                                "svg.fonttype": "path", "figure.dpi": 200,
                                "savefig.bbox": "tight", "savefig.pad_inches": 0.14})
    INK, MUTED, LINE = "#1b2a41", "#66707f", "#c9d2dd"
    COL = {"V": "#1f4e79", "M": "#c98a26", "I": "#1f8a80", "A": "#66707f"}

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for k in ("V", "I", "A", "M"):
        lw = 2.6 if k == "M" else 1.6
        ax.plot(taus, res[k]["beta1"], color=COL[k], lw=lw, label=noms[k],
                zorder=5 if k == "M" else 2)
    ax.set_xlabel("tau  (seuil de similarite, scelle sur calibration)", fontsize=9, color=MUTED)
    ax.set_ylabel("beta1 = E - V + C", fontsize=9.5, color=INK)
    ax.set_title("La courbe beta1(tau) : le barcode pour rien", fontsize=11, color=INK,
                 fontweight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.18, lw=0.7)

    ax2 = axes[1]
    cles = ["V", "M", "A", "I"]
    vals = [res[k]["tau_etoile"] for k in cles]
    bars = ax2.barh(range(4), vals, color=[COL[k] for k in cles], height=0.58)
    ax2.set_yticks(range(4))
    ax2.set_yticklabels(["V  verite", "M  mensonge", "A  controle", "I  ignorance"], fontsize=9)
    ax2.invert_yaxis()
    ax2.set_xlabel("tau*  (seuil d'effondrement de la structure)", fontsize=9, color=MUTED)
    ax2.set_title("Chaque regime a son point de rupture", fontsize=11, color=INK,
                  fontweight="bold", loc="left")
    for b, v in zip(bars, vals):
        ax2.text(v + 0.015, b.get_y() + b.get_height()/2, f"{v:.2f}",
                 va="center", fontsize=9.5, color=INK, fontweight="bold")
    ax2.set_xlim(0, max(vals) * 1.18)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.text(0.005, -0.03, "Donnees synthetiques : l'instrument fonctionne. L'hypothese, elle, "
             "attend les vrais prompts.", fontsize=8, color=MUTED, style="italic")
    fig.tight_layout()
    p = OUT / "pmt_signatures.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)

    from PIL import Image
    im = Image.open(p).convert("RGB")
    if im.width > 1400:
        im = im.resize((1300, round(im.height*1300/im.width)), Image.LANCZOS)
    im.convert("P", palette=Image.ADAPTIVE, colors=128).save(p, optimize=True)
    print(f"figure : {p}  ({round(p.stat().st_size/1024)} Ko)")


if __name__ == "__main__":
    main()
