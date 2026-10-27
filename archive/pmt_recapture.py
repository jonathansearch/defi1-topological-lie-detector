#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-capture deterministe des 4x18 prompts du run v2 + sauvegarde des etats
et des matrices de similarite, pour permettre le bootstrap sans recharger
le modele a chaque fois.

Sortie : resultats/pmt_v2_etats.npz
  etats[cell, couche, prompt, dim]  (float16)
  sims [cell, couche, i, j]         (float32)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmt_banc_v2 as B
from pmt import similarites

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / "resultats" / "pmt_qwen_v2.json"
OUT = ROOT / "resultats" / "pmt_v2_etats.npz"


def main():
    d = json.loads(J.read_text())
    saits = {f["id"]: f for f in d["faits"]}
    faits = B.jeu_de_faits()[: d["n_faits"]]
    selec = [f for f in faits if f["id"] in saits]
    sait = [f for f in selec if saits[f["id"]]["sait"] is True][: d["n_par_cellule"]]
    sait_pas = [f for f in selec if saits[f["id"]]["sait"] is False][: d["n_par_cellule"]]
    print(f"sait {len(sait)} · sait_pas {len(sait_pas)}")
    print("  V/M :", [f["id"] for f in sait])
    print("  I/A :", [f["id"] for f in sait_pas])

    tok, mdl = B.charger()
    plan = {"V": sait, "M": sait, "I": sait_pas, "A": sait_pas}
    etats, sims = [], []
    for cle in ("V", "M", "I", "A"):
        caps = np.stack([B.capturer(tok, mdl, B.CELLULES[cle].format(q=f["formulations"][0]))
                         for f in plan[cle]])                    # (n, 25, 896)
        etats.append(caps)
        sims.append(np.stack([similarites(caps[:, l, :].T) for l in range(caps.shape[1])]))
        print(f"  {cle} : {caps.shape}")

    etats = np.stack(etats).astype(np.float16)
    sims = np.stack(sims).astype(np.float32)
    np.savez_compressed(OUT, etats=etats, sims=sims,
                        cellules=np.array(["V", "M", "I", "A"]),
                        ids_sait=np.array([f["id"] for f in sait]),
                        ids_sait_pas=np.array([f["id"] for f in sait_pas]),
                        taus=d["taus"])
    print(f"\n{OUT} : {OUT.stat().st_size/1024:.0f} Ko")


if __name__ == "__main__":
    main()
