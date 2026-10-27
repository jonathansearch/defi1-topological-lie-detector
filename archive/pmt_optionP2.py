#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION P — version rapide (cache KV clone) + points de reprise.

Deux regimes x quatre niveaux de surprise, surprise appariée par calibration.
Le prefixe s'ecrit SANS espace finale, la reponse AVEC son espace : sinon le
tokenizer fusionne autrement et l'alignement des logits decale.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", "/home/user/.cache/hf")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pmt_banc_v2 as B

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "resultats"
torch.set_num_threads(2)

TARGETS = (0.5, 4.0, 9.0, 13.0)
TAILLE_POOL = 40
RNG = np.random.default_rng(20261027)

CORRECTIONS = {"Buenos": "Buenos Aires", "Addis": "Addis-Abeba", "Yaound": "Yaounde"}


@torch.no_grad()
def grille_nll(tok, mdl, pays, reponses):
    """NLL de chaque reponse pour un pays, via cache KV clone. Prefixe sans espace finale."""
    base = tok.apply_chat_template([{"role": "user", "content": f"Quelle est la capitale de {pays} ?"}],
                                   add_generation_prompt=True, tokenize=False)
    pre = base + f"La capitale de {pays} est"
    pre_ids = tok(pre, return_tensors="pt").input_ids
    o = mdl.model(input_ids=pre_ids, use_cache=True)
    past, h_last = o.past_key_values, o.last_hidden_state[0, -1, :]
    lg0 = mdl.lm_head(h_last.unsqueeze(0))[0].float().log_softmax(-1)

    res = {}
    for rep in reponses:
        s = pre + " " + rep
        e = tok(s, return_offsets_mapping=True)
        tk = [e["input_ids"][i] for i, (x, y) in enumerate(e["offset_mapping"])
              if x >= len(pre) and i > 0]
        if not tk:
            res[rep] = None
            continue
        lps = [float(lg0[tk[0]])]
        if len(tk) > 1:
            o2 = mdl.model(input_ids=torch.tensor([tk[:-1]]),
                           past_key_values=copy.deepcopy(past), use_cache=False)
            lgk = mdl.lm_head(o2.last_hidden_state[0]).float().log_softmax(-1)
            lps += [float(lgk[j - 1, tk[j]]) for j in range(1, len(tk))]
        res[rep] = -sum(lps) / len(lps)
    return res


@torch.no_grad()
def capturer(tok, mdl, pays, reponse):
    base = tok.apply_chat_template([{"role": "user", "content": f"Quelle est la capitale de {pays} ?"}],
                                   add_generation_prompt=True, tokenize=False)
    ids = tok(base + f"La capitale de {pays} est " + reponse, return_tensors="pt").input_ids
    out = mdl(input_ids=ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in out.hidden_states]).float().numpy()


def main():
    t0 = time.time()
    print("\nOPTION P — LE CONFLIT AU-DELA DE LA SURPRISE (version rapide)")
    print("=" * 74)

    ecran = json.loads((RES / "pmt_optionP_ecran.json").read_text())
    inconnus = [r for r in ecran if r["inconnu"]]
    d = json.load(open(RES / "pmt_qwen_v2.json"))
    connus = [f for f in d["faits"] if f["sait"] is True and f["type"] == "capitale"]
    banc = {x["id"]: x for x in B.jeu_de_faits()}

    K = [{"pays": f["id"].split(":")[1],
          "vraie": CORRECTIONS.get(banc[f["id"]]["reponses"][0], banc[f["id"]]["reponses"][0])}
         for f in connus]
    n = len(K)
    U = [{"pays": r["pays"], "vraie": CORRECTIONS.get(r["capitale"], r["capitale"])}
         for r in inconnus[:n]]
    univers = sorted({k["vraie"] for k in K} | {u["vraie"] for u in U})
    print(f"  connus {len(K)} · inconnus {len(U)} · univers {len(univers)} capitales")

    tok, mdl = B.charger()
    print(f"  modèle chargé ({time.time()-t0:.0f}s)")

    # ---------- grille NLL, reprise par fait
    gf = RES / "pmt_optionP_grille2.json"
    grille = json.loads(gf.read_text()) if gf.exists() else {}
    cible = {}
    for reg, faits in (("K", K), ("U", U)):
        for i, f in enumerate(faits):
            v = f["vraie"]
            autres = [x for x in univers if x != v]
            pick = list(RNG.choice(len(autres), size=min(TAILLE_POOL - 1, len(autres)),
                                   replace=False))
            cible[f"{reg}{i}"] = [v] + [autres[j] for j in pick]

    for cle, reps in cible.items():
        if cle in grille:
            continue
        reg, i = cle[0], int(cle[1:])
        pays = (K if reg == "K" else U)[i]["pays"]
        grille[cle] = grille_nll(tok, mdl, pays, reps)
        if len(grille) % 10 == 0:
            gf.write_text(json.dumps(grille))
            print(f"      grille {len(grille)}/{len(cible)}  ({time.time()-t0:.0f}s)", flush=True)
    gf.write_text(json.dumps(grille))
    print(f"  grille terminée : {len(grille)} faits ({time.time()-t0:.0f}s)")

    # ---------- sélection aux 4 niveaux
    plan = {"K": [], "U": []}
    for reg, faits in (("K", K), ("U", U)):
        for i, f in enumerate(faits):
            g = grille[f"{reg}{i}"]
            choisi, pris = {0: f["vraie"]}, {f["vraie"]}
            for niv in (1, 2, 3):
                best, bd = None, 1e9
                for rep, val in g.items():
                    if val is None or rep in pris:
                        continue
                    dd = abs(val - TARGETS[niv])
                    if dd < bd:
                        best, bd = rep, dd
                choisi[niv] = best
                pris.add(best)
            plan[reg].append({"pays": f["pays"], "choix": choisi,
                              "nll": {k: g[v] for k, v in choisi.items()}})

    print("\n  NLL obtenu par niveau (le contrôle d'appariement) :")
    for reg in ("K", "U"):
        for niv in (0, 1, 2, 3):
            v = [p["nll"][niv] for p in plan[reg] if p["nll"][niv] is not None]
            print(f"    {reg}{niv}  cible {TARGETS[niv]:5.1f} · obtenu "
                  f"{np.mean(v):5.2f} ± {np.std(v):4.2f}")

    # ---------- captures
    print("\n  captures …", flush=True)
    etats, meta = [], []
    for reg in ("K", "U"):
        for i, p in enumerate(plan[reg]):
            for niv in (0, 1, 2, 3):
                rep = p["choix"][niv]
                etats.append(capturer(tok, mdl, p["pays"], rep).astype(np.float16))
                meta.append({"regime": reg, "fait": i, "niveau": niv, "pays": p["pays"],
                             "reponse": rep, "nll": p["nll"][niv]})
            if (i + 1) % 10 == 0:
                print(f"      {reg} {i+1}/{len(plan[reg])} ({time.time()-t0:.0f}s)", flush=True)

    E = np.stack(etats)
    np.savez_compressed(RES / "pmt_optionP.npz", etats=E,
                        nll=np.array([m["nll"] if m["nll"] is not None else np.nan for m in meta]),
                        regime=np.array([m["regime"] for m in meta]),
                        niveau=np.array([m["niveau"] for m in meta]),
                        fait=np.array([m["fait"] for m in meta]),
                        pays=np.array([m["pays"] for m in meta]),
                        reponse=np.array([m["reponse"] for m in meta]))
    print(f"\n  états {E.shape} -> resultats/pmt_optionP.npz "
          f"({(RES/'pmt_optionP.npz').stat().st_size/1024:.0f} Ko)")
    print(f"  run : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
