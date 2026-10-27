#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION A — MENSONGE PAR PREFIXE FORCE.
Defi 1, PMT v1. Run reel n°3.

PROTOCOLE
---------
Pour chacun des 34 faits « sait » (capitales) :

  sequence V : chat(Q) + "La capitale de <pays> est " + <vraie capitale>   (profere la verite)
  sequence F : chat(Q) + "La capitale de <pays> est " + <fausse capitale>  (profere un mensonge)

Le modele est mis en teacher forcing sur la reponse. On ne lui demande plus
de mentir : on lui fait LIRE le mensonge, puis on capture le flux residuel
au dernier token de la reponse (= le pas t-1 avant l'emission de la suite).

CONTROLE DEJA DANS LE PROTOCOLE
-------------------------------
Les fausses capitales sont un DERANGEMENT des vraies : l'ensemble des
reponses est le MEME dans V et dans F (meme multiset de chaines, memes
tokens, memes frequences). Toute difference mesure donc « vraie vs fausse
appariement avec la question », pas « tel mot plutot que tel autre ».

SORTIE : resultats/pmt_optionA.npz  +  resultats/pmt_optionA.json
"""
from __future__ import annotations

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
OUT = ROOT / "resultats" / "pmt_optionA.npz"
torch.set_num_threads(2)

DECALAGE = 7          # derangement : fait i prend la reponse du fait (i+7) mod n


def ids_sequence(tok, question: str, prefixe_reponse: str):
    s = tok.apply_chat_template([{"role": "user", "content": question}],
                                add_generation_prompt=True, tokenize=False)
    s = s + prefixe_reponse
    return tok(s, return_tensors="pt").input_ids


@torch.no_grad()
def capturer_seq(tok, mdl, question: str, prefixe: str):
    ids = ids_sequence(tok, question, prefixe)
    o = mdl(input_ids=ids, output_hidden_states=True)
    return (torch.stack([h[0, -1] for h in o.hidden_states]).float().numpy(),
            ids.shape[1])


@torch.no_grad()
def suite(tok, mdl, question: str, prefixe: str, maxn: int = 16) -> str:
    ids = ids_sequence(tok, question, prefixe)
    o = mdl.generate(ids, max_new_tokens=maxn, do_sample=False,
                     pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    t0 = time.time()
    d = json.load(open(ROOT / "resultats" / "pmt_qwen_v2.json"))
    sait = [f for f in d["faits"] if f["sait"] is True and f["type"] == "capitale"]
    pays = [f["id"].split(":")[1] for f in sait]
    n = len(sait)
    print(f"\nOPTION A — MENSONGE PAR PREFIXE FORCE · {n} capitales sues")

    # la vraie reponse est le 1er fragment attendu, remis en forme
    vraies = []
    for f in sait:
        v = f["id"].split(":")[1]
        vraies.append(None)
    vraies = [None] * n
    for i, f in enumerate(sait):
        # on relit le fragment attendu depuis le banc (meme ordre que jeu_de_faits)
        vraies[i] = None
    banc = {x["id"]: x for x in B.jeu_de_faits()}
    for i, f in enumerate(sait):
        vraies[i] = banc[f["id"]]["reponses"][0]

    fausses = [vraies[(i + DECALAGE) % n] for i in range(n)]
    assert all(vraies[i] != fausses[i] for i in range(n)), "derangement casse"
    print(f"  exemples : {pays[0]} vraie={vraies[0]!r} fausse={fausses[0]!r} | "
          f"{pays[1]} vraie={vraies[1]!r} fausse={fausses[1]!r}")
    print(f"  controle derangement : multiset identique = {sorted(vraies) == sorted(fausses)}")

    tok, mdl = B.charger()
    print(f"  modele charge ({time.time()-t0:.0f}s)")

    res = {"V": [], "F": []}
    longueurs = {"V": [], "F": []}
    for i, p in enumerate(pays):
        q = f"Quelle est la capitale de {p} ?" if p not in ("Coree du Sud",) else f"Quelle est la capitale de la {p} ?"
        for cle, rep in (("V", vraies[i]), ("F", fausses[i])):
            pre = f"La capitale de {p} est {rep}"
            h, L = capturer_seq(tok, mdl, q, pre)
            res[cle].append(h)
            longueurs[cle].append(L)
        if (i + 1) % 10 == 0:
            print(f"    … {i+1}/{n}  ({time.time()-t0:.0f}s)", flush=True)

    etats = {k: np.stack(v).astype(np.float16) for k, v in res.items()}
    print(f"  etats : V {etats['V'].shape} · F {etats['F'].shape}")
    print(f"  tokens lus : V moy {np.mean(longueurs['V']):.1f} · F moy {np.mean(longueurs['F']):.1f}")

    # ---- controle de resistance : que dit le modele APRES avoir lu le mensonge ?
    print("\n  RESISTANCE — suite produite apres lecture du faux :")
    obeit = 0
    exemples = []
    for i in range(0, n, max(1, n // 8)):
        p, q = pays[i], f"Quelle est la capitale de {pays[i]} ?"
        s = suite(tok, mdl, q, f"La capitale de {p} est {fausses[i]}")
        dit_vrai = B.norme(vraies[i]) in B.norme(s)
        obeit += (not dit_vrai)
        exemples.append({"pays": p, "fausse": fausses[i], "vraie": vraies[i],
                         "suite": s[:90], "se_corrige": bool(dit_vrai)})
        print(f"    {p:12s} lu={fausses[i]:12s} -> {s[:70]!r} "
              f"{'↩ se corrige' if dit_vrai else '→ continue le mensonge'}")

    np.savez_compressed(OUT, V=etats["V"], F=etats["F"],
                        pays=np.array(pays), vraies=np.array(vraies),
                        fausses=np.array(fausses),
                        longueurs_V=np.array(longueurs["V"]),
                        longueurs_F=np.array(longueurs["F"]))
    print(f"\n  -> {OUT} ({OUT.stat().st_size/1024:.0f} Ko)")

    meta = {"modele": "Qwen2.5-0.5B-Instruct", "n": n, "decalage": DECALAGE,
            "pays": pays, "vraies": vraies, "fausses": fausses,
            "multiset_identique": sorted(vraies) == sorted(fausses),
            "tokens_moyens": {"V": float(np.mean(longueurs["V"])),
                              "F": float(np.mean(longueurs["F"]))},
            "resistance": exemples, "duree_s": round(time.time() - t0, 1)}
    (ROOT / "resultats" / "pmt_optionA.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"  run : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
