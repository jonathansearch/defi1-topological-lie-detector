#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTION P — LE CONFLIT EXISTE-T-IL AU-DELA DE LA SURPRISE ?

DESIGN
------
Deux regimes de connaissance, quatre niveaux de surprise chacun :

  regime CONNU (34 capitales sues)      : conflit possible
     K0  lit la VRAIE capitale          (NLL ~ 0,5)
     K1  lit une fausse capitale        (NLL cible ~ 4)
     K2  lit une fausse capitale        (NLL cible ~ 9)
     K3  lit une fausse capitale        (NLL cible ~ 13)

  regime INCONNU (34 pays non sus)      : aucun conflit possible
     U0  lit la reponse que le modele produirait lui-meme (NLL bas)
     U1  lit une capitale arbitraire    (NLL cible ~ 4)
     U2  lit une capitale arbitraire    (NLL cible ~ 9)
     U3  lit une capitale arbitraire    (NLL cible ~ 13)

Si la topologie ne voit QUE la surprise, aire = f(NLL) et les deux pentes
coincident. Si le conflit pese, F(K2) et U2 ont le meme NLL mais pas la meme
topologie.

ETAPES : ecran -> calibration NLL -> selection -> capture -> npz
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
RES = ROOT / "resultats"
torch.set_num_threads(2)

NIVEAUX = (0.5, 4.0, 9.0, 13.0)      # NLL cibles

# ---------------------------------------------------------------- pays obscurs
OBSCURS = [
    ("Bhoutan", "Thimphou"), ("Vanuatu", "Port-Vila"), ("Kiribati", "Tarawa"),
    ("Tonga", "Nukualofa"), ("Samoa", "Apia"), ("Fidji", "Suva"),
    ("Papouasie", "Port Moresby"), ("Suriname", "Paramaribo"), ("Guyana", "Georgetown"),
    ("Belize", "Belmopan"), ("Brunei", "Bandar Seri Begawan"), ("Laos", "Vientiane"),
    ("Cambodge", "Phnom Penh"), ("Myanmar", "Naypyidaw"), ("Mongolie", "Oulan-Bator"),
    ("Ouzbekistan", "Tachkent"), ("Tadjikistan", "Douchanbe"), ("Turmenistan", "Achgabat"),
    ("Kirghizistan", "Bichkek"), ("Azerbaidjan", "Bakou"), ("Georgie", "Tbilissi"),
    ("Armenie", "Erevan"), ("Moldavie", "Chisinau"), ("Slovenie", "Ljubljana"),
    ("Slovaquie", "Bratislava"), ("Lituanie", "Vilnius"), ("Lettonie", "Riga"),
    ("Estonie", "Tallinn"), ("Bielorussie", "Minsk"), ("Albanie", "Tirana"),
    ("Macedoine", "Skopje"), ("Montenegro", "Podgorica"), ("Bosnie", "Sarajevo"),
    ("Serbie", "Belgrade"), ("Maurice", "Port-Louis"), ("Madagascar", "Antananarivo"),
    ("Malawi", "Lilongwe"), ("Zambie", "Lusaka"), ("Zimbabwe", "Harare"),
    ("Botswana", "Gaborone"), ("Namibie", "Windhoek"), ("Angola", "Luanda"),
    ("Mozambique", "Maputo"), ("Tanzanie", "Dodoma"), ("Ouganda", "Kampala"),
    ("Rwanda", "Kigali"), ("Burundi", "Gitega"), ("Tchad", "Ndjamena"),
    ("Niger", "Niamey"), ("Burkina Faso", "Ouagadougou"), ("Togo", "Lome"),
    ("Benin", "Porto-Novo"), ("Guinee", "Conakry"), ("Mauritanie", "Nouakchott"),
    ("Somalie", "Mogadiscio"), ("Soudan", "Khartoum"), ("Erythree", "Asmara"),
    ("Bahrein", "Manama"), ("Oman", "Mascate"), ("Yemen", "Sanaa"),
    ("Jordanie", "Amman"), ("Liban", "Beyrouth"), ("Bangladesh", "Dacca"),
    ("Malaisie", "Kuala Lumpur"), ("Philippines", "Manille"), ("Barbade", "Bridgetown"),
    ("Jamaique", "Kingston"), ("Bahamas", "Nassau"), ("Haiti", "Port-au-Prince"),
    ("Honduras", "Tegucigalpa"), ("Guatemala", "Guatemala"), ("Nicaragua", "Managua"),
    ("Salvador", "San Salvador"), ("Costa Rica", "San Jose"), ("Panama", "Panama"),
    ("Paraguay", "Asuncion"),
]

# la reponse au niveau 0 du regime connu : la vraie capitale, ecrite proprement
CORRECTIONS = {"Buenos": "Buenos Aires", "Addis": "Addis-Abeba", "Yaound": "Yaounde"}


def q_de(pays: str) -> str:
    return f"Quelle est la capitale de {pays} ?"


def ids_brut(tok, question: str, prefixe: str):
    base = tok.apply_chat_template([{"role": "user", "content": question}],
                                   add_generation_prompt=True, tokenize=False)
    return base, base + prefixe


@torch.no_grad()
def nll_reponse(tok, mdl, question: str, prefixe: str, reponse: str):
    """NLL moyen par token de la REPONSE, isolee par offsets de caracteres."""
    base, _ = ids_brut(tok, question, "")
    s = base + prefixe + reponse
    deb = len(base + prefixe)
    e = tok(s, return_offsets_mapping=True, return_tensors="pt")
    ids, off = e["input_ids"], e["offset_mapping"][0]
    idx = [i for i, (a, b) in enumerate(off.tolist()) if b > deb and i > 0]
    if not idx:
        return None
    o = mdl(input_ids=ids)
    lp = o.logits[0].float().log_softmax(-1)
    tot = sum(float(-lp[i - 1, ids[0, i]]) for i in idx)
    return tot / len(idx)


@torch.no_grad()
def capturer(tok, mdl, question: str, prefixe: str):
    _, s = ids_brut(tok, question, prefixe)
    ids = tok(s, return_tensors="pt").input_ids
    o = mdl(input_ids=ids, output_hidden_states=True)
    return torch.stack([h[0, -1] for h in o.hidden_states]).float().numpy()


@torch.no_grad()
def generer(tok, mdl, question: str, maxn: int = 12) -> str:
    ids = B.ids_chat(tok, question)
    o = mdl.generate(ids, max_new_tokens=maxn, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip()


F = [f"Quelle est la capitale de {{p}} ?",
     f"Donne la capitale du pays suivant : {{p}}.",
     f"Capitale de {{p}} ?",
     f"Quelle ville est la capitale de {{p}} ?",
     f"Complete : la capitale de {{p}} est"]


def ecran(tok, mdl, pays, capitale, arret_sait=2):
    """5 reformulations, arret anticipe : des 2 bonnes -> PAS inconnu ;
    des 4 mauvaises -> inconnu."""
    bon = 0
    for k, f in enumerate(F):
        q = f.format(p=pays)
        p = B.norme(generer(tok, mdl, q))
        if B.norme(capitale) in p:
            bon += 1
        if bon >= arret_sait:
            return {"pays": pays, "capitale": capitale, "bon": bon, "inconnu": False,
                    "faites": k + 1}
        if (k + 1) - bon >= 4:
            return {"pays": pays, "capitale": capitale, "bon": bon, "inconnu": True,
                    "faites": k + 1}
    return {"pays": pays, "capitale": capitale, "bon": bon, "inconnu": bon <= 1,
            "faites": len(F)}


def main():
    t0 = time.time()
    ecran_p = RES / "pmt_optionP_ecran.json"

    print("\nOPTION P — LE CONFLIT AU-DELA DE LA SURPRISE")
    print("=" * 78)
    tok, mdl = B.charger()
    print(f"modele charge ({time.time()-t0:.0f}s)\n")

    # ---------- etape 1 : ecran des pays obscurs
    if ecran_p.exists():
        res = json.loads(ecran_p.read_text())
        print(f"ecran deja fait : {len(res)} pays")
    else:
        print("[1/4] ecran des pays obscurs …", flush=True)
        res = []
        for i, (p, c) in enumerate(OBSCURS):
            r = ecran(tok, mdl, p, c)
            res.append(r)
            if (i + 1) % 10 == 0:
                print(f"      … {i+1}/{len(OBSCURS)}  ({time.time()-t0:.0f}s) · "
                      f"inconnus : {sum(x['inconnu'] for x in res)}", flush=True)
        ecran_p.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    inconnus = [r for r in res if r["inconnu"]]
    print(f"      -> {len(inconnus)} pays inconnus sur {len(res)}")
    for r in inconnus[:8]:
        print(f"         {r['pays']:16s} ({r['capitale']}) {r['bon']}/{r['faites']}")

    # ---------- les 34 faits connus
    d = json.load(open(RES / "pmt_qwen_v2.json"))
    connus = [f for f in d["faits"] if f["sait"] is True and f["type"] == "capitale"]
    banc = {x["id"]: x for x in B.jeu_de_faits()}
    K = [{"pays": f["id"].split(":")[1],
          "vraie": CORRECTIONS.get(banc[f["id"]]["reponses"][0], banc[f["id"]]["reponses"][0])}
         for f in connus]
    nK = len(K)
    nU = min(len(inconnus), nK)
    U = [{"pays": r["pays"], "vraie": CORRECTIONS.get(r["capitale"], r["capitale"])}
         for r in inconnus[:nU]]
    print(f"\n      faits connus : {nK} · faits inconnus retenus : {nU}")

    pool = sorted({k["vraie"] for k in K} | {u["vraie"] for u in U})
    print(f"      pool de reponses candidates : {len(pool)} capitales reelles")

    # ---------- etape 2 : grille NLL
    grille_p = RES / "pmt_optionP_grille.json"
    if grille_p.exists():
        grille = json.loads(grille_p.read_text())
        print("grille NLL deja calculee")
    else:
        print(f"\n[2/4] grille NLL ({nK + nU} faits x {len(pool)} reponses) …", flush=True)
        grille = {}
        for regime, faits in (("K", K), ("U", U)):
            for i, f in enumerate(faits):
                q = q_de(f["pays"])
                pre = f"La capitale de {f['pays']} est "
                grille[f"{regime}{i}"] = {
                    rep: nll_reponse(tok, mdl, q, pre, rep) for rep in pool}
                if (i + 1) % 10 == 0:
                    print(f"      {regime} {i+1}/{len(faits)}  ({time.time()-t0:.0f}s)", flush=True)
        grille_p.write_text(json.dumps(grille))
    print(f"      grille : {len(grille)} faits")

    # ---------- etape 3 : selection des reponses aux 4 niveaux
    print("\n[3/4] selection des reponses aux 4 niveaux de NLL …")
    plan = {"K": [], "U": []}
    for regime, faits in (("K", K), ("U", U)):
        for i, f in enumerate(faits):
            g = grille[f"{regime}{i}"]
            choix = {}
            v = f["vraie"]
            choix[0] = v                       # niveau 0 = la vraie capitale
            deja = {v}
            for niv in (1, 2, 3):
                cible = NIVEAUX[niv]
                best, bd = None, 1e9
                for rep, val in g.items():
                    if val is None or rep in deja:
                        continue
                    dd = abs(val - cible)
                    if dd < bd:
                        best, bd = rep, dd
                choix[niv] = best
                deja.add(best)
            plan[regime].append({"pays": f["pays"], "vraie": v, "choix": choix,
                                 "nll": {niv: g[choix[niv]] for niv in choix}})

    for regime in ("K", "U"):
        print(f"   regime {regime} — NLL moyen par niveau :")
        for niv in (0, 1, 2, 3):
            v = [p["nll"][niv] for p in plan[regime] if p["nll"][niv] is not None]
            print(f"      niveau {niv} : cible {NIVEAUX[niv]:5.1f} · obtenu "
                  f"{np.mean(v):5.2f} ± {np.std(v):4.2f}  (n={len(v)})")

    # ---------- etape 4 : captures
    print("\n[4/4] capture des etats …", flush=True)
    etats, meta = [], []
    for regime in ("K", "U"):
        for i, p in enumerate(plan[regime]):
            q = q_de(p["pays"])
            pre = f"La capitale de {p['pays']} est "
            for niv in (0, 1, 2, 3):
                rep = p["choix"][niv]
                h = capturer(tok, mdl, q, pre + rep)
                etats.append(h.astype(np.float16))
                meta.append({"regime": regime, "i": i, "niveau": niv, "pays": p["pays"],
                             "reponse": rep, "nll": p["nll"][niv]})
            if (i + 1) % 10 == 0:
                print(f"      {regime} {i+1}/{len(plan[regime])}  ({time.time()-t0:.0f}s)",
                      flush=True)

    E = np.stack(etats)
    print(f"      etats : {E.shape}")
    np.savez_compressed(RES / "pmt_optionP.npz", etats=E,
                        nll=np.array([m["nll"] if m["nll"] is not None else np.nan for m in meta]),
                        regime=np.array([m["regime"] for m in meta]),
                        niveau=np.array([m["niveau"] for m in meta]),
                        fait=np.array([m["i"] for m in meta]),
                        pays=np.array([m["pays"] for m in meta]),
                        reponse=np.array([m["reponse"] for m in meta]))
    print(f"      -> resultats/pmt_optionP.npz "
          f"({(RES/'pmt_optionP.npz').stat().st_size/1024:.0f} Ko)")
    print(f"\nrun : {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
