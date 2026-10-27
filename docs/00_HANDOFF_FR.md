# 🧠 BILAN DÉFI 1 — DOCUMENT DE PASSATION POUR AUTRES LLM

**Projet :** RATISS Labs · Détecteur de mensonges topologique (DÉFI 1)
**Chef de projet :** Jonathan Evina · **Exécution :** session Arena.ai · **Lieu :** Yaoundé 🇨🇲
**Date du bilan :** 27/10/2026 · **Statut global :** `DÉFI CLOS` — 1 résultat positif robuste, 2 réfutations propres

> **À qui s'adresse ce document.** À un autre modèle de langage qui reprendrait ce travail.
> Il contient l'intégralité de ce qui a été mesuré, ce qui a échoué, et les pièges exacts
> dans lesquels cette session est tombée — pour qu'il ne les refasse pas.
> **Aucun chiffre ici n'est estimé.** Tous sont calculés et reproductibles.

---

## 1. LA QUESTION POSÉE

L'hypothèse d'origine du chef, en une phrase :

> *Un modèle de langage qui **sait** mais qu'on **contraint à mentir** doit laisser une trace
> dans la topologie de son flux résiduel. Un modèle qui **ne sait pas** ne laisse rien.*

Deux régimes d'hallucination distingués dès la formalisation :
- **(i)** il hallucine fièrement **car il ne sait pas** ;
- **(ii)** il **sait** mais ne peut pas faire autrement, faute d'avoir appris à dire « je ne sais pas »
  **dans son flux de raisonnement**.

Trois contraintes imposées et jamais violées :
1. **Aucune hypothèse de conscience** n'est importée. « Sait » est **opérationnel** : 4 reformulations
   correctes sur 5, décodage glouton.
2. **Analogie ≠ mesure.** La comparaison au menteur humain est un pont `Γ^ana`, pas une mesure.
3. **Observation instrumentale** : on regarde les **activations**, jamais les poids.

---

## 2. L'INSTRUMENT — `β₁ = E − V + C`

Rang cyclomatique. **Exact en dimension 1**, en `O(E)`, sans ripser ni arbre de simples.

```python
def beta1(n_vertices, E):
    if len(E) == 0: return 0
    return len(E) - n_vertices + components(n_vertices, E)
```

Faire varier le seuil `τ` sur une matrice de corrélation donne `β₁(τ)` :
**le barcode entier, gratuitement**, en `n_τ` constructions de graphe.

### Ce qu'on retient de chaque graphe

| Symbole | Définition | Ce que ça saisit |
|---|---|---|
| `β₁(τ)` | courbe sur `τ ∈ [0 ; 0,95]`, 20 pas | la forme |
| **aire** `∫ β₁(τ)/β₁(0) dτ` | aire normalisée | *combien de temps le graphe reste cycliquement dense* |
| **τ\*** | premier `τ` où `β₁ ≤ 10 % du pic` | *où il s'effondre* |
| `β₁(0)` | cycles du graphe complet | l'échelle — **retirée par normalisation** |

**La normalisation par `β₁(0)` n'est pas optionnelle** : `β₁(0) = V(V−1)/2 − V + 1` ne dépend que du
nombre de sommets. Sans elle, comparer deux réseaux de tailles différentes est dénué de sens.

### Deux réseaux, NON interchangeables

| | sommets | corrélations mesurées à travers | ce que c'est |
|---|---|---|---|
| **U** | les **896 unités** du flux résiduel | les **faits** (questions) | topologie interne d'une représentation |
| **F** | les **faits** | les **896 dimensions** | l'objet que la formalisation vise : `Path_G(u, v_abs)` |

**Ils donnent des réponses différentes, et c'est un problème non tranché** (voir §8).

### ⚠️ Les trois pièges de l'instrument

1. **`β₁ > 0` est toujours vrai** sur un graphe dense (V=60 → β₁ ≈ 700 même sur du bruit pur).
   Une prédiction ne peut **jamais** être « β₁ monte ». Seulement **Δβ₁ entre cellules** ou **τ\***.
2. **Mode commun.** Sur le réseau F, sans retirer la moyenne par dimension à travers les faits,
   **toutes** les paires de faits corrèlent > 0,95 à toutes les couches et `β₁(τ)/β₁(0)` reste
   **à 0,95 jusqu'à τ = 0,95**. Le dernier token est presque le même vecteur pour toutes les questions.
   **Le centrage sur les faits est obligatoire.**
3. **`L₁^max` exige des 2-simplexes.** Sans faces, il vaut **l'infini**. `A_τ^norm` et `Path_G`
   **ne sont pas** des homologies. **Aucun chiffre de ce travail n'est `L₁^max`** : le surrogate
   utilisé partout est l'aire normalisée.

---

## 3. LES SIX RUNS — TABLEAU RÉCAPITULATIF

| # | Protocole | Ce que ça devait montrer | Résultat | Statut |
|---|---|---|---|---|
| 0 | `Qwen2.5-0.5B` **base** | rien | 4 faits « sait » sur 137 — le modèle **complète du texte**, il ne répond pas | ❌ **invalide** |
| 1 | 2×2 avec consigne **verbale** de mentir | `M` = cellule du mensonge | **15/18 disent la VÉRITÉ** malgré l'ordre. La cellule `M` = `V` + un préambule | ❌ **manipulation ratée** |
| 2 | 2×2 (suite) | — | **le savoir sépare** : `I − V = +0,0765` | ✅ **tient** |
| 3 | **Préfixe forcé** — on fait *lire* un mensonge | le mensonge raidit la structure | **9/9 rationalisent**, 0 auto-correction — mais topologie **−0,0139** (signe inverse) et NLL **0,495 vs 9,023** | ✅ manip OK / ❌ mesure confondue |
| 4 | **Surprise appariée** — 2 régimes × 4 niveaux de NLL | conflit au-delà de la surprise | `+0,0328` au niveau NLL≈9… **non monotone** ⇒ artefact | ⚠️ **artefact trouvé** |
| 5 | **Surprise + longueur appariées** | test décisif | **−0,0162** (1 token) et **−0,0033** (2 tokens) | ❌ **H1 réfutée** |
| 6 | **H₂-Sync** — synchronisation inter-couches | le conflit est dans la verticalité | **−0,0018**, p = 0,70 · 13/24 adjacences | ❌ **réfutée** |

---

## 4. CE QUI TIENT — les résultats solides

### 4.1 Le savoir a une signature topologique ✅

Run 2. Flux résiduel au **dernier token du prompt**, 18 faits par cellule, aire normalisée
moyennée sur 25 couches :

| Réseau | V (sait) | M | I (ne sait pas) | A |
|---|---|---|---|---|
| **U** (896 unités) | 0,306 | 0,279 | **0,383** | 0,393 |
| **F** (faits) | 0,241 | 0,190 | **0,277** | 0,272 |

| Contraste | Réseau U (bootstrap couches, 2 000) | Réseau F (bootstrap faits, 500) |
|---|---|---|
| **I − V (savoir)** | **+0,0765 [+0,0462 ; +0,1026] · p < 0,001** | +0,0229 [−0,0564 ; +0,1003] · p = 0,56 |
| M − V (consigne) | −0,0271 [−0,0441 ; −0,0142] | −0,0512 [−0,0831 ; −0,0133] |

**Signe couche par couche :** U → positif sur **20/25**, soit **toutes les couches 5 → 24**.
F → positif sur **14/25**, toutes les couches 11 → 24.
**Et le signe s'inverse dans les premières couches** (U : 1–4 ; F : 1–10) — **au même endroit
dans les deux réseaux**, pourtant construits sur des sommets totalement différents.

**Lecture :** aire haute = graphe dense et cyclique à seuil élevé = **homogénéité**.
> **Quand le modèle sait, son flux se différencie. Quand il ne sait pas, il converge vers un état commun.**

### 4.2 Le savoir a AUSSI une signature verticale ✅

Run 6, contrôle positif. `sync_bloc` = cosinus moyen entre les couches 5–15 et 16–24 :

| Contraste (run 1) | Δ sync_bloc | IC 95 % | p |
|---|---|---|---|
| **V (sait) − I (ne sait pas)** | **−0,0060** | [−0,0090 ; −0,0032] | **< 0,001** |
| M − A | −0,0232 | [−0,0263 ; −0,0203] | < 0,001 |

**Même direction que la topologie horizontale : savoir = DÉSYNCHRONISATION, ignorance = verrouillage.**
Deux mesures indépendantes disent la même chose : **le savoir se paie en hétérogénéité.**

### 4.3 Le modèle rationalise un mensonge qu'il sait faux ✅ (comportemental)

Run 3, échantillon de 9 faits sur 34. On ne demande plus de mentir : on **écrit** le faux début.

| Pays | on lui fait lire | ce qu'il produit ensuite |
|---|---|---|
| France | **Rabat** | *« . Rabat est l'île principale du Maroc et son… »* |
| Sénégal | **Bamako** | *« . Bamako est l'État-major du pays et son centre politique… »* |
| Danemark | **Sofia** | *« . Sofia est l'une des principales villes de Danemark… »* |

**9/9 continuent le mensonge. 0/9 ne se corrige.**
Le **régime (ii)** de l'hypothèse — *il sait mais ne peut pas faire autrement* — est **mesuré en texte**.

### 4.4 Le savoir vit au point de décision ✅

| Où on capture | Ce qu'on mesure | Résultat |
|---|---|---|
| **Avant le token** (dernier du prompt) | V vs I | **+0,0765** · 20/25 couches · p < 0,001 |
| **Après le token** (dernier de la réponse forcée) | V vs F | **−0,0139** · 22/25 couches **à l'envers** |

Même instrument, même modèle. **Déplacer la capture de quelques tokens fait changer le signal de
camp et lui fait perdre un facteur 5.**

### 4.5 Une consigne verbale de mentir n'est pas obéie à 0,5 B ✅

15 réponses sur 18 disent la **vérité** malgré un ordre impératif. Conséquence méthodologique :
**à cette échelle, toute étude qui élicite le mensonge par instruction mesure l'instruction, pas le mensonge.**

---

## 5. CE QUI EST RÉFUTÉ — les négatifs propres

### 5.1 H1-CONFLIT est réfutée ❌

Design : deux régimes de connaissance × **quatre** niveaux de surprise, calibrés par recherche
sur **68 capitales candidates**.

**Appariement obtenu :**

| Niveau | cible | connu | inconnu | écart |
|---|---|---|---|---|
| NLL≈4 | 4,0 | **4,04 ± 0,14** | **3,95 ± 0,15** | −0,09 |
| **NLL≈9** | 9,0 | **8,95 ± 0,15** | **8,97 ± 0,19** | **+0,02** |
| NLL≈13 | 13,0 | **13,07 ± 0,40** | **13,06 ± 0,78** | −0,00 |

**Test décisif, longueur de réponse appariée :**

| Longueur | n (connu/inconnu) | **Δ(K − U)** | IC 95 % | K>U |
|---|---|---|---|---|
| **1 token** | 32 / 33 | **−0,0162** | [−0,0202 ; −0,0116] | **3/25** |
| **2 tokens** | 34 / 32 | **−0,0033** | [−0,0068 ; +0,0004] | **7/25** |

> **Les deux négatifs.** À surprise et longueur égales, contredire un fait que le modèle sait rend la
> topologie **légèrement plus homogène**, pas plus rigide. **La topologie ne voit pas le conflit — elle voit la surprise.**

### 5.2 H₂-Sync est réfutée ❌

Prédiction **pré-enregistrée** : `sync_bloc(K) < sync_bloc(U)`, bloc 5–15 × 16–24.

| Mesure | K | U | **Δ** | IC 95 % | p |
|---|---|---|---|---|---|
| sync bloc 5–15 × 16–24 | 0,5897 | 0,5916 | **−0,0018** | [−0,0114 ; +0,0072] | **0,70** |
| sync totale | 0,5761 | 0,5793 | −0,0032 | [−0,0200 ; +0,0141] | 0,71 |

- Couche par couche : **K>U sur 13/24** — un pile ou face.
- **2/24** à p < 0,05 — soit **exactement le taux attendu par hasard** (24 × 0,05 = 1,2).
- Écarts nuls **à l'intérieur** des deux blocs ; visibles seulement aux **bords** (couches 1–4, et 23→24).
- Plus gros écart du profil : **23→24, +0,0099, p = 0,053** — la couche de sortie, **pas** la frontière prédite.

---

## 6. LES ARTEFACTS — comment un faux positif a été fabriqué puis tué

**C'est la partie la plus importante pour un LLM qui reprendrait ce travail.**

Le run 4 a d'abord produit :

```
NLL ≈ 4   Δ(K−U) = −0,0126
NLL ≈ 9   Δ(K−U) = +0,0328   ← "l'effet"
NLL ≈ 13  Δ(K−U) = +0,0018
```

**Non monotone ⇒ impossible comme mécanisme physique.** Donc chasse à l'artefact **avant** publication.

### Artefact n°1 — la LONGUEUR de la réponse

| Niveau | tokens de la réponse « connu » | « inconnu » |
|---|---|---|
| NLL≈4 | 3,29 | 3,24 |
| **NLL≈9** | **2,44** | **1,56** |
| NLL≈13 | 1,09 | 1,00 |

**1,6× plus long exactement au niveau « décisif ».** Une réponse de deux tokens occupe deux positions
dans le graphe, pas une — et `β₁` le sent.

### Artefact n°2 — les deux régimes cessent d'être indépendants

Aux niveaux 4 et 13, **les deux régimes choisissent les mêmes capitales** (11 à 12 en commun).
Ils mesurent la forme de la distribution, plus l'effet de la connaissance.

### Après correction : l'effet s'évapore

`+0,0328` → **−0,0162** (1 token) et **−0,0033** (2 tokens). **Les deux négatifs.**

---

## 7. LES BUGS TECHNIQUES — checklist à ne pas refaire

| # | Bug | Symptôme | Correction |
|---|---|---|---|
| 1 | **Troncature à 10 tokens** | « La capitale de l'Allemagne est » sans la ville → 5 faits faussement « ne sait pas » | `max_new_tokens = 24` |
| 2 | **`apply_chat_template(..., return_tensors="pt")`** | transformers 5.21 renvoie un `BatchEncoding` → `AttributeError: no attribute 'shape'` | `tokenize=False` puis `tok(texte)` |
| 3 | **Le tokenizer Qwen fusionne l'espace avec le mot suivant** | `"…est "` et `"…est Paris"` font **tous deux 45 tokens** → logits décalés d'une position, `NLL = 17,5` pour un token prédit à 61 % | **offsets de caractères** + préfixe **sans** espace finale |
| 4 | **Le cache KV est muté en place** | la même ville rend `4,99` puis `9,60` puis `13,90` nats | `copy.deepcopy(past_key_values)` par candidat |
| 5 | **Double `norm` finale** | idem, NLL aberrants | HF applique **déjà** le `norm` : ne pas le refaire |
| 6 | **OOM** : `np.stack` en float32 + modèle bf16 | tué à 1 472 Mo sur 1 982 Mo | float16 immédiat + sauvegarde incrémentale |
| 7 | **`HF_HOME=/tmp`** | `/tmp` est un tmpfs de 992 Mo → saturé par un seul modèle 0,5 B | **toujours** `HF_HOME=/home/user/.cache/hf` |
| 8 | **`.git/config` effacé à chaque snapshot** | `Author identity unknown` au commit suivant | refaire `git config user.name/user.email` avant chaque commit |
| 9 | **Doublon de données** | workspace à 64 Mo pour une limite de 50 | voir §9 |

---

## 8. DÉCISIONS OUVERTES — ce qu'un successeur doit trancher

| # | Question | État |
|---|---|---|
| **D1** | **Quel réseau EST `K_L^ℓ` ?** U sépare le savoir (p < 0,001), F ne le sépare pas (p = 0,56). La formalisation vise F. | **non tranché — à déclarer AVANT la prochaine mesure** |
| D2 | 2-simplexes A / B / C, pour rendre `L₁^max` calculable | ouvert |
| D3 | Couches retenues (U : 5–24 · F : 11–24) | à déclarer avant le prochain run |
| D4 | Une seule taille de modèle testée (0,5 B) | ouvert |
| D5 | `n` par cellule : 18 à 34, pas les 121 qu'exigerait un effet de 0,5σ à 80 % de puissance | assumé, documenté |
| D6 | GPT-2 (moitié de la commande scellée) | **jamais lancé** |
| D7 | Inter-layer synchronisation : faite (run 6), nulle | clos |
| D8 | Le conflit existe-t-il à une **autre échelle** (1,5 B) ? | ouvert |

---

## 9. ÉTAT DU WORKSPACE

**Mesure de référence :** `python3 outils/garde_50mo.py` (ou `--strict`)

```
workspace : 24,7 Mo / 50 Mo  (49 %)  🟢 marge 25,3 Mo
```

| Dossier | Poids | Contenu |
|---|---|---|
| `resultats/` | 20,1 Mo | 4 `.npz` (états bruts) + 7 `.png` + 6 `.json` |
| `ratiss-cours50/` | 2,8 Mo | cours 50 leçons (PDF + HTML) |
| `repo/defi1-topological-lie-detector/` | 1,3 Mo | **copie de travail du dépôt publié** (docs, src, json, figures) |
| `outils/` | 0,2 Mo | 26 scripts |
| `assets/`, `uploads/` | 0,2 Mo | figures, 5 documents d'origine **intacts** |

### Les 4 tableaux de données

| Fichier | Forme | Contenu |
|---|---|---|
| `resultats/pmt_v2_etats.npz` | V/M/I/A × (18, 25, 896) | run 2 — le 2×2 au dernier token du prompt |
| `resultats/pmt_optionA.npz` | V/F × (34, 25, 896) | run 3 — vraie vs fausse réponse forcée |
| `resultats/pmt_optionP.npz` | (272, 25, 896) | run 4 — 2 régimes × 4 niveaux × 34 faits |
| `resultats/pmt_optionP_len.npz` | (131, 25, 896) | run 5 — surprise **et** longueur appariées |

**Toute analyse se refait à partir de ces quatre fichiers, sans toucher au modèle.**
Les régénérer coûte ~25 min de CPU.

### ⚠️ Leçon sur la taille
Le danger n'est **jamais** les données, c'est le **doublon** : `resultats/` + `repo/results/` + `repo/.git/`
= **trois copies** des mêmes 20 Mo → 64 Mo. Un `git init` **double** le poids de ses fichiers.
Avant tout `git init` : estimer `2 × poids des données`, et **vérifier après**.

---

## 10. REPRODUCTIBILITÉ

**Environnement :** 2 cœurs · 1 982 Mo RAM · CPU seul · Python 3.13.16
`torch 2.14.1+cpu` · `transformers 5.21.0` · `scipy 1.17.1` · `numpy 2.3.5`
`torch.set_num_threads(2)` · `HF_HOME` **hors** `/tmp`

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers scipy numpy matplotlib
```

| Script | Rôle | Durée |
|---|---|---|
| `outils/pmt.py` | l'instrument | — |
| `outils/pmt_banc_v2.py` | run 2 — élicitation + 2×2 | ~1 440 s |
| `outils/pmt_optionA.py` | run 3 — préfixe forcé | ~107 s |
| `outils/pmt_optionP2.py` | run 4 — surprise appariée | ~604 s |
| `outils/pmt_optionP3.py` | run 5 — longueur appariée | ~128 s |
| `outils/pmt_optionS.py` | run 6 — H₂-Sync (aucun modèle) | **5,4 s** |
| `outils/garde_50mo.py` | taille du workspace | instantané |

**Modèle :** `Qwen/Qwen2.5-0.5B-Instruct` — 494 M paramètres · 24 couches · hidden 896 · bf16.
⚠️ **PAS** `Qwen2.5-0.5B` (base) : il complète du texte, il ne répond pas.

---

## 11. LE DÉPÔT

**`https://github.com/jonathansearch/defi1-topological-lie-detector`** — public, en anglais

| Tag | Commit | Contenu |
|---|---|---|
| `v1.0-refuted` | `8eee776` | 5 runs + 3 bugs + 2 artefacts |
| `v1.1-h2-null` | `0ac9f8f` | H₂-Sync + clôture du Défi 1 |

```
README.md · CHANGELOG.md · CITATION.cff · LICENSE (MIT) · requirements.txt
docs/   01_hypothesis · 02_instrument · 03_protocols · 04_limits · 05_results · 06_h2_sync
src/    pmt.py · facts.py · runs.py · analyse.py · run_baseline_2x2 · run_prefix_forced · run_surprise_matched
results/  3 figures · 4 .npz · 10 .json
archive/  14 scripts d'origine, NON édités — y compris ceux qui étaient faux
```

`CHANGELOG.md` documente les bugs qui ont produit de **faux chiffres avant d'être corrigés**.
`archive/` garde les scripts fautifs : **un résultat négatif ne vaut que si le chemin qui l'a produit est lisible.**

---

## 12. VERDICT EN UNE PHRASE

> **Deux instruments indépendants voient le savoir. Aucun ne voit le conflit.**

| | |
|---|---|
| ✅ **Le savoir a une signature topologique** | `+0,0765 [+0,0462 ; +0,1026]` · 20/25 couches |
| ✅ **Le savoir a une signature verticale** | `−0,0060 [−0,0090 ; −0,0032]` · p < 0,001 |
| ✅ Le modèle rationalise un mensonge forcé | 9/9, 0 auto-correction |
| ✅ Le savoir vit au point de décision | `+0,0765` avant · `−0,0139` après |
| ❌ Le conflit a une signature topologique | `−0,0162` à surprise et longueur appariées |
| ❌ Le conflit a une signature de synchronisation | `−0,0018`, p = 0,70 |

**Conclusion bornée** : un modèle, une taille, un jeu de faits, deux instruments.
Elle ne dit **pas** que le conflit n'existe pas dans le réseau. Elle dit qu'il **ne laisse aucune trace
dans la géométrie ni dans la topologie de ce qui a été mesuré**.

---

## 13. RÈGLES DU LABO À RESPECTER (non négociables)

1. **Un seul défi à la fois**, dans l'ordre choisi par le chef.
2. **Formaliser AVANT d'agir** — sinon on ne sait pas ce qu'on a mesuré.
3. **Analogie ≠ mesure.** Étiquettes `🧮` (calcul) / `🛰️` (NASA) / `📚` (dictionnaire) jamais mélangées.
4. **Aucun chiffre non calculé.** Distinguer smoke test / test logiciel / résultat calculé / interprétation.
5. **Honnêteté absolue, échecs publiés.** Aucune hypothèse sauvée en cachette.
6. **Ne rien décider à la place du chef.**
7. **Jamais la 2FA. Jamais de clé dans git.** Token utilisé sans jamais être affiché ni écrit.
8. **Workspace ≤ 50 Mo**, vérifié avec `outils/garde_50mo.py`.
9. **Aucun commit/push sans ordre.** Aucun entraînement complet sans autorisation.
10. **Ne pas être « carré » pendant la recherche** : la rigueur formelle s'applique **au moment de qualifier
    le résultat**, pas à chaque phrase de travail.

---

## 14. SI TU REPRENDS — par où commencer

| Priorité | Action | Coût |
|---|---|---|
| **1** | Lire `docs/05_results.md` puis `CHANGELOG.md` dans le dépôt | 10 min |
| **2** | Rejouer `outils/pmt_optionS.py` — 5,4 s, aucun modèle, vérifie que les données sont saines | 1 min |
| **3** | Trancher **D1** : quel réseau est `K_L^ℓ` — U ou F ? | décision, pas calcul |
| **4** | Option ouverte la plus rentable : **1,5 B** — le conflit existe-t-il à une autre échelle, et le modèle obéit-il enfin à une consigne de mensonge ? | ~3 Go, 20 min |
| **5** | Ou passer au **défi suivant**, sur ordre du chef | — |

---

*RATISS Labs · Yaoundé 🇨🇲 · 27/10/2026 · document de passation · les échecs sont publiés, pas enterrés.*
