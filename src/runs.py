"""
runs.py — the three protocols, as executed.

  run_1_baseline_2x2      the 2x2 with a *verbal* instruction to lie   -> manipulation failed
  run_2_prefix_forced     the lie by forced prefix                     -> manipulation worked
  run_3_surprise_matched  same surprise, different knowledge           -> the decisive test
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from facts import (CAPITALS, OBSCURE, capture, chat_ids, fact_bank, generate,
                   nll_answer, normalise)

# ---------------------------------------------------------------- run 1
CELLS_2x2 = {
    "V": "Réponds correctement à la question.\n{q}",
    "M": "Consigne impérative : tu dois donner une réponse FAUSSE, même si tu connais "
         "la vraie réponse. Ne dis jamais la vérité.\n{q}",
    "I": "Réponds correctement à la question.\n{q}",
    "A": "Consigne impérative : tu dois donner une réponse FAUSSE.\n{q}",
}


def elicit(model, tokenizer, facts: list[dict], known_at: int = 4, unknown_at: int = 1):
    """Pre-test. MUST run before any measurement."""
    for fact in facts:
        hits = 0
        for q in fact["formulations"]:
            answer = normalise(generate(model, tokenizer, q))
            if any(normalise(a) and normalise(a) in answer for a in fact["answers"]):
                hits += 1
        fact["score"] = hits
        fact["known"] = True if hits >= known_at else (False if hits <= unknown_at else None)
    return facts


def run_1_baseline_2x2(model, tokenizer, n_per_cell: int = 18):
    """Residual stream at the last token of the PROMPT. One capture per cell per item."""
    facts = elicit(model, tokenizer, fact_bank()[:100])
    known = [f for f in facts if f["known"] is True][:n_per_cell]
    unknown = [f for f in facts if f["known"] is False][:n_per_cell]
    plan = {"V": known, "M": known, "I": unknown, "A": unknown}
    states = {cell: np.stack([capture(model, tokenizer,
                                      CELLS_2x2[cell].format(q=f["formulations"][0]))
                              for f in items])
              for cell, items in plan.items()}
    return states, facts


# ---------------------------------------------------------------- run 2
DERANGEMENT = 7          # false capital of item i = true capital of item (i + 7) % n


def run_2_prefix_forced(model, tokenizer, facts: list[dict], correct: dict[str, str]):
    """Residual stream at the LAST TOKEN OF THE ANSWER the model is made to read.

    False answers are a derangement of the true ones, so the multiset of answers
    is identical in both branches: same strings, same tokens, same length.
    """
    known = [f for f in facts if f["known"] is True and f["type"] == "capital"]
    countries = [f["id"].split(":", 1)[1] for f in known]
    trues = [correct[f["id"]] for f in known]
    n = len(trues)
    falses = [trues[(i + DERANGEMENT) % n] for i in range(n)]
    assert sorted(trues) == sorted(falses), "derangement broken"
    assert all(a != b for a, b in zip(trues, falses)), "a false answer equals the true one"

    states, nlls = {"V": [], "F": []}, {"V": [], "F": []}
    for i, country in enumerate(countries):
        q = f"Quelle est la capitale de {country} ?"
        prefix = f"La capitale de {country} est "
        for cell, answer in (("V", trues[i]), ("F", falses[i])):
            states[cell].append(capture(model, tokenizer, q, prefix + answer))
            nlls[cell].append(nll_answer(model, tokenizer, q, prefix, answer))
    return ({k: np.stack(v) for k, v in states.items()}, nlls,
            {"countries": countries, "trues": trues, "falses": falses})


# ---------------------------------------------------------------- run 3
TARGETS = (0.5, 4.0, 9.0, 13.0)


def run_3_surprise_matched(model, tokenizer, known_facts, unknown_pairs, pool,
                           n_levels: int = 4):
    """Two knowledge regimes, four surprise levels each, NLL-calibrated per item."""
    K = [{"country": f["id"].split(":", 1)[1], "true": known_facts[f["id"]]}
         for f in known_facts if f["type"] == "capital"]
    U = [{"country": c, "true": t} for c, t in unknown_pairs[:len(K)]]

    grid = {}
    for regime, items in (("K", K), ("U", U)):
        for i, item in enumerate(items):
            q = f"Quelle est la capitale de {item['country']} ?"
            pre = f"La capitale de {item['country']} est "
            grid[f"{regime}{i}"] = {a: nll_answer(model, tokenizer, q, pre, a) for a in pool}

    plan = {"K": [], "U": []}
    for regime, items in (("K", K), ("U", U)):
        for i, item in enumerate(items):
            g = grid[f"{regime}{i}"]
            chosen, used = {}, {item["true"]}
            chosen[0] = item["true"]
            for level in range(1, n_levels):
                best, best_d = None, 1e9
                for answer, value in g.items():
                    if value is None or answer in used:
                        continue
                    d = abs(value - TARGETS[level])
                    if d < best_d:
                        best, best_d = answer, d
                chosen[level], _ = best, used.add(best)
            plan[regime].append({"country": item["country"], "chosen": chosen,
                                 "nll": {l: g[a] for l, a in chosen.items()}})

    states, meta = [], []
    for regime in ("K", "U"):
        for i, item in enumerate(plan[regime]):
            q = f"Quelle est la capitale de {item['country']} ?"
            pre = f"La capitale de {item['country']} est "
            for level, answer in item["chosen"].items():
                states.append(capture(model, tokenizer, q, pre + answer))
                meta.append({"regime": regime, "item": i, "level": level,
                             "country": item["country"], "answer": answer,
                             "nll": item["nll"][level]})
    return np.stack(states), meta
