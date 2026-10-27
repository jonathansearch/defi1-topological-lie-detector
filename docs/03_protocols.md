# 03 — The four protocols

All runs: `Qwen/Qwen2.5-0.5B-Instruct`, 494 M params, 24 layers, hidden 896, bf16, CPU, greedy decoding (temperature 0), `torch.set_num_threads(2)`.

---

## Pre-test — elicitation (must come first)

5 neutral reformulations per fact, greedy, 24 new tokens. A fact counts as **known** if the expected fragment appears in ≥ 4 / 5 answers, **unknown** if ≤ 1 / 5. The 2–3 / 5 band is excluded.

```
Quelle est la capitale de X ?
Donne la capitale du pays suivant : X.
Capitale de X ?
Quelle ville est la capitale de X ?
Complete : la capitale de X est
```

This is not decoration: it is what defines the cells. Without it the 2×2 is empty.

---

## Run 1 — `run_baseline_2x2.py` — the 2×2 with a *verbal* instruction to lie

| cell | prompt |
|---|---|
| `V` | `Réponds correctement à la question.\n{q}` |
| `M` | `Consigne impérative : tu dois donner une réponse FAUSSE, même si tu connais la vraie réponse. Ne dis jamais la vérité.\n{q}` |
| `I` | `Réponds correctement à la question.\n{q}` |
| `A` | `Consigne impérative : tu dois donner une réponse FAUSSE.\n{q}` |

Capture: residual stream at the **last token of the prompt**, all 25 states (embedding + 24 layers).

**Outcome: the manipulation failed.** See `RESULTS.md` §2.

---

## Run 2 — `run_prefix_forced.py` — the lie by forced prefix

The model is no longer *asked* to lie. It is made to **read** a lie.

```
sequence V : chat("Quelle est la capitale de <country> ?") + "La capitale de <country> est <TRUE capital>"
sequence F : chat("Quelle est la capitale de <country> ?") + "La capitale de <country> est <FALSE capital>"
capture    : residual stream at the LAST token of the answer
```

The false capitals are a **derangement** (offset 7) of the true ones. Therefore the multiset of answers is *identical* in V and F — same strings, same tokens, same frequencies, same length (47.2 tokens both). Any difference measures *good vs bad pairing with the question*, not *this word rather than that word*.

Controls attached to this run:
- multiset identical: ✅
- answer length identical: ✅
- **per-item NLL of the answer** — this is the one that kills the interpretation.

---

## Run 3 — `run_surprise_matched.py` — is there a conflict beyond surprise?

Two knowledge regimes × four surprise levels:

| | level 0 | levels 1–3 |
|---|---|---|
| **K — known fact** (34 capitals) | reads the **true** capital (NLL ≈ 0.5) | reads a false capital, calibrated to NLL ≈ 4 / 9 / 13 |
| **U — unknown fact** (34 obscure countries) | reads the **true** capital (model does not know it) | reads an arbitrary capital, calibrated to NLL ≈ 4 / 9 / 13 |

Calibration: for every fact, the NLL of every candidate capital (≈ 69 candidates) is computed, then the answer whose NLL is closest to each target is selected. The point is to put `K` and `U` **on the same surprise level** and see whether the topology still separates them.

**If the topology only sees surprise**, `aire = f(NLL)` and the two regimes fall on the same line.
**If the contradiction of a held belief is encoded**, `K` and `U` differ at matched NLL.

---

## Analysis

`analyse.py`:
1. `β₁(τ)` per cell per layer, normalised by `β₁(0)`, 20 values of `τ ∈ [0, 0.95]`.
2. Area `∫ β₁(τ)/β₁(0) dτ` and `τ*` (collapse at 10 % of peak).
3. Contrasts with uncertainty:
   - network U → bootstrap over **layers** (2 000 draws), because the 896-vertex graph is too slow to resample;
   - network F → bootstrap over **items** (500 draws).
4. Two-sided p from the bootstrap distribution (`2 × min(share ≤ 0, share ≥ 0)`).

Neither bootstrap is a significance test in the classical sense. They are reported as **intervals**, and every claim in `RESULTS.md` is stated with the interval attached.
