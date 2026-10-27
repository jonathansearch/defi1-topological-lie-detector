# Changelog — what the investigation actually did

Not a release log. A log of the four runs, the three bugs, and the two artefacts,
in the order they appeared. Kept because a negative result is only worth something
if the path that produced it is readable.

---

## Run 0 — `Qwen2.5-0.5B` **base** — invalid, discarded

The sealed command named `Qwen2.5-0.5B`. Loaded, it turned out to be the **base**
checkpoint: it completes text instead of answering.

```
"La capitale de la France est"  ->  ['une', 'un', 'la', 'le', '-']
"7 + 5 = "                      ->  ['', '\\\\', '-', '(', '?']
```

Elicitation on 137 facts gave **4 known / 119 unknown / 14 grey**. The run was executed
anyway (493 s) and its numbers show no separation — as expected, since the prompts were
never questions. The output is kept as `archive/` trace only. **Not a result.**

→ switched to `Qwen2.5-0.5B-Instruct`.

---

## Run 1 — 2×2 with a verbal instruction to lie

### Bug 1 — truncation at 10 tokens

Answers were cut before the city name:

```
"Quelle est la capitale de Allemagne ?"  ->  "La capitale de l'Allemagne est"
```

Five known facts were scored **unknown** because of it. Fixed by raising `max_new_tokens`
to 24 — the model had been right, the harness had been deaf.

### Bug 2 — `apply_chat_template(..., return_tensors="pt")`

In transformers 5.21 this returns a `BatchEncoding`, not a tensor, and
`generate()` then fails with `AttributeError: 'BatchEncoding' object has no attribute 'shape'`.
Fixed: `tokenize=False`, then tokenise the string.

### Result

**The manipulation failed.** 15 of 18 answers in cell `M` state the **truth** despite an
imperative instruction to lie. The `M` cell is `V` plus a preamble — not a lie cell.

But the knowledge axis separates, strongly: `I − V = +0.0765 [+0.0462; +0.1026]`,
positive on **20/25 layers**.

### Bug 3 — saturating network F

With vertices = items and correlations measured across dimensions, **without** removing
the per-dimension common mode, every pair of items correlates > 0.95 at every layer and
`β₁(τ)/β₁(0)` stays at **0.95 down to τ = 0.95**. The last-token residual stream is
nearly the same vector for every question.

Fixed by centring each dimension across items. Without that step, the item network is
mathematically incapable of showing anything.

---

## Run 2 — the lie by forced prefix

Design: the model is no longer asked to lie; it is made to **read** a lie.

```
chat(Q) + "La capitale de <country> est " + <answer>
```

False answers are a **derangement** (offset 7) of the true ones, so the multiset of
answers is identical in both branches — same strings, same tokens, same length (47.2 both).

### It worked

**9/9** sampled facts continue the lie. **0/9** self-correct. The model rationalises:
*"Rabat est l'île principale du Maroc"*, *"Bamako est l'État-major du pays"*,
*"Sofia est l'une des principales villes de Danemark"*.

### The topology went the wrong way

`F − V = −0.0139 [−0.0186; −0.0093]`. `τ*` **down** from 0.46 to 0.43. `F > V` on 3/25 layers.
The lie does not stiffen the structure, it loosens it.

### And the control killed the interpretation

| | mean NLL over 34 facts |
|---|---|
| true answer | **0.495 nat** |
| false answer | **9.023 nats** |

**34/34 facts**, t = 11.84, a factor **5 000** in probability. Inside the residual stream,
on a known fact, *conflict and surprise are the same quantity*.

### Bug 4 — the tokenizer merges the space

`"La capitale de France est "` and `"… est Paris"` are **both 45 tokens**: Qwen merges
the trailing space with the following word. Splitting the sequence at the prefix boundary
therefore misaligns every logit by one position, and the first NLL attempt returned
nonsense (`NLL = 17.5` for a token the model predicts at 61 %).

Fixed: isolate answer tokens by **character offsets**, and write the prefix **without**
its trailing space.

---

## Run 3 — surprise-matched

Two knowledge regimes × four NLL levels, calibrated per item over 68 candidate capitals.
Matching achieved to the hundredth of a nat (4.04 vs 3.95 · 8.95 vs 8.97 · 13.07 vs 13.06).

### A first result, wrong

```
NLL ≈ 4   Δ(K−U) = −0.0126
NLL ≈ 9   Δ(K−U) = +0.0328   <- "the effect"
NLL ≈ 13  Δ(K−U) = +0.0018
```

**Non-monotone, therefore impossible as a mechanism.** So the artefact was hunted
before anything was published.

### Artefact 1 — length

At NLL ≈ 9, known answers are **2.44** tokens and unknown answers **1.56**. A 1.6×
difference exactly at the "decisive" level. A two-token answer occupies two positions in
the graph, not one, and β₁ feels it.

### Artefact 2 — the two regimes stop being independent

At levels 4 and 13 the two regimes pick **the same capitals** (11–12 in common). They are
then measuring the shape of the distribution, not the effect of knowledge.

### Bug 5 — the KV cache is mutated in place

Reusing `past_key_values` across candidate answers silently corrupted the grid: the same
city returned `4.99`, then `9.60`, then `13.90` nats. Fixed with `copy.deepcopy` per
candidate. Also, HF already applies the final `norm` — applying it again gave `NLL = 17.5`
where the truth was `0.5`.

Net effect of the fix: **3.4× faster grid, and correct**.

---

## Run 4 — surprise **and** length matched

Answers forced to exactly 1 token, then exactly 2 tokens, NLL ≈ 9 in both branches.

```
1 token : Δ(K−U) = −0.0162 [−0.0202; −0.0116]   K>U on  3/25
2 tokens: Δ(K−U) = −0.0033 [−0.0068; +0.0004]   K>U on  7/25
```

**Both negative. The +0.033 vanished.** H1-CONFLICT is refuted.

---

## What was never done

`L₁^max` and 2-simplices (open decision, and `L₁^max` is infinite without faces) ·
GPT-2 (half of the sealed command) · a second model size · inter-layer synchronisation ·
attention patterns · any comparison to humans.
