# 05 — Results, run by run

All numbers below are produced by `src/` from the arrays in `results/`.
Model: `Qwen/Qwen2.5-0.5B-Instruct`, 494 M params, 24 layers, hidden 896, bf16, CPU, greedy.

---

## Run 1 — the 2×2 with a verbal instruction to lie

**Elicitation first** (5 neutral reformulations), on 100 facts:

| Class | Count |
|---|---|
| known (≥ 4/5) | **75** |
| unknown (≤ 1/5) | **18** |
| grey band (2–3/5), excluded | 7 |

Cell size is therefore **18**, set by the rarest class — not by choice.

**The manipulation failed.** Verified after the fact on the 18 answers produced in cell `M`:

| Behaviour | Count |
|---|---|
| ignores the instruction, states the truth | **15 / 18** |
| obeys, states a false answer | 3 / 18 |

`cap:France → "La capitale de France est Paris."` · `cap:Allemagne → "…est Berlin."`

So `M` is `V` plus a 20-token preamble the model reads and does not apply. Every
`M − V` difference measures *reading an instruction*, not *lying*.

**But the knowledge axis separates, strongly** (residual stream at the last token
of the prompt, normalised area, mean over layers):

| Network | V | M | I | A |
|---|---|---|---|---|
| **U** (896 units) | 0.306 | 0.279 | **0.383** | 0.393 |
| **F** (18 items) | 0.241 | 0.190 | **0.277** | 0.272 |

| Contrast | Network U (bootstrap over layers, 2 000) | Network F (bootstrap over items, 500) |
|---|---|---|
| **I − V** (knowledge) | **+0.0765** [+0.0462; +0.1026] · p < 0.001 | +0.0229 [−0.0564; +0.1003] · p = 0.56 |
| M − V (instruction) | −0.0271 [−0.0441; −0.0142] | −0.0512 [−0.0831; −0.0133] |
| A − I | +0.0100 [+0.0018; +0.0186] | −0.0074 [−0.0507; +0.0269] |

Sign of `I − V`, layer by layer:

| Network U | 20/25 positive — **every layer from 5 to 24** |
|---|---|
| Network F | 14/25 positive — **every layer from 11 to 24** |

The sign **inverts in the first layers** (U: layers 1–4, F: layers 1–10). Two networks
built on completely different vertex sets flip at the same place. That is structure,
not coding coincidence.

**Reading:** a high area = the graph stays dense and cyclic at a high threshold =
many units still resemble each other when a lot is demanded = **homogeneity**.
Knowing differentiates; not knowing converges.

---

## Run 2 — the lie by forced prefix

34 known capitals. Sequences `chat(Q) + "La capitale de <country> est " + <answer>`,
the answer being either the true one or a **derangement** (offset 7) of the true ones.

| Control | Result |
|---|---|
| multiset of answers identical V / F | ✅ |
| answer length | **47.2 tokens both** |
| false answer equal to the true one | 0 / 34 |

### The manipulation worked — beautifully

Sample of 9 facts (one in four), what the model produces after reading the lie:

| Country | reads | continues with |
|---|---|---|
| France | **Rabat** | *". Rabat est l'île principale du Maroc et son…"* |
| Sénégal | **Bamako** | *". Bamako est l'État-major du pays et son centre politique…"* |
| Danemark | **Sofia** | *". Sofia est l'une des principales villes de Danemark…"* |

**9/9 continue the lie. 0/9 self-correct.** The model does not merely obey: it
**rationalises** a statement it knows to be false.

### The prediction was refuted in sign

| Network | V | F | Δ = F − V |
|---|---|---|---|
| U (896 units) | 0.212 | 0.198 | **−0.0139** [−0.0186; −0.0093] |
| F (34 items) | 0.617 | 0.594 | −0.0234 [−0.0702; +0.0231] · p = 0.27 |

`τ*`: U **0.46 → 0.43**, F 0.75 → 0.73 — it goes **down**, not up. `F > V` on
**3/25** and **2/25** layers.

The lie does not stiffen the structure. It slightly loosens it.

### And the control that kills the interpretation

Per-item NLL of the answer the model is made to read:

| | mean over 34 facts |
|---|---|
| true answer | **0.495 nat** |
| false answer | **9.023 nats** |
| more surprising in | **34 / 34 facts** · t = 11.84 |

In probability: 61 % vs 0.012 % — a factor **5 000**.

Dose–response by terciles of ΔNLL: no monotonicity, and the sign **inverts** in the
most surprised tercile. The effect is drowned.

---

## Run 3 — surprise-matched

Two regimes × four surprise levels, NLL calibrated per item against 68 candidate capitals.

| Level | target | known | unknown | gap |
|---|---|---|---|---|
| 1 | 4.0 | **4.04 ± 0.14** | **3.95 ± 0.15** | −0.09 |
| 2 | 9.0 | **8.95 ± 0.15** | **8.97 ± 0.19** | +0.02 |
| 3 | 13.0 | **13.07 ± 0.40** | **13.06 ± 0.78** | −0.00 |

Matched to the hundredth of a nat.

| Level | Δ(K − U) | 95 % CI | p | K>U |
|---|---|---|---|---|
| 4 | **−0.0126** | [−0.0146; −0.0105] | < 0.001 | 0/25 |
| **9** | **+0.0328** | [+0.0180; +0.0482] | < 0.001 | 19/25 |
| 13 | +0.0018 | [−0.0054; +0.0088] | 0.62 | 15/25 |

**Non-monotone, therefore impossible as a mechanism.** Artefact hunt:

| Level | answer tokens, known | answer tokens, unknown |
|---|---|---|
| 4 | 3.29 | 3.24 |
| **9** | **2.44** | **1.56** |
| 13 | 1.09 | 1.00 |

A 1.6× length difference exactly at the "decisive" level. And at levels 1 and 3 the
two regimes pick **the same capitals** (11–12 in common): they are no longer
independent probes.

---

## Run 4 — surprise **and** length matched

Answers forced to exactly **1 token**, then exactly **2 tokens**, NLL ≈ 9 in both branches.

| Length | n (known / unknown) | NLL | **Δ(K − U)** | 95 % CI | K>U |
|---|---|---|---|---|---|
| **1 token** | 32 / 33 | 9.27 / 8.98 | **−0.0162** | [−0.0202; −0.0116] | **3/25** |
| **2 tokens** | 34 / 32 | 8.78 / 8.71 | **−0.0033** | [−0.0068; +0.0004] | **7/25** |

**Both negative. The +0.033 vanishes.**

### Verdict

> **H1-CONFLICT is refuted.** At equal surprise and equal answer length, contradicting
> a fact the model **knows** produces a slightly *more homogeneous* topology, not a
> stiffer one. The topology of the residual stream does not see the conflict. It sees
> the surprise.

The known regime being *more* homogeneous than the unknown one at matched NLL and
length is consistent with run 1: a known fact, even contradicted, **tightens** the
representations around its answer. Knowing acts like a recall, not like a tension.

Do not over-read: −0.016 is small, the 2-token result sits at the edge (p = 0.086),
and **only one model size** was ever tested.
