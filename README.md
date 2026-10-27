# Topological Signatures of Knowledge and of Conflict in a Small Language Model

**Defi 1 — RATISS Labs · Yaoundé 🇨🇲**
Status: `SIMULATION` (runs executed on a real model) · one model · four runs

> A reporter's question, a small language model's residual stream, and one formula:
> `β₁ = E − V + C`. Does the topology of the residual stream know when the model
> **knows**, and does it know when the model is **lying**?
>
> Short answer: **yes for knowing, no for lying.**

---

## Results in one table

| Claim | Measure | Status |
|---|---|---|
| A known fact has a distinct topological signature | area `+0.0765 [+0.0462; +0.1026]`, positive on **20/25 layers** | ✅ holds |
| The model rationalises a lie it is forced to read | **9/9** continue the lie, 0/9 self-correct | ✅ holds |
| Knowing lives at the decision point, not after the token | `+0.0765` before the token · `−0.0139` after | ✅ observed |
| A verbal instruction to lie is not obeyed at 0.5 B | **15/18** answers say the truth | ✅ measured |
| A known fact also has a **vertical** signature | `sync V − I = −0.0060 [−0.0090; −0.0032]`, p < 0.001 | ✅ holds |
| **A conflict against a held belief has a topological signature** | `−0.0162 [−0.0202; −0.0116]` at matched NLL **and** matched length | ❌ **refuted** |
| **A conflict has an inter-layer synchronisation signature** | `−0.0018 [−0.0114; +0.0072]`, p = 0.70 | ❌ **refuted** |
| `L₁^max`, 2-simplices, GPT-2, second model size | — | ⬜ not measured |

Every number above is produced by the code in `src/` from the arrays in `results/`.
Nothing here is `L₁^max`: the surrogate used throughout is the normalised area
`∫ β₁(τ)/β₁(0) dτ` and the collapse scale `τ*`.

---

## The instrument

```
β₁ = E − V + C
```

Cyclomatic rank. Exact in dimension 1, O(E) with a union-find, no persistence
library. Sweeping the correlation threshold `τ` gives `β₁(τ)` — **the whole barcode,
for free**. Four numbers are kept per graph: the curve, its normalised area, `τ*`
(the scale where `β₁` falls to 10 % of its peak), and `β₁(0)` for normalisation.

Full details, including the three traps that cost real runs, in
[`docs/02_instrument.md`](docs/02_instrument.md).

---

## The four runs, and what each one destroyed

| # | Protocol | What it was supposed to show | What happened |
|---|---|---|---|
| 1 | 2×2 with a **verbal** instruction to lie | `M` = the lie cell | **Manipulation failed.** 15/18 answers said the truth. The `M` cell is `V` plus a preamble. |
| 2 | **Forced prefix** — make the model *read* a lie (derangement of the true answers, identical multiset) | the lie changes the topology | **Manipulation worked** (9/9 rationalisations). Topology went the **wrong way**: `F−V = −0.0139`, `τ*` **down**. And the per-item NLL is `0.495` vs `9.023` nats — the branches differ by a factor 5 000 in surprise. |
| 3 | **Surprise-matched**: two knowledge regimes × four NLL levels | conflict beyond surprise | An apparent `+0.0328` at NLL≈9 turned out to be a **length artefact** (answers 1.6× longer in the known regime). |
| 4 | **Length- and surprise-matched** re-run | the decisive test | `Δ = −0.0162` (1 token) and `−0.0033` (2 tokens). Both negative. **Refuted.** |
| 5 | **H₂-Sync** — inter-layer synchronisation on the captured states | conflict lives in the vertical axis | `Δ = −0.0018 [−0.0114; +0.0072]`, p = 0.70; 13/24 adjacencies, 2 significant out of 24 = chance. **Refuted.** The measure is not inert: it sees knowledge at p < 0.001. |

Each step is in [`docs/03_protocols.md`](docs/03_protocols.md); each failure is
documented with the number that killed it.

---

## What survives, stated honestly

**The knowledge result is the real one.** When the model knows a fact, the graph of
its residual stream stays cyclically dense at high threshold; when it does not know,
the graph collapses early. Positive on every layer from 5 to 24 on the 896-unit
network, and on layers 11 to 24 on the item network — with the sign **flipping in the
first four layers** in both networks, independently.

The reading: *knowing differentiates the representation; not knowing converges to a
common state.* Topology sees the difference between organised knowledge and noise.

**The lie result is a clean negative**, at one surprise level, two answer lengths, one
model. It may not generalise, and [`docs/04_limits.md`](docs/04_limits.md) says so in
eight numbered paragraphs written before the final run, not after.

---

## Reproducing

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers scipy numpy matplotlib

python src/run_baseline_2x2.py      # ~20 min CPU, elicitation + first 2x2
python src/run_prefix_forced.py     # ~2 min
python src/run_surprise_matched.py  # ~10 min
python src/analyse.py               # figures + tables
```

Machine used: 2 cores, 1 982 MB RAM, CPU only, `torch.set_num_threads(2)`.
`HF_HOME` **must** point outside `/tmp` (a 992 MB tmpfs is not enough for one 0.5 B model).

---

## Repository layout

```
src/        the four protocols and the instrument
docs/       hypothesis, instrument, protocols, limits, results
results/    raw arrays (.npz) and every figure
archive/    the eight scripts actually used during the investigation,
            preserved unedited — including the ones that were wrong
```

`archive/` is kept on purpose. A negative result is only worth something if the
path that produced it is readable.

---

## A note on what "knows" means here

Strictly operational: the model answers the same question correctly under **4 of 5**
neutral reformulations, greedily decoded, temperature 0. Nothing about belief,
intention, or awareness is implied anywhere in this work — by explicit constraint
on the original hypothesis.

---

*RATISS Labs · Yaoundé 🇨🇲 · failures are published, not buried.*

---

## Handoff

[`docs/00_HANDOFF_FR.md`](docs/00_HANDOFF_FR.md) — a complete French hand-off written for
another LLM taking over: every number, every refutation, the nine technical bugs, and the
decisions still open. If you are picking this work up, read that first.

## Changelog

[`CHANGELOG.md`](CHANGELOG.md) logs the four runs, the three bugs that produced wrong
numbers first (truncation at 10 tokens · the tokenizer merging the space · a mutated KV
cache), and the two artefacts (answer length · regimes collapsing onto the same answers).

## Figures

| File | Shows |
|---|---|
| `results/figures/pmt_v2_signatures.png` | run 1 — the knowledge axis, both networks, layer by layer |
| `results/figures/pmt_optionA_signatures.png` | run 2 — forced prefix: V vs F, and F−V per layer |
| `results/figures/pmt_optionP_final.png` | runs 3–4 — the apparent effect, the length artefact, and the corrected test |

## Data

| File | Contents |
|---|---|
| `results/pmt_qwen_v2.json` | run 1 — elicitation scores, per-layer area and `τ*`, all four cells |
| `results/pmt_v2_etats.npz` | run 1 — raw states `(4 cells, 18 items, 25 layers, 896)` |
| `results/pmt_optionA.npz` | run 2 — `V` / `F` states `(34, 25, 896)` and the true/false answers |
| `results/pmt_optionP.npz` | run 3 — 272 states, 2 regimes × 4 NLL levels × 34 items |
| `results/pmt_optionP_len.npz` | run 4 — 131 states at matched answer length |

## Known corrections applied

Three bugs found during the work, each of which had produced a wrong number first:

1. **Truncation at 10 tokens** cut answers before the city name, marking 5 known facts
   as unknown. Raised to 24.
2. **The Qwen tokenizer merges a leading space with the following word**, so splitting
   a sequence at `"est "` misaligns the logits. Answer tokens are isolated by
   **character offsets**, and the prefix is written without its trailing space.
3. **The KV cache is mutated in place** across candidate answers. It must be
   `copy.deepcopy`-ed per candidate, and HF already applies the final `norm` — applying
   it again doubled it.
