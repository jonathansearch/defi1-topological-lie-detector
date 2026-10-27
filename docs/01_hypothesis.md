# 01 — The hypothesis, in RATISS notation

**Domain:** `L` (latent / AI) · **Status of the whole document:** `HYPOTHESIS`

## 0. What is being claimed — and what is not

The source claim, split into seven statements, because confusing them is the main error available:

| # | Statement | Nature | Status |
|---|---|---|---|
| 1 | Looking at **weights** is anatomy; the physiology is in the **activations** | methodological premise | accepted, not tested |
| 2 | A human liar shows a peak; an ignorant human shows nothing | **analogy** (`Γ^ana`), not a measurement | analogy only |
| 3 | There are **two regimes of hallucination**: (i) it hallucinates confidently because it does not know; (ii) it knows but cannot do otherwise, having never learned to say "I don't know" *inside its reasoning stream* | the core claim | hypothesis |
| 4 | The route to abstention is missing from the flow | consequence of 3 | hypothesis |
| 5 | No hypothesis of human consciousness is to be imported | hard constraint | binding |
| 6 | Small model behaves naively, large model behaves constrained | pre-registered prediction | hypothesis |
| 7 | The programme: measure the flow, not the weights | programme | executed here |

**Constraint 5 is binding**: nothing below assigns the model beliefs, intention, or awareness. "Knows" is used operationally and means *exactly one thing*: the model answers the same question correctly under 4 of 5 neutral reformulations, greedily decoded.

## 1. Objects

- Activations `a_ℓ(x, t)` — residual stream of layer `ℓ`, at token position `t`, for prompt `x`.
- `G_L^{assoc}` — the association graph: units are vertices, an edge exists when the correlation between two units exceeds `τ`.
- `K_L^ℓ = Cl(G_L^{assoc})` — its clique complex, filtered by `τ`.
- `β_k(K^τ)` — Betti numbers of the filtered complex.
- `τ*` — the scale at which `β₁` collapses to 10 % of its peak.
- `Path_G(u, v_abs)` — the routes between a state `u` and the abstention state `v_abs`. **Routes, not a persistence invariant.**

⚠️ Two objects must never be mixed: working on **layer activations** is not working on **embeddings**. Different objects, different constructions, different repositories (`ratiss-lewm-integration`, `metac-bot-template` work on embeddings).

## 2. The 2×2

|  | answers truthfully | is constrained to answer falsely |
|---|---|---|
| **knows** | `V` | `M` |
| **does not know** | `I` | `A` |

`A` is the control: same instruction, no knowledge to contradict.

## 3. Pre-registered predictions

- **H1** — `Δ_conf = [L₁^max(M) − L₁^max(A)] − [L₁^max(V) − L₁^max(I)] > 0`
  *the lie costs topology only where there is a truth to contradict*
- **H2** — `AUC(𝒟|I, 𝒟|M) ≥ 0.80`, where `𝒟` is the discriminating signal.
- **H3** — `argmax_t Ψ(t) = t_c`, the collapse is located at the token immediately preceding the false token.
- **H4** — `Path_G(u, v_abs) = ∅`, the abstention route is absent.
- **H5** (open) — `δ_dérive = d(γ_M, γ_V)`.

## 4. Controls declared before measuring

`β₀` · norm · shuffle of the prompt↔answer pairing · length-matched prompts · pre-test (elicitation first) · null model · two model sizes · grid of `τ` with FDR.

## 5. Retained definition of "the lie" used in this repository

A lie is **not** an instruction the model ignores. It is **a text the model produces that contradicts a belief it demonstrably holds**, verified by the elicitation pre-test. This definition is what made run 3 (prefix forcing) possible and what invalidated run 2's `M` cell.
