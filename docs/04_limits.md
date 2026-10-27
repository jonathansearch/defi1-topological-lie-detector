# 04 — What this does not measure

Written before the final run, not after.

## 1. `L₁^max` is not computed

`L₁^max` is the longest-persistence feature of `H₁`. It requires **2-simplices**, which requires a decision (options A/B/C in the formalisation) that is still open. Without faces, `L₁^max` is **infinite**.

Everything called "area" here is a **surrogate**: `∫ β₁(τ)/β₁(0) dτ`. It is a property of the graph filtration, not a persistence diagram. `A_τ^norm` and `Path_G` are **not** homologies.

## 2. One model, one size, one family

`Qwen2.5-0.5B-Instruct` only. The pre-registered prediction H6 ("small = naive, large = constrained") is **untested**. GPT-2 — half of the sealed command — was **not run**.

## 3. Sample size

18 to 34 items per cell, not the 121 per cell that a 0.5σ effect at 80 % power would require. The intervals are wide and this repository says so on every line.

## 4. Networks U and F disagree, and this is unresolved

| contrast | network U (896 units) | network F (items) |
|---|---|---|
| `I − V` (knowledge) | +0.0765 [+0.0462; +0.1026] | +0.0229 [−0.0564; +0.1003] |
| `M − V` (instruction) | −0.0271 [−0.0441; −0.0142] | −0.0512 [−0.0831; −0.0133] |

The formalisation targets network F (`Path_G(u, v_abs)`). Network F is the one that **fails** to separate knowledge. Which network *is* `K_L^ℓ` is an open decision, and it must be declared **before** the next measurement, not after.

## 5. The conflict/surprise tangle is structural, not accidental

A model that "knows" is, by definition, a model that finds the false answer improbable. Inside the residual stream, **conflict and surprise are the same quantity** on a known fact. They can only be separated by moving to a regime where surprise exists without knowledge — which is exactly what protocol 3 does, and which introduces a different confound (knowledge itself).

## 6. At 0.5 B, one cannot ask for a lie and expect to be obeyed

Run 1 measured it: 15 of 18 answers said the **truth** despite an imperative instruction to lie. Any study that elicits lies by instruction at this scale is measuring the instruction, not the lie.

## 7. "Knows" is operational, never mental

The model is said to *know* a fact when 4/5 neutral reformulations extract the correct answer greedily. Nothing about belief, intention, or awareness is implied anywhere in this repository.

## 8. What was not measured either

- Synchronisation between layers.
- Attention patterns.
- Any 2-simplex construction.
- Any human comparison (`Γ^ana` is an analogy, and stays one).
