# 06 — H₂-Sync: does the vertical axis see the conflict?

**Pre-registered prediction** (written into the script before any computation):

> A forced lie creates a desynchronisation between the knowledge layers (5–15) and
> the generation layers (16–24), absent in the no-conflict regime.
> `sync_bloc(K) < sync_bloc(U)`.

Two-pass horizontal topology (β₁, run 4) had already found nothing at matched surprise
and matched answer length. This is the vertical test, on the same captured states.

## Measurements

| Measure | Definition |
|---|---|
| `sync_adj[ℓ]` | `cos(h_ℓ, h_{ℓ+1})`, 24 values |
| `sync_bloc` | mean `cos` over all pairs (layer in 5–15, layer in 16–24) |
| `sync_total` | mean `cos` over all off-diagonal pairs |

Contrasts on run 4 states: surprise (NLL ≈ 9) **and** answer length (1 token) matched.
Bootstrap over **facts**, 5 000 draws.

## Result

| Measure | K (conflict) | U (none) | Δ = K − U | 95 % CI | p |
|---|---|---|---|---|---|
| **sync bloc 5–15 × 16–24** | 0.5897 | 0.5916 | **−0.0018** | [−0.0114; +0.0072] | **0.70** |
| sync total | 0.5761 | 0.5793 | −0.0032 | [−0.0200; +0.0141] | 0.71 |

The sign is the predicted one. The size is **0.3 % of the baseline**, and the interval
straddles zero.

Layer by layer: `K > U` on **13 of 24** adjacencies — a coin flip. Two reach p < 0.05
(17→18 at 0.046, 20→21 at 0.016), which is exactly the rate expected from 24 tests with
no correction (24 × 0.05 = 1.2). Both are negative, and neither concerns the
knowledge↔generation boundary. The largest gap in the whole profile is at **23→24**
(Δ = +0.0099, p = 0.053) — the output layer, not the boundary.

Inside the predicted blocks, the differences are 0.000 to 0.002 everywhere.

## Surprise control

| Level | NLL | K | U | Δ |
|---|---|---|---|---|
| 0 | 1.40 | 0.5927 | 0.5607 | +0.0320 |
| 1 | 4.00 | 0.5690 | 0.5609 | +0.0081 |
| 2 | 8.96 | 0.5711 | 0.5907 | −0.0196 |
| 3 | 13.07 | 0.6088 | 0.6079 | +0.0009 |

Non-monotone, sign inverting. The −0.02 at NLL ≈ 9 is the same number as in the matched
test, where it had p = 0.70 — it is not distinguishable from noise.

## Positive control — the synchronisation measure is not inert

On run 1 (last token of the prompt, 18 facts per cell):

| Contrast | Δ sync_bloc | 95 % CI | p |
|---|---|---|---|
| **V (knows) − I (does not know)** | **−0.0060** | [−0.0090; −0.0032] | < 0.001 |
| M − A | −0.0232 | [−0.0263; −0.0203] | < 0.001 |

Vertical synchronisation **sees knowledge**, on both contrasts, and in the same direction
as the horizontal topology: knowing = desynchronisation, ignorance = lockstep. A second,
independent measurement saying that **knowing is paid in heterogeneity**.

So the instrument is not blind. It sees. It just does not see the conflict.

## Verdict

> **H₂-Sync refuted.** Neither the intra-layer topology (β₁) nor the inter-layer
> synchronisation separates a contradicted known fact from an improbable assertion about
> an unknown fact, at matched surprise and matched answer length.

**Known limitation:** one model, one size, one fact set, two instruments. This does not
say that conflict does not exist inside the network. It says that it leaves no trace in
the geometry or the topology of what was measured.
