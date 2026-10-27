# 02 — The instrument

## The single formula

```
β₁ = E − V + C
```

`E` = number of edges above threshold · `V` = number of vertices · `C` = number of connected components.

This is **Euler's formula for graphs**, i.e. the cyclomatic rank. For a 1-dimensional complex it is *exact*, not an approximation. It costs O(E) with a union-find.

```python
def beta1(n_vertices, E):
    if len(E) == 0: return 0
    return len(E) - n_vertices + components(n_vertices, E)
```

## Why sweep τ

Thresholding the correlation matrix at every `τ` and recomputing `β₁` gives `β₁(τ)` — **the whole barcode, for free**, without a simplex tree. One curve, `n_τ` graph builds, no persistence library.

## The four numbers that are kept

| Symbol | Definition | What it captures |
|---|---|---|
| `β₁(τ)` | curve over `τ ∈ [0, 0.95]`, 20 steps | shape |
| **area** `∫ β₁(τ)/β₁(0) dτ` | normalised area | *how long the graph stays cyclically dense* |
| **τ\*** | first `τ` with `β₁ ≤ 10 % of β₁(0)` | *where it collapses* |
| `β₁(0)` | number of cycles in the complete graph | scale, removed by normalisation |

Normalising by `β₁(0)` is mandatory: `β₁(0) = V(V−1)/2 − V + 1`, which depends only on the vertex count. Without normalisation, comparing two networks with different vertex counts is meaningless.

## Two networks, and they are not interchangeable

| | vertices | correlations measured across | what it is |
|---|---|---|---|
| **U** | the 896 residual-stream units | the *items* (questions) | internal topology of one representation |
| **F** | the *items* | the 896 dimensions | the object the formalisation targets: `Path_G(u, v_abs)` |

They disagree, and that disagreement is informative — see `04_limits.md`.

## Three traps found the hard way

1. **`β₁ > 0` is always true** on a dense graph (V=60 → β₁ ≈ 700 even on pure noise). A prediction can never be "β₁ rises". Only **differences between cells** or **`τ*`** can be predicted.
2. **Common mode.** On network F, without removing the per-dimension mean across items, every pair of items correlates > 0.95 at every layer and `β₁(τ)/β₁(0)` stays at **0.95 down to `τ = 0.95`**. The last-token residual is nearly the same vector for every question. Centring on the items is **not optional**.
3. **`L₁^max` needs 2-simplices.** Without faces it is infinite. `A_τ^norm` and `Path_G` are **not** homologies and must not be reported as such. None of the numbers here are `L₁^max`: the surrogate used is the normalised area.
