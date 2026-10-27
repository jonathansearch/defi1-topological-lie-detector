# results/

Source files are text (`.json`, `.png`) and are committed directly.

The four `.npz` arrays are **binary captures** (residual streams, float16) totalling
about 20 MB. They are kept in this repository's **history and on GitHub**, but are not
useful to read without the model, so they may be fetched on demand:

```bash
git clone https://github.com/jonathansearch/defi1-topological-lie-detector
```

| File | Shape | What it holds |
|---|---|---|
| `pmt_v2_etats.npz` | V/M/I/A × (18, 25, 896) | run 1 — the 2×2 at the last prompt token |
| `pmt_optionA.npz` | V/F × (34, 25, 896) | run 2 — true vs forced-false answers |
| `pmt_optionP.npz` | (272, 25, 896) | run 3 — 2 regimes × 4 NLL levels × 34 facts |
| `pmt_optionP_len.npz` | (131, 25, 896) | run 4 — surprise **and** length matched |

Every analysis in `src/` and `docs/` can be recomputed from these four files alone,
without touching a model. Regenerating them from scratch costs about 25 minutes of CPU.
