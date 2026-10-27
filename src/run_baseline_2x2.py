#!/usr/bin/env python3
"""Run 1 — the 2x2 with a VERBAL instruction to lie. Elicitation must run first."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from facts import fact_bank
from runs import elicit, run_1_baseline_2x2

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                               dtype=torch.bfloat16, low_cpu_mem_usage=True)
    mdl.eval()
    states, facts = run_1_baseline_2x2(mdl, tok)
    np.savez_compressed(OUT / "run1_2x2.npz", **states)
    print("known:", sum(f["known"] is True for f in facts),
          "unknown:", sum(f["known"] is False for f in facts),
          "grey:", sum(f["known"] is None for f in facts))
