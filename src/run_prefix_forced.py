#!/usr/bin/env python3
"""Run 2 — the lie by forced prefix. Derangement keeps the answer multiset identical."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs import run_2_prefix_forced

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                               dtype=torch.bfloat16, low_cpu_mem_usage=True)
    mdl.eval()
    facts = json.load(open(OUT / "run1_facts.json"))       # produced by run_baseline_2x2
    correct = json.load(open(OUT / "capital_names.json"))
    states, nlls, meta = run_2_prefix_forced(mdl, tok, facts, correct)
    np.savez_compressed(OUT / "run2_prefix.npz", **states,
                        nll_V=np.array(nlls["V"]), nll_F=np.array(nlls["F"]))
