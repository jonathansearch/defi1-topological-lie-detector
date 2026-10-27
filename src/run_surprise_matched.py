#!/usr/bin/env python3
"""Run 3 — two knowledge regimes x four surprise levels, NLL-calibrated per item."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from runs import run_3_surprise_matched

torch.set_num_threads(2)
OUT = Path(__file__).resolve().parents[1] / "results"

if __name__ == "__main__":
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    mdl = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                               dtype=torch.bfloat16, low_cpu_mem_usage=True)
    mdl.eval()
    known = json.load(open(OUT / "run1_known.json"))
    unknown = [tuple(x) for x in json.load(open(OUT / "unknown_pairs.json"))]
    pool = json.load(open(OUT / "capital_pool.json"))
    E, meta = run_3_surprise_matched(mdl, tok, known, unknown, pool)
    np.savez_compressed(OUT / "run3_surprise.npz", etats=E,
                        nll=np.array([m["nll"] for m in meta]),
                        regime=np.array([m["regime"] for m in meta]),
                        level=np.array([m["level"] for m in meta]))
