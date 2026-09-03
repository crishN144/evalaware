"""Step 5: does the honest number depend on layer choice or on the readout?

Two knobs are varied against the leave-one-family-out metric:
  layer    - a sweep, not just the argmax layer (which was itself picked on the
             leaky random-CV score and so is a mildly optimistic choice)
  pooling  - last-token vs mean-over-sequence residual readout

If the honest number is stable across both, the ~0.67-0.71 figure is a property
of the task, not of one arbitrary configuration.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config, confound, data, extract, probe

LAYERS = [0, 6, 12, 18, 24, 30, 35]


def report(models=None, layers=None, poolings=("last", "mean")) -> dict:
    models = models or [config.DEV_MODEL, config.MAIN_MODEL]
    layers = layers or LAYERS
    corpus = data.load_corpus()
    y = corpus.y
    out = {"layers": layers, "models": {}}

    for mid in models:
        s = mid.split("/")[-1]
        rec = {}
        for pool in poolings:
            if not extract.acts_path(mid, pool).exists():
                continue
            acts = extract.load_acts(mid, pool)
            per_layer = {}
            for li in layers:
                if li not in acts:
                    continue
                X = acts[li]
                random_cv = float(roc_auc_score(y, probe.oof_scores(X, y)))
                lofo, pooled = confound.leave_one_family_out(X, y, corpus.family)
                have = ~np.isnan(pooled)
                mixed = confound.mixed_family_auroc(pooled, y, corpus.family)
                per_layer[li] = {
                    "random_cv": random_cv,
                    "lofo_pooled": lofo["pooled_auroc"],
                    "mixed_macro": mixed.get("_macro"),
                    "smolagents": mixed.get("smolagents", {}).get("auroc"),
                    "gap": random_cv - (lofo["pooled_auroc"] or 0.0),
                }
                print(f"[sens] {s} {pool} L{li:2d}: random={random_cv:.4f} "
                      f"lofo={lofo['pooled_auroc']:.4f}", flush=True)
            vals = [v["lofo_pooled"] for v in per_layer.values()
                    if v["lofo_pooled"] is not None]
            rec[pool] = {"per_layer": per_layer,
                         "lofo_min": float(np.min(vals)) if vals else None,
                         "lofo_max": float(np.max(vals)) if vals else None,
                         "lofo_range": float(np.max(vals) - np.min(vals)) if vals else None,
                         "lofo_best_layer": (max(per_layer,
                                                 key=lambda k: per_layer[k]["lofo_pooled"])
                                             if vals else None)}
        out["models"][s] = rec

    p = config.ARTIFACTS / "sensitivity.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[sens] wrote {p}")
    return out
