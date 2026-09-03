"""Step 4: deployment reliability computed on HONEST (leave-one-family-out) scores.

reliability.py reports the same quantities on random-CV scores; those are
inflated by source leakage and must not be used for a deployment claim. This
module recomputes them on held-out-family predictions, which is what a monitor
deployed on an unseen source would actually experience.

Prior work (Nguyen et al. 2507.01786) proposed live monitoring without a
held-out-source split or a low-base-rate FPR analysis. This is that analysis.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config, data, reliability


def report(corpus=None, base_rates=(0.01, 0.05, 0.10), n_boot=None) -> dict:
    corpus = corpus or data.load_corpus()
    n_boot = n_boot or config.N_BOOTSTRAP
    y = corpus.y
    out = {"target_fpr": config.TARGET_FPR, "models": {}}

    for model in ("Qwen3-4B", "Qwen3-8B"):
        p = config.ARTIFACTS / f"lofo_pooled_{model}.npy"
        if not p.exists():
            continue
        s = np.load(p)
        m = ~np.isnan(s)
        ys, ss = y[m], s[m]

        rec = reliability.full_report(ys, ss, config.DEPLOY_BASE_RATE,
                                      config.TARGET_FPR, n_boot, config.SEED)
        rec["scores"] = "leave-one-family-out (honest)"
        rec["precision_by_base_rate"] = {
            str(b): reliability.precision_at_base_rate(
                rec["tpr_at_fpr"], rec["achieved_fpr"], b) for b in base_rates}

        # Same quantities on the leaky scores, for the side-by-side contrast.
        sw = config.ARTIFACTS / f"probe_sweep_{model}.json"
        if sw.exists():
            d = json.loads(sw.read_text())
            leaky = np.asarray(d["oof"][str(d["best_layer"])])
            lk = reliability.full_report(y, leaky, config.DEPLOY_BASE_RATE,
                                         config.TARGET_FPR, n_boot, config.SEED)
            rec["leaky_comparison"] = {
                "auroc": lk["auroc"], "tpr_at_fpr": lk["tpr_at_fpr"],
                "precision_at_base_rate": lk["precision_at_base_rate"],
                "ece": lk["ece"]}

        # How many false alarms per true catch, at a 1% base rate, per 10k transcripts.
        br = config.DEPLOY_BASE_RATE
        tp = rec["tpr_at_fpr"] * br * 10_000
        fp = rec["achieved_fpr"] * (1 - br) * 10_000
        rec["per_10k_transcripts_at_1pct"] = {
            "true_positives": float(tp), "false_positives": float(fp),
            "false_alarms_per_true_catch": float(fp / tp) if tp > 0 else None}

        out["models"][model] = rec
        print(f"[honest] {model}: AUROC {rec['auroc']:.4f}  "
              f"TPR@{rec['achieved_fpr']*100:.1f}%FPR {rec['tpr_at_fpr']:.4f}  "
              f"precision@1% {rec['precision_at_base_rate']:.4f}  ECE {rec['ece']:.4f}",
              flush=True)

    p = config.ARTIFACTS / "honest_reliability.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[honest] wrote {p}")
    return out
