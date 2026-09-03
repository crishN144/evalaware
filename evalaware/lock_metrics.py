"""Step 1: the locked headline table, with bootstrap CIs on the honest number.

Every metric here is recomputed from cached activations and predictions; nothing
is copied from a previous summary. Writes artifacts/locked_metrics.json.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from . import compare_blackbox, config, confound, data, extract, probe, reliability


def _slug(m):
    return m.split("/")[-1]


def lock(models=None, n_boot=None) -> dict:
    models = models or [config.DEV_MODEL, config.MAIN_MODEL]
    n_boot = n_boot or config.N_BOOTSTRAP
    corpus = data.load_corpus()
    y = corpus.y
    out = {"n": len(corpus), "n_eval": int(y.sum()), "n_deploy": int((y == 0).sum()),
           "n_families": len(set(corpus.family)),
           "n_categories": len(set(corpus.category)),
           "n_label_mixed_categories": sum(
               1 for c in set(corpus.category)
               if 0 < y[np.array(corpus.category) == c].mean() < 1),
           "models": {}}

    # Source-independent baselines (identical across models by construction).
    tfidf = np.load(config.ARTIFACTS / "tfidf_oof.npy")
    length = np.load(config.ARTIFACTS / "length_oof.npy")
    shared = {
        "tfidf": reliability.bootstrap_auroc(y, tfidf, n_boot, config.SEED),
        "length_only": reliability.bootstrap_auroc(y, length, n_boot, config.SEED),
        "human_author1": reliability.bootstrap_auroc(y, corpus.author1, n_boot, config.SEED),
        "human_author2": reliability.bootstrap_auroc(y, corpus.author2, n_boot, config.SEED),
    }
    out["baselines_shared"] = {
        k: {"auroc": float(roc_auc_score(y, v)), "ci95": [lo, hi]}
        for (k, (mean, lo, hi)), v in zip(
            shared.items(), [tfidf, length, corpus.author1, corpus.author2])}

    for mid in models:
        s = _slug(mid)
        rec = {}
        sweep = json.loads((config.ARTIFACTS / f"probe_sweep_{s}.json").read_text())
        bl = int(sweep["best_layer"])
        leaky = np.asarray(sweep["oof"][str(bl)])
        rec["best_layer"] = bl

        def entry(scores, mask=None):
            m = np.ones(len(y), bool) if mask is None else mask
            a = float(roc_auc_score(y[m], scores[m]))
            _, lo, hi = reliability.bootstrap_auroc(y[m], scores[m], n_boot, config.SEED)
            return {"auroc": a, "ci95": [lo, hi], "n": int(m.sum())}

        rec["random_cv"] = entry(leaky)

        # Recompute grouped CV and LOFO from the activations, not from cache.
        acts = extract.load_acts(mid)
        X = acts[bl]
        grouped = probe.oof_scores(X, y, groups=corpus.family)
        rec["grouped_cv"] = entry(grouped)

        lofo_summary, pooled = confound.leave_one_family_out(X, y, corpus.family)
        have = ~np.isnan(pooled)
        rec["lofo_pooled"] = entry(pooled, have)
        rec["lofo_macro_mixed_only"] = {
            "auroc": lofo_summary["macro_auroc"],
            "n_families_scored": lofo_summary["n_families_scored"]}

        mixed = confound.mixed_family_auroc(pooled, y, corpus.family)
        rec["mixed_family_macro"] = mixed.get("_macro")
        rec["mixed_family_detail"] = {k: v for k, v in mixed.items() if k != "_macro"}

        ja_path = config.ARTIFACTS / f"just_ask_{s}.json"
        if ja_path.exists():
            ja_map = json.loads(ja_path.read_text())
            ja = np.array([ja_map[i] for i in corpus.ids])
            rec["just_ask"] = entry(ja)

        np.save(config.ARTIFACTS / f"lofo_pooled_{s}.npy", pooled)
        np.save(config.ARTIFACTS / f"grouped_oof_{s}.npy", grouped)
        out["models"][s] = rec
        print(f"[lock] {s}: random={rec['random_cv']['auroc']:.4f} "
              f"lofo={rec['lofo_pooled']['auroc']:.4f} "
              f"CI{[round(x,3) for x in rec['lofo_pooled']['ci95']]}", flush=True)

    p = config.ARTIFACTS / "locked_metrics.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[lock] wrote {p}")
    return out
