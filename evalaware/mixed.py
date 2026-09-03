"""Step 3: the three label-mixed families.

These are the only places in the corpus where the label varies while the source
scaffold is held roughly fixed, so they carry most of the evidential weight.
Composition (each merged by data.family_of from same-scaffold categories):

    smolagents        smol_gaia_{manager,search}=eval  vs smol_real_{manager,search}=deploy
    aider             aider_full=eval                  vs aider_{commit,multi,manual}=deploy
    prompt_injection  mosscap/gandalf=eval             vs hackaprompt=deploy

All scores are leave-one-family-out: the whole family was held out of training.
"""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config, data, reliability

MIXED = ["smolagents", "aider", "prompt_injection"]


def _tokens(corpus):
    p = config.ARTIFACTS / "n_tokens.npy"
    return np.load(p) if p.exists() else np.array([len(t) for t in corpus.texts])


def report(corpus=None, n_boot=None) -> dict:
    corpus = corpus or data.load_corpus()
    n_boot = n_boot or 1000
    fam = np.array(corpus.family)
    cat = np.array(corpus.category)
    y = corpus.y
    nt = _tokens(corpus)
    out = {}

    for model in ("Qwen3-4B", "Qwen3-8B"):
        p = config.ARTIFACTS / f"lofo_pooled_{model}.npy"
        if not p.exists():
            continue
        s = np.load(p)
        rec = {}
        for f in MIXED:
            m = (fam == f) & ~np.isnan(s)
            if len(set(y[m])) < 2:
                continue
            auroc = float(roc_auc_score(y[m], s[m]))
            _, lo, hi = reliability.bootstrap_auroc(y[m], s[m], n_boot, config.SEED)
            subcats = {}
            for cc in sorted(set(cat[m])):
                k = m & (cat == cc)
                subcats[cc] = {
                    "n": int(k.sum()),
                    "label": "eval" if y[k].mean() == 1 else "deploy",
                    "mean_p_eval": float(s[k].mean()),
                    "median_tokens": float(np.median(nt[k])),
                    "agentic": float(np.mean(corpus.agentic[k])),
                    "human_a1": float(np.nanmean(corpus.author1[k])),
                    "human_a2": float(np.nanmean(corpus.author2[k])),
                }
            ev, dp = m & (y == 1), m & (y == 0)
            rec[f] = {
                "n": int(m.sum()), "n_eval": int(ev.sum()), "n_deploy": int(dp.sum()),
                "auroc": auroc, "ci95": [lo, hi],
                "inverted": auroc < 0.5,
                "mean_p_eval_on_eval_rows": float(s[ev].mean()),
                "mean_p_eval_on_deploy_rows": float(s[dp].mean()),
                "median_tokens_eval": float(np.median(nt[ev])),
                "median_tokens_deploy": float(np.median(nt[dp])),
                "agentic_rate_eval": float(corpus.agentic[ev].mean()),
                "agentic_rate_deploy": float(corpus.agentic[dp].mean()),
                "human_a1_auroc": float(roc_auc_score(y[m], corpus.author1[m])),
                "human_a2_auroc": float(roc_auc_score(y[m], corpus.author2[m])),
                "annotator_corr": float(np.corrcoef(corpus.author1[m],
                                                    corpus.author2[m])[0, 1]),
                "subcategories": subcats,
            }
        aurocs = [v["auroc"] for v in rec.values()]
        out[model] = {"per_family": rec,
                      "macro_auroc": float(np.mean(aurocs)) if aurocs else None,
                      "n_rows_total": int(sum(v["n"] for v in rec.values()))}

    p = config.ARTIFACTS / "mixed_families.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"[mixed] wrote {p}")
    return out
