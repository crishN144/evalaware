"""White-box probe vs the paper's black-box multiple-choice probe results.

mcq_transcripts/*.csv hold frontier-model answers to 5 probe questions x 5
epochs per transcript. Verified column semantics (checked against dataset.json,
exact match on all 858 shared rows):

    sample_id -> dataset `id`          joins the two sources
    is_eval   -> metadata.eval_category
    agent     -> metadata.scaffold is not None
    joe       -> metadata.eval_author1     (NOT `giles` - the names are swapped
    giles     -> metadata.eval_author2      relative to the author1/author2 order)
    human     -> mean(eval_author1, eval_author2)
    score     -> 1.0 / 0.5 / 0.0, the model's answer for one (pq_id, epoch)

858 of 976 dataset records appear in the CSVs; 19 CSV sample_ids are absent
from dataset.json. All comparisons are restricted to the intersection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from . import config


def available_models() -> list[str]:
    return sorted(p.stem for p in config.MCQ.glob("*.csv"))


def load_mcq(name: str) -> pd.DataFrame:
    return pd.read_csv(config.MCQ / f"{name}.csv")


def blackbox_scores(name: str) -> pd.DataFrame:
    """Mean score per sample_id, averaged over probe questions and epochs."""
    df = load_mcq(name)
    g = (df.groupby("sample_id")
           .agg(bb_score=("score", "mean"),
                is_eval=("is_eval", "first"),
                agent=("agent", "first"),
                human=("human", "first"),
                category=("category", "first"),
                n_obs=("score", "size"))
           .reset_index())
    return g


def compare(corpus, oof_scores, models=None) -> dict:
    """Align white-box probe scores with each black-box model on shared rows."""
    models = models or available_models()
    wb = pd.DataFrame({"sample_id": corpus.ids,
                       "wb_score": np.asarray(oof_scores),
                       "y": corpus.y,
                       "agentic": corpus.agentic})
    out = {}
    for name in models:
        bb = blackbox_scores(name)
        m = wb.merge(bb, on="sample_id", how="inner")
        if m.empty or m.y.nunique() < 2:
            out[name] = {"n": len(m), "error": "insufficient overlap"}
            continue
        assert (m.y.values == m.is_eval.values).all(), \
            f"{name}: label mismatch between dataset.json and mcq CSV"
        rec = {
            "n": int(len(m)),
            "blackbox_auroc": float(roc_auc_score(m.y, m.bb_score)),
            "whitebox_auroc": float(roc_auc_score(m.y, m.wb_score)),
            "human_auroc": float(roc_auc_score(m.y, m.human)),
            "corr_wb_bb": float(np.corrcoef(m.wb_score, m.bb_score)[0, 1]),
        }
        for stratum, mask in (("chat", m.agentic == 0), ("agentic", m.agentic == 1)):
            sub = m[mask]
            if sub.y.nunique() < 2:
                continue
            rec[f"blackbox_auroc_{stratum}"] = float(roc_auc_score(sub.y, sub.bb_score))
            rec[f"whitebox_auroc_{stratum}"] = float(roc_auc_score(sub.y, sub.wb_score))
        out[name] = rec
    return out
