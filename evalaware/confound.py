"""Does the probe generalise, or has it memorised source families?

Three tests, in increasing order of severity:

  1. leave_one_family_out - train on all source families but one, test on the
     held-out family. Random CV leaks: aider_commit / aider_multi / aider_full
     are the same corpus, so a random split puts near-twins on both sides.

  2. within_stratum - AUROC computed separately on chat-only and agentic-only
     rows. eval_type is badly confounded with scaffold in this corpus
     (alignment: 244 chat / 53 agentic; capability: 43 chat / 197 agentic), and
     eval_category itself correlates with length via scaffold.

  3. length_matched - resample deploy rows to match the eval length histogram,
     then re-score. Kills the "long => agentic => eval" shortcut.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config
from .probe import make_probe


def leave_one_family_out(X, y, families, min_per_side: int = 8):
    """Train on every source family but one, test on the held-out family.

    Random CV leaks badly here: no raw category in this corpus is label-mixed
    (0 of 60), so category identity determines the label outright, and a random
    split puts near-twins of every test row in training.

    Per-family AUROC is undefined for the 44 single-class families, so it is
    reported as None for them -- but their held-out predictions still go into a
    POOLED set scored once across all rows. `pooled_auroc` is the honest
    generalisation number; `macro_auroc` covers only the few mixed families.
    """
    families = np.asarray(families)
    y = np.asarray(y)
    pooled = np.full(len(y), np.nan)
    out = {}
    for fam in sorted(set(families)):
        te = families == fam
        tr = ~te
        if len(set(y[tr])) < 2:
            continue
        clf = make_probe()
        clf.fit(X[tr], y[tr])
        s = clf.predict_proba(X[te])[:, 1]
        pooled[te] = s
        rec = {"n": int(te.sum()), "pos_rate": float(y[te].mean()),
               "mean_score": float(s.mean())}
        if te.sum() >= min_per_side and len(set(y[te])) == 2:
            rec["auroc"] = float(roc_auc_score(y[te], s))
        else:
            rec["auroc"] = None
            rec["reason"] = ("single-class family" if len(set(y[te])) < 2
                             else "too few rows")
        out[fam] = rec

    scored = [v["auroc"] for v in out.values() if v.get("auroc") is not None]
    have = ~np.isnan(pooled)
    summary = {
        "per_family": out,
        "n_families": len(out),
        "n_families_scored": len(scored),
        "macro_auroc": float(np.mean(scored)) if scored else float("nan"),
        "median_auroc": float(np.median(scored)) if scored else float("nan"),
        "min_auroc": float(np.min(scored)) if scored else float("nan"),
        "pooled_auroc": (float(roc_auc_score(y[have], pooled[have]))
                         if have.sum() and len(set(y[have])) == 2 else None),
        "pooled_n": int(have.sum()),
    }
    return summary, pooled


def mixed_family_auroc(scores, y, families, min_n: int = 20):
    """AUROC inside each label-mixed family.

    These families use one scaffold for both eval and deploy rows, so the source
    corpus is roughly held constant while the label varies. With no label-mixed
    raw category in the corpus, this is the closest thing available to a test of
    whether the probe tracks eval-ness rather than corpus identity.
    """
    families = np.asarray(families)
    y = np.asarray(y)
    s = np.asarray(scores)
    out = {}
    for fam in sorted(set(families)):
        m = families == fam
        if m.sum() < min_n or len(set(y[m])) < 2:
            continue
        ok = m & ~np.isnan(s)
        if len(set(y[ok])) < 2:
            continue
        out[fam] = {"n": int(ok.sum()), "pos_rate": float(y[ok].mean()),
                    "auroc": float(roc_auc_score(y[ok], s[ok]))}
    if out:
        out["_macro"] = float(np.mean([v["auroc"] for v in out.values()]))
    return out


def within_stratum(oof_scores, y, agentic):
    """AUROC inside each scaffold stratum, using existing out-of-fold scores."""
    y = np.asarray(y)
    agentic = np.asarray(agentic)
    res = {}
    for name, m in (("chat", agentic == 0), ("agentic", agentic == 1)):
        if len(set(y[m])) < 2:
            res[name] = {"n": int(m.sum()), "auroc": None}
            continue
        res[name] = {"n": int(m.sum()),
                     "auroc": float(roc_auc_score(y[m], np.asarray(oof_scores)[m])),
                     "pos_rate": float(y[m].mean())}
    return res


def length_matched_auroc(oof_scores, y, lengths, n_bins: int = None,
                         seed: int = None, n_rep: int = 200):
    """AUROC on a subsample where eval/deploy share a length distribution.

    Bins log-length, then in each bin keeps equal numbers of each class. The
    correction is only as good as the binning: coarse bins leave eval rows
    systematically longer than deploy rows *within* a bin, and the shortcut
    survives. `residual_log_len_gap` reports the mean within-bin log-length
    difference actually achieved -- if it is not near zero, the reported AUROC
    still contains length signal and n_bins should be raised.
    """
    n_bins = config.LENGTH_MATCH_BINS if n_bins is None else n_bins
    seed = config.SEED if seed is None else seed
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(oof_scores)
    lg = np.log1p(np.asarray(lengths, dtype=float))
    edges = np.quantile(lg, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_id = np.clip(np.digitize(lg, edges[1:-1]), 0, n_bins - 1)

    usable = [b for b in range(n_bins)
              if ((bin_id == b) & (y == 1)).any() and ((bin_id == b) & (y == 0)).any()]
    gaps = [lg[(bin_id == b) & (y == 1)].mean() - lg[(bin_id == b) & (y == 0)].mean()
            for b in usable]

    vals, n_last = [], 0
    for _ in range(n_rep):
        keep = []
        for b in usable:
            m = bin_id == b
            pos, neg = np.flatnonzero(m & (y == 1)), np.flatnonzero(m & (y == 0))
            k = min(len(pos), len(neg))
            keep.append(rng.choice(pos, k, replace=False))
            keep.append(rng.choice(neg, k, replace=False))
        if not keep:
            break
        idx = np.concatenate(keep)
        n_last = len(idx)
        if len(set(y[idx])) < 2:
            continue
        vals.append(roc_auc_score(y[idx], s[idx]))

    if not vals:
        return {"auroc": None, "n_per_rep": 0, "n_usable_bins": len(usable)}
    return {"auroc": float(np.mean(vals)),
            "auroc_sd": float(np.std(vals)),
            "n_per_rep": int(n_last),
            "n_reps": len(vals),
            "n_bins": int(n_bins),
            "n_usable_bins": len(usable),
            "residual_log_len_gap": float(np.mean(gaps)) if gaps else float("nan")}
