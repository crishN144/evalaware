"""Reliability: ROC, TPR at a fixed FPR, precision at a deployment base rate,
calibration/ECE, and bootstrap confidence intervals.

AUROC alone is not a deployment claim. A monitor that flags evaluation-shaped
transcripts runs against a population where evals are rare; at a 1% base rate a
classifier with excellent AUROC can still be mostly wrong when it fires.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def tpr_at_fpr(y, s, target_fpr: float):
    fpr, tpr, thr = roc_curve(y, s)
    i = np.searchsorted(fpr, target_fpr, side="right") - 1
    i = max(i, 0)
    return float(tpr[i]), float(thr[i]), float(fpr[i])


def precision_at_base_rate(tpr: float, fpr: float, base_rate: float):
    """Bayes-adjusted precision if evals occur at `base_rate` in the wild."""
    num = tpr * base_rate
    den = num + fpr * (1.0 - base_rate)
    return float(num / den) if den > 0 else float("nan")


def expected_calibration_error(y, p, n_bins: int = 10):
    """Equal-width ECE. Probes are fit on a ~55/45 corpus; their outputs are not
    calibrated for any other base rate, and ECE here says how far off they are."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(p)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p > lo) & (p <= hi) if lo > 0 else (p >= lo) & (p <= hi)
        if not m.any():
            bins.append((lo, hi, 0, np.nan, np.nan))
            continue
        conf, acc = p[m].mean(), y[m].mean()
        ece += (m.sum() / n) * abs(acc - conf)
        bins.append((float(lo), float(hi), int(m.sum()), float(conf), float(acc)))
    return float(ece), bins


def bootstrap_auroc(y, s, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile CI over stratified resamples (keeps class balance stable)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True),
                              rng.choice(neg, len(neg), replace=True)])
        vals[b] = roc_auc_score(y[idx], s[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(np.mean(vals)), float(lo), float(hi)


def full_report(y, s, base_rate: float, target_fpr: float,
                n_boot: int, seed: int) -> dict:
    auroc = float(roc_auc_score(y, s))
    mean, lo, hi = bootstrap_auroc(y, s, n_boot, seed)
    tpr, thr, achieved_fpr = tpr_at_fpr(y, s, target_fpr)
    ece, bins = expected_calibration_error(y, s)
    return {
        "auroc": auroc,
        "auroc_ci95": [lo, hi],
        "auroc_bootstrap_mean": mean,
        "target_fpr": target_fpr,
        "achieved_fpr": achieved_fpr,
        "tpr_at_fpr": tpr,
        "threshold_at_fpr": thr,
        "deploy_base_rate": base_rate,
        "precision_at_base_rate": precision_at_base_rate(tpr, achieved_fpr, base_rate),
        "ece": ece,
        "calibration_bins": bins,
        "n": int(len(y)),
        "n_pos": int(np.sum(y)),
    }
