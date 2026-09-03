"""Per-layer logistic-regression probe on the residual stream."""
from __future__ import annotations

import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from . import config


def make_probe(C: float = 1.0):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=C, random_state=config.SEED),
    )


def oof_scores(X, y, groups=None, C: float = 1.0, n_splits=None):
    """Out-of-fold P(eval). Grouped CV when `groups` is given."""
    n_splits = n_splits or config.CV_FOLDS
    clf = make_probe(C)
    if groups is None:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                             random_state=config.SEED)
        return cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    cv = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    return cross_val_predict(clf, X, y, cv=cv, groups=groups,
                             method="predict_proba")[:, 1]


def layer_sweep(acts: dict[int, np.ndarray], y, groups=None, C: float = 1.0):
    """AUROC at every extracted layer. Returns (results, oof_by_layer)."""
    results, oof = {}, {}
    for li in sorted(acts):
        s = oof_scores(acts[li], y, groups=groups, C=C)
        oof[li] = s
        results[li] = float(roc_auc_score(y, s))
        print(f"[probe] layer {li:2d}  AUROC={results[li]:.4f}", flush=True)
    return results, oof


def best_layer(results: dict[int, float]) -> int:
    return max(results, key=results.get)


def fit_full(X, y, C: float = 1.0):
    clf = make_probe(C)
    clf.fit(X, y)
    return clf


def save_sweep(results, oof, ids, model_id, tag="sweep"):
    path = config.ARTIFACTS / f"probe_{tag}_{model_id.split('/')[-1]}.json"
    path.write_text(json.dumps({
        "auroc_by_layer": {str(k): v for k, v in results.items()},
        "best_layer": best_layer(results),
        "best_auroc": max(results.values()),
        "oof": {str(k): v.tolist() for k, v in oof.items()},
        "ids": list(ids),
    }))
    print(f"[probe] wrote {path}")
    return path
