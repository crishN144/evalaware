import numpy as np
import pytest

from evalaware import reliability as R


def test_perfectly_calibrated_input_has_near_zero_ece():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 200_000)
    y = (rng.uniform(0, 1, 200_000) < p).astype(int)
    ece, bins = R.expected_calibration_error(y, p, n_bins=10)
    assert ece < 0.01
    assert sum(b[2] for b in bins) == len(p)


def test_systematically_overconfident_input_has_large_ece():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 10_000)
    p = np.where(y == 1, 0.99, 0.98)      # confident and nearly always wrong
    ece, _ = R.expected_calibration_error(y, p)
    assert ece > 0.4


def test_precision_at_base_rate_matches_bayes():
    # tpr=1.0, fpr=0.01, base=0.01 -> 0.01 / (0.01 + 0.01*0.99)
    got = R.precision_at_base_rate(1.0, 0.01, 0.01)
    assert got == pytest.approx(0.01 / (0.01 + 0.0099), rel=1e-9)


def test_rare_events_make_a_good_classifier_mostly_wrong():
    """The point of reporting precision at a deployment base rate."""
    assert R.precision_at_base_rate(0.9, 0.05, 0.01) < 0.20


def test_tpr_at_fpr_respects_the_budget():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(500), np.zeros(500)].astype(int)
    s = np.r_[rng.normal(1, 1, 500), rng.normal(0, 1, 500)]
    tpr, thr, fpr = R.tpr_at_fpr(y, s, 0.01)
    assert fpr <= 0.01 + 1e-9
    assert 0.0 <= tpr <= 1.0


def test_bootstrap_ci_brackets_the_point_estimate():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(300), np.zeros(300)].astype(int)
    s = np.r_[rng.normal(1, 1, 300), rng.normal(0, 1, 300)]
    auroc = roc_auc_score(y, s)
    mean, lo, hi = R.bootstrap_auroc(y, s, n_boot=400, seed=0)
    assert lo < auroc < hi
    assert hi - lo < 0.25
