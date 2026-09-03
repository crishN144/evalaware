"""The null tests. If these fail, no downstream number can be believed."""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from evalaware import confound, probe


@pytest.fixture
def separable():
    rng = np.random.default_rng(0)
    n, d = 400, 64
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    X = rng.normal(0, 1, (n, d))
    X[y == 1, :3] += 1.5          # signal spread over a few dims, as in a real probe
    return X, y


def test_probe_recovers_a_real_signal(separable):
    X, y = separable
    s = probe.oof_scores(X, y)
    assert roc_auc_score(y, s) > 0.90


def test_shuffled_labels_give_chance_auroc():
    """The single most important guard: no leakage through CV or scaling."""
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 128))          # pure noise, more dims than rows/2
    y = rng.integers(0, 2, 300)
    s = probe.oof_scores(X, y)
    assert 0.35 < roc_auc_score(y, s) < 0.65


def test_scaler_is_fit_inside_the_fold_not_on_all_data():
    """A StandardScaler fit outside CV leaks test statistics into training."""
    clf = probe.make_probe()
    names = [s[0] for s in clf.steps]
    assert names[0] == "standardscaler"
    assert names[-1] == "logisticregression"


def test_layer_sweep_returns_one_auroc_per_layer(separable):
    X, y = separable
    acts = {0: X, 4: X, 8: X}
    res, oof = probe.layer_sweep(acts, y)
    assert set(res) == {0, 4, 8}
    assert set(oof) == {0, 4, 8}
    assert all(0.0 <= v <= 1.0 for v in res.values())
    assert probe.best_layer(res) in res


def test_grouped_cv_never_trains_on_the_test_group():
    """Grouped CV must drop AUROC when the signal is group identity alone."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 40)
    y = (groups % 2 == 0).astype(int)          # label is a pure function of group
    X = np.zeros((400, 8))
    for g in range(10):                        # each group gets its own offset
        X[groups == g, g % 8] = 1.0
    s_group = probe.oof_scores(X, y, groups=groups)
    assert roc_auc_score(y, s_group) < 0.75


def test_leave_one_family_out_flags_a_memorised_family():
    rng = np.random.default_rng(0)
    fams = np.repeat([f"f{i}" for i in range(6)], 50)
    y = rng.integers(0, 2, 300)
    X = rng.normal(0, 1, (300, 16))            # no real signal
    out, pooled = confound.leave_one_family_out(X, y, fams)
    assert out["n_families_scored"] >= 4
    assert 0.30 < out["macro_auroc"] < 0.70
    assert 0.30 < out["pooled_auroc"] < 0.70
    assert out["pooled_n"] == 300


def test_length_matched_kills_a_pure_length_shortcut():
    """A 'probe' that only reads length must collapse once lengths are matched."""
    rng = np.random.default_rng(0)
    n = 1200
    y = rng.integers(0, 2, n)
    lengths = np.where(y == 1, rng.lognormal(8, 0.5, n), rng.lognormal(6, 0.5, n))
    s = lengths / lengths.max()                # score == length, nothing else
    raw = roc_auc_score(y, s)
    matched = confound.length_matched_auroc(s, y, lengths, n_rep=50)
    assert raw > 0.85
    assert matched["auroc"] < 0.60
    assert matched["auroc"] < raw - 0.30
    # matching quality must be reported, and must actually be tight
    assert abs(matched["residual_log_len_gap"]) < 0.05


def test_within_stratum_splits_by_scaffold():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    ag = rng.integers(0, 2, 200)
    s = rng.uniform(0, 1, 200)
    out = confound.within_stratum(s, y, ag)
    assert set(out) == {"chat", "agentic"}
    assert out["chat"]["n"] + out["agentic"]["n"] == 200


def test_lofo_pools_predictions_from_single_class_families():
    """44 of 47 real families are single-class; per-family AUROC is undefined
    there, but their held-out rows must still reach the pooled score."""
    rng = np.random.default_rng(0)
    fams = np.repeat([f"f{i}" for i in range(8)], 25)
    y = np.zeros(200, dtype=int)
    y[np.isin(fams, ["f0", "f1", "f2", "f3"])] = 1      # every family is pure
    X = rng.normal(0, 1, (200, 8)) + y[:, None] * 0.8
    out, pooled = confound.leave_one_family_out(X, y, fams)
    assert out["n_families_scored"] == 0                # none are mixed
    assert np.isnan(out["macro_auroc"])
    assert out["pooled_n"] == 200                       # but all rows scored
    assert out["pooled_auroc"] is not None


def test_mixed_family_auroc_only_scores_label_mixed_families():
    rng = np.random.default_rng(0)
    fams = np.array(["pure"] * 60 + ["mixed"] * 60)
    y = np.r_[np.ones(60), rng.integers(0, 2, 60)].astype(int)
    s = rng.uniform(0, 1, 120)
    out = confound.mixed_family_auroc(s, y, fams)
    assert "pure" not in out
    assert "mixed" in out
    assert out["mixed"]["n"] == 60
    assert "_macro" in out


def test_mixed_family_auroc_ignores_nan_scores():
    fams = np.array(["a"] * 40)
    y = np.r_[np.ones(20), np.zeros(20)].astype(int)
    s = np.r_[np.full(10, np.nan), np.linspace(1, 0, 30)]
    out = confound.mixed_family_auroc(s, y, fams)
    assert out["a"]["n"] == 30
