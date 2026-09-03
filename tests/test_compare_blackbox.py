import numpy as np
import pytest

from evalaware import compare_blackbox as CB


def test_mcq_csvs_are_present():
    models = CB.available_models()
    assert len(models) >= 5
    assert "gpt4o" in models


@pytest.fixture(scope="module")
def mcq():
    return CB.load_mcq("gpt4o")


def test_join_key_and_label_columns_agree_with_dataset(corpus, mcq):
    """Pins the column semantics documented in compare_blackbox's docstring."""
    meta = {i: (y, a, a1, a2) for i, y, a, a1, a2 in
            zip(corpus.ids, corpus.y, corpus.agentic, corpus.author1, corpus.author2)}
    df = mcq.drop_duplicates("sample_id")
    df = df[df.sample_id.isin(meta)]
    assert len(df) >= 800

    y = np.array([meta[s][0] for s in df.sample_id])
    ag = np.array([meta[s][1] for s in df.sample_id])
    a1 = np.array([meta[s][2] for s in df.sample_id])
    a2 = np.array([meta[s][3] for s in df.sample_id])

    assert (df.is_eval.values == y).all()
    assert (df.agent.values == ag).all()
    # joe == author1 and giles == author2 -- the names are swapped relative to
    # the author1/author2 ordering. Getting this backwards silently corrupts
    # every human-vs-model comparison.
    assert np.allclose(df.joe.values, a1)
    assert np.allclose(df.giles.values, a2)
    assert np.allclose(df.human.values, (a1 + a2) / 2)


def test_blackbox_scores_average_over_probe_questions_and_epochs(mcq):
    g = CB.blackbox_scores("gpt4o")
    assert g.n_obs.max() <= 25          # 5 probe questions x 5 epochs
    assert g.bb_score.between(0, 1).all()
    assert g.sample_id.is_unique


def test_compare_restricts_to_the_intersection(corpus):
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, len(corpus))
    out = CB.compare(corpus, s, models=["gpt4o"])
    rec = out["gpt4o"]
    assert 800 <= rec["n"] <= 976
    assert 0.0 <= rec["blackbox_auroc"] <= 1.0
    assert 0.4 < rec["whitebox_auroc"] < 0.6      # random scores -> chance
