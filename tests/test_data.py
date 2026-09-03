import numpy as np
import pytest

from evalaware import data


def test_flatten_content_handles_every_observed_shape():
    assert data.flatten_content("hi") == "hi"
    assert data.flatten_content(None) == ""
    assert data.flatten_content([{"type": "text", "text": "a"},
                                 {"type": "text", "text": "b"}]) == "a b"
    # blocks without a text key must not raise
    assert isinstance(data.flatten_content([{"type": "image"}]), str)


def test_corpus_shape_and_label_counts(corpus):
    # Pinned from the decrypted dataset.json; a change means the data moved.
    assert len(corpus) == 976
    assert corpus.y.sum() == 537              # eval
    assert (corpus.y == 0).sum() == 439       # deploy
    assert corpus.agentic.sum() == 476
    assert (corpus.agentic == 0).sum() == 500
    assert sum(t == "capability" for t in corpus.eval_type) == 240
    assert sum(t == "alignment" for t in corpus.eval_type) == 297


def test_every_transcript_renders_non_empty(corpus):
    assert all(isinstance(t, str) for t in corpus.texts)
    assert min(len(t) for t in corpus.texts) > 0


def test_eval_type_empty_exactly_on_deploy_rows(corpus):
    for t, y in zip(corpus.eval_type, corpus.y):
        assert (t == "") == (y == 0)


def test_families_collapse_near_duplicate_sources(corpus):
    fam = dict(zip(corpus.category, corpus.family))
    assert fam["aider_commit"] == fam["aider_multi"] == fam["aider_full"]
    assert fam["smol_gaia_search"] == fam["smol_real_manager"]
    assert fam["ethics_justice"] == fam["ethics_virtue"]
    assert len(set(corpus.family)) < len(set(corpus.category))


def test_human_labels_present_and_in_range(corpus):
    for a in (corpus.author1, corpus.author2):
        assert np.isfinite(a).all()
        assert a.min() >= 0.0 and a.max() <= 1.0


def test_subset_is_consistent(corpus):
    m = corpus.agentic == 1
    sub = corpus.subset(m)
    assert len(sub) == m.sum()
    assert set(sub.ids) <= set(corpus.ids)
    assert (sub.y == corpus.y[m]).all()
