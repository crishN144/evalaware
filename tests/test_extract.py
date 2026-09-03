import numpy as np
import pytest

from evalaware import config, data, extract


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.DEV_MODEL)


def test_short_text_passes_through_unchanged(tok):
    s = "user: hello\n\nassistant: hi"
    assert extract.truncate_text(s, tok) == s


def test_long_text_is_capped_and_keeps_head_and_tail(tok):
    head = "SYSTEM_MARKER_HEAD " + ("alpha " * 4000)
    tail = ("omega " * 4000) + " SYSTEM_MARKER_TAIL"
    out = extract.truncate_text(head + tail, tok, max_tokens=512, head_tokens=128)
    n = len(tok(out, add_special_tokens=False)["input_ids"])
    assert n <= 512 + 32                     # + elision marker slack
    assert "SYSTEM_MARKER_HEAD" in out       # framing survives
    assert "SYSTEM_MARKER_TAIL" in out       # readout position survives
    assert config.ELISION.strip() in out


def test_truncation_covers_the_whole_real_corpus(tok, corpus):
    """p90 is 16.5k tokens and the longest record is 78.8k, over max_position."""
    long_ids = np.argsort([-len(t) for t in corpus.texts])[:15]
    for i in long_ids:
        out = extract.truncate_text(corpus.texts[i], tok)
        n = len(tok(out, add_special_tokens=False)["input_ids"])
        assert n <= config.MAX_TOKENS + 32


def test_layers_are_in_range():
    assert max(config.LAYERS) < config.N_LAYERS
    assert min(config.LAYERS) >= 0
    assert len(set(config.LAYERS)) == len(config.LAYERS)


# ------------------------------------------------------------------ GPU only
@pytest.mark.gpu
def test_residual_accessor_returns_batch_seq_hidden():
    """Pins the transformers-v5 breakage.

    Qwen3DecoderLayer.forward returns a bare Tensor, so `.output[0]` indexes the
    batch dimension and silently yields (seq, hidden). With batch=1 the wrong
    shape still 'looks fine', which is exactly why this is a test.
    """
    import torch
    lm = extract.load_model(config.DEV_MODEL)
    prompts = ["The capital of France is", "Hello there friend"]
    with lm.trace(prompts):
        good = lm.model.layers[8].output.save()
        bad = lm.model.layers[8].output[0].save()
    assert good.ndim == 3
    assert good.shape[0] == len(prompts)
    assert good.shape[-1] == lm.config.hidden_size
    assert bad.ndim == 2                      # batch dim indexed away
    assert bad.shape[0] == good.shape[1]


@pytest.mark.gpu
def test_extract_batch_shapes_and_left_padding():
    lm = extract.load_model(config.DEV_MODEL)
    assert lm.tokenizer.padding_side == "left"
    texts = ["short", "a considerably longer prompt with many more tokens in it"]
    got = extract.extract_batch(lm, texts, [0, 8])
    for li in (0, 8):
        assert got[li].shape == (2, lm.config.hidden_size)
        assert np.isfinite(got[li]).all()
