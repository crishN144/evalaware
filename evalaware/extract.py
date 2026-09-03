"""Residual-stream extraction via nnsight, cached to artifacts/.

IMPORTANT (transformers >= 5): Qwen3DecoderLayer.forward returns a bare Tensor,
not a tuple. The legacy nnsight idiom `layers[i].output[0]` therefore indexes the
BATCH dimension and silently yields (seq, hidden) instead of (batch, seq, hidden).
Always use `layers[i].output`. tests/test_extract.py pins this.
"""
from __future__ import annotations

import gc

import numpy as np
import torch

from . import config


def free_model(lm) -> None:
    """Drop a loaded model and return its memory to the GPU.

    Stages run in one process, so a model left resident from an earlier stage
    is still holding ~16 GiB when the next stage loads its own copy. Without
    this, the second load sees a nearly full card and the preflight refuses.
    """
    try:
        del lm
    except NameError:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].replace(".", "-")


def acts_path(model_id: str, pooling: str = "last"):
    suffix = "" if pooling == "last" else f"_{pooling}"
    return config.ARTIFACTS / f"acts_{model_slug(model_id)}{suffix}.npz"


# ------------------------------------------------------------------ truncation
def truncate_text(text: str, tokenizer,
                  max_tokens=None, head_tokens=None) -> str:
    """Head+tail truncation at the token level, re-rendered to text.

    The head carries the system prompt and task framing; the tail carries the
    position whose residual we read. Dropping either loses signal, so we keep
    both and elide the middle.
    """
    max_tokens = max_tokens or config.MAX_TOKENS
    head_tokens = head_tokens or config.HEAD_TOKENS
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    tail_tokens = max_tokens - head_tokens
    head = tokenizer.decode(ids[:head_tokens], skip_special_tokens=True)
    tail = tokenizer.decode(ids[-tail_tokens:], skip_special_tokens=True)
    return head + config.ELISION + tail


# ------------------------------------------------------------------ extraction
# Rough bf16 weight footprints, GiB. Used only for the preflight check.
_WEIGHT_GIB = {"Qwen3-4B": 8.0, "Qwen3-8B": 16.4}


def free_gpu_gib(device: int = 0) -> float:
    free, _total = torch.cuda.mem_get_info(device)
    return free / 1024 ** 3


def require_free_gpu_memory(model_id: str, headroom: float = 4.0) -> None:
    """Fail fast if a co-tenant job has taken the GPU's memory.

    On AIRE, `--gres=gpu:l40s:1` allocates a device but does NOT fence off its
    memory: another job can already be resident on the same physical card. A
    45 GiB L40S showing only 10 GiB free is normal, not a bug. Checking here
    turns a 40-line CUDA traceback deep inside from_pretrained into one line.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device visible")
    need = _WEIGHT_GIB.get(model_slug(model_id), 8.0) + headroom
    free = free_gpu_gib()
    if free < need:
        raise RuntimeError(
            f"only {free:.1f} GiB free on this GPU but {model_id} needs about "
            f"{need:.1f} GiB (weights + headroom). Another job is sharing the "
            f"card. Resubmit to land on a different one, or use a smaller model."
        )
    print(f"[extract] preflight ok: {free:.1f} GiB free, need ~{need:.1f} GiB")


def load_model(model_id: str, device_map=None):
    """device_map=None -> "cuda" if one card has room, else "auto" to shard."""
    from nnsight import LanguageModel

    if device_map is None:
        need = _WEIGHT_GIB.get(model_slug(model_id), 8.0) + 4.0
        if torch.cuda.is_available() and torch.cuda.device_count() > 1 \
                and free_gpu_gib() < need:
            print(f"[extract] card 0 has {free_gpu_gib():.1f} GiB free; "
                  f"sharding with device_map='auto'")
            device_map = "auto"
        else:
            device_map = "cuda"
            require_free_gpu_memory(model_id)
    elif device_map == "cuda":
        require_free_gpu_memory(model_id)

    dtype = getattr(torch, config.DTYPE)
    lm = LanguageModel(model_id, device_map=device_map, dtype=dtype, dispatch=True)
    lm.tokenizer.padding_side = "left"          # last real token is always at -1
    if lm.tokenizer.pad_token is None:
        lm.tokenizer.pad_token = lm.tokenizer.eos_token
    return lm


@torch.no_grad()
def extract_batch(lm, texts: list[str], layers: list[int],
                  pooling: str = "last") -> dict[int, np.ndarray]:
    """Residual stream at each requested layer, pooled over the sequence.

    pooling="last" reads the final token (padding is left-side, so index -1 is
    always a real token). pooling="mean" averages over non-pad positions, which
    is a different readout and a genuine sensitivity check rather than a
    reparametrisation of the same one.
    """
    saved = {}
    with lm.trace(texts):
        for li in layers:
            # `.output` -> (batch, seq, hidden). Do NOT write `.output[0]`.
            h = lm.model.layers[li].output
            saved[li] = (h[:, -1, :] if pooling == "last" else h.mean(dim=1)).save()
    return {li: saved[li].float().cpu().numpy() for li in layers}


def extract_corpus(corpus, model_id: str, layers=None, batch_size: int = 4,
                   overwrite: bool = False, log_every: int = 20,
                   pooling: str = "last"):
    """Extract and cache activations for every record. Returns {layer: (n,hid)}."""
    layers = layers or config.LAYERS
    out_path = acts_path(model_id, pooling)
    if out_path.exists() and not overwrite:
        print(f"[extract] cache hit {out_path}")
        return load_acts(model_id, pooling)

    lm = load_model(model_id)
    tok = lm.tokenizer
    texts = [truncate_text(t, tok) for t in corpus.texts]

    # Long sequences dominate cost; grouping by length keeps padding waste down.
    order = np.argsort([len(tok(t, add_special_tokens=False)["input_ids"])
                        for t in texts])
    chunks = {li: [] for li in layers}
    done = 0
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        batch = [texts[i] for i in idx]
        got = extract_batch(lm, batch, layers, pooling=pooling)
        for li in layers:
            chunks[li].append(got[li])
        done += len(idx)
        if done % log_every < batch_size:
            print(f"[extract] {done}/{len(order)}", flush=True)

    inv = np.argsort(order)              # restore original record order
    acts = {li: np.concatenate(chunks[li], axis=0)[inv] for li in layers}

    np.savez_compressed(
        out_path,
        ids=np.array(corpus.ids),
        layers=np.array(layers),
        **{f"layer_{li}": acts[li] for li in layers},
    )
    print(f"[extract] wrote {out_path}")
    free_model(lm)
    return acts


def load_acts(model_id: str, pooling: str = "last") -> dict[int, np.ndarray]:
    z = np.load(acts_path(model_id, pooling), allow_pickle=False)
    return {int(li): z[f"layer_{int(li)}"] for li in z["layers"]}


def load_acts_ids(model_id: str, pooling: str = "last") -> list[str]:
    z = np.load(acts_path(model_id, pooling), allow_pickle=False)
    return [str(x) for x in z["ids"]]
