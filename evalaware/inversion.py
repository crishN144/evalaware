"""Step 2: evidence for the prompt_injection inversion. Gathers, does not conclude.

The `prompt_injection` family scores BELOW chance under leave-one-family-out:
the probe assigns higher P(eval) to the deploy-labelled rows than to the
eval-labelled ones. This module dumps the numbers and the raw transcripts so the
inversion can be adjudicated by reading, not by trusting a metric.

Family composition (all chat, all n=56):
    mosscap_prompt_injection      19  eval
    gandalf_ignore_instructions    9  eval
    gandalf_summarization          9  eval
    hackaprompt_dataset           19  deploy
"""
from __future__ import annotations

import json
import textwrap

import numpy as np
from sklearn.metrics import roc_auc_score

from . import compare_blackbox, config, data

FAMILY = "prompt_injection"
EVAL_CATS = ["mosscap_prompt_injection", "gandalf_ignore_instructions",
             "gandalf_summarization"]
DEPLOY_CATS = ["hackaprompt_dataset"]


def _scores(corpus):
    out = {}
    for s in ("Qwen3-4B", "Qwen3-8B"):
        p = config.ARTIFACTS / f"lofo_pooled_{s}.npy"
        if p.exists():
            out[f"probe_lofo_{s}"] = np.load(p)
        sw = config.ARTIFACTS / f"probe_sweep_{s}.json"
        if sw.exists():
            d = json.loads(sw.read_text())
            out[f"probe_leaky_cv_{s}"] = np.asarray(d["oof"][str(d["best_layer"])])
        ja = config.ARTIFACTS / f"just_ask_{s}.json"
        if ja.exists():
            mp = json.loads(ja.read_text())
            out[f"just_ask_{s}"] = np.array([mp[i] for i in corpus.ids])
    out["human_author1"] = corpus.author1
    out["human_author2"] = corpus.author2
    out["human_mean"] = (corpus.author1 + corpus.author2) / 2
    return out


def report(corpus=None) -> dict:
    corpus = corpus or data.load_corpus()
    fam = np.array(corpus.family)
    cat = np.array(corpus.category)
    y = corpus.y
    m = fam == FAMILY
    sc = _scores(corpus)

    rep = {"n": int(m.sum()), "n_eval": int(y[m].sum()),
           "n_deploy": int((y[m] == 0).sum()), "auroc_within_family": {},
           "mean_score_by_subcategory": {}, "blackbox_within_family": {}}

    for name, s in sc.items():
        rep["auroc_within_family"][name] = float(roc_auc_score(y[m], s[m]))

    for c in EVAL_CATS + DEPLOY_CATS:
        k = cat == c
        rep["mean_score_by_subcategory"][c] = {
            "n": int(k.sum()),
            "label": "eval" if y[k].mean() == 1 else "deploy",
            **{name: float(s[k].mean()) for name, s in sc.items()},
        }

    ids = set(np.array(corpus.ids)[m])
    for name in compare_blackbox.available_models():
        bb = compare_blackbox.blackbox_scores(name)
        bb = bb[bb.sample_id.isin(ids)]
        if len(bb) < 20 or bb.is_eval.nunique() < 2:
            continue
        rep["blackbox_within_family"][name] = {
            "n": int(len(bb)),
            "auroc": float(roc_auc_score(bb.is_eval, bb.bb_score)),
        }

    # How many independent judges invert (AUROC < 0.5) on this family?
    judges = {**{k: v for k, v in rep["auroc_within_family"].items()},
              **{k: v["auroc"] for k, v in rep["blackbox_within_family"].items()}}
    rep["n_judges"] = len(judges)
    rep["n_judges_inverting"] = sum(1 for v in judges.values() if v < 0.5)

    p = config.ARTIFACTS / "inversion_prompt_injection.json"
    p.write_text(json.dumps(rep, indent=2))
    print(f"[inversion] wrote {p}")
    return rep


def dump_transcripts(corpus=None, per_side: int = 20, chars: int = 2600) -> str:
    """Write raw transcripts from both sides to a readable file."""
    corpus = corpus or data.load_corpus()
    cat = np.array(corpus.category)
    y = corpus.y
    sc = _scores(corpus)
    probe = sc.get("probe_lofo_Qwen3-4B")

    lines = []
    lines.append("prompt_injection family - raw transcripts, both sides")
    lines.append("=" * 78)
    lines.append("")
    lines.append("The benchmark labels mosscap/gandalf as EVAL and hackaprompt as DEPLOY.")
    lines.append("The probe, most frontier models, and one of two human annotators")
    lines.append("rank them the other way round. Read and judge for yourself:")
    lines.append("does the hackaprompt content read as organic real-world usage?")
    lines.append("")
    lines.append("probe_score = P(eval) from Qwen3-4B, whole family held out of training.")
    lines.append("")

    for side, cats in (("EVAL-labelled", EVAL_CATS), ("DEPLOY-labelled", DEPLOY_CATS)):
        idx = np.flatnonzero(np.isin(cat, cats))
        rng = np.random.default_rng(config.SEED)
        take = idx if len(idx) <= per_side else rng.choice(idx, per_side, replace=False)
        lines.append("")
        lines.append("#" * 78)
        lines.append(f"# {side}  ({len(take)} shown of {len(idx)})")
        lines.append("#" * 78)
        for i in sorted(take):
            lines.append("")
            lines.append("-" * 78)
            lines.append(f"id={corpus.ids[i]}   category={corpus.category[i]}   "
                         f"label={'eval' if y[i] else 'deploy'}")
            lines.append(f"probe_score={probe[i]:.3f}   "
                         f"human_a1={corpus.author1[i]:.3f}  human_a2={corpus.author2[i]:.3f}")
            lines.append("-" * 78)
            body = corpus.texts[i]
            if len(body) > chars:
                body = body[:chars] + f"\n[... {len(corpus.texts[i]) - chars} more chars ...]"
            lines.append(textwrap.indent(body, "  "))

    p = config.ARTIFACTS / "inversion_transcripts.txt"
    p.write_text("\n".join(lines))
    print(f"[inversion] wrote {p}  ({len(lines)} lines)")
    return str(p)
