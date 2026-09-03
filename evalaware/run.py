"""Thin orchestrator. Each stage writes to artifacts/ and can run alone.

    python -m evalaware.run --stage all   --model Qwen/Qwen3-4B
    python -m evalaware.run --stage probe --model Qwen/Qwen3-8B
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from . import (baselines, compare_blackbox, config, confound, data, extract,
               figures, probe, reliability)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def write(name: str, obj) -> None:
    p = config.ARTIFACTS / name
    p.write_text(json.dumps(obj, indent=2, default=_json_default))
    print(f"[run] wrote {p}")


def stage_extract(corpus, model_id, args):
    return extract.extract_corpus(corpus, model_id, batch_size=args.batch_size,
                                  overwrite=args.overwrite)


def stage_probe(corpus, model_id, args):
    acts = extract.load_acts(model_id)
    ids = extract.load_acts_ids(model_id)
    assert ids == corpus.ids, "activation cache is stale; rerun --stage extract"

    results, oof = probe.layer_sweep(acts, corpus.y)
    probe.save_sweep(results, oof, corpus.ids, model_id)

    bl = probe.best_layer(results)
    write(f"probe_summary_{model_id.split('/')[-1]}.json", {
        "model": model_id,
        "auroc_by_layer": results,
        "best_layer": bl,
        "best_auroc": results[bl],
    })
    return acts, results, oof, bl


def stage_baselines(corpus, model_id, args):
    out = {}
    tfidf = baselines.tfidf_baseline(corpus)
    out["tfidf_auroc"] = float(roc_auc_score(corpus.y, tfidf))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    length_scores, n_tokens = baselines.length_baseline(corpus, tokenizer=tok)
    out["length_only_auroc"] = float(roc_auc_score(corpus.y, length_scores))
    out["median_tokens_eval"] = float(np.median(n_tokens[corpus.y == 1]))
    out["median_tokens_deploy"] = float(np.median(n_tokens[corpus.y == 0]))

    out["human_author1_auroc"] = float(roc_auc_score(
        corpus.y[~np.isnan(corpus.author1)], corpus.author1[~np.isnan(corpus.author1)]))
    out["human_author2_auroc"] = float(roc_auc_score(
        corpus.y[~np.isnan(corpus.author2)], corpus.author2[~np.isnan(corpus.author2)]))

    np.save(config.ARTIFACTS / "tfidf_oof.npy", tfidf)
    np.save(config.ARTIFACTS / "length_oof.npy", length_scores)
    np.save(config.ARTIFACTS / "n_tokens.npy", n_tokens)

    if args.just_ask:
        ja = baselines.just_ask(corpus, model_id, batch_size=args.batch_size)
        out["just_ask_auroc"] = float(roc_auc_score(corpus.y, ja))

    write(f"baselines_{model_id.split('/')[-1]}.json", out)
    return out


def stage_reliability(corpus, model_id, args):
    sweep = json.loads((config.ARTIFACTS /
                        f"probe_sweep_{model_id.split('/')[-1]}.json").read_text())
    bl = str(sweep["best_layer"])
    s = np.asarray(sweep["oof"][bl])
    rep = reliability.full_report(
        corpus.y, s, config.DEPLOY_BASE_RATE, config.TARGET_FPR,
        config.N_BOOTSTRAP, config.SEED)
    rep["layer"] = int(bl)
    write(f"reliability_{model_id.split('/')[-1]}.json", rep)
    return rep


def stage_confound(corpus, model_id, args):
    acts = extract.load_acts(model_id)
    sweep = json.loads((config.ARTIFACTS /
                        f"probe_sweep_{model_id.split('/')[-1]}.json").read_text())
    bl = int(sweep["best_layer"])
    X = acts[bl]
    s = np.asarray(sweep["oof"][str(bl)])

    n_tokens_path = config.ARTIFACTS / "n_tokens.npy"
    n_tokens = (np.load(n_tokens_path) if n_tokens_path.exists()
                else np.array([len(t) for t in corpus.texts]))

    lofo, pooled = confound.leave_one_family_out(X, corpus.y, corpus.family)
    out = {
        "layer": bl,
        "random_cv_auroc": float(roc_auc_score(corpus.y, s)),
        "leave_one_family_out": lofo,
        # Scored on held-out-family predictions, not random-CV ones.
        "mixed_family_auroc_lofo": confound.mixed_family_auroc(pooled, corpus.y, corpus.family),
        "within_stratum": confound.within_stratum(s, corpus.y, corpus.agentic),
        "length_matched": confound.length_matched_auroc(s, corpus.y, n_tokens),
    }
    np.save(config.ARTIFACTS / f"lofo_pooled_{model_id.split('/')[-1]}.npy", pooled)
    # Grouped CV as a second, cheaper leakage check.
    s_grouped = probe.oof_scores(X, corpus.y, groups=corpus.family)
    out["grouped_cv_auroc"] = float(roc_auc_score(corpus.y, s_grouped))
    write(f"confound_{model_id.split('/')[-1]}.json", out)
    return out


def stage_compare(corpus, model_id, args):
    sweep = json.loads((config.ARTIFACTS /
                        f"probe_sweep_{model_id.split('/')[-1]}.json").read_text())
    bl = str(sweep["best_layer"])
    s = np.asarray(sweep["oof"][bl])
    out = compare_blackbox.compare(corpus, s)
    write(f"compare_blackbox_{model_id.split('/')[-1]}.json", out)
    return out


def stage_figures(corpus, model_id, args):
    figures.make_all([config.DEV_MODEL, config.MAIN_MODEL])


STAGES = {
    "extract": stage_extract,
    "baselines": stage_baselines,
    "probe": stage_probe,
    "reliability": stage_reliability,
    "confound": stage_confound,
    "compare": stage_compare,
    "figures": stage_figures,
}
ORDER = ["extract", "probe", "baselines", "reliability", "confound", "compare",
         "figures"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=config.DEV_MODEL)
    ap.add_argument("--stage", default="all",
                    choices=["all"] + list(STAGES))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--just-ask", action="store_true",
                    help="run the black-box just-ask baseline (needs a GPU)")
    args = ap.parse_args()

    corpus = data.load_corpus()
    print(f"[run] corpus n={len(corpus)} pos={corpus.y.sum()} "
          f"families={len(set(corpus.family))}")

    stages = ORDER if args.stage == "all" else [args.stage]
    for name in stages:
        print(f"\n{'=' * 60}\n[run] STAGE {name}\n{'=' * 60}", flush=True)
        STAGES[name](corpus, args.model, args)


if __name__ == "__main__":
    main()
