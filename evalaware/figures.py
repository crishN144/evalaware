"""Plots for the write-up. Reads artifacts/, writes figures/.

Every figure is built from cached JSON, so this needs no GPU and can be rerun
after any stage without recomputing anything.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from . import config

PALETTE = {"probe": "#2b6cb0", "tfidf": "#a0aec0", "length": "#cbd5e0",
           "human": "#2f855a", "just_ask": "#c05621", "bad": "#c53030"}


def _slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def _load(name: str):
    p = config.ARTIFACTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fig_layer_sweep(models: list[str]):
    """AUROC vs depth, with the surface baseline drawn as the real bar to clear."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mid in models:
        s = _load(f"probe_sweep_{_slug(mid)}.json")
        if not s:
            continue
        d = {int(k): v for k, v in s["auroc_by_layer"].items()}
        xs = sorted(d)
        ax.plot(xs, [d[x] for x in xs], marker="o", ms=4, label=_slug(mid))
    b = _load(f"baselines_{_slug(models[0])}.json") or {}
    if "tfidf_auroc" in b:
        ax.axhline(b["tfidf_auroc"], ls="--", c=PALETTE["tfidf"],
                   label=f"TF-IDF surface ({b['tfidf_auroc']:.3f})")
    if "human_author1_auroc" in b:
        ax.axhline(b["human_author1_auroc"], ls=":", c=PALETTE["human"],
                   label=f"human annotator ({b['human_author1_auroc']:.3f})")
    ax.set_xlabel("layer"); ax.set_ylabel("AUROC (random CV)")
    ax.set_title("Eval-awareness probe by depth")
    ax.set_ylim(0.5, 1.0); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(config.FIGURES / "layer_sweep.png", dpi=160)
    plt.close(fig)


def fig_generalisation(models: list[str]):
    """The headline honesty plot: random CV vs every leakage-controlled number."""
    keys = [("random_cv_auroc", "random CV"),
            ("grouped_cv_auroc", "grouped CV"),
            ("__lofo", "leave-one-family-out\n(pooled)"),
            ("__mixed", "label-mixed families\n(held-out)"),
            ("__len", "length-matched")]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    w = 0.8 / len(models)
    for i, mid in enumerate(models):
        c = _load(f"confound_{_slug(mid)}.json")
        if not c:
            continue
        vals = []
        for k, _ in keys:
            if k == "__lofo":
                vals.append(c["leave_one_family_out"]["pooled_auroc"])
            elif k == "__mixed":
                vals.append(c["mixed_family_auroc_lofo"]["_macro"])
            elif k == "__len":
                vals.append(c["length_matched"]["auroc"])
            else:
                vals.append(c[k])
        x = np.arange(len(keys)) + i * w
        cols = [PALETTE["bad"] if j == 0 else PALETTE["probe"]
                for j in range(len(keys))]
        ax.bar(x, vals, w * .9, color=cols, alpha=.85 if i == 0 else .55,
               label=_slug(mid))
        for xi, v in zip(x, vals):
            ax.text(xi, v + .01, f"{v:.3f}", ha="center", fontsize=7)
    ax.axhline(0.5, c="k", lw=.8, ls=":")
    ax.set_xticks(np.arange(len(keys)) + w * (len(models) - 1) / 2)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=8)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.05)
    ax.set_title("Random CV overstates the probe; red bar is the leaky number")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    fig.tight_layout(); fig.savefig(config.FIGURES / "generalisation.png", dpi=160)
    plt.close(fig)


def fig_roc_and_calibration(model_id: str):
    s = _load(f"probe_sweep_{_slug(model_id)}.json")
    r = _load(f"reliability_{_slug(model_id)}.json")
    if not (s and r):
        return
    from . import data
    corpus = data.load_corpus()
    y = corpus.y
    p = np.asarray(s["oof"][str(s["best_layer"])])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    a1.plot(fpr, tpr, c=PALETTE["probe"], label=f"probe (AUC {r['auroc']:.3f})")
    for nm, arr, col in [("TF-IDF", "tfidf_oof.npy", PALETTE["tfidf"]),
                         ("length only", "length_oof.npy", PALETTE["length"])]:
        f = config.ARTIFACTS / arr
        if f.exists():
            fx, tx, _ = roc_curve(y, np.load(f))
            a1.plot(fx, tx, c=col, lw=1.2, ls="--", label=nm)
    a1.plot([0, 1], [0, 1], "k:", lw=.8)
    a1.axvline(r["target_fpr"], c=PALETTE["bad"], lw=.8, ls=":")
    a1.set_xlabel("FPR"); a1.set_ylabel("TPR")
    a1.set_title(f"ROC - {_slug(model_id)} layer {r['layer']}")
    a1.legend(fontsize=8); a1.grid(alpha=.3)

    bins = [b for b in r["calibration_bins"] if b[2] > 0]
    xs = [(b[0] + b[1]) / 2 for b in bins]
    a2.plot([0, 1], [0, 1], "k:", lw=.8, label="perfect")
    a2.plot(xs, [b[4] for b in bins], marker="o", c=PALETTE["probe"],
            label=f"probe (ECE {r['ece']:.3f})")
    a2.set_xlabel("predicted P(eval)"); a2.set_ylabel("observed frequency")
    a2.set_title("Calibration"); a2.legend(fontsize=8); a2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"roc_calibration_{_slug(model_id)}.png", dpi=160)
    plt.close(fig)


def fig_blackbox_compare(model_id: str):
    cb = _load(f"compare_blackbox_{_slug(model_id)}.json")
    if not cb:
        return
    rows = [(k, v) for k, v in cb.items() if "blackbox_auroc" in v]
    rows.sort(key=lambda kv: kv[1]["blackbox_auroc"])
    names = [k for k, _ in rows]
    bb = [v["blackbox_auroc"] for _, v in rows]
    wb = rows[0][1]["whitebox_auroc"]
    hu = rows[0][1]["human_auroc"]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(names, bb, color=PALETTE["just_ask"], alpha=.8,
            label="black-box MCQ probe (paper)")
    ax.axvline(wb, c=PALETTE["probe"], lw=2,
               label=f"white-box probe, random CV ({wb:.3f})")
    ax.axvline(hu, c=PALETTE["human"], lw=1.5, ls=":",
               label=f"human ({hu:.3f})")
    c = _load(f"confound_{_slug(model_id)}.json")
    if c:
        lofo = c["leave_one_family_out"]["pooled_auroc"]
        ax.axvline(lofo, c=PALETTE["bad"], lw=2, ls="--",
                   label=f"white-box, held-out family ({lofo:.3f})")
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("AUROC")
    ax.set_title(f"White-box probe vs black-box asking ({_slug(model_id)})")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(config.FIGURES / f"blackbox_compare_{_slug(model_id)}.png", dpi=160)
    plt.close(fig)


def make_all(models: list[str]):
    have = [m for m in models
            if (config.ARTIFACTS / f"probe_sweep_{_slug(m)}.json").exists()]
    if not have:
        print("[figures] nothing to plot")
        return
    fig_layer_sweep(have)
    fig_generalisation(have)
    for m in have:
        fig_roc_and_calibration(m)
        fig_blackbox_compare(m)
    for p in sorted(config.FIGURES.glob("*.png")):
        print(f"[figures] {p}")
