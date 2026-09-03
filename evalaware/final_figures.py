"""Step 6: the four publication figures. Captions state the HONEST reading.

Every figure is drawn from cached JSON so this needs no GPU and is rerunnable.
Figures are labelled so a leaky number can never be mistaken for a headline.
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from . import config, data

C_LEAK = "#c53030"      # leaky / not to be quoted
C_HON = "#2b6cb0"       # honest
C_HUM = "#2f855a"
C_GREY = "#a0aec0"
C_BB = "#c05621"


def _j(name):
    p = config.ARTIFACTS / name
    return json.loads(p.read_text()) if p.exists() else None


def _cap(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7.4,
             wrap=True, color="#2d3748")


# ------------------------------------------------------------------ (a)
def fig_a_leaky_vs_honest():
    lm = _j("locked_metrics.json")
    mf = _j("mixed_families.json")
    if not lm:
        return
    models = list(lm["models"])
    labels = ["random CV\n(LEAKY)", "TF-IDF words\n(no model)", "grouped CV",
              "leave-one-\nfamily-out", "smolagents only\n(cleanest test)"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    w = 0.38
    for i, s in enumerate(models):
        r = lm["models"][s]
        smol = (mf.get(s, {}).get("per_family", {}).get("smolagents", {}).get("auroc")
                if mf else None)
        vals = [r["random_cv"]["auroc"],
                lm["baselines_shared"]["tfidf"]["auroc"],
                r["grouped_cv"]["auroc"],
                r["lofo_pooled"]["auroc"],
                smol]
        errs = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        ci = [r["random_cv"]["ci95"], lm["baselines_shared"]["tfidf"]["ci95"],
              r["grouped_cv"]["ci95"], r["lofo_pooled"]["ci95"], None]
        lo = [v - c[0] if c else 0 for v, c in zip(vals, ci)]
        hi = [c[1] - v if c else 0 for v, c in zip(vals, ci)]
        x = np.arange(len(labels)) + i * w
        cols = [C_LEAK, C_GREY, C_HON, C_HON, C_HON]
        ax.bar(x, vals, w * .9, color=cols, alpha=1.0 if i == 0 else .55,
               yerr=[lo, hi], capsize=2.5, ecolor="#4a5568",
               edgecolor="k", linewidth=.4, label=s)
        for xi, v in zip(x, vals):
            if v is not None:
                ax.text(xi, v + .022, f"{v:.3f}", ha="center", fontsize=7.2)
    ax.axhline(.5, c="k", lw=.9, ls=":")
    ax.text(4.55, .515, "chance", fontsize=7, color="#4a5568")
    ax.set_xticks(np.arange(len(labels)) + w / 2)
    ax.set_xticklabels(labels, fontsize=8.4)
    ax.set_ylabel("AUROC"); ax.set_ylim(0, 1.08)
    ax.set_title("Eval-awareness probe: leaky vs honest evaluation", fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    _cap(fig, "HONEST READING: the red bar is an artefact. No source category in this benchmark is label-mixed (0/60), so random CV\n"
              "lets the probe win by recognising the source corpus; a TF-IDF bag of words gets 0.936 with no model at all. Under a\n"
              "held-out-source split the signal falls to 0.67-0.71 at this layer/readout (0.67-0.74 across all layers and both poolings,\n"
              "see fig_b), and to ~0.63 on smolagents - the one family holding scaffold and length fixed while the label varies, though\n"
              "that estimate is itself unstable (0.49-0.78 across configs, n=159). Error bars are bootstrap 95% CIs.")
    fig.savefig(config.FIGURES / "fig_a_leaky_vs_honest.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ (b)
def fig_b_by_layer():
    sens = _j("sensitivity.json")
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    styles = {"Qwen3-4B": "-", "Qwen3-8B": "--"}
    for s in ("Qwen3-4B", "Qwen3-8B"):
        sw = _j(f"probe_sweep_{s}.json")
        if sw:
            d = {int(k): v for k, v in sw["auroc_by_layer"].items()}
            xs = sorted(d)
            ax.plot(xs, [d[x] for x in xs], styles[s], color=C_LEAK, lw=1.6,
                    marker="o", ms=3, label=f"{s} random CV (leaky)")
        if sens and s in sens["models"]:
            pl = sens["models"][s].get("last", {}).get("per_layer", {})
            xs = sorted(int(k) for k in pl)
            ys = [pl[str(x)]["lofo_pooled"] if str(x) in pl
                  else pl[x]["lofo_pooled"] for x in xs]
            ax.plot(xs, ys, styles[s], color=C_HON, lw=1.8, marker="s", ms=4,
                    label=f"{s} leave-one-family-out (honest)")
    lm = _j("locked_metrics.json")
    if lm:
        ax.axhline(lm["baselines_shared"]["tfidf"]["auroc"], color=C_GREY, ls=":",
                   lw=1.3, label=f"TF-IDF words ({lm['baselines_shared']['tfidf']['auroc']:.3f})")
    ax.axhline(.5, c="k", lw=.9, ls=":")
    ax.set_xlabel("layer"); ax.set_ylabel("AUROC"); ax.set_ylim(0.4, 1.0)
    ax.set_title("AUROC by layer: the leaky-honest gap is present at every depth",
                 fontsize=11)
    ax.legend(fontsize=7.6, loc="lower right"); ax.grid(alpha=.25)
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    _cap(fig, "HONEST READING: both models peak in LATE layers, and scaling 4B->8B does not improve the honest number - it lowers it\n"
              "(0.714 -> 0.668 last-token; 0.743 -> 0.730 at each readout's best layer), consistent with fitting corpus identity harder\n"
              "rather than representing eval-awareness. The late-layer peak and non-monotonic scaling independently corroborate Manek\n"
              "(2606.29196) on Qwen-2.5; this method is not novel here and is not claimed to be. The gap between red and blue is source\n"
              "leakage and it does not close at any depth. Mean-pooling (not shown) is both higher and flatter than last-token.")
    fig.savefig(config.FIGURES / "fig_b_by_layer.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ (c)
def fig_c_reliability():
    hr = _j("honest_reliability.json")
    if not hr:
        return
    corpus = data.load_corpus()
    y = corpus.y
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))

    a = axes[0]
    for s, col in (("Qwen3-4B", C_HON), ("Qwen3-8B", "#63b3ed")):
        p = config.ARTIFACTS / f"lofo_pooled_{s}.npy"
        if not p.exists():
            continue
        sc = np.load(p)
        m = ~np.isnan(sc)
        fpr, tpr, _ = roc_curve(y[m], sc[m])
        a.plot(fpr, tpr, color=col, lw=1.7,
               label=f"{s} honest ({hr['models'][s]['auroc']:.3f})")
        sw = _j(f"probe_sweep_{s}.json")
        if sw:
            lk = np.asarray(sw["oof"][str(sw["best_layer"])])
            f2, t2, _ = roc_curve(y, lk)
            a.plot(f2, t2, color=C_LEAK, lw=1.1, ls="--", alpha=.75,
                   label=f"{s} leaky ({hr['models'][s]['leaky_comparison']['auroc']:.3f})")
    a.axvline(0.01, color="k", ls=":", lw=1)
    a.text(.014, .06, "1% FPR budget", fontsize=7)
    a.plot([0, 1], [0, 1], "k:", lw=.8)
    a.set_xlabel("false positive rate"); a.set_ylabel("true positive rate")
    a.set_title("ROC", fontsize=10); a.legend(fontsize=7); a.grid(alpha=.25)

    a = axes[1]
    brs = [0.01, 0.05, 0.10]
    w = 0.35
    for i, s in enumerate(hr["models"]):
        v = [hr["models"][s]["precision_by_base_rate"][str(b)] for b in brs]
        a.bar(np.arange(len(brs)) + i * w, v, w * .9,
              color=C_HON, alpha=1 - .45 * i, edgecolor="k", lw=.4, label=s)
        for xi, vv in zip(np.arange(len(brs)) + i * w, v):
            a.text(xi, vv + .012, f"{vv:.2f}", ha="center", fontsize=7)
    a.set_xticks(np.arange(len(brs)) + w / 2)
    a.set_xticklabels([f"{int(b*100)}%" for b in brs])
    a.set_xlabel("eval base rate in the wild"); a.set_ylabel("precision at 1% FPR")
    a.set_title("Precision vs base rate (honest scores)", fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=.25, axis="y"); a.set_ylim(0, 1)

    a = axes[2]
    a.plot([0, 1], [0, 1], "k:", lw=.9, label="perfect calibration")
    for s, col in (("Qwen3-4B", C_HON), ("Qwen3-8B", "#63b3ed")):
        r = hr["models"].get(s)
        if not r:
            continue
        bins = [b for b in r["calibration_bins"] if b[2] > 0]
        a.plot([(b[0] + b[1]) / 2 for b in bins], [b[4] for b in bins],
               marker="o", ms=4, color=col, label=f"{s} (ECE {r['ece']:.3f})")
    a.set_xlabel("predicted P(eval)"); a.set_ylabel("observed frequency")
    a.set_title("Calibration (honest scores)", fontsize=10)
    a.legend(fontsize=7.5); a.grid(alpha=.25)

    fig.tight_layout(rect=[0, 0.13, 1, 1])
    m4 = hr["models"]["Qwen3-4B"]["per_10k_transcripts_at_1pct"]
    _cap(fig, "HONEST READING: at a realistic 1% eval base rate the probe is not a usable monitor. Using held-out-source predictions, "
              f"Qwen3-4B catches only {hr['models']['Qwen3-4B']['tpr_at_fpr']*100:.0f}% of evals at a 1% FPR budget and precision is "
              f"{hr['models']['Qwen3-4B']['precision_at_base_rate']:.2f} - per 10,000 transcripts that is {m4['true_positives']:.0f} true catches against "
              f"{m4['false_positives']:.0f} false alarms\n({m4['false_alarms_per_true_catch']:.1f} false alarms per true catch). Qwen3-8B is worse. ECE ~0.28-0.31 means the scores are badly "
              "calibrated as well as weak. The leaky curve (dashed) is what a random-CV evaluation would have promised.")
    fig.savefig(config.FIGURES / "fig_c_reliability.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ (d)
def fig_d_vs_blackbox(model="Qwen3-4B"):
    cb = _j(f"compare_blackbox_{model}.json")
    lm = _j("locked_metrics.json")
    hr = _j("honest_reliability.json")
    if not (cb and lm):
        return
    rows = sorted([(k, v) for k, v in cb.items() if "blackbox_auroc" in v],
                  key=lambda kv: kv[1]["blackbox_auroc"])
    names = [k for k, _ in rows]
    bb = [v["blackbox_auroc"] for _, v in rows]
    n = rows[0][1]["n"]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(names, bb, color=C_BB, alpha=.85, edgecolor="k", lw=.4,
            label=f"black-box frontier models, prompted (n={n} shared rows)")
    for i, v in enumerate(bb):
        ax.text(v + .004, i, f"{v:.3f}", va="center", fontsize=7.2)
    hu = rows[0][1]["human_auroc"]
    ax.axvline(hu, color=C_HUM, lw=1.8, ls=":", label=f"human annotators ({hu:.3f})")
    wb = lm["models"][model]["random_cv"]["auroc"]
    ax.axvline(wb, color=C_LEAK, lw=2,
               label=f"white-box probe, random CV - LEAKY ({wb:.3f})")
    ho = lm["models"][model]["lofo_pooled"]["auroc"]
    ax.axvline(ho, color=C_HON, lw=2.4,
               label=f"white-box probe, held-out source - HONEST ({ho:.3f})")
    ax.set_xlim(0.45, 1.0); ax.set_xlabel("AUROC")
    ax.set_title(f"White-box probe ({model}) vs black-box prompting of frontier models",
                 fontsize=11)
    ax.grid(alpha=.25, axis="x")
    # Legend outside the axes: inside, it covers the lowest-scoring model bars.
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, fontsize=7.4, ncol=2, loc="lower center",
               bbox_to_anchor=(0.5, 0.145), frameon=False)
    fig.tight_layout(rect=[0, 0.235, 1, 1])
    _cap(fig, "HONEST READING: the comparison is NOT like-for-like and the probe should not be claimed to beat these models. The black-box\n"
              "numbers are frontier models prompted zero-shot on transcripts they never trained on; the white-box probe is a supervised\n"
              "classifier fit on this very corpus. Compared on the only common footing - generalising to an unseen source - the probe\n"
              f"({ho:.3f}) lands BELOW every frontier model shown and far below humans ({hu:.3f}). The red line is the leaky number.")
    fig.savefig(config.FIGURES / f"fig_d_vs_blackbox_{model}.png", dpi=200)
    plt.close(fig)


def make_all():
    fig_a_leaky_vs_honest()
    fig_b_by_layer()
    fig_c_reliability()
    fig_d_vs_blackbox()
    for p in sorted(config.FIGURES.glob("fig_*.png")):
        print(f"[final] {p}")
