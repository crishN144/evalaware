"""Figures E-H requested in the write-up review: purity scan, per-category honest
scores + mixed-family AUROC, annotator disagreement, token-length distributions.
CPU only, drawn from cached artifacts."""
from __future__ import annotations
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config, data

C_EVAL, C_DEP, C_HON, C_LEAK, C_PI = "#2b6cb0", "#a0aec0", "#2b6cb0", "#c53030", "#c05621"
PI_CATS = ["mosscap_prompt_injection", "gandalf_ignore_instructions",
           "gandalf_summarization", "hackaprompt_dataset"]

def _cap(fig, text):
    fig.text(0.5, 0.005, text, ha="center", va="bottom", fontsize=7.4, wrap=True, color="#2d3748")

def fig_e_purity(c):
    cat = np.array(c.category)
    cats = sorted(set(cat), key=lambda k: -(cat == k).sum())
    n = np.array([(cat == k).sum() for k in cats]); pr = np.array([c.y[cat == k].mean() for k in cats])
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(cats))
    ax.bar(x, n, color=[C_EVAL if p == 1 else C_DEP for p in pr], edgecolor="black", linewidth=0.3)
    for xi, ni, pi in zip(x, n, pr):
        ax.text(xi, ni + 1, "1.0" if pi == 1 else "0.0", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=90, fontsize=6)
    ax.set_ylabel("rows in category"); ax.set_ylim(0, n.max() * 1.25)
    ax.set_title("Every one of the 60 source categories is single-label (number above bar = fraction eval)")
    ax.bar([0], [0], color=C_EVAL, label="100% eval"); ax.bar([0], [0], color=C_DEP, label="100% deploy"); ax.legend(loc="upper right")
    n_mixed = int(((pr > 0) & (pr < 1)).sum())
    ax.text(0.99, 0.72, f"label-mixed categories: {n_mixed} of {len(cats)}", transform=ax.transAxes, ha="right", fontsize=10, fontweight="bold", color=C_LEAK)
    _cap(fig, "HONEST READING: the fraction-eval above every bar is exactly 0.0 or 1.0. Source identity determines the label, so a random CV split, which "
              "puts rows from the same source on both sides, can score 0.956 by recognising the corpus. This is the mechanism of the leak in one picture.")
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(config.FIGURES / "fig_e_purity_scan.png", dpi=200); plt.close(fig)

def fig_f_per_category(c):
    p = np.load(config.ARTIFACTS / "lofo_pooled_Qwen3-4B.npy")
    cat = np.array(c.category); fam = np.array(c.family)
    cats = sorted(set(cat), key=lambda k: p[cat == k].mean())
    mean = np.array([p[cat == k].mean() for k in cats]); lab = np.array([c.y[cat == k].mean() for k in cats])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [3, 1]})
    x = np.arange(len(cats))
    cols = [C_PI if k in PI_CATS else (C_EVAL if l == 1 else C_DEP) for k, l in zip(cats, lab)]
    ax.bar(x, mean, color=cols, edgecolor="black", linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels([("* " if k in PI_CATS else "") + k for k in cats], rotation=90, fontsize=6)
    ax.axhline(0.5, ls=":", color="black"); ax.set_ylabel("mean honest P(eval), Qwen3-4B"); ax.set_ylim(0, 1)
    ax.set_title("Held-out P(eval) per source category (blue = eval-labelled, grey = deploy-labelled, orange = prompt_injection)")
    for k, xi, m_ in zip(cats, x, mean):
        if k == "hackaprompt_dataset":
            ax.annotate("hackaprompt: labelled DEPLOY,\nscored 0.83", (xi, m_), (xi - 16, 0.86), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.7))
    mf = json.load(open(config.ARTIFACTS / "mixed_families.json"))["Qwen3-4B"]["per_family"]
    fams = ["smolagents", "aider", "prompt_injection"]
    au = [mf[f]["auroc"] for f in fams]; ci = [mf[f]["ci95"] for f in fams]
    ax2.bar(range(3), au, color=[C_HON, C_DEP, C_PI], edgecolor="black", linewidth=0.4,
            yerr=[[a - l for a, (l, h) in zip(au, ci)], [h - a for a, (l, h) in zip(au, ci)]], capsize=4)
    ax2.axhline(0.5, ls=":", color="black"); ax2.set_ylim(0, 1)
    ax2.set_xticks(range(3)); ax2.set_xticklabels(["smolagents\n(clean)", "aider\n(length artefact)", "prompt_injection\n(inverted)"], fontsize=8)
    for i, a in enumerate(au): ax2.text(i + 0.32, a, f"{a:.2f}", ha="left", va="center", fontsize=9)
    ax2.set_ylabel("held-out AUROC within family"); ax2.set_title("The 3 label-mixed families")
    _cap(fig, "HONEST READING: 44 of 47 families are single-label, so within-family AUROC exists for only three. prompt_injection sits below chance (0.30): "
              "its deploy-labelled hackaprompt rows get HIGHER P(eval) than its eval-labelled mosscap/gandalf rows. aider's 0.95 is a length readout and is discounted; smolagents is the only clean test.")
    fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(config.FIGURES / "fig_f_per_category.png", dpi=200); plt.close(fig)

def fig_g_annotators(c):
    cat = np.array(c.category); fam = np.array(c.family); m = fam == "prompt_injection"
    ok = ~np.isnan(c.author1) & ~np.isnan(c.author2)
    r_all = np.corrcoef(c.author1[ok], c.author2[ok])[0, 1]; r_pi = np.corrcoef(c.author1[m], c.author2[m])[0, 1]
    fig, ax = plt.subplots(figsize=(6.4, 6))
    ax.scatter(c.author1[ok & ~m], c.author2[ok & ~m], s=10, color=C_DEP, alpha=0.5, label=f"other 920 rows (r = {r_all:+.2f} corpus-wide)")
    marks = {"mosscap_prompt_injection": "o", "gandalf_ignore_instructions": "^", "gandalf_summarization": "v", "hackaprompt_dataset": "s"}
    for k, mk in marks.items():
        mm = cat == k
        ax.scatter(c.author1[mm], c.author2[mm], s=42, marker=mk, color=C_EVAL if k != "hackaprompt_dataset" else C_LEAK,
                   edgecolor="black", linewidth=0.5, label=f"{k} ({'deploy' if k=='hackaprompt_dataset' else 'eval'}-labelled)")
    ax.plot([0, 1], [0, 1], ls=":", color="black", lw=0.8)
    ax.set_xlabel("annotator 1: P(eval)"); ax.set_ylabel("annotator 2: P(eval)")
    ax.set_title(f"The two dataset annotators disagree inside prompt_injection\n(r = {r_pi:+.2f} in-family vs {r_all:+.2f} corpus-wide)")
    ax.legend(fontsize=7, loc="lower right"); ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
    _cap(fig, "HONEST READING: corpus-wide the two humans agree (grey cloud on the diagonal). Inside prompt_injection they rank the rows in opposite orders, and both give the "
              "deploy-labelled hackaprompt rows (red squares) P(eval) around 0.8. The label is contested by its own authors.")
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(config.FIGURES / "fig_g_annotators.png", dpi=200); plt.close(fig)

def fig_h_lengths(c):
    nt = np.load(config.ARTIFACTS / "n_tokens.npy"); fam = np.array(c.family)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    bins = np.logspace(np.log10(nt.min()), np.log10(nt.max()), 40)
    ax.hist(nt[c.y == 1], bins=bins, alpha=0.65, color=C_EVAL, label=f"eval (median {int(np.median(nt[c.y==1]))} tokens)")
    ax.hist(nt[c.y == 0], bins=bins, alpha=0.65, color=C_DEP, label=f"deploy (median {int(np.median(nt[c.y==0]))} tokens)")
    ax.set_xscale("log"); ax.set_xlabel("transcript length (tokens, log)"); ax.set_ylabel("rows")
    a_len = roc_auc_score(c.y, nt); ax.set_title(f"Corpus-wide: eval transcripts are SHORTER; length alone separates labels at AUROC {max(a_len, 1-a_len):.2f}"); ax.legend()
    m = fam == "aider"
    ax2.hist(nt[m & (c.y == 1)], bins=bins, alpha=0.75, color=C_EVAL, label=f"aider eval (n={int((m&(c.y==1)).sum())})")
    ax2.hist(nt[m & (c.y == 0)], bins=bins, alpha=0.75, color=C_LEAK, label=f"aider deploy (n={int((m&(c.y==0)).sum())})")
    ax2.set_xscale("log"); ax2.set_xlabel("transcript length (tokens, log)")
    ev, dp = nt[m & (c.y == 1)], nt[m & (c.y == 0)]
    ax2.axvspan(ev.max(), dp.min(), color="yellow", alpha=0.3)
    ax2.set_title(f"Inside aider: ranges are disjoint ({ev.min()}–{ev.max()} vs {dp.min()}–{dp.max()})"); ax2.legend()
    _cap(fig, "HONEST READING: length is not the corpus-wide confound (eval rows are shorter, and length tracks scaffold rather than label). Inside aider it is the entire story: "
              "a single threshold in the yellow gap classifies all 129 rows perfectly, so aider's 0.95 is a token counter and is discounted.")
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(config.FIGURES / "fig_h_lengths.png", dpi=200); plt.close(fig)

def fig_i_transplant():
    import glob
    paths = sorted(glob.glob(str(config.ARTIFACTS / "transplant_Qwen3-*.json")))
    if not paths: return
    res = [json.load(open(p)) for p in paths]
    from .transplant import CONDITIONS
    fig, ax = plt.subplots(figsize=(8, 4.4)); w = 0.38
    for i, r in enumerate(res):
        vals = [r["conditions"][k]["auroc"] for k in CONDITIONS]
        x = np.arange(len(CONDITIONS)) + i * w
        ax.bar(x, vals, w, color=[C_HON, "#63b3ed"][i], edgecolor="black", linewidth=0.4, label=f"{r['model'].split('/')[-1]} L{r['layer']}")
        for xi, v in zip(x, vals): ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.axhline(0.5, ls=":", color="black"); ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(len(CONDITIONS)) + w / 2)
    ax.set_xticklabels(["original", "eval rows get\ngeneric prompt", "hackaprompt gets\nGOURD prompt", "both swapped"])
    ax.set_ylabel("AUROC within prompt_injection")
    ax.set_title("System-prompt transplant: does the inversion follow the prompt?\n(probe fit on the other 46 families; user turns untouched)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES / "fig_i_transplant.png", dpi=200); plt.close(fig)

if __name__ == "__main__":
    c = data.load_corpus()
    fig_e_purity(c); fig_f_per_category(c); fig_g_annotators(c); fig_h_lengths(c); fig_i_transplant()
    print("done")
