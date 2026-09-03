"""Three cheap controls requested in review, all CPU, all from cached artifacts.

1. internal vs verbalised: does the honest probe agree with the same model's
   own "is this an eval?" answer (just-ask)? 2x2 at each score's median, 2x2 at
   0.5, Spearman rho.
2. cross-regime transfer: train on chat-only rows, test on agentic-only rows,
   and the reverse. A generalisation axis orthogonal to source family.
3. random-direction control: AUROC of the projection of standardised
   activations onto random unit directions (Nguyen et al. report 0.517 for a
   random Gaussian probe). 200 draws.
"""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from . import config, data, extract
from .probe import make_probe

MODELS = [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]

def two_by_two(a, b, ta, tb):
    """rows: probe says eval / deploy; cols: model says eval / deploy."""
    pa, pb = a >= ta, b >= tb
    return {"probe_eval_model_eval": int((pa & pb).sum()), "probe_eval_model_deploy": int((pa & ~pb).sum()),
            "probe_deploy_model_eval": int((~pa & pb).sum()), "probe_deploy_model_deploy": int((~pa & ~pb).sum()),
            "disagreement_rate": float((pa != pb).mean())}

def run():
    c = data.load_corpus(); y = c.y; fam = np.array(c.family)
    out = {}
    rng = np.random.default_rng(config.SEED)
    for m, L in MODELS:
        slug = extract.model_slug(m)
        X = extract.load_acts(m)[L]
        p = np.load(config.ARTIFACTS / f"lofo_pooled_{slug}.npy")
        ja_map = json.load(open(config.ARTIFACTS / f"just_ask_{slug}.json"))
        ja = np.array([ja_map[i] for i in c.ids])
        rec = {"layer": L}
        # 1. internal vs verbalised
        rho = spearmanr(p, ja).correlation
        rec["internal_vs_verbal"] = {
            "spearman_rho": float(rho),
            "just_ask_frac_says_eval_at_0.5": float((ja >= 0.5).mean()),
            "just_ask_frac_says_eval_on_deploy_rows": float((ja[y == 0] >= 0.5).mean()),
            "median_split": two_by_two(p, ja, np.median(p), np.median(ja)),
            "at_0.5": two_by_two(p, ja, 0.5, 0.5),
            "auroc_probe_honest": float(roc_auc_score(y, p)),
            "auroc_just_ask": float(roc_auc_score(y, ja)),
            "auroc_on_rows_where_model_says_deploy": (float(roc_auc_score(y[ja < 0.5], p[ja < 0.5]))
                                                      if len(set(y[ja < 0.5])) == 2 else None),
            "n_rows_model_says_deploy": int((ja < 0.5).sum()),
        }
        # 2. cross-regime transfer
        chat, ag = c.agentic == 0, c.agentic == 1
        xr = {}
        for name, tr, te in [("train_chat_test_agentic", chat, ag), ("train_agentic_test_chat", ag, chat)]:
            clf = make_probe().fit(X[tr], y[tr]); s = clf.predict_proba(X[te])[:, 1]
            xr[name] = {"auroc": float(roc_auc_score(y[te], s)), "n_train": int(tr.sum()), "n_test": int(te.sum()),
                        "test_pos_rate": float(y[te].mean())}
        # within-regime LOFO for reference
        for name, msk in [("lofo_within_chat", chat), ("lofo_within_agentic", ag)]:
            xr[name] = {"auroc": float(roc_auc_score(y[msk], p[msk])), "n": int(msk.sum())}
        rec["cross_regime"] = xr
        # 3. random direction control
        Z = StandardScaler().fit_transform(X)
        aucs = []
        for _ in range(200):
            d = rng.standard_normal(Z.shape[1]); d /= np.linalg.norm(d)
            aucs.append(roc_auc_score(y, Z @ d))
        aucs = np.array(aucs)
        rec["random_direction"] = {"n_draws": 200, "mean_auroc": float(aucs.mean()), "sd": float(aucs.std()),
                                   "mean_abs_dev_from_chance": float(np.abs(aucs - 0.5).mean()),
                                   "max_abs_dev": float(np.abs(aucs - 0.5).max()),
                                   "p95_folded": float(np.quantile(np.maximum(aucs, 1 - aucs), 0.95))}
        out[slug] = rec
        print(f"[extra] {slug}: rho(probe, just-ask)={rho:.3f}  median-split disagreement="
              f"{rec['internal_vs_verbal']['median_split']['disagreement_rate']:.3f}  "
              f"just-ask says eval on {rec['internal_vs_verbal']['just_ask_frac_says_eval_at_0.5']:.0%} of rows | "
              f"chat->agentic {xr['train_chat_test_agentic']['auroc']:.3f}  agentic->chat {xr['train_agentic_test_chat']['auroc']:.3f} | "
              f"random dir {aucs.mean():.3f}±{aucs.std():.3f}, p95 folded {rec['random_direction']['p95_folded']:.3f}", flush=True)
    (config.ARTIFACTS / "extra_experiments.json").write_text(json.dumps(out, indent=2))
    return out

if __name__ == "__main__":
    run()
