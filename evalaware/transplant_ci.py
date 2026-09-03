"""Bootstrap 95% CIs for the transplant AUROCs (presentation support for pub_i).

Recomputes per-row scores with the same held-out fit transplant.score() uses,
asserts the point AUROCs equal the ones already in transplant_<model>.json, and
writes CIs to transplant_ci.json. The original JSON is not modified.
"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from . import config, data, extract, inversion
from .probe import make_probe
from .reliability import bootstrap_auroc
from .transplant import CONDITIONS, acts_path

def run():
    c = data.load_corpus(); fam = np.array(c.family); out = {}
    for m, L in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
        slug = extract.model_slug(m)
        X = extract.load_acts(m)[L]; tr = fam != inversion.FAMILY
        clf = make_probe().fit(X[tr], c.y[tr])
        z = np.load(acts_path(m), allow_pickle=False); y = z["y"]
        ref = json.load(open(config.ARTIFACTS / f"transplant_{slug}.json"))
        out[slug] = {}
        for k in CONDITIONS:
            p = clf.predict_proba(z[f"{k}__layer_{L}"])[:, 1]
            a = float(roc_auc_score(y, p))
            assert abs(a - ref["conditions"][k]["auroc"]) < 1e-6, (slug, k, a, ref["conditions"][k]["auroc"])
            _, lo, hi = bootstrap_auroc(y, p, config.N_BOOTSTRAP, config.SEED); ci = (lo, hi)
            out[slug][k] = {"auroc": a, "ci95": [float(ci[0]), float(ci[1])]}
            print(f"[ci] {slug} {k:12s} {a:.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]", flush=True)
    (config.ARTIFACTS / "transplant_ci.json").write_text(json.dumps(out, indent=2))

if __name__ == "__main__":
    run()
