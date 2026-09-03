#!/bin/bash
# Re-derive every headline number from the raw dataset or raw activations, in one run.
# Nothing here reads a cached result JSON. Writes a timestamped log you can keep.
#
#   source /mnt/scratch/bgxp240/interp/activate.sh
#   ./verify.sh            # all six claims, ~30-40 min on a 16-core node (claim 3 is the slow one)
#   ./verify.sh --quick    # skip claim 3's random-CV refit, ~10 min
#
# Prerequisite: artifacts/acts_Qwen3-4B.npz and acts_Qwen3-8B.npz (the only GPU step).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # threaded BLAS oversubscribes on many small refits
LOG="artifacts/verify_run_$(date +%Y%m%d_%H%M).log"
QUICK=${1:-}
exec > >(tee "$LOG") 2>&1
echo "### verify.sh started $(date)  host $(hostname)  python $(python -V 2>&1)"

step() { echo; echo "================================================================"; echo "## $1"; echo "================================================================"; }

step "TEST SUITE (32 CPU tests; pins fold-internal scaling, the shuffled-label null, the annotator mapping)"
python -m pytest -q -m "not gpu" 2>&1 | tail -2

step "CLAIM 1 — 0.956 is leakage: 0 of 60 categories label-mixed; TF-IDF with no model ~0.936"
python - <<'PY'
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from evalaware import data
c = data.load_corpus(); y = c.y; cat = np.array(c.category)
mixed = [x for x in set(cat) if 0 < y[cat == x].mean() < 1]
print("categories:", len(set(cat)), " label-MIXED categories:", len(mixed))
print("distinct positive rates across categories:", sorted({round(float(y[cat == x].mean()), 3) for x in set(cat)}))
clf = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1,2), sublinear_tf=True, min_df=2),
                    LogisticRegression(max_iter=2000, random_state=0))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
s = cross_val_predict(clf, c.texts, y, cv=cv, method="predict_proba")[:, 1]
print("TF-IDF random-CV AUROC:", round(roc_auc_score(y, s), 4))
PY

step "CLAIM 2 — honest (leave-one-family-out) AUROC ~0.714 / ~0.668 from raw activations"
python - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score
from evalaware import data, extract, confound
c = data.load_corpus()
for m, layer in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
    X = extract.load_acts(m)[layer]
    summ, pooled = confound.leave_one_family_out(X, c.y, c.family)
    ok = ~np.isnan(pooled)
    print(m, "layer", layer, "LOFO pooled AUROC:", round(roc_auc_score(c.y[ok], pooled[ok]), 4),
          "| n scored:", int(ok.sum()), "| families:", summ["n_families"])
PY

if [ "$QUICK" != "--quick" ]; then
step "CLAIM 3 — scaling: leaky rises 4B->8B, honest falls (random-CV refit on 4096-dim activations; slow)"
python - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score
from evalaware import data, extract, probe, confound
c = data.load_corpus()
for m, L in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
    X = extract.load_acts(m)[L]
    leaky = roc_auc_score(c.y, probe.oof_scores(X, c.y))
    _, p = confound.leave_one_family_out(X, c.y, c.family)
    ok = ~np.isnan(p)
    print(f"{m}: leaky={leaky:.4f}  honest={roc_auc_score(c.y[ok], p[ok]):.4f}")
PY
else
step "CLAIM 3 — skipped (--quick). Run without --quick, or: sbatch job_claim3.sh"
fi

step "CLAIM 4 — prompt_injection inversion: probe ~0.30, survives PWNED removal, 9/9 frontier models invert"
python - <<'PY'
import numpy as np, re
from sklearn.metrics import roc_auc_score
from evalaware import data, extract, confound, compare_blackbox
c = data.load_corpus(); fam = np.array(c.family); y = c.y
X = extract.load_acts("Qwen/Qwen3-4B")[34]
_, p = confound.leave_one_family_out(X, y, c.family)
m = fam == "prompt_injection"
print("probe LOFO AUROC in family :", round(roc_auc_score(y[m], p[m]), 3))
pwn = np.array([bool(re.search("PWNED", t, re.I)) for t in c.texts])
keep = m & ~(pwn & (np.array(c.category) == "hackaprompt_dataset"))
print("   with PWNED rows removed :", round(roc_auc_score(y[keep], p[keep]), 3), f"(n={int(keep.sum())})")
print("PWNED in hackaprompt: %.1f%%   in rest of corpus: %.1f%%" % (100*pwn[np.array(c.category)=="hackaprompt_dataset"].mean(), 100*pwn[~m].mean()))
print("human a1:", round(roc_auc_score(y[m], c.author1[m]), 3), " human a2:", round(roc_auc_score(y[m], c.author2[m]), 3),
      " annot corr in-family:", round(np.corrcoef(c.author1[m], c.author2[m])[0,1], 3), " corpus-wide:", round(np.corrcoef(c.author1, c.author2)[0,1], 3))
ids = set(np.array(c.ids)[m])
for name in compare_blackbox.available_models():
    bb = compare_blackbox.blackbox_scores(name); bb = bb[bb.sample_id.isin(ids)]
    print(f"   {name:24s} {roc_auc_score(bb.is_eval, bb.bb_score):.3f}")
PY
echo "(the reading half of claim 4 is not scriptable: python -c \"from evalaware import inversion; inversion.dump_transcripts()\" then read artifacts/inversion_transcripts.txt)"

step "CLAIM 5 — at a 1% base rate: precision ~0.185 (4B), ~0.092 (8B); Bayes check by hand"
python - <<'PY'
import numpy as np
from evalaware import data, extract, confound, reliability
c = data.load_corpus()
for m, L in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
    X = extract.load_acts(m)[L]
    _, p = confound.leave_one_family_out(X, c.y, c.family)
    ok = ~np.isnan(p)
    r = reliability.full_report(c.y[ok], p[ok], 0.01, 0.01, 2000, 0)
    tp = r["tpr_at_fpr"] * 0.01 * 10000; fp = r["achieved_fpr"] * 0.99 * 10000
    print(f"{m}: AUROC {r['auroc']:.4f}  TPR@{r['achieved_fpr']*100:.2f}%FPR {r['tpr_at_fpr']:.4f}  precision@1% {r['precision_at_base_rate']:.4f}  ECE {r['ece']:.4f}  -> {tp:.0f} TP vs {fp:.0f} FP = {fp/tp:.1f} false alarms per catch")
tpr, fpr, base = 0.2048, 0.0091, 0.01
print("Bayes by hand, 4B: precision =", tpr*base / (tpr*base + fpr*(1-base)))
PY

step "CLAIM 6 — aider's 0.948 is a length artefact: length-only AUROC 0.000, disjoint token ranges"
python -c "
import numpy as np
from sklearn.metrics import roc_auc_score
from evalaware import data
c=data.load_corpus(); fam=np.array(c.family); nt=np.load('artifacts/n_tokens.npy')
m=fam=='aider'; y=c.y
print('length-only AUROC within aider:', round(roc_auc_score(y[m], nt[m]),3))
ev,dp = nt[m&(y==1)], nt[m&(y==0)]
print(f'eval tokens [{ev.min()},{ev.max()}]  deploy tokens [{dp.min()},{dp.max()}]')
print('ranges overlap:', bool(ev.max()>dp.min() and dp.max()>ev.min()))"

step "CLAIM 7 — the transplant's unchanged condition reproduces the cached LOFO score (extraction is stable)"
if [ -f artifacts/transplant_acts_Qwen3-4B.npz ]; then
python -m evalaware.transplant score 2>/dev/null | grep -E "original"
else
echo "transplant activations not present; run: sbatch job_transplant.sh"
fi

echo; echo "### verify.sh finished $(date)   log: $LOG"
