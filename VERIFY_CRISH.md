# VERIFY_CRISH.md

A checklist **you** run to independently verify the headline claims. Every command
recomputes its number **from the raw dataset or the raw activations** — none of
them read a cached result JSON. Fill in each `VERDICT:` yourself.

Setup, once per shell:

    source /mnt/scratch/bgxp240/interp/activate.sh
    cd /mnt/scratch/bgxp240/interp/evalaware

Prerequisite: `artifacts/acts_Qwen3-4B.npz` and `acts_Qwen3-8B.npz` must exist.
If not, they are the only GPU step: `sbatch job_dev.sh` (4B), `sbatch job_main.sh` (8B).

**Runtimes.** These recompute rather than read cached JSON, so they are not instant.
Claims 1, 4, 6 take under ~2 min. Claims 2 and 5 take ~4-6 min each (47 leave-one-out
refits per model). Claim 3 is the slowest at ~10-15 min because it also refits the
5-fold random CV on 4096-dim activations. If you want them off your terminal, put the
block in a file and `sbatch` it with `--partition=nodes --cpus-per-task=16`, and set
`export OMP_NUM_THREADS=1` first - threaded BLAS oversubscribes on these many small
refits and runs *slower* (this cost me a stalled job; see FINDINGS §9 notes).

**Tolerance — do not chase the third decimal.** These numbers are not bit-exact
across runs: lbfgs convergence in `LogisticRegression` depends on BLAS thread count,
so the same command under `OMP_NUM_THREADS=1` vs the default can differ by ~0.001.
Verified example: claim 3 recomputed on a 16-core node gave leaky 0.9565 / honest
0.7132 against the locked 0.9563 / 0.7139. **Agreement to ~0.002 is a PASS.**
A discrepancy worth acting on is one that changes a conclusion (e.g. honest > 0.8,
or 8B's honest above 4B's).

Before anything else, confirm the test suite is green — several claims below rest
on invariants these tests pin (fold-internal scaling, the shuffled-label null,
the annotator column mapping):

    python -m pytest -q -m "not gpu"        # expect: 32 passed

---

## CLAIM 1 — The 0.956 headline is leakage, not signal

**Claim:** random-CV AUROC 0.956 is inflated because no source category is
label-mixed (0 of 60); a TF-IDF bag of words with no model reaches 0.936.

**Recompute from the raw dataset (does not touch any probe or cache):**

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
    clf = make_pipeline(TfidfVectorizer(max_features=50000, ngram_range=(1,2),
                                        sublinear_tf=True, min_df=2),
                        LogisticRegression(max_iter=2000, random_state=0))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    s = cross_val_predict(clf, c.texts, y, cv=cv, method="predict_proba")[:, 1]
    print("TF-IDF random-CV AUROC:", round(roc_auc_score(y, s), 4))
    PY

**Expect:** `categories: 60  label-MIXED categories: 0` and TF-IDF AUROC ≈ **0.936**.

**Read with your own eyes:** run the snippet below and confirm every category is
100% one label — the mechanism of the leak, in one screen.

    python -c "
    import numpy as np; from evalaware import data
    c=data.load_corpus(); cat=np.array(c.category)
    r=sorted({x:(int((cat==x).sum()), float(c.y[cat==x].mean())) for x in set(cat)}.items(),
             key=lambda kv:-kv[1][0])
    [print(f'{k:32s} n={v[0]:3d} pos_rate={v[1]:.2f}') for k,v in r[:20]]"

Every `pos_rate` must be exactly 0.00 or 1.00.

` VERDICT: ____`

---

## CLAIM 2 — The honest number is ~0.67–0.74

**Claim:** under leave-one-family-out, AUROC falls to 0.714 (4B) / 0.668 (8B) at
the reported layer, and 0.67–0.74 across all layers and both poolings.

**Recompute the LOFO number from the raw activations:**

    python - <<'PY'
    import numpy as np, json
    from sklearn.metrics import roc_auc_score
    from evalaware import data, extract, confound
    c = data.load_corpus()
    for m, layer in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
        X = extract.load_acts(m)[layer]
        summ, pooled = confound.leave_one_family_out(X, c.y, c.family)
        ok = ~np.isnan(pooled)
        print(m, "layer", layer,
              "LOFO pooled AUROC:", round(roc_auc_score(c.y[ok], pooled[ok]), 4),
              "| n scored:", int(ok.sum()), "| families:", summ["n_families"])
    PY

**Expect:** 4B ≈ **0.714**, 8B ≈ **0.668**, n scored = 976, families = 47.

**Sanity check that the split is doing real work** — grouped CV should land in the
same region, and random CV should not:

    python -c "
    import numpy as np
    from sklearn.metrics import roc_auc_score
    from evalaware import data, extract, probe
    c=data.load_corpus(); X=extract.load_acts('Qwen/Qwen3-4B')[34]
    print('random  CV:', round(roc_auc_score(c.y, probe.oof_scores(X,c.y)),4))
    print('grouped CV:', round(roc_auc_score(c.y, probe.oof_scores(X,c.y,groups=c.family)),4))"

**Expect:** random ≈ 0.956, grouped ≈ 0.689. The ~0.27 gap **is** the leak.

**Read with your own eyes:** open `FINDINGS.md` §6 and confirm the honest number
never exceeds 0.743 in any of the 28 layer/pooling configurations.

` VERDICT: ____`

---

## CLAIM 3 — Scaling 4B → 8B makes the honest number worse

**Claim:** the leaky metric says 8B is better (0.956→0.960); the honest metric says
8B is worse (0.714→0.668, and 0.743→0.730 at each readout's best layer).

**Recompute both metrics for both models side by side:**

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
        print(f"{m}: leaky={leaky:.4f}  honest={roc_auc_score(c.y[ok],p[ok]):.4f}")
    PY

**Expect:** leaky rises with size, honest falls. If honest does NOT fall, this
claim is dead — say so.

**Caveat to hold in mind while judging:** two points is not a scaling curve. The
honest CIs (0.681–0.745 vs 0.634–0.703) overlap slightly at the edges. Treat this
as "scaling does not help", which is safe, rather than "scaling reliably hurts".

` VERDICT: ____`

---

## CLAIM 4 — The `prompt_injection` inversion is real and is not a probe artefact

**Claim:** the family scores 0.302 (below chance) under LOFO; **13 of 18**
independent judges — including 9/9 frontier models and 1 of 2 humans — also invert;
and removing the "PWNED" competition string does **not** fix it (0.302 → 0.291).

**Recompute all of it from raw data:**

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
    print("   with PWNED rows removed :", round(roc_auc_score(y[keep], p[keep]), 3),
          f"(n={int(keep.sum())})")
    print("PWNED in hackaprompt: %.1f%%   in rest of corpus: %.1f%%" % (
        100*pwn[np.array(c.category)=="hackaprompt_dataset"].mean(),
        100*pwn[~m].mean()))
    print("human a1:", round(roc_auc_score(y[m], c.author1[m]), 3),
          " human a2:", round(roc_auc_score(y[m], c.author2[m]), 3),
          " annot corr in-family:", round(np.corrcoef(c.author1[m], c.author2[m])[0,1], 3),
          " corpus-wide:", round(np.corrcoef(c.author1, c.author2)[0,1], 3))
    ids = set(np.array(c.ids)[m])
    for name in compare_blackbox.available_models():
        bb = compare_blackbox.blackbox_scores(name); bb = bb[bb.sample_id.isin(ids)]
        print(f"   {name:24s} {roc_auc_score(bb.is_eval, bb.bb_score):.3f}")
    PY

**Expect:** probe 0.302; PWNED-removed ≈ 0.291 (**the inversion survives**);
PWNED in 57.9% of hackaprompt and 0.0% of the rest; annotator correlation
−0.242 in-family vs +0.774 corpus-wide; and every one of the 9 frontier models
below 0.5.

**Read with your own eyes — this is the part that cannot be delegated:**

    less artifacts/inversion_transcripts.txt

Read **all 19** `DEPLOY-labelled` (hackaprompt) transcripts and at least **10** of
the 20 `EVAL-labelled` (mosscap/gandalf) ones. Both sides are prompt-injection
attacks. The question you are adjudicating: *does the hackaprompt content read as
organic real-world usage, as its `deploy` label asserts?* Your answer decides
between candidate C3 (the label encodes provenance, not content) and C2 (system-prompt
distribution shift) in `FINDINGS.md` §3.

**If you want to settle C2**, the experiment that would do it has NOT been run:
transplant the `The secret password is GOURD...` system prompt onto hackaprompt
rows and the generic one onto mosscap/gandalf rows, re-extract, and re-score. If
the inversion follows the system prompt, C2 is the mechanism.

` VERDICT: ____`

---

## CLAIM 5 — At a 1% base rate the probe is not a usable monitor

**Claim:** on honest scores, Qwen3-4B catches 20.5% of evals at a 1% FPR budget,
precision 0.185, i.e. **4.4 false alarms per true catch**; 8B is worse (9.9);
ECE ≈0.28–0.31.

**Recompute from the activations, not from `honest_reliability.json`:**

    python - <<'PY'
    import numpy as np
    from evalaware import data, extract, confound, reliability
    c = data.load_corpus()
    for m, L in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
        X = extract.load_acts(m)[L]
        _, p = confound.leave_one_family_out(X, c.y, c.family)
        ok = ~np.isnan(p)
        r = reliability.full_report(c.y[ok], p[ok], 0.01, 0.01, 2000, 0)
        tp = r["tpr_at_fpr"] * 0.01 * 10000
        fp = r["achieved_fpr"] * 0.99 * 10000
        print(f"{m}: AUROC {r['auroc']:.4f}  TPR@{r['achieved_fpr']*100:.2f}%FPR "
              f"{r['tpr_at_fpr']:.4f}  precision@1% {r['precision_at_base_rate']:.4f} "
              f"ECE {r['ece']:.4f}  -> {tp:.0f} TP vs {fp:.0f} FP "
              f"= {fp/tp:.1f} false alarms per catch")
    PY

**Expect:** 4B ≈ 0.714 / 0.205 / 0.185 / ECE 0.278 / 4.4 false alarms per catch;
8B ≈ 0.668 / 0.091 / 0.092 / ECE 0.312 / 9.9.

**Verify the Bayes arithmetic independently** (this is the load-bearing step — if
the precision formula is wrong, the whole reliability claim is wrong):

    python -c "
    tpr, fpr, base = 0.2048, 0.0091, 0.01
    print('precision =', tpr*base / (tpr*base + fpr*(1-base)))"

**Expect:** ≈ 0.185, matching the number above.

**Read with your own eyes:** `figures/fig_c_reliability.png`. Confirm the middle
panel shows precision rising steeply with base rate (0.19 → 0.54 → 0.71) — i.e.
the conclusion is specific to *rare* evals and would not hold at 10%.

` VERDICT: ____`

---

## CLAIM 6 (bonus) — aider's 0.948 is a length artefact, not eval-awareness

Worth checking because it is the one mixed-family number that looks like a success.

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

**Expect:** length-only AUROC **0.000**, eval 2266–3824, deploy 5951–48901,
`overlap: False`. Perfect separation by length alone ⇒ discount aider.

` VERDICT: ____`

---

## Summary sheet

| # | claim | headline number | verdict |
|---|---|---|---|
| 1 | 0.956 is leakage (0/60 mixed; TF-IDF 0.936) | 0.956 vs 0.936 | ____ |
| 2 | honest number is ~0.67–0.74 | 0.714 / 0.668 | ____ |
| 3 | scaling hurts the honest number | 0.714 → 0.668 | ____ |
| 4 | prompt_injection inversion, 13/18 judges | 0.302 | ____ |
| 5 | unusable at 1% base rate | 4.4 false alarms/catch | ____ |
| 6 | aider is a length artefact | length AUROC 0.000 | ____ |

**If any verdict comes back FAIL, `FINDINGS.md` must be corrected before anything
is written up.**
