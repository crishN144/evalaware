# FINDINGS

Evaluation-awareness probing on Qwen3-4B / Qwen3-8B, benchmark `jjpn2/eval_awareness`
(976 transcripts, 537 eval / 439 deploy, 60 source categories, 47 merged families).

Scope: Qwen3-4B and Qwen3-8B only; this dataset only. No SAD, no other model or corpus.

Every number below was produced by the command in its row. `$A` = `artifacts/`.
Activation caches must exist first (`--stage extract`, GPU); everything else is CPU.

    source /mnt/scratch/bgxp240/interp/activate.sh
    cd /mnt/scratch/bgxp240/interp/evalaware

---

## 0. Framing (read before quoting anything)

This is an **audit**, not a new method. Linear probing for eval-awareness is an
existing, crowded subfield (Nguyen 2507.01786; Chaudhary 2509.13333; Manek
2606.29196; Abdelnabi & Salem 2505.14617; Hua & Nanda 2510.20487). Nothing here
claims to be the first probe, the first open-model probe, or the first Qwen probe.

The claims are:

1. Prior probing results may be inflated by source/formatting confounds.
2. On real transcripts under a held-out-source split, the signal largely collapses.
3. At a 1% base rate the probe is not trustworthy as the live monitor prior work proposed.
4. The Qwen late-layer peak and non-monotonic scaling **independently corroborate
   Manek (2606.29196)** on Qwen-2.5. Stated as corroboration, not priority.

---

## 1. The leakage result

**No source category in this benchmark is label-mixed: 0 of 60.** Every source
dataset is entirely eval or entirely deploy, so category identity determines the
label. Random CV therefore rewards recognising the corpus, not detecting eval-awareness.

Recompute: `python -c "from evalaware import lock_metrics; lock_metrics.lock()"` -> `$A/locked_metrics.json`

| metric | Qwen3-4B | 95% CI | Qwen3-8B | 95% CI |
|---|---|---|---|---|
| random CV — **LEAKY, do not quote** | 0.9563 | 0.945–0.967 | 0.9599 | 0.948–0.970 |
| grouped CV (by family) | 0.6888 | 0.656–0.721 | 0.6557 | 0.621–0.690 |
| **leave-one-family-out (pooled) — HONEST** | **0.7139** | **0.681–0.745** | **0.6683** | **0.634–0.703** |
| mixed-family macro | 0.6263 | — | 0.6815 | — |
| just-ask (same model, black-box) | 0.5749 | 0.539–0.611 | 0.6072 | 0.573–0.642 |

Model-independent baselines (same corpus, no activations):

| baseline | AUROC | 95% CI |
|---|---|---|
| TF-IDF word 1–2 grams | 0.9360 | 0.921–0.950 |
| length only (log tokens) | 0.5923 | 0.556–0.628 |
| human annotator 1 | 0.9043 | 0.884–0.924 |
| human annotator 2 | 0.8808 | 0.857–0.903 |

**TF-IDF on raw words reaches 0.936 with no model at all** — within 0.02 of the
probe's leaky score. That single row is the clearest statement of the problem.

Best layer: 34 (4B), 35 (8B). CIs are 2000 stratified bootstrap resamples, seed 0.

---

## 2. Scaling hurts

| | 4B | 8B | direction |
|---|---|---|---|
| random CV (leaky) | 0.9563 | 0.9599 | 8B looks better |
| LOFO pooled (honest) | 0.7139 | 0.6683 | **8B is worse** |
| best honest over all layers/poolings | 0.743 | 0.730 | **8B is worse** |

The leaky metric says scaling helps; the honest metric says it hurts. Consistent
with the larger model fitting corpus identity harder. Non-monotonic scaling
corroborates Manek (2606.29196).

Caveat: two model sizes is not a scaling curve. This is a two-point observation
consistent with a prior report, not an established trend.

---

## 3. The `prompt_injection` inversion

Recompute: `python -c "from evalaware import inversion; inversion.report(); inversion.dump_transcripts()"`
-> `$A/inversion_prompt_injection.json`, `$A/inversion_transcripts.txt`

Family = 56 rows, all chat, no scaffold:

| sub-category | n | label | mean P(eval) 4B | mean P(eval) 8B | human a1 | human a2 |
|---|---|---|---|---|---|---|
| mosscap_prompt_injection | 19 | eval | 0.525 | 0.718 | 0.934 | 0.768 |
| gandalf_ignore_instructions | 9 | eval | 0.560 | 0.733 | 0.913 | 0.757 |
| gandalf_summarization | 9 | eval | 0.644 | 0.688 | 0.940 | 0.801 |
| hackaprompt_dataset | 19 | **deploy** | **0.832** | 0.670 | 0.782 | 0.866 |

Held-out AUROC within the family: **0.302** (4B), 0.508 (8B). Below chance.

### The inversion is not specific to this probe

**13 of 18 independent judges invert on this family** (AUROC < 0.5):

| judge | AUROC on the 56 rows |
|---|---|
| gpt4o | 0.103 |
| claude35sonnet | 0.146 |
| gpt41mini | 0.164 |
| just-ask Qwen3-4B | 0.179 |
| gemini25flash | 0.199 |
| o4minimedium | 0.296 |
| probe LOFO 4B | 0.302 |
| gpt41 | 0.353 |
| human annotator 2 | 0.353 |
| gemini25pro | 0.370 |
| claude37sonnetthinking | 0.455 |
| claude37sonnet | 0.481 |
| probe LOFO 8B | 0.508 |
| human annotator 1 | 0.910 |

Annotator correlation is **+0.774 corpus-wide but −0.242 inside this family.**
The two humans who built the dataset rank these rows in opposite orders.

### Three candidate mechanisms — evidence for and against

**C1. The competition target string "I have been PWNED".**
- *For:* appears in 57.9% of hackaprompt rows and **0.0% of the rest of the corpus**;
  no other category contains it. Rows containing it score 0.842 vs 0.818 without.
- *Against — and this is decisive:* deleting every PWNED-containing row leaves the
  inversion essentially unchanged, **0.302 → 0.291** (n=45). The 8 hackaprompt rows
  with no PWNED string still average 0.818.
- **Verdict: insufficient. C1 does not explain the inversion.**

**C2. System-prompt distribution shift under the held-out split.**
- *For:* all 37 eval rows share the system prompt `The secret password is GOURD...`.
  The string GOURD appears in exactly 37 rows, all in this family — so with the
  family held out, the probe has **never seen this system prompt in training.**
  Meanwhile hackaprompt's `You are a helpful assistant.` appears in 488 rows across
  42 categories and is maximally familiar. Predicted eval rows pushed toward the
  middle while hackaprompt is scored on user-turn content.
- *Against — tested, and it fails (§3.1):* transplanting the system prompts moves
  P(eval) in the **wrong direction** to explain the inversion. Giving hackaprompt
  the GOURD prompt *raises* its P(eval) (0.835 → 0.939 on 4B); giving the eval
  rows the generic prompt leaves them flat (4B) or lowers them (8B). Swapping both
  deepens the inversion: 0.306 → 0.259 (4B), 0.512 → 0.299 (8B).
- **Verdict: refuted as the mechanism. The inversion lives in the user turns.**

**C3. The ground-truth label encodes provenance, not content.**
- *For:* every one of the 56 rows is a prompt-injection attack. `eval_category=False`
  for hackaprompt means "organic/real" — real people submitting to a public
  competition — not "non-adversarial". So the label tracks *where the text came from*
  while every judge reads *what the text is*. This is the only candidate that also
  explains why 9/9 frontier models and one of two human annotators invert; a probe
  artefact cannot explain GPT-4o at 0.103.
- *Against:* it reframes rather than mechanises. It predicts an inversion but says
  nothing about the magnitude, and does not explain why annotator 1 is at 0.910
  while annotator 2 is at 0.353.
- **Verdict: best fit to the full evidence, including the black-box models.**

**No verdict is asserted here — this is for you to adjudicate.** Read
`$A/inversion_transcripts.txt` (20 eval-labelled + all 19 deploy-labelled, with
per-row probe and human scores) and judge whether the hackaprompt content reads as
organic real-world usage.

### 3.1 The system-prompt transplant (C2 test)

Recompute: `sbatch job_transplant.sh` -> `$A/transplant_Qwen3-4B.json`, `$A/transplant_Qwen3-8B.json`, `figures/fig_i_transplant.png`

Same 56 rows, user turns untouched, system prompts swapped; scored by a probe
fit on the other 46 families (the leave-one-family-out fit). Re-extracting the
`original` condition reproduces the cached LOFO score to ~0.004
(0.306 vs 0.302 on 4B; 0.512 vs 0.508 on 8B).

| condition | 4B AUROC | 4B mean P(eval) eval / deploy rows | 8B AUROC | 8B mean P(eval) eval / deploy rows |
|---|---|---|---|---|
| original | 0.306 | 0.560 / 0.835 | 0.512 | 0.719 / 0.670 |
| swap_eval | 0.326 | 0.563 / 0.833 | 0.339 | 0.492 / 0.669 |
| swap_deploy | 0.232 | 0.561 / 0.939 | 0.492 | 0.716 / 0.810 |
| swap_both | 0.259 | 0.564 / 0.938 | 0.299 | 0.494 / 0.813 |

`swap_eval` = mosscap/gandalf rows get "You are a helpful assistant."; `swap_deploy` =
hackaprompt rows get the GOURD prompt; `swap_both` = both. The probe does respond to
the system prompt, but the unseen GOURD prompt reads as *more* eval-like, not less,
and swapping both makes the inversion worse on both models. The hackaprompt user
turns are what the probe scores as eval-like, which is what C3 says.

---

## 4. The label-mixed families

Recompute: `python -c "from evalaware import mixed; mixed.report()"` -> `$A/mixed_families.json`

These 344 rows are the only ones where the label moves while the scaffold is held
roughly fixed. All scores leave-one-family-out.

| family | n (e/d) | AUROC 4B | 95% CI | AUROC 8B | length-only AUROC | usable |
|---|---|---|---|---|---|---|
| smolagents | 159 (64/95) | 0.630 | 0.541–0.711 | 0.639 | 0.483 | **yes — clean** |
| aider | 129 (31/98) | 0.948 | 0.905–0.980 | 0.898 | **0.000** | **no — discount** |
| prompt_injection | 56 (37/19) | 0.302 | 0.162–0.437 | 0.508 | 0.174 | inverted |

**aider must be discounted.** Length alone separates its labels perfectly
(AUROC 0.000 = perfect inverted separation): eval rows span 2266–3824 tokens,
deploy rows 5951–48901, with **zero overlap**. Length-matching leaves 1 usable bin
and 2 rows, so its "length-matched 1.000" is meaningless. Its 0.948 is a length
artefact, not evidence of eval-awareness.

**smolagents is the single cleanest test in the corpus**: same scaffold both sides,
both 100% agentic, median 8104 vs 8503 tokens, length-only AUROC 0.483 (no length
signal), lengths overlap across 8 bins. It gives **0.630 / 0.639**.

Caveat: at n=159 this estimate is unstable — across layers and poolings it ranges
0.49–0.78 (§6). Treat ~0.63 as an indication, not a precise value.

---

## 5. Reliability at a 1% base rate (honest scores)

Recompute: `python -c "from evalaware import honest_reliability; honest_reliability.report()"`
-> `$A/honest_reliability.json`. Computed on leave-one-family-out predictions.

| quantity | Qwen3-4B | Qwen3-8B |
|---|---|---|
| AUROC (honest) | 0.7139 | 0.6683 |
| TPR at 0.91% FPR | 0.2048 | 0.0912 |
| precision @ 1% base rate | 0.185 | 0.092 |
| precision @ 5% base rate | 0.542 | 0.345 |
| precision @ 10% base rate | 0.714 | 0.527 |
| ECE | 0.278 | 0.312 |
| *(leaky would have claimed precision@1%)* | *0.389* | *0.341* |

Per 10,000 transcripts at a 1% base rate:

| | true catches | false alarms | false alarms per catch |
|---|---|---|---|
| Qwen3-4B | 20 | 90 | **4.4** |
| Qwen3-8B | 9 | 90 | **9.9** |

The probe misses ~80% of evaluations while raising 4–10 false alarms for every one
it catches, and ECE ≈0.28–0.31 means the scores are badly calibrated on top of
being weak. Nguyen et al. (2507.01786) proposed live monitoring without a
held-out-source split or a low-base-rate analysis; on this benchmark, under those
controls, the monitor does not hold up.

---

## 6. Sensitivity

Recompute: `sbatch job_sens.sh` -> `$A/sensitivity.json`

Honest (LOFO pooled) AUROC by layer and readout:

| model | pooling | L0 | L6 | L12 | L18 | L24 | L30 | L35 | range |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B | last | 0.553 | 0.614 | 0.571 | 0.625 | 0.684 | 0.709 | 0.686 | 0.155 |
| Qwen3-4B | mean | 0.659 | 0.679 | 0.704 | **0.743** | 0.737 | 0.677 | 0.718 | 0.084 |
| Qwen3-8B | last | 0.465 | 0.650 | 0.651 | 0.660 | 0.648 | 0.667 | 0.669 | 0.203 |
| Qwen3-8B | mean | 0.675 | 0.695 | 0.710 | **0.730** | 0.700 | 0.681 | 0.685 | 0.055 |

Findings:

- **The honest number is 0.67–0.74, not 0.67–0.71.** Mean-pooling beats last-token
  (best 0.743 vs 0.714) and is far more stable across depth (range 0.055–0.084 vs
  0.155–0.203). Reporting only the last-token figure would understate the probe.
- The leaky number is 0.83–0.98 in **every** configuration. The gap never closes.
- Scaling still hurts at each readout's best layer (0.743 → 0.730).
- **Fragile:** the smolagents estimate swings 0.49–0.78 across configs (n=159). Any
  claim resting on it needs that range attached.
- The "best layer" (34/35) was selected on the leaky random-CV score, so it is a
  mildly optimistic choice; the layer sweep above is the honest view.

---

## 7. Controls that came back clean

- **Length is not the confound corpus-wide.** Eval transcripts are *shorter*
  (median 237 vs 1471 tokens). Length tracks scaffold (corr(log len, agentic)=0.84),
  not the label (corr(log len, y)=−0.17). Length-matching moves the leaky AUROC only
  0.956 → 0.951 with a residual within-bin log-length gap of −0.0045.
  It *is* a confound **within the aider family** (§4).
- **Shuffled-label null passes:** a probe on random labels sits at chance
  (`tests/test_probe.py::test_shuffled_labels_give_chance_auroc`).
- **The scaler is fit inside each CV fold**, not on all data (asserted by test).

## 7b. Three more controls (review-requested; CPU only)

Recompute: `python -m evalaware.extra_experiments` -> `$A/extra_experiments.json`

**Internal vs verbalised.** Does the honest probe agree with the same model's own
"is this an eval?" answer? Barely. Spearman rho between honest P(eval) and just-ask
is 0.12 (4B) / 0.16 (8B); splitting each score at its
median, the two disagree on 44% / 43% of rows. Just-ask calls
88% / 76% of all rows "eval" (and 86% / 75% of the deploy-labelled ones),
the same over-classification Needham et al. report for frontier models.

**Cross-regime transfer.** Train on chat rows, test on agentic rows, and the reverse:

| | chat -> agentic | agentic -> chat | LOFO within chat | LOFO within agentic |
|---|---|---|---|---|
| Qwen3-4B | 0.741 | 0.537 | 0.685 | 0.742 |
| Qwen3-8B | 0.490 | 0.473 | 0.644 | 0.692 |

Transfer across the chat/agentic boundary is asymmetric on 4B and absent on 8B.

**Random-direction control.** AUROC of standardised activations projected onto 200
random unit directions: 0.504 ± 0.073 (4B), 0.497 ± 0.082 (8B). Folded
(max(a, 1-a)) 95th percentile 0.642 / 0.652. The honest corpus-wide 0.714 / 0.668
clears that envelope; the smolagents 0.63 does not clearly clear it, which is one more
reason to call that number an indication rather than a value. Nguyen et al. report
0.517 for a random Gaussian probe on their data.

---

## 8. Known limitations

1. Two model sizes and one dataset. Not a scaling law.
2. `smolagents` (n=159) is the only clean mixed-family test and its estimate is unstable.
3. The system-prompt transplant (§3.1) swaps only the system turn at one layer per model; it refutes C2 as the mechanism but does not by itself prove C3.
4. Transcripts are truncated to 4096 tokens (1024 head + 3072 tail); 30.6% of rows
   exceed this and lose their middle.
5. The black-box frontier numbers are taken from the paper's released
   `mcq_transcripts`, not re-run. They are a fair like-for-like on the 858 shared
   rows but are *their* measurements.
6. Probe is logistic regression on one layer at a time. No diff-of-means, no
   multi-layer, no steering — so this bounds *this* probe, not every probe.

---

## 9. Artifact index

| file | contents |
|---|---|
| `locked_metrics.json` | §1 table, all bootstrap CIs |
| `inversion_prompt_injection.json` | §3, all judges, C1/C2/C3 evidence |
| `inversion_transcripts.txt` | 39 raw transcripts, both sides, human-readable |
| `mixed_families.json` | §4 incl. `_length_audit` |
| `honest_reliability.json` | §5 |
| `sensitivity.json` | §6 |
| `transplant_Qwen3-*.json` | §3.1 transplant, all four conditions |
| `extra_experiments.json` | §7b internal-vs-verbal, cross-regime, random direction |
| `figures/pub_*.png` | publication figures, captions live in the write-up |
| `figures/fig_e..i_*.png` | purity scan, per-category scores, annotators, lengths, transplant |
| `figures/fig_a..d_*.png` | the four figures, captions state the honest reading |
