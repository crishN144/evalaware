# evalaware

Probing Qwen3 residual streams for evaluation-awareness, on the
`jjpn2/eval_awareness` benchmark (976 transcripts, 537 eval / 439 deploy).

**Results live in [FINDINGS.md](FINDINGS.md).**
**To check them yourself, run [VERIFY_CRISH.md](VERIFY_CRISH.md).**

This is an audit of an existing method, not a new one. Linear probing for
eval-awareness is a crowded subfield (Nguyen 2507.01786; Chaudhary 2509.13333;
Manek 2606.29196; Abdelnabi & Salem 2505.14617; Hua & Nanda 2510.20487). No
priority is claimed for probing, for open models, or for Qwen. The contribution
is the confound audit plus low-base-rate reliability on a benchmark none of them
used. The Qwen late-layer peak and non-monotonic scaling corroborate Manek.

    source /mnt/scratch/bgxp240/interp/activate.sh
    cd /mnt/scratch/bgxp240/interp/evalaware
    python -m pytest -q -m "not gpu"        # 32 tests, ~70s, no GPU needed
    python -m evalaware.run --stage all --model Qwen/Qwen3-4B --just-ask

Stages run in order and each caches to `artifacts/`, so any one can be rerun
alone: `extract -> probe -> baselines -> reliability -> confound -> compare`.
Only `extract` and `--just-ask` need a GPU.

## READ THIS BEFORE QUOTING ANY NUMBER

**No raw category in this corpus is label-mixed — 0 of 60.** Every source
dataset is entirely eval or entirely deploy, so category identity determines
the label outright. Any classifier that recognises the source corpus scores
near-perfectly without representing anything about evaluation-awareness.

Consequences, all measured on Qwen3-4B layer 34:

| number | value | what it means |
|---|---|---|
| random-CV AUROC | 0.956 | **do not quote this** - near-twins on both sides of the split |
| TF-IDF surface baseline | 0.936 | word n-grams alone get almost all of it |
| grouped CV (by family) | 0.689 | |
| leave-one-family-out, pooled | 0.714 | the honest generalisation number (0.67-0.74 across layers/poolings) |
| mixed-family macro | 0.626 | the three families where label varies within source |

The three label-mixed *families* exist only because `data.family_of` merges
categories that share a scaffold: smolagents (gaia=eval vs real=deploy), aider
(full=eval vs commit/multi=deploy), prompt_injection (mosscap/gandalf=eval vs
hackaprompt=deploy). Those 344 rows are the only place in the corpus where the
label moves while the source is roughly held fixed, so they carry most of the
evidential weight and deserve more than the rest combined.

## Controls that came back clean

Length is **not** the confound, contrary to the obvious worry. Eval transcripts
are *shorter* (median 237 vs 1471 tokens); length tracks scaffold
(corr(log len, agentic)=0.84), not the label (corr(log len, y)=-0.17).
Length-matching moves AUROC 0.956 -> 0.951, and `length_matched` reports the
achieved within-bin length gap so under-matching is visible rather than assumed
(see `config.LENGTH_MATCH_BINS`).

## transformers v5 gotcha, pinned by a test

`Qwen3DecoderLayer.forward` returns a bare Tensor, not a tuple. The legacy
nnsight idiom `layers[i].output[0]` indexes the BATCH dimension and silently
yields `(seq, hidden)`. Use `layers[i].output`.
`tests/test_extract.py::test_residual_accessor_returns_batch_seq_hidden` fails
if this regresses.

## mcq_transcripts column semantics, pinned by a test

`joe` == `eval_author1` and `giles` == `eval_author2` — swapped relative to the
author1/author2 ordering. `human` is their mean; `is_eval`/`agent` reproduce
`eval_category`/`scaffold` exactly. 858 of 976 records overlap.
Getting the annotator mapping backwards silently corrupts every human
comparison, so `tests/test_compare_blackbox.py` asserts it.

## Layout

    config.py            paths, layers, seeds, constants
    data.py              load, flatten content, format transcript, labels, families
    extract.py           nnsight extraction -> artifacts/acts_<model>.npz
    baselines.py         just-ask, TF-IDF, length-only
    probe.py             per-layer logistic probe, AUROC
    reliability.py       ROC, TPR@FPR, precision at deploy base rate, ECE, bootstrap
    confound.py          leave-one-family-out, within-stratum, length-matched
    compare_blackbox.py  probe vs the paper's mcq_transcripts
    lock_metrics.py      headline table + bootstrap CIs      -> locked_metrics.json
    inversion.py         prompt_injection inversion evidence -> inversion_*.{json,txt}
    mixed.py             the 3 label-mixed families          -> mixed_families.json
    honest_reliability.py  1% base-rate reliability, honest  -> honest_reliability.json
    sensitivity.py       layer x pooling sweep               -> sensitivity.json
    final_figures.py     the four publication figures        -> figures/fig_*.png
    run.py               orchestrator

## Added after review (Sep 2026)

| module / file | what |
|---|---|
| `evalaware/transplant.py`, `job_transplant.sh` | system-prompt transplant on the 56 prompt_injection rows (C2 test). FINDINGS §3.1 |
| `evalaware/extra_experiments.py` | internal-vs-verbalised 2x2, cross-regime transfer, random-direction control. FINDINGS §7b |
| `evalaware/pub_figures.py` | publication figures `figures/pub_*.png`, one semantic colour system, captions live in the write-up |
| `evalaware/extra_figures.py` | earlier review figures `figures/fig_e..i_*.png` (captions baked in) |
| `build_writeup_v3.py`, `build_writeup_v4.py` | docx builders; v4 is the current write-up |

Environment: `source /mnt/scratch/bgxp240/interp/activate.sh` first. The default conda env has no sklearn and pytest collection fails outright without it.

### Figure fonts

`evalaware/pub_figures.py` asks for Inter (text) and JetBrains Mono (code), both OFL-licensed,
falling back to DejaVu if absent. To match the shipped PNGs, drop the static TTFs into
`~/.fonts/evalaware/ttf/` (Inter-{Regular,SemiBold,Bold}.ttf, JetBrainsMono-Regular.ttf) and
delete matplotlib's `fontlist*.json` cache once. Sources: github.com/rsms/inter (v4.1),
github.com/JetBrains/JetBrainsMono (v2.304).
