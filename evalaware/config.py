"""Paths, constants, and run configuration. No logic beyond derived paths."""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT      = Path("/mnt/scratch/bgxp240/interp")
REPO      = ROOT / "evalaware"
DATA      = ROOT / "data" / "dataset.json"
MCQ       = ROOT / "data" / "mcq_transcripts"
ARTIFACTS = REPO / "artifacts"
FIGURES   = REPO / "figures"

for _p in (ARTIFACTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- models
DEV_MODEL  = "Qwen/Qwen3-4B"    # debug here first
MAIN_MODEL = "Qwen/Qwen3-8B"    # headline numbers here

# Both Qwen3-4B and Qwen3-8B have 36 layers (hidden 2560 / 4096).
N_LAYERS = 36
LAYERS   = list(range(0, N_LAYERS, 2)) + [N_LAYERS - 1]   # 0,2,...,34,35

SEED = 0

# ---------------------------------------------------------------- tokenisation
# Transcript lengths are extremely skewed: p50=548, p90=16.5k, max=78.8k tokens.
# We keep a head window (system prompt / task framing is highly informative) and
# a tail window (the last token is what we read the residual stream at).
MAX_TOKENS  = 4096
HEAD_TOKENS = 1024
TAIL_TOKENS = MAX_TOKENS - HEAD_TOKENS
ELISION     = "\n\n[... transcript truncated ...]\n\n"

# ---------------------------------------------------------------- probing
CV_FOLDS   = 5
N_BOOTSTRAP = 2000
# Deployment-realistic base rate used for precision reporting in reliability.py.
DEPLOY_BASE_RATE = 0.01
# Report TPR at this false-positive rate.
TARGET_FPR = 0.01

# Bins used to length-match eval vs deploy in confound.py. Coarse bins leave a
# large within-bin length gap and under-correct: on a synthetic length-only
# score, 10 bins leaves residual AUROC 0.76 (gap +0.178 log-tokens) while 50
# bins gives 0.53 (gap +0.016). confound.length_matched_auroc reports the
# achieved gap so under-matching is visible rather than assumed.
LENGTH_MATCH_BINS = 50

DTYPE = "bfloat16"
