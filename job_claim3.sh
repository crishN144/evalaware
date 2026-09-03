#!/bin/bash
#SBATCH --job-name=ea_claim3
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_claim3_%j.log
source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u - <<'PY'
import numpy as np
from sklearn.metrics import roc_auc_score
from evalaware import data, extract, probe, confound
c = data.load_corpus()
for m, L in [("Qwen/Qwen3-4B", 34), ("Qwen/Qwen3-8B", 35)]:
    X = extract.load_acts(m)[L]
    leaky = roc_auc_score(c.y, probe.oof_scores(X, c.y))
    _, p = confound.leave_one_family_out(X, c.y, c.family)
    ok = ~np.isnan(p)
    print(f"{m}: leaky={leaky:.4f}  honest={roc_auc_score(c.y[ok],p[ok]):.4f}", flush=True)
PY
echo "### DONE rc=$?"
