#!/bin/bash
#SBATCH --job-name=ea_meanpool
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_meanpool_%j.log
source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
echo "### host $(hostname)"
python - <<'PY'
from evalaware import data, extract
c = data.load_corpus()
for m in ["Qwen/Qwen3-4B", "Qwen/Qwen3-8B"]:
    print(f"### mean-pool extraction {m}", flush=True)
    extract.extract_corpus(c, m, batch_size=4, pooling="mean")
PY
echo "### DONE rc=$?"
