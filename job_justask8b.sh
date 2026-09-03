#!/bin/bash
#SBATCH --job-name=ea_ja8b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_ja8b_%j.log
source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
echo "### host $(hostname)"
python -m evalaware.run --stage baselines --model Qwen/Qwen3-8B --just-ask
echo "### DONE rc=$?"
