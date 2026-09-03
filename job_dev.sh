#!/bin/bash
#SBATCH --job-name=ea_dev4b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_dev_%j.log

source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
echo "### host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "########## GPU TESTS ##########"
python -m pytest -q -m gpu 2>&1 | tail -20 || exit 1

echo; echo "########## PIPELINE Qwen3-4B ##########"
python -m evalaware.run --stage all --model Qwen/Qwen3-4B --batch-size 4 --just-ask
echo "### DONE rc=$?"
