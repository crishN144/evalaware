#!/bin/bash
#SBATCH --job-name=ea_transplant
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_transplant_%j.log
source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
echo "### host $(hostname)"; nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -u -m evalaware.transplant extract
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python -u -m evalaware.transplant score
echo "### DONE rc=$?"
