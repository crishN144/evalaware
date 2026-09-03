#!/bin/bash
#SBATCH --job-name=ea_main8b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=gpu013
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_main_%j.log

source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
echo "### host $(hostname)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo; echo "########## GPU TESTS ##########"
python -c "print(1)"

echo; echo "########## PIPELINE Qwen3-8B ##########"
python -m evalaware.run --stage all --model Qwen/Qwen3-8B --batch-size 4 --just-ask
echo "### DONE rc=$?"
