#!/bin/bash
#SBATCH --job-name=ea_sens
#SBATCH --partition=nodes
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/scratch/bgxp240/interp/evalaware/artifacts/job_sens_%j.log
source /mnt/scratch/bgxp240/interp/activate.sh
cd /mnt/scratch/bgxp240/interp/evalaware
# One BLAS thread per process: the refits are many and small, so threaded BLAS
# oversubscribes and thrashes (168 CPU-min in 10 wall-min on the login node).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "### host $(hostname) cpus=$SLURM_CPUS_PER_TASK"
python -u -c "from evalaware import sensitivity as S; S.report()"
echo "### DONE rc=$?"
