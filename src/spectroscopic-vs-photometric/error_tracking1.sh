#!/bin/bash

#SBATCH --job-name=error-tracking1-halos
#SBATCH --account=hywu_cluster_sims_0001
#SBATCH --array=0-1
#SBATCH --time=2:00:00
#SBATCH --mem=330G
#SBATCH --output=error-tracking_%A_%a.txt
#SBATCH --partition=dev
#SBATCH --chdir="/users/smritir/Corrfunc-Research/source/spectroscopic-vs-photometric"
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=all
#SBATCH --mail-user=smritir@smu.edu

module purge
module load miniforge
conda activate mamba_env

export OMP_NUM_THREAD=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python3 error_tracking1.py $SLURM_ARRAY_TASK_ID