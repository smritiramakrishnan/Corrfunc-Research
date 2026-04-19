#!/bin/bash

#SBATCH --job-name=wp-count-limit
#SBATCH --account=hywu_cluster_sims_0001
#SBATCH --array=11-24%5
#SBATCH --output=wp-counts_%A_%a.txt
#SBATCH --time=1-00:00:00
#SBATCH --partition=standard-s
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --chdir="/users/smritir/Corrfunc-Research/source/spectroscopic-vs-photometric"
#SBATCH --mail-type=all
#SBATCH --mail-user=smritir@smu.edu

module purge
module load miniforge
conda activate mamba_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python3 wp_cross_counts.py $SLURM_ARRAY_TASK_ID
