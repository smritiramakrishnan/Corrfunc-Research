#!/bin/bash

#SBATCH --job-name=wp-cluster-counts
#SBATCH --account=account_name
#SBATCH --array=0-24%5
#SBATCH --partition=partition_name
#SBATCH --output=wp-cluster-counts_%A_%a.txt
#SBATCH --time=1-00:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-task=32
#SBATCH --chdir=file_dir
#SBATCH --mail-type=all
#SBATCH --mail-user=user_mail

module purge
module load miniforge
conda activate mamba_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python3 wp_cluster_counts.py $SLURM_ARRAY_TASK_ID
