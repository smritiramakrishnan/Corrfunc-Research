#!/bin/bash

#SBATCH --job-name=wp-across-subvolumes
#SBATCH --account=account_name
#SBATCH --output=output_%A_%a.txt
#SBATCH --array=0-24%5
#SBATCH --partition=partition_name
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --chdir=file_dir
#SBATCH --mail-type=all
#SBATCH --mail-user=user_mail

module purge
module load miniforge
conda activate mamba_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python3 wp_cross_subvol.py $SLURM_ARRAY_TASK_ID