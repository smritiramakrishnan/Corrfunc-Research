#!/bin/bash

#SBATCH --job-name=wp-across-subvolumes
#SBATCH --account=hywu_cluster_sims_0001
#SBATCH --output=output_%A_%a.txt
#SBATCH --array=20-24
#SBATCH --partition=dev
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --chdir="/users/smritir/Corrfunc-Research/source/spectroscopic-vs-photometric"
#SBATCH --mail-type=all
#SBATCH --mail-user smritir@smu.edu

module purge
module load miniforge
conda activate mamba_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

srun python3 wp_cross_subvol.py $SLURM_ARRAY_TASK_ID