#!/bin/bash
#SBATCH --job-name=annulus_Re10_t100
#SBATCH --account=pi_bthomases_smith_edu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=18:00:00
#SBATCH --output=%x_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT_80

set -e

module purge
module load conda/latest
conda activate /work/pi_bthomases_smith_edu/bthomases_smith_edu/conda-envs/dedalus3

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PMIX_MCA_psec=^munge

cd /work/pi_bthomases_smith_edu/bthomases_smith_edu/repos/dedalus_NS_annulus
mpirun python NS_annulus_Re10_long.py
