#!/bin/bash
#SBATCH --job-name=annulus_big
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out

module purge
module load conda/latest
conda activate /work/pi_shuzhou_umass_edu/bthomases_smith_edu/conda-envs/dedalus3

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export PMIX_MCA_psec=^munge

cd /work/pi_shuzhou_umass_edu/bthomases_smith_edu/repos/dedalus_NS_annulus

mpirun python NS_annulus_mpi_biggertest.py
