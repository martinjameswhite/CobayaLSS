#!/bin/bash
#SBATCH -J Cobaya
#SBATCH -N 1
#SBATCH -t 1:30:00
#SBATCH -o Cobaya.out
#SBATCH -e Cobaya.err
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -A <repo>
#
# May need to module load python also.
source activate cobaya
#
export PYTHONPATH=${PYTHONPATH}:/path/to/CobayaLSS/lss_likelihood
export OMP_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=4
#
# Example chain:
fb=boss_s01_z038_sig8_xi
#
# To start a run:
rm -rf chains/${fb}.*
srun -N ${SLURM_NNODES} --ntasks-per-node 8 -c 4 cobaya-run ${fb}.yaml
#
# To restart:
#srun -N ${SLURM_NNODES} --ntasks-per-node 8 -c 4 cobaya-run chains/${fb}
#
