#!/bin/bash
#SBATCH -J Cobaya
#SBATCH -N 1
#SBATCH -t 0:10:00
#SBATCH -o Cobaya.out
#SBATCH -e Cobaya.err
#SBATCH -q debug
#SBATCH -C haswell
#SBATCH -A desi
#
source activate cobaya
#
export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/mwhite/Fitting/Cobaya/lss_likelihood
export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8
#
rm -rf chains/debug.*
#
srun -N ${SLURM_NNODES} --ntasks-per-node 1 -c 8 \
  cobaya-run debug_cobaya.yaml
#
