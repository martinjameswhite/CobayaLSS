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
export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/sfschen/CobayaLSS/lss_likelihood
export PYTHONPATH=${PYTHONPATH}:/global/homes/s/sfschen/Python/velocileptors
export NUMEXPR_MAX_THREADS=8
#
rm -rf chains/debug.*
#
srun -N ${SLURM_NNODES} --ntasks-per-node 1 -c 8 \
  cobaya-run debug_cobaya.yaml
#
