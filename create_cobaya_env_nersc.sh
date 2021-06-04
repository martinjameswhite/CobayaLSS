#!/bin/bash
#
# This works best if run from an interactive shell, not a Jupyter "shell".
#
# I needed to use PrgEnv-gnu in order to get the Planck lensing
# likelihood to compile.
# 
# Set up an empty environment for Cobaya (cloning the base ensures
# mpi4py is properly included in its NERSC form).
#
conda create --name cobaya --clone base
#
# Switch to the environment.
source activate cobaya
#
# Install some basic stuff
conda install -c conda-forge pyfftw -y
conda install healpy -y
#
# Set up the environment for Jupyter.
conda install ipykernel ipython jupyter # Should already be there.
python3 -m ipykernel install --user --name cobaya --display-name Cobaya-env
#
# Now install Cobaya
python3 -m pip install cobaya  # --upgrade
#
# and any "cosmo" packages it wants
cobaya-install cosmo -p /global/cscratch1/sd/mwhite/Cobaya/Packages
#
# Install velocileptors,
#python3 -m pip install git+https://github.com/sfschen/velocileptors
# and Anzu if you want.
#conda install -c conda-forge pyccl chaospy -y
#python3 -m pip install -v git+https://github.com/kokron/anzu
#
