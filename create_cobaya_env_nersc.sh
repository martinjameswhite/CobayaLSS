#!/bin/bash
#
# Need to run this from an interactive shell, not a Jupyter "shell"
# since those don't have the PrgEnv installed.
#
# I needed to use PrgEnv-gnu in order to get the Planck lensing
# likelihood to compile.
# 
conda create --name cobaya --clone base
#
source activate cobaya
# Install some basic stuff
conda install -c conda-forge pyfftw -y
conda install healpy -y
# Use the "Cobaya" version of CLASS (installed with cobaya-install below)
###cd /global/cfs/cdirs/m68/mwhite/class/python/
###python setup.py install
# Set up the environment for Jupyter.
conda install ipykernel ipython jupyter # Should already be there.
python -m ipykernel install --user --name cobaya --display-name Cobaya-env
#
python -m pip install cobaya  # --upgrade
#
cobaya-install cosmo -m /global/cscratch1/sd/mwhite/Cobaya/Packages
#
# Install velocileptors. 
#python3 -m pip install git+https://github.com/sfschen/velocileptors
# and Anzu
#conda install -c conda-forge pyccl chaospy -y
#python3 -m pip install -v git+https://github.com/kokron/anzu
#
