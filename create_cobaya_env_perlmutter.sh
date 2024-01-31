#!/bin/bash
#
# Install Cobaya environment on NERSC Perlmutter.
#
# Set up an empty environment for Cobaya (cloning the nersc-mpi4py
# ensures mpi4py is properly included in its NERSC form).
# Due to compiler issues, we need to use the conda-forge channel
# when possible.
#
conda create --name cobaya --clone nersc-mpi4py
#
# Switch to the environment.
conda activate cobaya
#
# Install some basic stuff
conda install -c conda-forge numpy scipy matplotlib -y
conda install -c conda-forge astropy sympy pandas cython -y
conda install -c conda-forge pyfftw healpy -y
#
# Set up the environment for Jupyter.
conda install -c conda-forge ipykernel ipython jupyter -y
python3 -Xfrozen_models=off -m ipykernel install --user --name cobaya --display-name Cobaya-env
#
# Now install Cobaya -- we use the "main" branch of the GitHub.
#python3 -m pip install cobaya  # --upgrade
python3 -m pip install git+https://github.com/CobayaSampler/cobaya
#
# and any "cosmo" packages it wants, here in $SCRATCH/Cobaya
cobaya-install cosmo -p $SCRATCH/Cobaya/Packages
# you likely need to "upgrade" the packages:
cobaya-install cosmo --upgrade -p $SCRATCH/Cobaya/Packages
#
# Install velocileptors,
#python3 -m pip install git+https://github.com/sfschen/velocileptors
# and Anzu if you want.
#conda install -c conda-forge pyccl chaospy -y
#python3 -m pip install -v git+https://github.com/kokron/anzu
# and findiff for the Taylor series emulators
python3 -m pip install --upgrade findiff
#
