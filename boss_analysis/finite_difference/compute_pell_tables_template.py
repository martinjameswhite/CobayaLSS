import numpy as np

from classy import Class
from linear_theory import f_of_a, D_of_a
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD

# k vector to use:
kvec = np.concatenate( ([0.0005,],\
                        np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                        np.arange(0.03,0.51,0.01)) )

sigma8_z0 = 0.8
ki, pi = np.loadtxt('boss_template_pk.txt', unpack=True)

def compute_pell_tables(pars, z=0.61, OmM_fid=0.31 ):

    fsigma8, apar, aperp = pars
    
    # Compute the implied value of fz
    Dz = D_of_a(1./(1+z), OmegaM=OmM_fid)
    fz = fsigma8 / (Dz * sigma8_z0)
    
    # Now do the RSD
    modPT = LPT_RSD(ki, Dz**2 * pi, kIR=0.2,\
                cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=1, jn=5)
    modPT.make_pltable(fz, kv=kvec, apar=apar, aperp=aperp, ngauss=3)
    
    return modPT.p0ktable, modPT.p2ktable, modPT.p4ktable
    