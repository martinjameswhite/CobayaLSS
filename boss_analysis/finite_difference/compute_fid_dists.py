import numpy as np
import sys

from classy import Class

# Reference Cosmology:
z = float(sys.argv[1])
print(z)
Omega_M = 0.31
fb = 0.1571
h = 0.6766
ns = 0.9665
speed_of_light = 2.99792458e5


pkparams = {
    'output': 'mPk',
    'P_k_max_h/Mpc': 20.,
    'z_pk': '0.0,10',
    'A_s': np.exp(3.040)*1e-10,
    'n_s': 0.9665,
    'h': h,
    'N_ur': 3.046,
    'N_ncdm': 0,#1,
    #'m_ncdm': 0,
    'tau_reio': 0.0568,
    'omega_b': h**2 * fb * Omega_M,
    'omega_cdm': h**2 * (1-fb) * Omega_M}

import time
t1 = time.time()
pkclass = Class()
pkclass.set(pkparams)
pkclass.compute()

Hz_fid = pkclass.Hubble(z) * speed_of_light / h # this H(z) in units km/s/(Mpc/h) = 100 * E(z)
chiz_fid = pkclass.angular_distance(z) * (1.+z) * h # this is the comoving radius in units of Mpc/h 

print(Hz_fid, chiz_fid)

np.savetxt('fid_dists_z_%.2f.txt'%(z), (Hz_fid, chiz_fid) )