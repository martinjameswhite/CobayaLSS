import numpy as np

from classy import Class

def compute_sigma8(pars, lnA0 = 3.047):
    
    OmegaM, h= pars
    
    omega_b = 0.02242

    lnAs =  lnA0
    ns = 0.9665

    nnu = 1
    nur = 2.033
    mnu = 0.06
    omega_nu = 0.0106 * mnu
        
    omega_c = (OmegaM - omega_b/h**2 - omega_nu/h**2) * h**2

    pkparams = {
        'output': 'mPk',
        'P_k_max_h/Mpc': 20.,
        'z_pk': '0.0,10',
        'A_s': np.exp(lnAs)*1e-10,
        'n_s': ns,
        'h': h,
        'N_ur': nur,
        'N_ncdm': nnu,
        'm_ncdm': mnu,
        'tau_reio': 0.0568,
        'omega_b': omega_b,
        'omega_cdm': omega_c}

    pkclass = Class()
    pkclass.set(pkparams)
    pkclass.compute()
    
    sigma8 = pkclass.sigma8()

    return sigma8
    