import numpy as np
from cobaya.theory import Theory
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from scipy.interpolate import interp1d
from scipy.special import spherical_jn
from scipy.integrate import simps

from linear_theory import*
from pnw_dst import pnw_dst


class RSDCalculatorNu(Theory):

    file_base_name = 'RSDCalculator'

    # yaml variables
    kmin: float
    kmax: float
    nk: int
    ngauss: int
    nmax: int
    kIR: float

    def initialize(self):
        """called from __init__ to initialize"""

        # this kvec is optimized SPECIFICALLY for the BOSS analysis
        
        self.kvec = np.concatenate( ([0.0005,],\
                                     np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                                     np.arange(0.03,0.51,0.01)) )
        
        #self.k = np.logspace(np.log10(self.kmin),
        #                     np.log10(self.kmax),
        #                     self.nk)

    def initialize_with_provider(self, provider):
        """
        Initialization after other components initialized, using Provider class
        instance which is used to return any dependencies (see calculate below).
        """
        self.provider = provider

    def get_requirements(self):
        """
        Return dictionary of derived parameters or other quantities that are needed
        by this component and should be calculated by another theory class.
        """

        return {}

    def must_provide(self, **requirements):
        for i, t in enumerate(requirements):
            if t=='pt_pk_ell_model':
                self.z = np.array(requirements[t]['z'])
                self.chiz_fid = np.array(requirements[t]['chiz_fid'])
                self.hz_fid = np.array(requirements[t]['hz_fid'])

        return {'Pk_interpolator': {'k_max': 30, 'z':self.z,\
                                    'nonlinear': False,\
                                    'vars_pairs': [['delta_nonu','delta_nonu']]},
                'sigma8_z': {'z': self.z},
                'fsigma8': {'z': self.z},
                'Hubble': {'z': self.z},
                'rdrag': None,
                'angular_diameter_distance': {'z': self.z},
                'H0': None,\
                'omegam': None,\
                'omnuh2': None,}

    def get_can_provide(self):

        return ['pt_pk_ell_model']

    def calculate(self, state, want_derived=True, **params_values_dict):

        pk_lin_interp = self.provider.get_Pk_interpolator(nonlinear=False)

        # Get cosmological parameters
        OmM = self.provider.get_param('omegam')
        hub  = self.provider.get_Hubble(0)[0]/100.
        omnuh2 = self.provider.get_param('omnuh2')
        fnu =  omnuh2/ hub**2 /OmM
        
        # linear growth rate using analytic formula
        #sigma8_z = self.provider.get_sigma8_z(self.z)
        #fsigma8 = self.provider.get_fsigma8(self.z)
        f = f_of_a(1/(1.+self.z), OmegaM=OmM) * (1 - 0.6 * fnu)

        # AP
                    
        Hz   = self.provider.get_Hubble(self.z)[0]/self.provider.get_Hubble(0)[0]
        chiz = self.provider.get_comoving_radial_distance(self.z)[0]*hub
        apar,aperp = self.hz_fid/Hz,chiz/self.chiz_fid
        
        #H0 = self.provider.get_param('H0')
        #h = H0 / 100.
        #apar = self.hz_fid/(self.provider.get_Hubble(self.z) / H0)
        #aperp = (self.provider.get_angular_diameter_distance(self.z) *
        #         (1 + self.z) * h) / self.chiz_fid

        
        # Full Shape PT Module
        state['pt_pk_ell_model'] = []
        
        for i, z in enumerate(self.z):
            pk = pk_lin_interp.P(z, self.k * h) * h**3
            lpt = LPT_RSD(ki, pi, kIR=0.2,\
                          cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, jn=5)
            lpt.make_pltable(f, kv=self.kvec, apar=apar, aperp=aperp, ngauss=2)
            state['pt_pk_ell_model'].append(lpt)
            
        # Zeldovich Reconstruction
        tate['pt_recon_zel_mod'] = []
        
        for i, z in enumerate(self.z):
            knw, pnw = pnw_dst(ki, pi, ii_l =75, ii_r = 250)
            pw = pi - pnw
            
            qbao   = pp.get_param('rdrag') * hub # want this in Mpc/h units
            
            t1 = time.time()

            j0 = spherical_jn(0,ki*qbao)
            Sk = np.exp(-0.5*(ki*15)**2)

            sigmadd = simps( 2./3 * pi * (1-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)
            sigmass = simps( 2./3 * pi * (-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)

            sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
            sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
            sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term
        
            Zels = {'R': self.bao_R,
                    'fz':ff,\
                    'alphas': (apar, aperp),\
                    'klin': ki, 'pnw': pnw, 'pw': pw,\
                    'sigmas': (sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds)}
            
            state['pt_recon_zel_mod'].append(Zels)
