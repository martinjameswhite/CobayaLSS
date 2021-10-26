import numpy as np
from cobaya.theory import Theory
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from scipy.interpolate import interp1d


class RSDCalculator(Theory):

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

        self.k = np.logspace(np.log10(self.kmin),
                             np.log10(self.kmax),
                             self.nk)

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

        return {'Pk_interpolator': {'k_max': 10,
                                    'z': np.linspace(0.0, self.zmax, 100),
                                    'nonlinear': False},
                'sigma8_z': {'z': self.z},
                'fsigma8': {'z': self.z},
                'Hubble': {'z': self.z},
                'angular_diameter_distance': {'z': self.z},
                'H0': None}

    def get_can_provide(self):

        return ['pt_pk_ell_model']

    def calculate(self, state, want_derived=True, **params_values_dict):

        pk_lin_interp = self.provider.get_Pk_interpolator(nonlinear=False)

        # linear growth rate
        sigma8_z = self.provider.get_sigma8_z(self.z)
        fsigma8 = self.provider.get_fsigma8(self.z)
        f = fsigma8 / sigma8_z

        # AP
        H0 = self.provider.get_param('H0')
        h = H0 / 100.
        apar = self.hz_fid/(self.provider.get_Hubble(self.z) / H0)
        aperp = (self.provider.get_angular_diameter_distance(self.z) *
                 (1 + self.z) * h) / self.chiz_fid

        state['pt_pk_ell_model'] = []
        for i, z in enumerate(self.z):
            pk = pk_lin_interp.P(z, self.k * h) * h**3
            lpt = LPT_RSD(self.k, pk, kIR=self.kIR, cutoff=10,
                          extrap_min=-4, extrap_max = 3,
                          threads=1, jn=5)
            lpt.make_pltable(f[i], nmax=self.nmax, kmin=self.kmin,
                             kmax=self.kmax, nk=self.nk,
                             apar=apar[i], aperp=aperp[i])
            state['pt_pk_ell_model'].append(lpt)
