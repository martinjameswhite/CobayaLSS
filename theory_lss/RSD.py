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
    fiducial_exp_geo_filename: str

    def initialize(self):
        """called from __init__ to initialize"""

        self.k = np.logspace(np.log10(self.kmin),
                             np.log10(self.kmax),
                             self.nk)

        nus, ws = np.polynomial.legendre.leggauss(2*self.ngauss)
        self.nus = nus[0:self.ngauss]

        self.nspec = 19

        #fiducial distances/hubble params
        fid_exp_geo = np.genfromtxt(self.fiducial_exp_geo_filename, names=True)
        self.fid_hz_spline = interp1d(fid_exp_geo['z'], fid_exp_geo['h'])
        self.fid_chi_spline = interp1d(fid_exp_geo['z'], fid_exp_geo['chi'])


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

        if 'pknu_basis_spectrum_grid' in requirements:
            self.z = requirements['pknu_basis_spectrum_grid']['z']
            return {'Pk_interpolator': {'k_max': 10,
                                        'z': self.z,
                                        'nonlinear': False},
                    'sigma8_z': {'z': self.z},
                    'fsigma8_z': {'z': self.z},
                    'Hubble': {'z': self.z},
                    'angular_diameter_distance':{'z': self.z},
                    'H0': None}


    def get_can_provide(self):

        return ['pknu_basis_spectrum_grid']

    def calculate(self, state, want_derived=True, **params_values_dict):

        k = self.k
        nu = self.nu

        pk_lin_interp = self.provider.get_Pk_interpolator(nonlinear=False)

        #linear growth rate
        sigma8_z = self.provider.get_sigma8_z(self.z)
        fsigma8 = self.provider.get_fsigma8(self.z)
        f = fsigma8 / sigma8_z

        #AP
        hub = self.provider.get_param('H0')
        hz_fid = self.fid_hz_spline(self.z)
        chi_fid = self.fid_chi_spline(self.z)
        apar = hz_fid/(self.provider.get_Hubble(self.z) * 2997.925 / hub)
        aperp = (self.provider.get_angular_diameter_distance(self.z) *
                 (1 + self.z) * hub) / da_fid

        state['pknu_basis_spectrum_k'] = self.k
        state['pknu_basis_spectrum_nu'] = self.nu
        state['pknu_basis_spectrum_grid']= np.zeros((len(z), len(k), len(nu),
                                                     self.nspec)

        for i, z in enumerate(self.z):
            pk = pk_lin_interp.P(z, self.k)
            lpt = LPT_RSD(self.k, pk, kIR=self.kIR)
            lpt.make_pltable(f[i], nmax=self.nmax, kmin=self.kmin,
                             kmax=self.kmax, nk=self.nk,
                             apar=apar[i],aperp=aperp[i])

            state['pknu_basis_spectrum_grid'][i,...] = lpt.pknutable
