import numpy as np
from cobaya.theory import Theory
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD


class RSDCalculator(Theory):

    file_base_name = 'RSDCalculator'

    # yaml variables
    kmin: float
    kmax: float
    nk: int
    zmin: float
    zmax: float
    nz: float
    ngauss: int
    nmax: int
    z: Union[Sequence, np.ndarray]

    def initialize(self):
        """called from __init__ to initialize"""

        if hasattr(self, 'z'):
            if self.z == []:

                if hasattr(self, 'zmin'):
                    zmin = self.zmin

                    try:
                        zmax = self.zmax
                        nz = self.nz
                        self.z = np.linspace(zmin, zmax, nz)
                    except Exception as e:
                        print(e)
                        raise(e)
            self.z = np.array(self.z)

        self.z = np.hstack([self.z, np.zeros(1)])
        self.z = np.unique(self.z)

        self.k = np.logspace(np.log10(self.kmin),
                             np.log10(self.kmax),
                             self.nk)

        nus, ws = np.polynomial.legendre.leggauss(2*self.ngauss)
        self.nus = nus[0:self.ngauss]

        self.nspec = 19


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
        return {'Pk_interpolator': {'k_max': 10,
                                    'z': self.z,
                                    'nonlinear': False},
                'sigma8_z': {'z': self.z},
                'fsigma8_z': {'z': self.z}}

    def must_provide(self, **requirements):

        return {}

    def get_can_provide(self):

        return ['pknu_basis_spectrum_grid']

    def calculate(self, state, want_derived=True, **params_values_dict):

        state['pknu_basis_spectrum_grid'] = {}

        k = self.k
        nu = self.nu

        pk_lin_interp = self.provider.get_Pk_interpolator(nonlinear=False)
        sigma8_z = self.provider.get_sigma8_z(self.z)
        fsigma8 = self.provider.get_fsigma8(self.z)
        f = fsigma8 / sigma8_z

        state['pknu_basis_spectrum_grid'] = {}
        state['pknu_basis_spectrum_grid']['k'] = self.k
        state['pknu_basis_spectrum_grid']['nu'] = self.nu
        state['pknu_basis_spectrum_grid']['power'] = np.zeros((len(z), len(k), len(nu), self.nspec)

        for i, z in enumerate(self.z):
            pk = pk_lin_interp.P(z, self.k)
            lpt = LPT_RSD(self.k, pk, kIR=self.kIR)
            lpt.make_pltable(f[i], nmax=self.nmax, kmin=self.kmin,
                             kmax=self.kmax, nk=self.nk,
                             apar=1,aperp=1)

            state['pknu_basis_spectrum_grid']['power'][i,...] = lpt.pknutable