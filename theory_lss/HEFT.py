import numpy as np
from cobaya.theory import Theory
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from velocileptors.LPT import moment_expansion_fftw
from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
#from anzu.emu_funcs import LPTEmulator
from cobaya.typing import Union, Sequence
import sys

class PTCalculator(Theory):

    file_base_name = 'PTCalculator'

    # yaml variables
    kmin: float
    kmax: float
    nk: int
    zmin: float
    zmax: float
    nz: float
    kecleft: bool
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

        self.nspec = 10

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
        if ('pt_spectrum_grid' in requirements) | ('pt_spectrum_interpolator' in requirements):
            reqs = {'Pk_interpolator': {'k_max': 10,
                                        'z': self.z,
                                        'nonlinear': False},
                    'sigma8_z': {'z': self.z},
                    'Hubble': {'z': [0.0]}}

        if 'pt_spectrum_interpolator' in requirements:

            self.get_interpolator = True

        return reqs

    def get_can_provide(self):

        return ['pt_spectrum_grid', 'pt_spectrum_interpolator']

    def calculate(self, state, want_derived=True, **params_values_dict):

        state['pt_spectrum_grid'] = {}

        k = self.k
        pk_lin_interp = self.provider.get_Pk_interpolator(nonlinear=False)
        sigma8_z = self.provider.get_sigma8_z(self.z)
        D = sigma8_z/sigma8_z[0]
        h = self.provider.get_Hubble(0.0) / 100
        
        state['pt_spectrum_grid']['k'] = self.k
        state['pt_spectrum_grid']['power'] = np.zeros((self.nz,
                                                       self.nk,
                                                       self.nspec))
        if self.kecleft:
            pk = pk_lin_interp.P(0, self.k * h)
            cleftobj = RKECLEFT(self.k, pk)

            for i, z in enumerate(self.z):
                cleftobj.make_ptable(D=D[i], kmin=k[0], kmax=k[-1], nk=1000)
                cleftpk = cleftobj.pktable
                state['pt_spectrum_grid']['power'][i, :, ...] = cleftpk[:,1:self.nspec+1]

        else:
            for i, z in enumerate(self.z):
                pk = pk = pk_lin_interp.P(z, self.k)
                cleftobj = CLEFT(self.k, pk, N=2700, jn=10, cutoff=1)
                cleftobj.make_ptable()
                cleftpk = cleftobj.pktable[:, 1:self.nspec+1]

                # Different cutoff for other spectra, because otherwise different
                # large scale asymptote

                cleftobj = CLEFT(k, pk, N=2700, jn=5, cutoff=10)
                cleftobj.make_ptable()

                cleftpk[:, 2:] = cleftobj.pktable[:, 4:self.nspec+1]
                state['pt_spectrum_grid']['power'][i, :, ...] = cleftpk

        state['pt_spectrum_grid']['power'][:, :, 1] /= 2
        state['pt_spectrum_grid']['power'][:, :, 5] /= 0.25
        state['pt_spectrum_grid']['power'][:, :, 6] /= 2
        state['pt_spectrum_grid']['power'][:, :, 7] /= 2

        self.pt_spectrum_grid = state['pt_spectrum_grid']
        
        if self.get_interpolator:
            state['pt_spectrum_interpolator'] = []

            for i in range(self.nspec):
                state['pt_spectrum_interpolator'].append(PowerSpectrumInterpolator(self.z,
                                                                               self.k,
                                                                               state['pt_spectrum_grid']['power'][:, :, i]))
        self.pt_spectrum_interpolator = state['pt_spectrum_interpolator']

