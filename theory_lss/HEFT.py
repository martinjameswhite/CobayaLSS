import numpy as np
from cobaya.theory import Theory
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from velocileptors.LPT.cleft_fftw import CLEFT
from velocileptors.EPT.cleft_kexpanded_resummed_fftw import RKECLEFT
from anzu.emu_funcs import LPTEmulator
from cobaya.typing import Union, Sequence
import warnings

class HEFTCalculator(Theory):

    file_base_name = 'HEFTCalculator'

    # yaml variables
    kmin: float
    kmax: float
    nk: int
    zmin: float
    zmax: float
    nz: float
    kecleft: bool
    heft: bool
    z: Union[Sequence, np.ndarray]
    use_pcb: bool

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
        self.get_eft_interpolator = False
        self.get_heft_interpolator = False

        if self.heft:
            if self.use_pcb:
                raise ValueError('You requested that we use P_cb but Anzu does not support massive neutrinos.\
                                  Either set use_heft=False or use_pcb=False.')

            self.emu = LPTEmulator(kecleft=self.kecleft)

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

        reqs = {}

        if self.use_pcb:
            self.lin_pk_pairs = [['delta_nonu', 'delta_nonu']]
        else:
            self.lin_pk_pairs = [['delta_tot', 'delta_tot']]


        if 'heft_spectrum_interpolator' in requirements:
            if requirements['heft_spectrum_interpolator'] is None:
                zs = self.z
            elif 'z' in requirements['heft_spectrum_interpolator']:
                zs = requirements['heft_spectrum_interpolator']['z']
                self.z = zs
                self.nz = len(self.z)
            else:
                zs = self.z

            self.get_heft_interpolator = True
            reqs = {'Pk_interpolator': {'k_max': 10,
                                        'z': zs,
                                        'nonlinear': False,
                                        'vars_pairs': self.lin_pk_pairs},
                    'sigma8_z': {'z': zs},
                    'Hubble': {'z': [0.0]}}
            reqs.update({'ombh2': None, 'omch2': None, 'w': None,
                         'ns': None, 'sigma8': None, 'H0': None,
                         'N_eff': None})
            
        elif 'eft_spectrum_interpolator' in requirements:

            if requirements['eft_spectrum_interpolator'] is None:
                zs = self.z
            elif 'z' in requirements['eft_spectrum_interpolator']:
                zs = requirements['eft_spectrum_interpolator']['z']
                self.z = zs
                self.nz = len(self.z)
            else:
                zs = self.z

            self.get_eft_interpolator = True
            reqs = {'Pk_interpolator': {'k_max': 10,
                                        'z': zs,
                                        'nonlinear': False,
                                        'vars_pairs': self.lin_pk_pairs},
                    'sigma8_z': {'z': zs},
                    'Hubble': {'z': [0.0]}}

        elif 'heft_spectrum_grid' in requirements:

            if requirements['heft_spectrum_grid'] is None:
                zs = self.z
            elif 'z' in requirements['heft_spectrum_grid']:
                zs = requirements['heft_spectrum_grid']['z']
                self.z = zs
                self.nz = len(self.z)
            else:
                zs = self.z

            reqs = {'Pk_interpolator': {'k_max': 10,
                                        'z': zs,
                                        'nonlinear': False,
                                        'vars_pairs': self.lin_pk_pairs},
                    'sigma8_z': {'z': zs},
                    'Hubble': {'z': [0.0]}}
            reqs.update({'ombh2': None, 'omch2': None, 'w': None,
                            'ns': None, 'sigma8': None, 'H0': None,
                            'N_eff': None})            
            
        elif 'eft_spectrum_grid' in requirements:

            if requirements['eft_spectrum_grid'] is None:
                zs = self.z
            elif 'z' in requirements['eft_spectrum_grid']:
                zs = requirements['eft_spectrum_grid']['z']
                self.z = zs
                self.nz = len(self.z)
            else:
                zs = self.z

            reqs = {'Pk_interpolator': {'k_max': 10,
                                        'z': zs,
                                        'nonlinear': False,
                                        'vars_pairs': self.lin_pk_pairs},
                    'sigma8_z': {'z': zs},
                    'Hubble': {'z': [0.0]}}
            
        return reqs

    def get_can_provide(self):

        return ['eft_spectrum_grid', 'eft_spectrum_interpolator',
                'heft_spectrum_grid', 'heft_spectrum_interpolator']

    def calculate(self, state, want_derived=True, **params_values_dict):

        k = self.k
        pk_lin_interp = self.provider.get_Pk_interpolator(nonlinear=False,var_pair=self.lin_pk_pairs[0])
        sigma8_z = self.provider.get_sigma8_z(self.z)
        D = sigma8_z/sigma8_z[0]
        h = self.provider.get_Hubble(0.0) / 100

        state['eft_spectrum_k'] = self.k
        state['eft_spectrum_grid'] = np.zeros((self.nz,
                                               self.nspec,
                                               self.nk))
        if self.kecleft:
            pk = pk_lin_interp.P(0, self.k * h) * h**3
            cleftobj = RKECLEFT(self.k, pk)

            for i, z in enumerate(self.z):
                cleftobj.make_ptable(D=D[i], kmin=k[0], kmax=k[-1], nk=1000)
                cleftpk = cleftobj.pktable.T
                state['eft_spectrum_grid'][i, ...] = cleftpk[1:self.nspec+1, :]

        else:
            for i, z in enumerate(self.z):
                pk = pk_lin_interp.P(z, self.k * h) * h**3
                cleftobj = CLEFT(self.k, pk, N=2700, jn=10, cutoff=1)
                cleftobj.make_ptable()
                cleftpk = cleftobj.pktable[:, 1:self.nspec+1].T

                # Different cutoff for other spectra, because otherwise different
                # large scale asymptote

                cleftobj = CLEFT(k, pk, N=2700, jn=5, cutoff=10)
                cleftobj.make_ptable()

                cleftpk[:, 2:] = cleftobj.pktable[:, 4:self.nspec+1].T
                state['eft_spectrum_grid'][i, ...] = cleftpk

        state['eft_spectrum_grid'][:, 1, :] /= 2
        state['eft_spectrum_grid'][:, 5, :] /= 0.25
        state['eft_spectrum_grid'][:, 6, :] /= 2
        state['eft_spectrum_grid'][:, 7, :] /= 2

        self.eft_spectrum_grid = state['eft_spectrum_grid']

        if self.get_eft_interpolator:
            state['eft_spectrum_interpolator'] = []

            for i in range(self.nspec):
                state['eft_spectrum_interpolator'].append(PowerSpectrumInterpolator(self.z,
                                                                                    self.k,
                                                                                    state['eft_spectrum_grid'][:, i, :]))

            self.eft_spectrum_interpolator = state['eft_spectrum_interpolator']

        if self.heft:
            cosmo_params = ['ombh2', 'omch2',
                            'w', 'ns', 'sigma8', 'H0', 'N_eff']
            cosmo = [self.provider.get_param(par) for par in cosmo_params]
            cosmo.append(1.0)
            cosmo = np.array(cosmo)

            X = np.tile(cosmo, len(self.z)).reshape((len(self.z), -1))
            a = 1/(1 + self.z)
            X[:, -1] = a

            emu_spec = self.emu.predict(
                self.k, X, spec_lpt=state['eft_spectrum_grid'])
            state['heft_spectrum_grid'] = emu_spec

            if self.get_heft_interpolator:
                state['heft_spectrum_interpolator'] = []

                for i in range(self.nspec):
                    state['heft_spectrum_interpolator'].append(PowerSpectrumInterpolator(self.z,
                                                                                         self.k,
                                                                                         state['heft_spectrum_grid'][:, i, :]))

