import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.typing import Union, Sequence, Optional
from cobaya.theories.cosmo import PowerSpectrumInterpolator
from itertools import permutations, product
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import yaml

from emulator import Emulator
from predict_cl import AngularPowerSpectra

datavector_requires = {'p0': ['z_fid', 'chiz_fid', 'hz_fid'],
                       'p2': ['z_fid', 'chiz_fid', 'hz_fid'],
                       'p4': ['z_fid', 'chiz_fid', 'hz_fid'],
                       'c_kk': [],
                       'c_dk': ['nz_d'],
                       'c_dd': ['nz_d']}


class HarmonicSpaceWLxRSD(Likelihood):
    # From yaml file.
    heft: bool
    datavector_info_filename: str
    zmin_proj: float
    zmax_proj: float
    nz_proj: int
    nchi_proj: int
    emulator_weights: Optional[Dict[str, str]]
    use_samples: Optional[Union[Sequence, np.ndarray]]

    # k = convergence
    # d = (galaxy) density
    # All assuming galaxy-CMB lensing
    # still need to implement galaxy-galaxy lensing

#    datavector_requires = {'p0': ['window_p0', 'z_fid', 'chiz_fid', 'hz_fid', 'k_min', 'k_max'],
#                           'p2': ['window_p2', 'z_fid', 'chiz_fid', 'hz_fid', 'k_min', 'k_max'],
#                           'p4': ['window_p4', 'z_fid', 'chiz_fid', 'hz_fid', 'k_min', 'k_max'],
#                           'c_kk': ['window_kk', 'ell_min', 'ell_max'],
#                           'c_dk': ['window_dk', 'nz_d', 'ell_min', 'ell_max'],
#                           'c_gd': ['window_dd', 'nz_d', 'ell_min', 'ell_max']}

    # ignore scale cuts and windows for now

    def initialize(self):
        """Sets up the class."""
        self.z = np.linspace(self.zmin_proj,
                             self.zmax_proj,
                             self.nz_proj)
        self.nk = 100
        self.k = np.logspace(-3, 0, self.nk)
        self.compute_c_kk = False
        self.compute_c_dk = False
        self.compute_c_dd = False
        self.compute_p0 = False
        self.compute_p2 = False
        self.compute_p4 = False
        self.load_data()
        self.setup_projection()

    def load_data(self):
        """Loads the required data."""

        with open(self.datavector_info_filename, 'r') as fp:
            datavector_info = yaml.load(fp)

        # must specify a file with the actual correlation functions
        # in it. Should have the following columns: datavector type
        # (e.g. p0), redshift bin number for each sample being
        # correlated, separation values (e.g. k, ell, etc.), and
        # actual values for the spectra/correlation functions
        dt = np.dtype([('spectrum_type', 'U10'), ('zbin0', np.int),
                       ('zbin1', np.int), ('separation', np.float), ('value', np.float)])

        self.spectra = np.genfromtxt(datavector_info['spectra_filename'],
                                     names=True, dtype=dt)
        self.spectrum_types = np.unique(self.spectra['spectrum_type'])
        self.spectrum_info = {}
        self.n_dv = len(self.spectra)
        self.compute_cell = False
        self.compute_pell = False

        if ('c_kk' in self.spectrum_types) | \
            ('c_dk' in self.spectrum_types) | \
                ('c_dd' in self.spectrum_types):
            self.compute_cell = True
        else:
            self.compute_cell = False

        if ('p0' in self.spectrum_types) | \
            ('p2' in self.spectrum_types) | \
                ('p4' in self.spectrum_types):
            self.compute_pell = True
        else:
            self.compute_pell = False

        requirements = []

        for t in self.spectrum_types:
            setattr(self, 'compute_{}'.format(t), True)
            idx = self.spectra['spectrum_type'] == t
            nbins0 = len(np.unique(self.spectra[idx]['zbin0']))
            nbins1 = len(np.unique(self.spectra[idx]['zbin1']))
            z0 = self.spectra['zbin0'][idx][0]
            z1 = self.spectra['zbin1'][idx][0]
            idx &= (self.spectra['zbin0'] == z0) & (
                self.spectra['zbin0'] == z1)
            ndv_per_bin = np.sum(idx)
            sep = self.spectra[idx]['separation']

            self.spectrum_info[t] = {'nbins0': nbins0, 'nbins1': nbins1,
                                     'ndv_per_bin': ndv_per_bin,
                                     'separation': sep}

            requirements.extend(datavector_requires[t])

        requirements = np.unique(np.array(requirements))

        for r in requirements:
            if r in ['z_fid', 'chiz_fid', 'hz_fid']:
                setattr(self, r, datavector_info[r])
            else:
                setattr(self, r, np.genfromtxt(
                    datavector_info[r], dtype=None, names=True))

        # Always need a covariance matrix. This should be a text file
        # with columns specifying the two data vector types, and four redshift
        # bin indices for each element, as well as a column for the elements
        # themselves
        dt = np.dtype([('spectrum_type0', 'U10'), ('spectrum_type1', 'U10'), ('zbin00', np.int),
                       ('zbin01', np.int), ('zbin10', np.int), ('zbin11', np.int), ('value', np.float)])
        cov_raw = np.genfromtxt(
            datavector_info['covariance_filename'], names=True, dtype=dt)

        self.cov = np.zeros((self.n_dv, self.n_dv))

        spec_type_counter = []
        zbin_counter = {}

        for t0, t1 in product(self.spectrum_types, self.spectrum_types):
            if (t0, t1) in spec_type_counter:
                continue

            if (t0, t1) not in zbin_counter.keys():
                zbin_counter[(t0, t1)] = []

            n00, n01, n10, n11 = np.arange(self.spectrum_info[t0]['nbins0']), \
                np.arange(self.spectrum_info[t0]['nbins1']), \
                np.arange(self.spectrum_info[t1]['nbins0']), \
                np.arange(self.spectrum_info[t1]['nbins1'])

            for zb00, zb01, zb10, zb11 in product(n00, n01, n10, n11):
                if ((zb00, zb01), (zb10, zb11)) in zbin_counter[(t0, t1)]:
                    continue

                idxi = (self.spectra['spectrum_type'] == t0) & \
                       (self.spectra['zbin0'] == zb00) & \
                       (self.spectra['zbin1'] == zb01)

                idxj = (self.spectra['spectrum_type'] == t1) & \
                       (self.spectra['zbin0'] == zb10) & \
                       (self.spectra['zbin1'] == zb11)

                covidx = (((cov_raw['spectrum_type0'] == t0) &
                           (cov_raw['zbin00'] == zb00) &
                           (cov_raw['zbin01'] == zb01) &
                           (cov_raw['spectrum_type1'] == t1) &
                           (cov_raw['zbin10'] == zb10) &
                           (cov_raw['zbin11'] == zb11)))  # |
#                          ((cov_raw['spectrum_type0'] == t0) & \
#                         (cov_raw['zbin00'] == zb00) & \
#                         (cov_raw['zbin01'] == zb01) & \
#                         (cov_raw['spectrum_type1'] == t1) & \
#                         (cov_raw['zbin10'] == zb10) & \
#                         (cov_raw['zbin11'] == zb11)))

                try:
                    self.cov[idxi, idxj] = cov_raw[covidx]['value']
                    zbin_counter[(t0, t1)].extend(
                        permutations(((zb00, zb01), (zb10, zb11))))
                except:
                    pass

            spec_type_counter.extend(permutations((t0, t1)))

        self.cinv = np.linalg.inv(self.cov)

        # check if we have emulators for these statistics
        if hasattr(self, 'emulator_weights'):
            self.emulators = {}
            for stat in self.emulator_weights:
                # Only need one emulator per real space
                # stat.
                if stat in ['p_mm', 'p_dm', 'p_dd']:
                    emu_path = self.emulator_weights[stat]
                    emu = Emulator(emu_path)
                    self.emulators[stat] = emu
                    self.cell_emulators = True

                elif stat in ['p0', 'p2', 'p4']:
                    self.emulators[stat] = []
                    for i in range(self.spectrum_info[stat]['nbins0']):
                        emu_path = self.emulator_weights[stat][i]
                        emu = Emulator(emu_path)
                        self.emulators[stat].append(emu)
                    self.pell_emulators = True

    def setup_projection(self):

        self.ndbins = self.spectrum_info['c_dk']['nbins0']
        if not hasattr(self, 'use_samples'):
            self.use_samples = np.arange(self.ndbins)

        self.projection_models = []

        for i in range(self.ndbins):
            if i not in self.use_samples:
                continue
            dndz = Spline(self.nz_d['z'],
                          self.nz_d['nz{}'.format(i)])(self.z)
            aps = AngularPowerSpectra(self.z, dndz,
                                      Nchi=self.nchi_proj)
            self.projection_models.append(aps)

    def get_requirements(self):

        reqs = {}
        if self.compute_cell:
            if not self.cell_emulators:
                if self.heft:
                    reqs.update({'heft_spectrum_interpolator': {'z': self.z},
                                 'heft_spectrum_grid': {'z': self.z}})
                else:
                    reqs.update({'eft_spectrum_interpolator': {'z': self.z}})
            else:
                reqs.update('logA': None, 'ns': None,
                            'H0': None, 'w': None,
                            'ombh2': None, 'omch2': None)

            z = np.concatenate([self.z, [1098.]])
            reqs.update({'comoving_radial_distance': {'z': z},
                         'Hubble': {'z': self.z},
                         'H0': None,
                         'omegam': None})

        if self.compute_pell:
            if not self.pell_emulators:
                reqs['pt_pk_ell_model'] = {}
                reqs['pt_pk_ell_model']['z'] = self.z_fid
                reqs['pt_pk_ell_model']['chiz_fid'] = self.chiz_fid
                reqs['pt_pk_ell_model']['hz_fid'] = self.hz_fid
            else:
                reqs.update('logA': None, 'ns': None,
                            'H0': None, 'w': None,
                            'ombh2': None, 'omch2': None)

        return reqs

    def combine_real_space_spectra(self, k, spectra, bias_params, cross=False):

        pkvec = np.zeros((14, spectra.shape[1], spectra.shape[2]))
        pkvec[:10, ...] = spectra
        # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]

        # Higher derivative terms
        pkvec[10:, ...] = -k[np.newaxis, :,
                             np.newaxis]**2 * pkvec[nabla_idx, ...]

        b1, b2, bs, bk2, sn = bias_params
        if not cross:
            bterms = [1,
                      2*b1, b1**2,
                      b2, b2*b1, 0.25*b2**2,
                      2*bs, 2*bs*b1, bs*b2, bs**2,
                      2*bk2, 2*bk2*b1, bk2*b2, 2*bk2*bs]
        else:
            # hm correlations only have one kind of <1,delta_i> correlation
            bterms = [1,
                      b1, 0,
                      b2/2, 0, 0,
                      bs, 0, 0, 0,
                      bk2, 0, 0, 0]

        p = np.einsum('b, bkz->kz', bterms, pkvec)

        if not cross:
            p += sn

        return p

    def combine_pkell_spectra(self, bvec, pktable):
        '''
        Returns pell, assuming AP parameters from input p{ell}ktable
        '''

        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = bvec
        bias_monomials = np.array([1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs,
                                   b2*bs, bs**2, b3, b1*b3, alpha0, alpha2,
                                   alpha4, alpha6, sn, sn2, sn4])
        p = np.sum(pktable * bias_monomials, axis=1)

        return p

    def logp(self, **params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""

        if self.compute_cell:

            # do I need a little h here?
            z = np.concatenate([self.z, [1098.]])
            chiz = self.provider.get_comoving_radial_distance(z)
            chistar = chiz[-1]
            chiz = chiz[:-1]
            omega_m = self.provider.get_param('omegam')

            ez = self.provider.get_Hubble(self.z) /\
                self.provider.get_param('H0')

            if not self.emulate_cell:
                if self.heft:
                    cell_spec_interpolator = self.provider.get_result(
                        'heft_spectrum_interpolator')
                else:
                    cell_spec_interpolator = self.provider.get_result(
                        'eft_spectrum_interpolator')

                n_spec_cell = len(cell_spec_interpolator)
                self.rs_power_spectra = np.zeros(
                    (self.ndbins, self.nz_proj, self.nk, 3))
            else:
                cosmo_params = [self.provider.get_param(p) for p in
                                ['logA', 'ns', 'H0', 'w', 'ombh2', 'omch2']]

                self.rs_power_spectra = np.zeros(
                    (self.ndbins, self.nz_proj, self.emulators['p_mm'].nk, 3))

            # only looping over lens bins for now, since
            # galaxy lensing, and lens x lens cross corr
            # not implemented yet
            for i in range(self.ndbins):
                if i not in self.use_samples:
                    continue
                b1 = params_values['b1_{}'.format(i)]
                b2 = params_values['b2_{}'.format(i)]
                bs = params_values['bs_{}'.format(i)]
                bk = params_values['bk_{}'.format(i)]
                sn = params_values['sn_{}'.format(i)]
                bias_params = [b1, b2, bs, bk, sn]
                if not self.emulate_cells:

                    spectra = np.zeros((n_spec_cell, self.nk, self.nz_proj))
                    for j in range(n_spec_cell):
                        spectra[j, ...] = cell_spec_interpolator[j].P(
                            self.z, self.k).T

                    pmm = spectra[0, ...]
                    pdm = self.combine_real_space_spectra(
                        self.k, spectra, bias_params, cross=True)
                    pdd = self.combine_real_space_spectra(
                        self.k, spectra, bias_params, cross=False)
                    k = self.k
                else:
                    params = cosmo_params + bias_params

                    cparam_grid = np.zeros(len(self.z), len(cosmo_params)+1)
                    param_grid = np.zeros(len(self.z), len(params)+1)

                    params = np.array(params)
                    cosmo_params = np.array(params)

                    cparam_grid[:, :-1] = cosmo_params
                    param_grid[:, :-1] = params
                    cparam_grid[:, -1] = self.z
                    param_grid[:, -1] = self.z

                    k, pmm = self.emulators['p_mm'](cparam_grid)
                    k, pdm = self.emulators['p_dm'](param_grid)
                    k, pdd = self.emulators['p_dd'](param_grid)

                pmm_spline = PowerSpectrumInterpolator(self.z, k, pmm.T)
                self.rs_power_spectra[i, :, :, 0] = pmm.T

                pdm_spline = PowerSpectrumInterpolator(self.z, k, pdm.T)
                self.rs_power_spectra[i, :, :, 1] = pdm.T

                pdd_spline = PowerSpectrumInterpolator(self.z, k, pdd.T)
                self.rs_power_spectra[i, :, :, 2] = pdd.T

                lval, Cdd, Cdk, Ckk = self.projection_models[i](
                    pmm_spline, pdm_spline,
                    pdd_spline,
                    chiz, ez, omega_m, chistar)

                # just interpolate onto desired ell for now.
                # need to implement windowing.
                if self.compute_c_kk:
                    Ckk_spline = interp1d(np.log10(lval), np.arcsinh(Ckk),
                                          fill_value='extrapolate')
                    self.spectrum_info['c_kk']['{}_0_model'.format(i)] = np.sinh(Ckk_spline(
                        np.log10(self.spectrum_info['c_kk']['separation'])))

                if self.compute_c_dk:
                    Cdk_spline = interp1d(np.log10(lval), np.arcsinh(Cdk),
                                          fill_value='extrapolate')
                    self.spectrum_info['c_dk']['{}_0_model'.format(i)] = np.sinh(Cdk_spline(
                        np.log10(self.spectrum_info['c_dk']['separation'])))

                if self.compute_c_dd:
                    Cdd_spline = interp1d(np.log10(lval), np.arcsinh(Cdd),
                                          fill_value='extrapolate')
                    self.spectrum_info['c_dd']['{}_0_model'.format(i)] = np.sinh(Cdd_spline(
                        np.log10(self.spectrum_info['c_dd']['separation'])))

        if self.compute_pell:

            if not self.pell_emulators:
                pt_models = self.provider.get_result(
                    'pt_pk_ell_model')

            for i in range(self.ndbins):
                if i not in self.use_samples:
                    continue
                b1 = params_values['b1_{}'.format(i)]
                b2 = params_values['b2_{}'.format(i)]
                bs = params_values['bs_{}'.format(i)]
                b3 = params_values['b3_{}'.format(i)]

                alpha0 = params_values['alpha0_{}'.format(i)]
                alpha2 = params_values['alpha2_{}'.format(i)]
                alpha4 = params_values['alpha4_{}'.format(i)]
                alpha6 = params_values['alpha6_{}'.format(i)]

                sn = params_values['sn_{}'.format(i)]
                sn2 = params_values['sn2_{}'.format(i)]
                sn4 = params_values['sn4_{}'.format(i)]

                bias_params = [b1, b2, bs, b3, alpha0, alpha2, alpha4,
                               alpha6, sn, sn2, sn4]

                # compute multipoles and interpolate onto desired k values
                # need to implement window/wide angle stuff still.
                if not self.pell_emulators:
                    lpt = pt_models[i]
                    k, p0, p2, p4 = lpt.combine_bias_terms_pkell(bias_params)
                else:
                    params = cosmo_params + bias_params

                    param_grid = np.zeros(len(self.z), len(params)+1)

                    params = np.array(params)
                    param_grid[:, :-1] = params
                    param_grid[:, -1] = self.z

                    k, p0 = self.emulators['p0'][i](param_grid)
                    k, p2 = self.emulators['p2'][i](param_grid)
                    k, p4 = self.emulators['p4'][i](param_grid)

                if not hasattr(self, 'pkell_spectra'):
                    self.pkell_spectra = np.zeros((self.ndbins, len(k), 3))

                self.pkell_spectra[i, :, 0] = p0
                self.pkell_spectra[i, :, 1] = p2
                self.pkell_spectra[i, :, 2] = p4

                if self.compute_p0:
                    p0_spline = interp1d(np.log10(k), np.arcsinh(p0),
                                         fill_value='extrapolate')
                    self.spectrum_info['p0']['{}_model'.format(i)] = np.sinh(p0_spline(
                        np.log10(self.spectrum_info['p0']['separation'])))

                if self.compute_p2:
                    p2_spline = interp1d(np.log10(k), np.arcsinh(p2),
                                         fill_value='extrapolate')
                    self.spectrum_info['p2']['{}_model'.format(i)] = np.sinh(p2_spline(
                        np.log10(self.spectrum_info['p2']['separation'])))

                if self.compute_p4:
                    p4_spline = interp1d(np.log10(k), np.arcsinh(p4),
                                         fill_value='extrapolate')
                    self.spectrum_info['p4']['{}_model'.format(i)] = np.sinh(p4_spline(
                        np.log10(self.spectrum_info['p4']['separation'])))

        # package everything up into one big model datavector
        model = []
        for t in self.spectrum_types:
            for i in range(self.ndbins):
                if i not in self.use_samples:
                    continue
                if t[0] == 'p':
                    model.append(self.spectrum_info[t]['{}_model'.format(i)])
                else:
                    model.append(self.spectrum_info[t]['{}_0_model'.format(i)])

        model = np.hstack(model)
        self.model_pred = model

        chi2 = np.dot(self.spectra['value'] - model,
                      np.dot(self.cinv, self.spectra['value'] - model))

        return(-0.5*chi2)
