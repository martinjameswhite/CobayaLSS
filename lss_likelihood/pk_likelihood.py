import numpy as np
from cobaya.likelihood import Likelihood
from cobaya.typing import Sequence
import sys


class SimPkLikelihood(Likelihood):
    # From yaml file.
    datafile: str
    covfile: str
    zfid:  float
    dataset: Sequence[str]
    use_heft: bool
    kmin: float
    kmax: float


    def initialize(self):
        """Sets up the class."""
        self.loadData()

    def loadData(self):
        """Loads the required data."""
        data = np.loadtxt(self.datafile)
        data = data.T

        self.k = data[:, 0]
        min_idx = self.k.searchsorted(self.kmin)
        max_idx = self.k.searchsorted(self.kmax)
        self.k = self.k[min_idx:max_idx]
        self.p_gg = data[min_idx:max_idx, 1]
        self.p_gm = data[min_idx:max_idx, 2]
        self.p_mm = data[min_idx:max_idx, -1]

        datavec = []

        for dset in self.dataset:
            if dset == 'p_gg':
                datavec.append(self.p_gg)
            elif dset == 'p_gm':
                datavec.append(self.p_gm)
            elif dset == 'p_mm':
                datavec.append(self.p_mm)

        self.datavec = np.hstack(datavec)

        self.ndv = len(self.datavec)

        #self.cov = np.loadtxt(self.covfile)
        #self.cov = self.cov[:self.ndv, :self.ndv]
        #dummy cov for now
#        self.cov = np.diag(np.ones(self.ndv))
        dk = self.k[1:] - self.k[:-1]
        dk = np.hstack([dk, np.array([dk[-1]])])
        dk = np.hstack([dk, dk])
        vol = 1000**3
        self.cov = 2 * np.pi**2 / (np.hstack([self.k, self.k]) * dk * vol) * 2 * np.diag(self.datavec**2)
        self.cinv = np.linalg.inv(self.cov)

    def get_requirements(self):

        if not self.use_heft:
            return {'pt_spectrum_interpolator': None}
        else:
            raise(NotImplementedError)

    def combine_pt_spectra(self, k, spectra, bias_params, cross=False):

        pkvec = np.zeros((len(k), 14))
        pkvec[:, :10] = spectra
#        print(pkvec.shape)
        # IDs for the <nabla^2, X> ~ -k^2 <1, X> approximation.
        nabla_idx = [0, 1, 3, 6]

        # Higher derivative terms
        pkvec[:, 10:] = -k[:,np.newaxis]**2 * pkvec[:,nabla_idx]

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

        p = np.einsum('b, kb->k', bterms, pkvec)

        if not cross:
            p += sn

        return p

    def logp(self, **params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""

        sys.stdout.flush()

        pt_spec_interpolator = self.provider.get_result('pt_spectrum_interpolator')
        n_spec = len(pt_spec_interpolator)
        pt_spectra = np.zeros((len(self.k), n_spec))

        for i in range(n_spec):
            pt_spectra[:, i] = pt_spec_interpolator[i].P(self.zfid, self.k)

        b1 = params_values['b1']
        b2 = params_values['b2']
        bs = params_values['bs']
        bk = params_values['bk']
        sn = params_values['sn']

        bias_params = [b1, b2, bs, bk, sn]

        model = []
        for dset in self.dataset:
            if dset == 'p_gg':
                p_gg = self.combine_pt_spectra(self.k, pt_spectra, bias_params)
                model.append(p_gg)

            elif dset == 'p_gm':
                p_gm = self.combine_pt_spectra(self.k, pt_spectra, bias_params,
                                               cross=True)
                model.append(p_gm)

            elif dset == 'p_mm':
                model.append(pt_spectra[:, 0])

        model = np.hstack(model)

        chi2 = np.dot(self.datavec - model,
                      np.dot(self.cinv, self.datavec - model))

        return(-0.5*chi2)
