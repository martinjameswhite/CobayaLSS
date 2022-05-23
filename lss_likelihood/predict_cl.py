#!/usr/bin/env python3
#
# Code to compute angular power spectra using Limber's approximation.
#
import numpy as np
from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


class AngularPowerSpectra():
    """Computes angular power spectra using the Limber approximation."""

    def compute_mag_bias_kernels(self, Nchi=101):
        """Returns magnification bias kernel if 's' is the slope of
           the number counts dlog10N/dm."""

        self.w_mag = [np.zeros(Nchi)] * self.n_lens

        cmax = np.max(self.chival) * 1.1
        def zupper(x): return np.linspace(x, cmax, Nchi)
        chivalp = np.array(list(map(zupper, self.chival))).transpose()
        zvalp = self.zchi(chivalp)

        for i in range(self.n_lens):
            dndz_n = np.interp(
                zvalp, self.zz, self.dndz_lens[i], left=0, right=0)
            Ez = self.E_of_z(zvalp)
            g = (chivalp-self.chival[np.newaxis, :])/chivalp
            g *= dndz_n*Ez/2997.925
            g = self.chival * simps(g, x=chivalp, axis=0)
            self.w_mag[i] = self.smag[i] * 1.5*(self.OmM)/2997.925**2*(1+self.zval)*g

    def compute_galaxy_convergence_kernels(self, Nchi=101):
        self.w_k = [np.zeros(Nchi)] * self.n_source

        cmax = np.max(self.chival) * 1.1
        def zupper(x): return np.linspace(x, cmax, Nchi)
        chivalp = np.array(list(map(zupper, self.chival))).transpose()
        zvalp = self.zchi(chivalp)

        for i in range(self.n_source):
            dndz_n = np.interp(
                zvalp, self.zz, self.dndz_source[i], left=0, right=0)
            Ez = self.E_of_z(zvalp)
            g = (chivalp-self.chival[np.newaxis, :])/chivalp
            g *= dndz_n*Ez/2997.925
            g = self.chival * simps(g, x=chivalp, axis=0)
            self.w_k[i] = 1.5*(self.OmM)/2997.925**2*(1+self.zval)*g

    def compute_galaxy_density_kernels(self, Nchi=101):
        """Returns magnification bias kernel not including 's'
           (the slope of the number counts dlog10N/dm)"""

        self.w_d = [np.zeros(Nchi)] * self.n_lens
        self.zeff_d = [None] * self.n_lens
        ez = self.E_of_z(self.zz)

        for i in range(self.n_lens):
            self.w_d[i] = Spline(self.zz, self.dndz_lens[i] * ez)(self.zval)
            self.w_d[i] /= simps(self.w_d[i], x=self.chival)
            self.zeff_d[i] = simps(self.zval*self.w_d[i]
                                   ** 2/self.chival**2, x=self.chival)
            self.zeff_d[i] /= simps(self.w_d[i]**2 /
                                    self.chival**2, x=self.chival)

    def compute_galaxy_ia_kernels(self, Nchi=101):
        """Returns magnification bias kernel not including 's'
           (the slope of the number counts dlog10N/dm)"""

        self.w_ia = [None] * self.n_source
        ez = self.E_of_z(self.zz)

        for i in range(self.n_source):
            self.w_ia[i] = Spline(self.zz, self.dndz_source[i] * ez)(self.zval)
            self.w_ia[i] /= simps(self.w_ia[i], x=self.chival)
            self.w_ia[i] *= self.a_ia * (1 + self.zval) / (1 +
                                 0.62) ** self.eta_ia * 0.0134 / self.Dz(self.zval)

    def compute_cmb_convergence_kernel(self, Nchi=101):
        self.w_cmbk = 1.5*self.OmM*(1.0/2997.925)**2*(1+self.zval)
        self.w_cmbk *= self.chival*(self.chistar-self.chival)/self.chistar

    def __init__(self, z, dndz_lens, dndz_source, Nchi=101,
                 d_x_cmbk=True, cmbk_x_cmbk=True):
        """
        Set up the class.
        dndz_lens: A numpy array containing dN/dz vs z for all lens bins.
        dndz_source: A numpy array containing dN/dz vs z for all source bins.
        """
        # Copy the arguments, setting up the z-range.
        self.zz = z
        self.dndz_lens = dndz_lens
        self.dndz_source = dndz_source
        self.compute_d_x_cmbk = d_x_cmbk
        self.compute_cmbk_x_cmbk = cmbk_x_cmbk

        if self.dndz_lens is not None:
            self.n_lens = len(self.dndz_lens)
            for i in range(self.n_lens):
                self.dndz_lens[i] = self.dndz_lens[i]/simps(self.dndz_lens[i], x=self.zz, axis=-1)
        else:
            self.n_lens = 0
        
        if self.dndz_source is not None:
            self.n_source = len(dndz_source)
            for i in range(self.n_source):
                self.dndz_source[i] = self.dndz_source[i] / simps(self.dndz_source[i], x=self.zz, axis=-1)
        else:
            self.n_source = 0
        
        self.chistar = None
        self.Nchi = Nchi

    def set_geometry(self, chiz, ez, omega_m, chistar):
        self.OmM = omega_m
        # Set up the chi(z) array and z(chi) spline.
        self.chiz = chiz
        self.zchi = Spline(self.chiz, self.zz)
        self.E_of_z = Spline(self.zz, ez)
        # Work out W(chi) for the objects whose dNdz is supplied.
        chimin = np.min(self.chiz) + 1e-5
        chimax = np.max(self.chiz)
        self.chival = np.linspace(chimin, chimax, self.Nchi)
        self.zval = self.zchi(self.chival)
        self.chistar = chistar

        self.compute_cmb_convergence_kernel()
        self.compute_galaxy_density_kernels()
        self.compute_galaxy_convergence_kernels()
        self.compute_mag_bias_kernels()
        self.compute_galaxy_ia_kernels()

    def __call__(self, Pmm, Pgm, Pgg,
                 chiz, ez, omega_m, chistar, Dz, smag,
                 a_ia, eta_ia, Nell=50, Lmax=1001):
        """Computes C_l^{gg} and C_l^{kg}."""

        self.a_ia = a_ia
        self.eta_ia = eta_ia
        self.Dz = Spline(self.zz, Dz)
        self.smag = (5*smag-2.)        
        
        # recompute kernels if cosmology changed
        if chistar != self.chistar:
            self.set_geometry(chiz, ez, omega_m, chistar)
        

        ell = np.logspace(1, np.log10(Lmax), Nell)  # More ell's are cheap.
        Cdd = np.zeros((Nell, self.n_lens, self.Nchi))
        Cdk = np.zeros((Nell, self.n_lens * self.n_source, self.Nchi))
        Ckk = np.zeros((Nell, self.n_source * self.n_source, self.Nchi))
        Cdcmbk = np.zeros((Nell, self.n_lens, self.Nchi))
        Ccmbkcmbk = np.zeros((Nell, self.Nchi))

        for i, chi in enumerate(self.chival):
            kval = (ell+0.5)/chi            
            Ccmbkcmbk[:, i] = self.w_cmbk[i]**2 / \
                chi**2 * Pmm.P(self.zval[i], kval)

        for i in range(self.n_lens):
            for j, chi in enumerate(self.chival):
                kval = (ell+0.5)/chi
                f1f2 = self.w_d[i][j]*self.w_d[i][j]/chi**2 * Pgg[i].P(self.zeff_d[i], kval)
                f1m2 = self.w_d[i][j] * self.w_mag[i][j]/chi**2 * Pgm[i].P(self.zeff_d[i], kval)
                m1f2 = self.w_mag[i][j] * self.w_d[i][j] / chi**2 * Pgm[i].P(self.zeff_d[i], kval)
                m1m2 = self.w_mag[i][j]**2 / chi**2 * Pmm.P(self.zval[j], kval)
                Cdd[:, i, j] = f1f2 + f1m2 + m1f2 + m1m2

        for i in range(self.n_lens):
            for j, chi in enumerate(self.chival):
                kval = (ell+0.5)/chi
                f1f2 = self.w_d[i][j] * self.w_cmbk[j] / chi**2 * Pgm[i].P(self.zeff_d[i], kval)
                m1f2 = self.w_mag[i][j] * self.w_cmbk[i] / chi**2 * Pmm.P(self.zval[j], kval)
                Cdcmbk[:, i, j] = f1f2 + m1f2

        for i in range(self.n_lens):
            for j in range(self.n_source):
                for k, chi in enumerate(self.chival):
                    kval = (ell+0.5)/chi
                    f1f2 = self.w_d[i][k] * self.w_k[j][k] / chi**2 * Pgm[i].P(self.zeff_d[i], kval)
                    f1i2 = self.w_d[i][k] * self.w_ia[j][k] / chi**2 * Pgm[i].P(self.zeff_d[i], kval)
                    f1m2 = self.w_d[j][k] * self.w_mag[j][k] / chi**2 * Pgm[i].P(self.zeff_d[i], kval)
                    m1i2 = self.w_mag[i][k] * self.w_ia[j][k] / chi**2 * Pmm.P(self.zval[k], kval)
                    Cdk[:, i * self.n_source + j, k] = f1f2 + f1i2 + f1m2 + m1i2

        for i in range(self.n_source):
            for j in range(self.n_source):
                for k, chi in enumerate(self.chival):
                    kval = (ell + 0.5) / chi
                    f1f2 = self.w_k[i][k] * self.w_k[j][k] / chi**2 * Pmm.P(self.zval[k], kval)
                    f1i2 = self.w_k[i][k] * self.w_ia[j][k] / chi**2 * Pmm.P(self.zval[k], kval)
                    i1f2 = self.w_ia[i][k] * self.w_k[j][k] / chi**2 * Pmm.P(self.zval[k], kval)
                    i1i2 = self.w_ia[i][k] * self.w_ia[j][k] / chi**2 * Pmm.P(self.zval[k], kval)
                    Ckk[:, i * self.n_source + j, k] = f1f2 + f1i2 + i1f2 + i1i2


        # and then just integrate them.

        Ccmbkcmbk = simps(Ccmbkcmbk, x=self.chival, axis=-1)        
        if self.n_lens > 0:
            Cdd = simps(Cdd, x=self.chival, axis=-1).reshape(-1, self.n_lens)
            Cdcmbk = simps(Cdcmbk, x=self.chival, axis=-1).reshape(-1, self.n_lens)            
            if self.n_source > 0:
                Cdk = simps(Cdk, x=self.chival, axis=-1).reshape(-1, self.n_lens * self.n_source)
            else:
                Cdk = None
        else:
            Cdd = None
            Cdcmbk = None

        if self.n_source > 0:
            Ckk = simps(Ckk, x=self.chival, axis=-1).reshape(-1, self.n_source * self.n_source)
        else:
            Ckk = None

        # Now interpolate onto a regular ell grid.
        lval = np.arange(Lmax)
        if Cdd is not None:
            Cdd = Spline(ell, Cdd)(lval).reshape(-1, self.n_lens)
        if Cdk is not None:
            Cdk = Spline(ell, Cdk)(lval).reshape(-1, self.n_lens * self.n_source)
        if Ckk is not None:
            Ckk = Spline(ell, Ckk)(lval).reshape(-1, self.n_source * self.n_source)
        if Cdcmbk is not None:
            Cdcmbk = Spline(ell, Cdcmbk)(lval).reshape(-1, self.n_lens)
        if Ccmbkcmbk is not None:
            Ccmbkcmbk = Spline(ell, Ccmbkcmbk)(lval)

        return((lval, Cdd, Cdk, Ckk, Cdcmbk, Ccmbkcmbk))
