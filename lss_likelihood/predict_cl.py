#!/usr/bin/env python3
#
# Code to compute angular power spectra using Limber's approximation.
#
import numpy as np
import sys

from scipy.integrate import simps
from scipy.interpolate import InterpolatedUnivariateSpline as Spline


class AngularPowerSpectra():
    """Computes angular power spectra using the Limber approximation."""

    def lagrange_spam(self, z):
        """Returns the weights to apply to each z-slice to interpolate to z.
           -- Not currently used, could be used if need P(k,z)."""
        dz = self.zlist[:, None] - self.zlist[None, :]
        singular = (dz == 0)
        dz[singular] = 1.0
        fac = (z - self.zlist) / dz
        fac[singular] = 1.0
        return(fac.prod(axis=-1))

    def mag_bias_kernel(self, s, Nchi_mag=101):
        """Returns magnification bias kernel if 's' is the slope of
           the number counts dlog10N/dm."""
        zval = self.zchi(self.chival)
        cmax = np.max(self.chival) * 1.1
        def zupper(x): return np.linspace(x, cmax, Nchi_mag)
        chivalp = np.array(list(map(zupper, self.chival))).transpose()
        zvalp = self.zchi(chivalp)
        dndz_n = np.interp(zvalp, self.zz, self.dndz, left=0, right=0)
        Ez = self.E_of_z(zvalp)
        g = (chivalp-self.chival[np.newaxis, :])/chivalp
        g *= dndz_n*Ez/2997.925
        g = self.chival * simps(g, x=chivalp, axis=0)
        self.mag_kern = 1.5*(self.OmM)/2997.925**2*(1+zval)*g

    def __init__(self, z, dndz, Nchi=101):
        """Set up the class.
            dndz: A numpy array containing dN/dz vs at z."""
        # Copy the arguments, setting up the z-range.
        self.zz = z
        self.dndz = dndz
        # Normalize dN/dz.
        self.dndz = self.dndz/simps(self.dndz, x=self.zz)
        self.chistar = None

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
        self.fchi = Spline(self.zz, self.dndz * ez)(self.zval)
        self.fchi /= simps(self.fchi, x=self.chival)
        # and W(chi) for the CMB
        self.chistar = chistar
        self.fcmb = 1.5*self.OmM*(1.0/2997.925)**2*(1+self.zval)
        self.fcmb *= self.chival*(self.chistar-self.chival)/self.chistar
        # Compute the effective redshift.
        self.zeff = simps(self.zval*self.fchi**2/self.chival**2, x=self.chival)
        self.zeff /= simps(self.fchi**2/self.chival**2, x=self.chival)

    def __call__(self, Pmm, Pgm, Pgg,
                 chiz, ez, omega_m, chistar, smag=0.4,
                 Nell=50, Lmax=1001):
        """Computes C_l^{gg} and C_l^{kg}."""

        #recompute kernels if cosmology changed
        if chistar != self.chistar:
            self.setup_geometry(chiz, ez, omega_m, chistar)
            self.mag_bias_kernel(smag)  # Magnification bias kernel.

        fmag = self.mag_kern * (5*smag-2.)
        ell = np.logspace(1, np.log10(Lmax), Nell)  # More ell's are cheap.
        Cgg, Ckg, Ckk = np.zeros(
            (Nell, self.Nchi)), np.zeros((Nell, self.Nchi))

        # Work out the integrands for C_l^{gg} and C_l^{kg}.
        for i, chi in enumerate(self.chival):
            kval = (ell+0.5)/chi        # The vector of k's.
            f1f2 = self.fchi[i]*self.fchi[i]/chi**2 * Pgg(self.zval[i],kval)
            f1m2 = self.fchi[i] * fmag[i]/chi**2 * Pgm(self.zval[i],kval)
            m1f2 = fmag[i]*self.fchi[i]/chi**2 * Pgm(self.zval[i],kval)
            m1m2 = fmag[i] * fmag[i]/chi**2 * Pmm(self.zval[i],kval)
            Cgg[:, i] = f1f2 + f1m2 + m1f2 + m1m2
            f1f2 = self.fchi[i]*self.fcmb[i]/chi**2 * Pgm(self.zval[i],kval)
            m1f2 = fmag[i]*self.fcmb[i]/chi**2 * Pmm(self.zval[i],kval)
            Ckg[:, i] = f1f2 + m1f2
            Ckk[:, i] = self.fcmb[i]*self.fcmb[i]/chi**2 * Pmm(self.zval[i],kval)

        # and then just integrate them.
        Cgg = simps(Cgg, x=self.chival, axis=-1)
        Ckg = simps(Ckg, x=self.chival, axis=-1)
        Ckk = simps(Ckk, x=self.chival, axis=-1)

        # Now interpolate onto a regular ell grid.
        lval = np.arange(Lmax)
        Cgg = Spline(ell, Cgg)(lval)
        Ckg = Spline(ell, Ckg)(lval)
        Ckk = Spline(ell, Ckk)(lval)

        return((lval, Cgg, Ckg, Ckk))
        #
