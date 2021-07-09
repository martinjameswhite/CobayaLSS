#!/usr/bin/env python3
#
# Code to compute angular power spectra using Limber's approximation,
# ignoring higher-order corrections such as curved sky or redshift-space
# distortions (that predominantly affect low ell).
#
import numpy as np
import sys

from scipy.integrate   import simps
from scipy.interpolate import InterpolatedUnivariateSpline as Spline






class AngularPowerSpectra():
    """Computes angular power spectra using the Limber approximation."""
    def lagrange_spam(self,z):
        """Returns the weights to apply to each z-slice to interpolate to z.
           -- Not currently used, could be used if need P(k,z)."""
        dz = self.zlist[:,None] - self.zlist[None,:]
        singular = (dz == 0)
        dz[singular] = 1.0
        fac = (z - self.zlist) / dz
        fac[singular] = 1.0
        return(fac.prod(axis=-1))
        #
    def T_AK(self,x):
        """The "T" function of Adachi & Kasai (2012), used below."""
        b1,b2,b3=2.64086441,0.883044401,0.0531249537
        c1,c2,c3=1.39186078,0.512094674,0.0394382061
        x3   = x**3
        x6   = x3*x3
        x9   = x3*x6
        tmp  = 2+b1*x3+b2*x6+b3*x9
        tmp /= 1+c1*x3+c2*x6+c3*x9
        tmp *= x**0.5
        return(tmp)
        #
    def chi_of_z(self,zz):
        """The comoving distance to redshift zz, in Mpc/h.
           Uses the Pade approximate of Adachi & Kasai (2012) to compute chi
           for a LCDM model, ignoring massive neutrinos."""
        s_ak = (self.OmX/self.OmM)**0.3333333
        tmp  = self.T_AK(s_ak)-self.T_AK(s_ak/(1+zz))
        tmp *= 2997.925/(s_ak*self.OmM)**0.5
        return(tmp)
        #
    def E_of_z(self,zz):
        """The dimensionless Hubble parameter at zz."""
        Ez = (self.OmM*(1+zz)**3 + self.OmX)**0.5
        return(Ez)
        #
    def mag_bias_kernel(self,s,Nchi_mag=101):
        """Returns magnification bias kernel if 's' is the slope of
           the number counts dlog10N/dm."""
        zval    = self.zchi(self.chival)
        cmax    = np.max(self.chival) * 1.1
        zupper  = lambda x: np.linspace(x,cmax,Nchi_mag)
        chivalp = np.array(list(map(zupper,self.chival))).transpose()
        zvalp   = self.zchi(chivalp)
        dndz_n  = np.interp(zvalp,self.zz,self.dndz,left=0,right=0)
        Ez      = self.E_of_z(zvalp)
        g       = (chivalp-self.chival[np.newaxis,:])/chivalp
        g      *= dndz_n*Ez/2997.925
        g       = self.chival * simps(g,x=chivalp,axis=0)
        mag_kern= 1.5*(self.OmM)/2997.925**2*(1+zval)*g*(5*s-2.)
        return(mag_kern)
        #
    def shot3to2(self):
        """Returns the conversion from 3D shotnoise to 2D shotnoise power."""
        Cshot = self.fchi**2/self.chival**2
        Cshot = simps(Cshot,x=self.chival)
        return(Cshot)
    def __init__(self,OmM,dndz,Nchi=101,Nz=251):
        """Set up the class.
            OmM:  The value of Omega_m(z=0) for the cosmology.
            dndz: A numpy array (Nbin,2) containing dN/dz vs. z."""
        # Copy the arguments, setting up the z-range.
        self.pofk = None
        self.Nchi = Nchi
        self.OmM  = OmM
        self.OmX  = 1.0-OmM
        self.zmin = dndz[ 0,0]
        self.zmax = dndz[-1,0]
        self.zz   = np.linspace(self.zmin,self.zmax,Nz)
        self.dndz = Spline(dndz[:,0],dndz[:,1])(self.zz)
        # Normalize dN/dz.
        self.dndz = self.dndz/simps(self.dndz,x=self.zz)
        # Set up the chi(z) array and z(chi) spline.
        self.chiz = np.array([self.chi_of_z(z) for z in self.zz])
        self.zchi = Spline(self.chiz,self.zz)
        # Work out W(chi) for the objects whose dNdz is supplied.
        chimin    = np.min(self.chiz) + 1e-5
        chimax    = np.max(self.chiz)
        self.chival= np.linspace(chimin,chimax,self.Nchi)
        zval      = self.zchi(self.chival)
        self.fchi = Spline(self.zz,self.dndz*self.E_of_z(self.zz))(zval)
        self.fchi/= simps(self.fchi,x=self.chival)
        # and W(chi) for the CMB
        self.chistar= self.chi_of_z(1098.)
        self.fcmb = 1.5*self.OmM*(1.0/2997.925)**2*(1+zval)
        self.fcmb*= self.chival*(self.chistar-self.chival)/self.chistar
        # Compute the effective redshift.
        self.zeff = simps(zval*self.fchi**2/self.chival**2,x=self.chival)
        self.zeff/= simps(     self.fchi**2/self.chival**2,x=self.chival)
        #
    def __call__(self,PggEmu,PgmEmu,PmmEmu,pk_pars,smag=0.4,Nell=64,Lmax=1001):
        """Computes C_l^{gg} and C_l^{kg} given emulators for P_{ij}."""
        fmag   = self.mag_bias_kernel(smag) # Magnification bias kernel.
        ell    = np.logspace(1,np.log10(Lmax),Nell) # More ell's are cheap.
        Cgg,Ckg= np.zeros( (Nell,self.Nchi) ),np.zeros( (Nell,self.Nchi) )
        Pgg    = Spline(*PggEmu(pk_pars+[self.zeff]))     # Extrapolates.
        Pgm    = Spline(*PgmEmu(pk_pars+[self.zeff]))     # Extrapolates.
        Pmm    = Spline(*PmmEmu(pk_pars[:6]+[self.zeff])) # Extrapolates.
        # Work out the integrands for C_l^{gg} and C_l^{kg}.
        for i,chi in enumerate(self.chival):
            kval     = (ell+0.5)/chi        # The vector of k's.
            f1f2     = self.fchi[i]*self.fchi[i]/chi**2 * Pgg(kval)
            f1m2     = self.fchi[i]*     fmag[i]/chi**2 * Pgm(kval)
            m1f2     =      fmag[i]*self.fchi[i]/chi**2 * Pgm(kval)
            m1m2     =      fmag[i]*     fmag[i]/chi**2 * Pmm(kval)
            Cgg[:,i] = f1f2 + f1m2 + m1f2 + m1m2
            f1f2     = self.fchi[i]*self.fcmb[i]/chi**2 * Pgm(kval)
            m1f2     =      fmag[i]*self.fcmb[i]/chi**2 * Pmm(kval)
            Ckg[:,i] = f1f2 + m1f2
        # and then just integrate them.
        Cgg = simps(Cgg,x=self.chival,axis=-1)
        Ckg = simps(Ckg,x=self.chival,axis=-1)
        # Now interpolate onto a regular ell grid.
        lval= np.arange(Lmax)
        Cgg = Spline(ell,Cgg)(lval)
        Ckg = Spline(ell,Ckg)(lval)
        return( (lval,Cgg,Ckg) )
        #

