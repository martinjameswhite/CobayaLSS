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
from scipy.special     import hyp2f1


def D_of_z(OmM,zz):
    """Scale-independent growth factor for flat LCDM."""
    aa = 1./(1.+zz)
    rr = (1-OmM)/OmM
    t1 = hyp2f1(1./3,1,11./6,-aa**3*rr)
    t2 = hyp2f1(1./3,1,11./6,-rr)
    return( aa * t1/t2 )
    #





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
    def mag_bias_kernel(self,s,Nchi_mag=101):
        """Returns magnification bias kernel if 's' is the slope of
           the number counts dlog10N/dm."""
        zval    = self.zchi(self.chival)
        cmax    = np.max(self.chival) * 1.1
        zupper  = lambda x: np.linspace(x,cmax,Nchi_mag)
        chivalp = np.array(list(map(zupper,self.chival))).transpose()
        zvalp   = self.zchi(chivalp)
        dndz_n  = np.interp(zvalp,self.zz,self.dndz,left=0,right=0)
        Ez      = self.Eofz(zvalp)
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
    def __init__(self,OmM,chi_of_z,E_of_z,dndz,Nchi=201,Nz=251):
        """Set up the class.
            OmM:  The value of Omega_m(z=0) for the cosmology.
            chi_of_z: A function returning radial distance in Mpc/h given z.
            E_of_z: A function returning H(z)/H(0) given z.
            dndz: A numpy array (Nbin,2) containing dN/dz vs. z."""
        # Copy the arguments, setting up the z-range.
        self.Eofz = E_of_z
        self.Nchi = Nchi
        self.OmM  = OmM
        self.OmX  = 1.0-OmM
        self.zmin = np.min([0.05,dndz[0,0]])
        self.zmax = dndz[-1,0]
        self.zz   = np.linspace(self.zmin,self.zmax,Nz)
        self.dndz = Spline(dndz[:,0],dndz[:,1],ext=1)(self.zz)
        # Normalize dN/dz.
        self.dndz = self.dndz/simps(self.dndz,x=self.zz)
        # Set up the chi(z) array and z(chi) spline.
        self.chiz = chi_of_z(self.zz)
        self.zchi = Spline(self.chiz,self.zz)
        # Work out W(chi) for the objects whose dNdz is supplied.
        chimin    = np.min(self.chiz)
        chimax    = np.max(self.chiz)
        self.chival= np.linspace(chimin,chimax,Nchi)
        zval      = self.zchi(self.chival)
        self.fchi = Spline(self.zz,self.dndz*E_of_z(self.zz))(zval)
        self.fchi/= simps(self.fchi,x=self.chival)
        # and W(chi) for the CMB
        self.chistar= chi_of_z(1098.)
        self.fcmb = 1.5*self.OmM*(1.0/2997.925)**2*(1+zval)
        self.fcmb*= self.chival*(self.chistar-self.chival)/self.chistar
        # Compute the effective redshift.
        self.zeff = simps(zval*self.fchi**2/self.chival**2,x=self.chival)
        self.zeff/= simps(     self.fchi**2/self.chival**2,x=self.chival)
        # and save linear growth.
        self.ld2  = (D_of_z(self.OmM,zval)/D_of_z(self.OmM,self.zeff))**2
        #
    def __call__(self,PggEmu,PgmEmu,PmmEmu,cpars,bparsA,bparsX,\
                 smag=0.4,Nell=64,Lmax=1001):
        """Computes C_l^{gg} and C_l^{kg} given emulators for P_{ij} and
           cosmological parameters (cpars) plus bias params for auto (bparsA)
            and cross (bparsX) and the magnification slope (smag)."""
        # Set up arrays to hold kernels for C_l.
        ell    = np.logspace(1,np.log10(Lmax),Nell) # More ell's are cheap.
        Cgg,Ckg= np.zeros( (Nell,self.Nchi) ),np.zeros( (Nell,self.Nchi) )
        # The magnification bias kernel.
        fmag   = self.mag_bias_kernel(smag)
        # Fit splines to our P(k).  The spline extrapolates as needed.
        pars   = np.array(cpars+bparsA+[self.zeff])
        Pgg    = Spline(*PggEmu(pars))
        pars   = np.array(cpars+bparsX+[self.zeff])
        Pgm    = Spline(*PgmEmu(pars))
        pars   = np.array(cpars+[self.zeff])
        Pmm    = Spline(*PmmEmu(pars),ext=1) # Extrapolate with zeros.
        # Work out the integrands for C_l^{gg} and C_l^{kg}.
        for i,chi in enumerate(self.chival):
            kval     = (ell+0.5)/chi        # The vector of k's.
            f1f2     = self.fchi[i]*self.fchi[i]/chi**2 * Pgg(kval)
            f1m2     = self.fchi[i]*     fmag[i]/chi**2 * Pgm(kval)
            m1f2     =      fmag[i]*self.fchi[i]/chi**2 * Pgm(kval)
            m1m2     =      fmag[i]*     fmag[i]/chi**2 * Pmm(kval)*self.ld2[i]
            Cgg[:,i] = f1f2 + f1m2 + m1f2 + m1m2
            f1f2     = self.fchi[i]*self.fcmb[i]/chi**2 * Pgm(kval)
            m1f2     =      fmag[i]*self.fcmb[i]/chi**2 * Pmm(kval)*self.ld2[i]
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

