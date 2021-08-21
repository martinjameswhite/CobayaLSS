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

# Model the (real-space) power spectrum using LPT.
from velocileptors.LPT.cleft_fftw import CLEFT
# Model the (real-space) power spectrum using Anzu.
from anzu.emu_funcs import LPTEmulator



class LPTPowerSpectra():
    """Computes the (real-space) power spectrum, P(k,z) [Mpc/h units]."""
    def calculate_pk(self,b1,b2,bs,b3,alpha_a,alpha_x,sn):
        """Returns k,Pgg,Pgm,Pmm using pre-computed CLEFT kernels."""
        cleft  = self.cleft
        kk,pgg = cleft.combine_bias_terms_pk(b1,b2,bs,b3,alpha_a,sn)
        kk,pgm = cleft.combine_bias_terms_pk_crossmatter(b1,b2,bs,b3,alpha_x)
        return((kk,pgg,pgm,self.cleft.pktable[:,1]))
        #
    def __init__(self,klin,plin):
        """klin,plin: Arrays containing Plinear [Mpc/h units]."""
        # Copy the arguments.
        self.klin = klin
        self.plin = plin
        # Set up the CLEFT class -- this can take a little while.
        self.cleft= CLEFT(self.klin,self.plin)
        self.cleft.make_ptable(nk=250)
        #
    def __call__(self,pk_pars):
        """Computes the three P(k,z).  Returns k,Pgg,Pgm,Pmm [Mpc/h units]."""
        return(self.calculate_pk(*pk_pars))
        #





class HalobridPowerSpectra():
    """Computes the (real-space) power spectrum, P(k,z) [Mpc/h units]."""
    def combine_bias_terms_pk_gg(self,b1,b2,bs,sn):
        """A CLEFT helper function to return P_{gg}."""
        kv  = self.cleft.pktable[:,0]
        ret = (1+b1)**2*self.hf +\
              1.*b2*self.cleft.pktable[:, 4]+\
              b1*b2*self.cleft.pktable[:, 5]+\
              b2*b2*self.cleft.pktable[:, 6]+\
              1.*bs*self.cleft.pktable[:, 7]+\
              b1*bs*self.cleft.pktable[:, 8]+\
              b2*bs*self.cleft.pktable[:, 9]+\
              bs*bs*self.cleft.pktable[:,10]+sn
        return(kv,ret)
        #
    def combine_bias_terms_pk_gm(self,b1,b2,bs):
        """A CLEFT helper function to return P_{gm}."""
        bb  = 1+b1
        kv  = self.cleft.pktable[:,0]
        ret = (1+b1)*self.hf +\
              0.5*b2*self.cleft.pktable[:,4]+0.5*bs*self.cleft.pktable[:,7]
        return(kv,ret)
        #
    def calculate_pk(self,b1,b2,bs,sn):
        """Returns k,Pgg,Pgm,Pmm using HaloFit+pre-computed CLEFT kernels."""
        kk,pgg = self.combine_bias_terms_pk_gg(b1,b2,bs,sn)
        kk,pgm = self.combine_bias_terms_pk_gm(b1,b2,bs)
        return((kk,pgg,pgm,self.hf))
        #
    def __init__(self,klin,plin,halofit):
        """klin,plin,halofit: Arrays containing power spectra [Mpc/h units]."""
        # Copy the arguments.
        self.klin = klin
        self.plin = plin
        # Set up the CLEFT class -- this can take a little while.
        self.cleft= CLEFT(self.klin,self.plin)
        self.cleft.make_ptable(nk=250)
        # Resample the halofit spectrum to the common k grid.
        self.hf = Spline(klin,halofit)(self.cleft.pktable[:,0])
        #
    def __call__(self,pk_pars):
        """Computes the three P(k,z).  Returns k,Pgg,Pgm,Pmm [Mpc/h units]."""
        return(self.calculate_pk(*pk_pars))
        #





class AnzuPowerSpectra():
    """Computes the (real-space) power spectrum, P(k,z) [Mpc/h units]."""
    def combine_bias_terms_pk_gg(self,b1,b2,bs,bn,sn):
        """A helper function to return P_{gg}."""
        # Order returned by Anzu is:
        #    1-1, delta-1 , delta-delta, delta2-1, delta2-delta, delta2-delta2
        #   s2-1,  s2-delta, s2-delta2, s2-s2,
        #   nd-nd, nd-delta, nd-delta2, nd-s2
        kv  = self.kv
        ret = 1.000*self.pktable[ 0,:] + 2   *b1*self.pktable[ 1,:]  +\
              1.*b2*self.pktable[ 3,:] + 2   *bs*self.pktable[ 6,:]  +\
              b1*b1*self.pktable[ 2,:] +   b1*b2*self.pktable[ 4,:]  +\
            2*b1*bs*self.pktable[ 7,:] +   b2*b2*self.pktable[ 5,:]/4+\
              b2*bs*self.pktable[ 8,:] +   bs*bs*self.pktable[ 9,:]  +\
              2.*bn*self.pktable[10,:] + 2*b1*bn*self.pktable[11,:]  +\
              b2*bn*self.pktable[12,:] + 2*bs*bn*self.pktable[13,:]  +\
              sn
        return(kv,ret)
        #
    def combine_bias_terms_pk_gm(self,b1,b2,bs,bn):
        """A helper function to return P_{gm}."""
        kv  = self.kv
        ret = 1.*self.pktable[ 0,:] + b1*self.pktable[1,:] +\
          0.5*b2*self.pktable[ 3,:] + bs*self.pktable[6,:] +\
              bn*self.pktable[10,:]
        return(kv,ret)
        #
    def combine_bias_terms_pk_mm(self):
        """A helper function to return P_{mm}."""
        kv  = self.kv
        ret = self.pktable[0,:]
        return(kv,ret)
        #
    def calculate_pk(self,b1,b2,bs,bn,sn):
        """Returns k,Pgg,Pgm,Pmm using Anzu."""
        # Combine pre-computed terms to get the spectra.
        kk,pgg = self.combine_bias_terms_pk_gg(b1,b2,bs,bn,sn)
        kk,pgm = self.combine_bias_terms_pk_gm(b1,b2,bs,bn)
        kk,pmm = self.combine_bias_terms_pk_mm()
        return((kk,pgg,pgm,pmm))
        #
    def __init__(self,z_eff,pars=None):
        """Initialize the class.
           z_eff is the (effective) redshift at which to evaluate P(k).
           If no cosmological parameters are passed it uses a default
           set otherwise "pars" should hold wb,wc,ns,sig8,hub."""
        # Store z_eff as a scale factor.
        self.aeff= 1.0/(1.0+z_eff)
        # Set the k-values we want to return.
        self.kv  = np.logspace(-3.0,0.0,256)
        # Set up the Anzu class
        self.emu = LPTEmulator(kecleft=True)
        # Fill in the basis spectra using Anzu -- this can take a little while.
        # Upon completion "pktable" is an (Nspec,Nk) array, extended assuming
        # <X, nabla^2 delta> ~ -k^2 <X, 1>.
        if pars is None:
            wb,wc,ns,sig8,hub = 0.022,0.119-0.0006442,0.96824-0.005,0.771,0.677
        else:
            wb,wc,ns,sig8,hub = pars
        cospars  = np.atleast_2d([wb,wc,-1.,ns,sig8,100*hub,3.046,self.aeff])
        pkvec    = self.emu.predict(self.kv,cospars)[0,:,:]
        # Now add the nabla terms assuming <X, nabla^2 delta> ~ -k^2 <X, 1>.
        # We ignore <nabla^2 del,nabla^2 del> so we have only 14 basis specta.
        self.pktable      = np.zeros( (14,len(self.kv)) )
        self.pktable[:10] = pkvec.copy()
        self.pktable[10:] = -self.kv**2 * pkvec[ [0,1,3,6] ]
        #
    def __call__(self,pk_pars):
        """Computes the three P(k,z).  Returns k,Pgg,Pgm,Pmm [Mpc/h units]."""
        return(self.calculate_pk(*pk_pars))
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
    def set_pk(self,klin,plin,halofit=None,pars=None):
        """Sets the linear theory power spectrum or interpolator.
            klin: A numpy array (Nk) containing kk [Mpc/h units].
            plin: A numpy array (Nk) containing Plinear [Mpc/h units].
            halofit: A numpy array (Nk) containing HaloFit [Mpc/h units].
                     If this is None, then pure LPT is used, otherwise
                     a hybrid method is used.
            If both klin and plin are None, then Anzu is used.
            For Anzu if pars is None use a fixed cosmology, else change
            the cosmology."""
        if halofit is None:
            if (klin is None) and (plin is None):  # Use Anzu.
                if pars is None:
                    self.pofk = AnzuPowerSpectra(self.zeff)
                else:
                    self.pofk = AnzuPowerSpectra(self.zeff,pars=pars)
            else: # Use LPT.
                self.pofk = LPTPowerSpectra(klin,plin)
        else: # Use Hybrid
            self.pofk = HalobridPowerSpectra(klin,plin,halofit)
        #
    def __init__(self,OmM,chi_of_z,E_of_z,dndz,Nchi=201,Nz=251):
        """Set up the class.
            OmM:  The value of Omega_m(z=0) for the cosmology.
            chi_of_z: A function returning radial distance in Mpc/h given z.
            E_of_z: A function returning H(z)/H(0) given z.
            dndz: A numpy array (Nbin,2) containing dN/dz vs. z."""
        # Copy the arguments, setting up the z-range.
        self.Eofz = E_of_z
        self.pofk = None
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
        chimin    = np.min(self.chiz) + 1e-5
        chimax    = np.max(self.chiz)
        self.chival= np.linspace(chimin,chimax,self.Nchi)
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
        #
    def __call__(self,pk_pars,smag=0.4,Nell=64,Lmax=1001):
        """Computes C_l^{gg} and C_l^{kg}."""
        fmag    = self.mag_bias_kernel(smag) # Magnification bias kernel.
        ell     = np.logspace(1,np.log10(Lmax),Nell) # More ell's are cheap.
        Cgg,Ckg = np.zeros( (Nell,self.Nchi) ),np.zeros( (Nell,self.Nchi) )
        Pofk    = self.pofk(pk_pars)        # Computes all P(k,z)'s.
        Pgg     = Spline(Pofk[0],Pofk[1])   # Extrapolates as needed.
        Pgm     = Spline(Pofk[0],Pofk[2])   # Extrapolates as needed.
        Pmm     = Spline(Pofk[0],Pofk[3],ext=1)# Extrapolates with zeros.
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


