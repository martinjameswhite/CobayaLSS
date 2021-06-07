#
# Models.
#

import numpy      as np
import predict_cl as T	# T is for "theory".
import sys
import os

from classy import Class

from scipy.interpolate import InterpolatedUnivariateSpline as Spline





class Model():
    """The class that handles model predictions."""
    def __call__(self,fast,slow):
        """The model prediction for parameter set p."""
        OmM,hub,sig8                         = slow
        b1,b2,bs,bn,alpha_a,alpha_x,sn0,smag = fast
        if self.name.startswith("clpt_sig8"):
            # This is a special case where don't need to call
            # Bolztmann code again since rescaling Plin is easy.
            bs,b3  = 0.0,0.0
            biases = [b1,b2,bs,b3]
            cterms = [alpha_a,alpha_x]
            stoch  = [sn0]
            pars   = biases + cterms + stoch
            #
            if self.old_slow is None:
                self.old_slow = np.array(slow) + 1
            if not np.allclose(slow,self.old_slow):
                # Need to rescale P(k) to match requested sig8 value.  This
                # is interpreted as [total matter] sigma8(z=0).
                # Since we're only using linear P(k) this is easy, don't
                # need to recompute everything.
                zz = self.zeff
                hh = self.cc.h()
                Af = (sig8/self.cc.sigma8())**2
                ki = np.logspace(-3.0,1.5,750)
                pi = np.array([self.cc.pk_cb_lin(k*hh,zz)*hh**3*Af for k in ki])
                # Only need to overwrite P(k), distances, etc. unchanged.
                self.aps.set_pk(ki,pi)
                self.old_slow = slow.copy()
            # Use the model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name.startswith("clpt"): # But not clpt_sig8
            bs,b3  = 0.0,0.0
            biases = [b1,b2,bs,b3]
            cterms = [alpha_a,alpha_x]
            stoch  = [sn0]
            pars   = biases + cterms + stoch
            #
            if self.old_slow is None:
                self.old_slow = np.array(slow) + 1
            if not np.allclose(slow,self.old_slow):
                # Recompute model.
                wb = self.cc.omega_b()
                ns = self.cc.n_s()
                wnu= 0.0006442013903673842
                wc = OmM*hub**2 - wb - wnu
                cc = Class()
                cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':self.zeff,\
                        'non linear': 'halofit',\
                        'A_s':2e-9, 'n_s':ns, 'h':hub,\
                        'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                        'z_reio': 7.0, 'omega_b':wb, 'omega_cdm':wc})
                cc.compute()
                # Need to rescale P(k) to match requested sig8 value.  Since
                # only use linear P this is easy.
                Af = (sig8/cc.sigma8())**2
                ki = np.logspace(-3.0,1.5,750)
                pi = np.array([cc.pk_cb_lin(k*hub,self.zeff)*hub**3*Af \
                               for k in ki])
                # Need to remake APS class.
                self.aps = T.AngularPowerSpectra(OmM,self.dndz)
                self.aps.set_pk(ki,pi)
                # Update old_slow.
                self.old_slow = slow.copy()
            # Use the model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name.startswith("anzu"):
            bpar = [b1,b2,bs,bn,sn0]
            #
            if self.old_slow is None:
                self.old_slow = np.array(slow) + 1
            if not np.allclose(slow,self.old_slow):
                # Recompute model.
                # Use defaults for non-varying parameters.
                # We found a better match to our fiducial cosmology if we
                # subtract non-zero wnu and slightly tilt ns.
                wnu  = 0.0006442013903673842
                wb   = self.cc.omega_b()
                ns   = self.cc.n_s() - 0.005
                wc   = OmM*hub**2 - wb - wnu
                cpar = [wb,wc,ns,sig8,hub]
                # Need to remake APS class.
                self.aps= T.AngularPowerSpectra(self.OmM,self.dndz)
                self.aps.set_pk(None,None,None,pars=cpar)
                # Update old_slow.
                self.old_slow = slow.copy()
            # Use the model to compute predictions.
            ell,clgg,clgk = self.aps(bpar,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        else:
            raise RuntimeError("Unknown model "+self.name)
        return(tt)
        #
    def __init__(self,modelname,dndz,fid_class):
        """This sets up the model:
           modelname: A unique model name.
           dndz:      A two-column numpy array of z,dN/dz.
           fid_class: A Class instance holding the fiducial cosmology."""
        self.name = modelname
        self.dndz = dndz
        self.OmM  = fid_class.Omega_m()
        self.omh3 = 0.09633   # Can be used to derive h from OmegaM.
        self.cc   = fid_class # Keep a copy.
        cc        = fid_class # A shorter name.
        # Use the angular power spectrum class to get z_eff, this assumes
        # a "fiducial" cosmology but we hold z_eff fixed after this.
        self.aps      = T.AngularPowerSpectra(self.OmM,dndz)
        self.zeff     = self.aps.zeff
        self.old_slow = None
        #
