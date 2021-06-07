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
    def __call__(self,p):
        """The model prediction for parameter set p."""
        if self.name=="clpt_bias": # b1,b2,alpha_a,alpha_x,sn0,smag
            b1,b2,alpha_a,alpha_x,sn0,smag = p
            bs,b3,sn= 0.0,0.0,0.0
            biases = [b1,b2,bs,b3]
            cterms = [alpha_a,alpha_x]
            stoch  = [sn]
            pars   = biases + cterms + stoch
            # P(k), distances, kernels, etc. unchanged.
            # Don't need to change aps.
            # Use the model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            clgg  += sn0*1e-9
            tt     = np.array([ell,clgg,clgk]).T
        elif self.name=="hybr_bias": # b1,b2,bs,sn0,smag
            b1,b2,bs,sn0,smag = p
            pars   = [b1,b2,bs,sn0]
            # P(k), distances, kernels, etc. unchanged.
            # Don't need to change aps.
            # Use the model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt            = np.array([ell,clgg,clgk]).T
        elif self.name=="anzu_bias": # b1,b2,bs,bn,sn,smag
            b1,b2,bs,bn,sn0,smag = p
            pars = [b1,b2,bs,bn,sn0]
            # P(k), distances, kernels, etc. unchanged.
            # Don't need to change aps.
            # Use the model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name=="clpt_sig8": # sig8,b1,b2,alpha_a,alpha_x,sn,smag
            sig8,b1,b2,alpha_a,alpha_x,sn0,smag = p
            bs,b3  = 0.0,0.0
            biases = [b1,b2,bs,b3]
            cterms = [alpha_a,alpha_x]
            stoch  = [sn0]
            pars   = biases + cterms + stoch
            if np.abs(sig8-self.oldsig8)>0.001:
                # Need to rescale P(k) to match requested sig8 value.  This
                # is interpreted as [total matter] sigma8(z=0).
                # Since we're only using linear P(k) this is easy.
                zz = self.zeff
                hh = self.cc.h()
                Af = (sig8/self.cc.sigma8())**2
                ki = np.logspace(-3.0,1.5,750)
                pi = np.array([self.cc.pk_cb_lin(k*hh,zz)*hh**3*Af for k in ki])
                # Only need to overwrite P(k), distances, kernels, etc. unchanged.
                self.aps.set_pk(ki,pi)
                self.oldsig8 = sig8
            # Now use the PT model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name=="hybr_sig8": # sig8,b1,b2,bs,sn,smag
            sig8,b1,b2,bs,sn0,smag = p
            pars = [b1,b2,bs,sn0]
            # Need to rescale P(k) to match requested sig8 value.  This
            # is interpreted as [total matter] sigma8(z=0).
            # Since need halofit we need to remake class.
            Af = (sig8/self.cc.sigma8())**2
            hub= self.cc.h()
            ns = self.cc.n_s()
            zz = self.zeff
            wb = self.cc.omega_b()
            wnu= 0.0006442013903673842
            wc = self.OmM*hub**2 - wb - wnu
            cc = Class()
            cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':self.zeff,\
                    'non linear': 'halofit',\
                    'A_s':2e-9*Af, 'n_s':ns, 'h':hub,\
                    'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                    'z_reio': 7.0, 'omega_b':wb, 'omega_cdm':wc})
            cc.compute()
            ki = np.logspace(-3.0,1.5,750)
            pi = np.array([cc.pk_cb_lin(k*hub,zz)*hub**3 for k in ki])
            hf = np.array([cc.pk_cb(k*hub,zz)*hub**3 for k in ki])
            # Only need to overwrite P(k).  Distances, kernels, etc. unchanged.
            self.aps.set_pk(ki,pi,hf)
            # Now use the model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name=="anzu_sig8": # sig8,b1,b2,bs,bn,sn,smag
            sig8,b1,b2,bs,bn,sn0,smag = p
            bpar = [b1,b2,bs,bn,sn0]
            # self.aps= T.AngularPowerSpectra(self.OmM,self.dndz)
            # Use defaults for non-sigma8 parameters.
            # We found a better match to our fiducial cosmology if we
            # subtract non-zero wnu and slightly tilt ns.
            wnu  = 0.0006442013903673842
            hub  = self.cc.h()
            ns   = self.cc.n_s() - 0.005
            wb   = self.cc.omega_b()
            wc   = self.OmM*hub**2 - wb - wnu
            cpar = [wb,wc,ns,sig8,hub]
            if np.abs(sig8-self.oldsig8)>0.001:
                self.aps.set_pk(None,None,None,pars=cpar)
                self.oldsig8 = sig8
            # Now use the model to compute predictions.
            ell,clgg,clgk = self.aps(bpar,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name=="clpt_lcdm": # OmM,sig8,b1,b2,alpha_a,alpha_x,sn,smag
            OmM,sig8,b1,b2,alpha_a,alpha_x,sn0,smag = p
            hub    = (self.omh3/OmM)**0.333333
            bs,b3  = 0.0,0.0
            biases = [b1,b2,bs,b3]
            cterms = [alpha_a,alpha_x]
            stoch  = [sn0]
            pars   = biases + cterms + stoch
            #
            if (np.abs(OmM -self.oldOmM )>0.001)|\
               (np.abs(hub -self.oldhub )>0.001)|\
               (np.abs(sig8-self.oldsig8)>0.001):
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
                # Update oldhub, etc.
                self.oldOmM  = OmM
                self.oldhub  = hub
                self.oldsig8 = sig8
            # Now use the PT model to compute predictions.
            ell,clgg,clgk = self.aps(pars,smag,Lmax=1251)
            tt = np.array([ell,clgg,clgk]).T
        elif self.name=="anzu_lcdm": # OmM,sig8,b1,b2,bs,bn,sn,smag
            OmM,sig8,b1,b2,bs,bn,sn0,smag = p
            hub  = (self.omh3/OmM)**0.333333
            bpar = [b1,b2,bs,bn,sn0]
            #
            if (np.abs(OmM -self.oldOmM )>0.001)|\
               (np.abs(hub -self.oldhub )>0.001)|\
               (np.abs(sig8-self.oldsig8)>0.001):
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
                # Update oldhub, etc.
                self.oldOmM  = OmM
                self.oldhub  = hub
                self.oldsig8 = sig8
            # Now use the model to compute predictions.
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
        # a "fiducial" cosmology but we hold it fixed after this.
        self.aps = T.AngularPowerSpectra(self.OmM,dndz)
        self.zeff= self.aps.zeff
        self.oldOmM  =  -100.
        self.oldhub  =  -100.
        self.oldsig8 =  -100.
        if modelname.startswith("anzu"):
            self.aps.set_pk(None,None,None)
        elif modelname.startswith("hybr"):
            # For fits involving the same power spectrum pre-load the PT class
            hub = fid_class.h()
            ki  = np.logspace(-3.0,1.5,1024)
            pi  = np.array([cc.pk_cb_lin(k*hub,self.zeff)*hub**3 for k in ki])
            hf  = np.array([cc.pk_cb(k*hub,self.zeff)*hub**3 for k in ki])
            self.aps.set_pk(ki,pi,hf)
        else:
            # For fits involving the same power spectrum pre-load the PT class
            hub = fid_class.h()
            ki  = np.logspace(-3.0,1.5,1024)
            pi  = np.array([cc.pk_cb_lin(k*hub,self.zeff)*hub**3 for k in ki])
            self.aps.set_pk(ki,pi,halofit=None)
        #
