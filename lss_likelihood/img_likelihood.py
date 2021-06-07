import numpy as np
import predict_cl  as T       # T is for "theory".
import cl_model    as M
import sys
import os

from classy import Class

from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from cobaya.likelihood import Likelihood


class ClLikelihood(Likelihood):
    # From yaml file.
    mname: str
    clsfn: str
    covfn: str
    wlafn: str
    wlxfn: str
    dndzfn:str
    ityp:  str
    isamp: int
    acut:  int
    xcut:  int
    #
    def initialize(self):
        """Sets up the class."""
        # Load the data and invert the covariance matrix.
        self.loadData()
        self.cinv= np.linalg.inv(self.cov)
        #
        # Set up a fiducial cosmology and Class instance, then
        # compute the distance and Hubble parameter at zeff for the
        # "fiducial" cosmology.
        hub = 0.677
        cc  = Class()
        #cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':'0,10',\
        #        'A_s':2.11e-9, 'n_s':0.96824, 'h':0.6760,\
        #        'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
        #        'tau_reio': 0.057, 'omega_b':0.022447, 'omega_cdm':0.11923})
        #cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':'0,10',\
        #        'A_s':2.11e-9, 'n_s':0.96824, 'h':hub,\
        #        'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
        #        'z_reio': 7.0, 'omega_b':0.022, 'omega_cdm':0.119})
        #
        cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':'0,10',\
                'non linear': 'halofit',\
                'A_s':1.90e-9, 'n_s':0.96824, 'h':hub,\
                'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                'z_reio': 7.0, 'omega_b':0.022, 'omega_cdm':0.119})
        cc.compute()
        self.OmM     = cc.Omega0_m()
        self.omh3    = 0.09633# For setting h if not otherwise given.
        self.sig8_fid= cc.sigma8()
        # Set up a model instance and use this to get z_eff.
        self.model= M.Model(self.mname,self.dndz,cc)
        self.zeff = self.model.aps.zeff
    def logp(self,**params_values):
        """
        Given a dictionary of nuisance parameter values params_values
        return a log-likelihood.
        """
        #H0_theory = self.provider.get_param("H0")
        #cls = self.provider.get_Cl(ell_factor=True)
        OmM  = params_values['Omega_m']
        if OmM<=0: OmM=0.3
        hub  = params_values.get('hub',(self.omh3/OmM)**0.3333)
        sig8 = params_values['sig8']
        b1   = params_values['b1']
        b2   = params_values['b2']
        bs   = params_values['bs']
        bn   = params_values['bn']
        alpA = params_values['alpha_a']
        alpX = params_values['alpha_x']
        sn   = params_values['lgSN'];  sn = 10.0**sn
        smag = params_values['smag']
        #
        # Set the fast and slow parameters.
        fast = [b1,b2,bs,bn,alpA,alpX,sn,smag]
        slow = [OmM,hub,sig8]
        #
        # Get the "theory".
        tt = self.model(fast,slow)
        # Now compute chi^2.
        thy  = self.observe(tt)
        chi2 = np.dot(self.dd-thy,np.dot(self.cinv,self.dd-thy))
        #
        return(-0.5*chi2)
        #
    def loadData(self):
        """Load the data, covariance and windows from files."""
        dd       = np.loadtxt(self.clsfn)
        self.cov = np.loadtxt(self.covfn)
        self.wla = np.loadtxt(self.wlafn)
        self.wlx = np.loadtxt(self.wlxfn)
        self.dndz= np.loadtxt(self.dndzfn)
        # Now pack things and modify the covariance matrix to
        # "drop" some data points.
        self.xx = dd[:,0]
        self.dd = np.append(dd[:,1],dd[:,2])
        for i in np.nonzero(self.xx>self.acut)[0]:           # Auto
            ii = i + 0*self.xx.size
            self.cov[ii, :] = 0
            self.cov[ :,ii] = 0
            self.cov[ii,ii] = 1e15
        for i in np.nonzero(self.xx>self.xcut)[0]:           # Cross
            ii = i + 1*self.xx.size
            self.cov[ii, :] = 0
            self.cov[ :,ii] = 0
            self.cov[ii,ii] = 1e15
        #
    def observe(self,tt):
        """Applies the window function and binning matrices."""
        lmax = self.wla.shape[1]
        ells = np.arange(lmax)
        # Have to stack auto and cross.
        obs1 = np.dot(self.wla,np.interp(ells,tt[:,0],tt[:,1],right=0))
        obs2 = np.dot(self.wlx,np.interp(ells,tt[:,0],tt[:,2],right=0))
        obs  = np.concatenate([obs1,obs2])
        return(obs)
        #
