import numpy as np
import sys
import os

import predict_cl as T # For theory.

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood


class ClLikelihood(Likelihood):
    # From yaml file.
    clsfn: str
    covfn: str
    wlafn: str
    wlxfn: str
    acut:  int
    xcut:  int
    #
    def initialize(self):
        """Sets up the class."""
        # Load the data and invert the covariance matrix.
        self.loadData()
        self.cinv= np.linalg.inv(self.cov)
    def get_requirements(self):
        """What we need."""
        req = {'pt_cell_mod': None,\
               'b1': None,\
               'b2': None,\
               'bs': None,\
               'bn': None,\
               'alpha_a': None,\
               'alpha_x': None,\
               'SN': None,\
               'smag': None\
              }
        return(req)
    def logp(self,**params_values):
        """
        Given a dictionary of nuisance parameter values params_values
        return a log-likelihood.
        """
        thy  = self.predict()
        obs  = self.observe(thy)
        chi2 = np.dot(self.dd-obs,np.dot(self.cinv,self.dd-obs))
        #
        return(-0.5*chi2)
        #
    def loadData(self):
        """Load the data, covariance and windows from files."""
        dd       = np.loadtxt(self.clsfn)
        self.cov = np.loadtxt(self.covfn)
        self.wla = np.loadtxt(self.wlafn)
        self.wlx = np.loadtxt(self.wlxfn)
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
    def predict(self):
        """Predicts the theory C_ell's."""
        pp   = self.provider
        modPT= pp.get_result('pt_cell_mod')
        b1   = pp.get_param('b1')
        b2   = pp.get_param('b2')
        SN   = pp.get_param('SN')
        smag = pp.get_param('smag')
        if pp.get_result('pt_modelname').startswith('clpt'):
            alpA   = pp.get_param('alpha_a')
            alpX   = pp.get_param('alpha_x')
            bs,b3  = 0.0,0.0
            biases = [b1,b2,bs,b3]
            cterms = [alpA,alpX]
            stoch  = [SN]
            pars   = biases + cterms + stoch
        elif pp.get_result('pt_modelname').startswith('anzu'):
            bs   = pp.get_param('bs')
            bn   = pp.get_param('bn')
            pars = [b1,b2,bs,bn,SN]
        #
        ell,clgg,clgk = modPT(pars,smag,Lmax=1251)
        tt = np.array([ell,clgg,clgk]).T
        return(tt)
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







class PT_cell_theory(Theory):
    """A class to return a PT C_ell module."""
    # From yaml file.
    model:  str
    dndzfn: str
    #
    def initialize(self):
        """Sets up the class."""
        self.dndz = np.loadtxt(self.dndzfn)
        self.zmin = np.min(self.dndz[:,0])
        self.zmax = np.max(self.dndz[:,0])
    def get_requirements(self):
        """What we need in order to provide C_ell."""
        zg  = np.linspace(self.zmin,self.zmax,21,endpoint=True)
        req = {\
               'Pk_interpolator': {'k_max': 30,'z': zg,\
                                   'nonlinear': False},\
               'sigma8_z': {'z': [0]},\
               'Hubble': {'z': [0]},\
               'omegam': None\
              }
        return(req)
    def get_can_provide(self):
        """What do we provide: a PT class that can compute C_ell."""
        return ['pt_cell_mod']
    def calculate(self,state,want_derived=True,**params_values_dict):
        """Create and initialize the PT class."""
        # Make shorter names and get params.
        pp  = self.provider
        OmM = pp.get_param('omegam')
        hub = self.provider.get_Hubble(0)[0]/100.
        # Set up the APS, including its zeff.
        # For now chi(z) is assumed LCDM, but we could
        # pass a Spline or something from the provider.
        aps = T.AngularPowerSpectra(OmM,self.dndz)
        if self.model.startswith('clpt'):
            # Get Plin.
            ki  = np.logspace(-3.0,1.5,750)
            pi  = pp.get_Pk_interpolator(nonlinear=False)
            pi  = pi.P(aps.zeff,ki*hub)*hub**3
            # and set the power spectrum module in APS:
            aps.set_pk(ki,pi)
        elif self.model.startswith('anzu'):
            wb   = pp.get_param('ombh2')
            wc   = pp.get_param('omch2')
            ns   = pp.get_param('ns') - 0.005
            sig8 = pp.get_sigma8_z(0)[0]
            cpar = [wb,wc,ns,sig8,hub]
            aps.set_pk(None,None,None,pars=cpar)
        # Save the PT model in the state.
        state['pt_cell_mod'] = aps
        state['cell_modelname']=self.model
        #
