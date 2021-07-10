import numpy as np
import sys
import os

from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from emulator          import Emulator
from predict_cl        import AngularPowerSpectra
from cobaya.likelihood import Likelihood




class GxKLikelihood(Likelihood):
    # From yaml file.
    model:  str
    clsfn:  str
    covfn:  str
    Pggfn:  str
    Pgmfn:  str
    Pmmfn:  str
    suffx:  list
    dndzfn: list
    wlafn:  list
    wlxfn:  list
    acut:   list
    xcut:   list
    #
    def initialize(self):
        """Sets up the class."""
        # Load the data and invert the covariance matrix.
        self.loadData()
        self.cinv   = np.linalg.inv(self.cov)
        self.PggEmu = Emulator(self.Pggfn)
        self.PgmEmu = Emulator(self.Pgmfn)
        self.PmmEmu = Emulator(self.Pmmfn)
    def get_requirements(self):
        """What we require."""
        zgrid= np.logspace(0,3.1,64) - 1.0
        reqs = {\
               'logA':     None,\
               'omch2':    None,\
               'ombh2':    None,\
               'H0':       None,\
               'ns':       None,\
               'omegam':   None,\
               'Hubble':   {'z': zgrid},\
               'comoving_radial_distance': {'z': zgrid}\
               }
        # Build the parameter names we require for each sample.
        for suf in self.suffx:
            for pref in ['b1','b2',\
                         'bs','bn','alpha_a','alpha_x',\
                         'SN','smag']:
                reqs[pref+'_'+suf] = None
        return(reqs)
    def logp(self,**params_values):
        """Return the log-likelihood."""
        pp  = self.provider
        OmM = pp.get_param('omegam')
        hub = pp.get_param('H0')/100.0
        # Make splines for chi(z) and E(z), converting to Mpc/h.
        zgrid = np.logspace(0,3.1,64)-1.0
        chiz  = pp.get_comoving_radial_distance(zgrid)*hub
        chiz  = Spline(zgrid,chiz)
        Eofz  = pp.get_Hubble(zgrid)
        Eofz  = Spline(zgrid,Eofz/(100*hub))
        #
        obs = np.array([],dtype='float')
        for i,suf in enumerate(self.suffx):
            aps = AngularPowerSpectra(OmM,chiz,Eofz,self.dndz[i])
            # Fill in the parameter list, starting with the
            # cosmological parameters.
            pars = [pp.get_param('logA'),\
                    pp.get_param('ns'),\
                    pp.get_param('H0'),\
                    -1.0,\
                    pp.get_param('ombh2'),\
                    pp.get_param('omch2')]
            # Extract some common parameters.
            b1  = pp.get_param('b1_'+suf)
            b2  = pp.get_param('b2_'+suf)
            sn  = pp.get_param('SN_'+suf)
            smag= pp.get_param('smag_'+suf)
            #
            # Do some parameter munging depending upon the model name
            # to fill in the rest of pars.
            # NOTE: the redshift at which to evaluate this is filled
            # in by the angular power spectrum code, so is not passed here.
            if self.model.startswith('clpt'):
                alpA  = pp.get_param('alpha_a_'+suf)
                alpX  = pp.get_param('alpha_x_'+suf)
                pars += [b1,b2,alpA,alpX,sn]
            elif self.model.startswith('anzu'):
                bs    = pp.get_param('bs_'+suf)
                bn    = pp.get_param('bn_'+suf)
                pars += [b1,b2,bs,bn,sn]
            else:
                raise RuntimeError("Unknown model.")
            # and call APS to get a prediction,
            ell,clgg,clgk = aps(self.PggEmu,self.PgmEmu,self.PmmEmu,\
                                pars,smag,Lmax=1251)
            thy = np.array([ell,clgg,clgk]).T
            # then "observe" it, appending the observations to obs.
            obs = np.append(obs,self.observe(thy,self.wla[i],self.wlx[i]))
        # Now compute chi^2 and return -ln(L)
        chi2 = np.dot(self.dd-obs,np.dot(self.cinv,self.dd-obs))
        return(-0.5*chi2)
        #
    def loadData(self):
        """Load the data, covariance and windows from files."""
        dd        = np.loadtxt(self.clsfn)
        self.cov  = np.loadtxt(self.covfn)
        self.dndz = []
        for fn in self.dndzfn:
            self.dndz.append(np.loadtxt(fn))
        self.wla = []
        for fn in self.wlafn:
            self.wla.append(np.loadtxt(fn))
        self.wlx = []
        for fn in self.wlxfn:
            self.wlx.append(np.loadtxt(fn))
        # Now pack things and modify the covariance matrix to
        # "drop" some data points.
        Nsamp   = (dd.shape[1]-1)//2
        if Nsamp!=len(self.dndz):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.wla):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.wlx):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.acut):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        if Nsamp!=len(self.xcut):
            raise RuntimeError("Inconsistent inputs: Nsamp mismatch.")
        self.xx = dd[:,0]
        self.dd = dd[:,1:].T.flatten()
        for j in range(Nsamp):
            for i in np.nonzero(self.xx>self.acut[j])[0]:           # Auto
                ii = i + (2*j+0)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
            for i in np.nonzero(self.xx>self.xcut[j])[0]:           # Cross
                ii = i + (2*j+1)*self.xx.size
                self.cov[ii, :] = 0
                self.cov[ :,ii] = 0
                self.cov[ii,ii] = 1e15
        #
    def observe(self,tt,wla,wlx):
        """Applies the window function and binning matrices."""
        lmax = wla.shape[1]
        ells = np.arange(lmax)
        # Have to stack auto and cross.
        obs1 = np.dot(wla,np.interp(ells,tt[:,0],tt[:,1],right=0))
        obs2 = np.dot(wlx,np.interp(ells,tt[:,0],tt[:,2],right=0))
        obs  = np.concatenate([obs1,obs2])
        return(obs)
        #
