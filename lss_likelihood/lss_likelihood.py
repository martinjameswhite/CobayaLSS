import numpy as np

from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from velocileptors.LPT.lpt_rsd_fftw          import LPT_RSD
from velocileptors.LPT.moment_expansion_fftw import MomentExpansion



class XiLikelihood(Likelihood):
    # From yaml file.
    datfn: str
    covfn: str
    zfid:  float
    mcut:  float
    qcut:  float
    chiz_fid: float
    Hz_fid: float
    #
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        self.omh3    = 0.09633# For setting h if not otherwise given.
        self.old_slow= None
    def get_requirements(self):
        zg  = np.linspace(0,self.zfid,8,endpoint=True)
        req = {\
                'Pk_interpolator': {'k_max': 30,'z': zg,\
                                   'nonlinear': False},\
               'sigma8_z': {'z': [0.0,self.zfid]},\
               'fsigma8':  {'z': [self.zfid]},\
               'Hubble':   {'z': [0.0,self.zfid]},\
               'comoving_radial_distance': {'z': [self.zfid]},\
               'omegam': None\
              }
        return(req)
    def logp(self,**params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""
        hub  = self.provider.get_Hubble(0)[0]/100.
        sig8 = self.provider.get_sigma8_z(0)[0]
        OmM  = params_values['omegam']
        b1   = params_values['bsig8']/sig8 - 1.0
        b2   = params_values['b2']
        bs   = params_values['bs']
        alp0 = params_values['alpha0']
        alp2 = params_values['alpha2']
        # Divide the parameters into fast and slow and pass both.
        slow = [OmM,hub,sig8]
        fast = [b1,b2,bs,alp0,alp2]
        thy  = self.predict(fast,slow)
        obs  = self.observe(thy)
        chi2 = np.dot(self.dd-obs,np.dot(self.cinv,self.dd-obs))
        #
        return(-0.5*chi2)
        #
    def loadData(self):
        """Loads the data and error."""
        # First load the data.
        dd      = np.loadtxt(self.datfn)
        self.xx = dd[:,0]
        # Generate the data and error vectors.
        self.dd  = np.concatenate((dd[:,1],dd[:,2]))
        # Now load the covariance matrix.
        self.cov = np.loadtxt(self.covfn)
        # Now we have the covariance matrix but we're only going to want some
        # of the entries in computing chi^2.
        for i in np.nonzero(self.xx<self.mcut)[0]:     # Monopole.
            self.cov[i,:] = 0
            self.cov[:,i] = 0
            self.cov[i,i] = 1e15
        for i in np.nonzero(self.xx<self.qcut)[0]:     # Quadrupole.
            ii = i + self.xx.size
            self.cov[ii, :] = 0
            self.cov[ :,ii] = 0
            self.cov[ii,ii] = 1e15
        self.cinv = np.linalg.inv(self.cov)
        #
    def predict(self,fast,slow):
        OmM,hub,sig8           = slow
        b1,b2,bs,alpha0,alpha2 = fast
        bias = [b1,b2,bs,0.]
        cterm= [alpha0,alpha2,0,0]
        stoch= [0,0,0]
        bpars= bias + cterm + stoch
        # Make shorter names.
        pp   = self.provider
        zfid = self.zfid
        if self.old_slow is None:
            self.old_slow = np.array(slow) + 1
        if not np.allclose(slow,self.old_slow):
            # Get cosmological parameters
            s8   = pp.get_sigma8_z(self.zfid)[0]
            fs8  = pp.get_fsigma8(self.zfid)[0]
            ff   = fs8 / s8
            # and Plin.
            ki   = np.logspace(-3.0,1.5,750)
            pi   = pp.get_Pk_interpolator(nonlinear=False)
            pi   = pi.P(self.zfid,ki*hub)*hub**3
            # Work out the A-P scaling to the fiducial cosmology.
            Hz   = pp.get_Hubble(self.zfid)[0]/pp.get_Hubble(0)[0]
            chiz = pp.get_comoving_radial_distance(self.zfid)[0]*hub
            apar,aperp = self.Hz_fid/Hz,chiz/self.chiz_fid
            # Now generate and save the PT model
            self.modPT = LPT_RSD(ki,pi,kIR=0.2,one_loop=True,shear=True)
            self.modPT.make_pltable(ff,apar=apar,aperp=aperp,\
                                    kmin=1e-3,kmax=0.8,nk=100,nmax=5)
            # and update old_slow
            self.old_slow = slow.copy()
        #
        xi0,xi2,xi4 = self.modPT.combine_bias_terms_xiell(bpars)
        if np.isnan(xi0).any()|np.isnan(xi2).any()|np.isnan(xi4).any():
            xi0,xi2,xi4 = self.modPT.combine_bias_terms_xiell(bpars,\
                            method='gauss_poly')
        ss = np.linspace(20.0,150.,150)
        xi0= np.interp(ss,xi0[0],xi0[1])
        xi2= np.interp(ss,xi2[0],xi2[1])
        xi4= np.interp(ss,xi4[0],xi4[1])
        tt = np.array([ss,xi0,xi2]).T
        return(tt)
        #
    def observe(self,tt):
        """Do the binning into the observed s bins."""
        # Now integrate/average across the bin -- do each seperately and then
        # splice/combine.
        thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
        thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
        xx   = self.xx
        dx   = xx[1]-xx[0]
        tmp0 = np.zeros_like(xx)
        tmp2 = np.zeros_like(xx)
        for i in range(xx.size):
            ss     = np.linspace(xx[i]-dx/2,xx[i]+dx/2,100)
            ivol   = 3.0/((xx[i]+dx/2)**3-(xx[i]-dx/2)**3)
            tmp0[i]= np.trapz(ss**2*thy0(ss),x=ss)*ivol
            tmp2[i]= np.trapz(ss**2*thy2(ss),x=ss)*ivol
        thy0 = tmp0
        thy2 = tmp2
        # Since we have extrapolated wildly, set out-of-bounds points to data.
        npnt = xx.size
        ww   = np.nonzero( (xx<tt[0,0])|(xx>tt[-1,0]) )[0]
        if len(ww)>0:
            thy0[ww] = self.dd[ww]
            thy2[ww] = self.dd[ww+npnt]
        # Append quadrupole to monopole.
        obs = np.concatenate((thy0,thy2))
        return(obs)
        #











class PkLikelihood(Likelihood):
    # From yaml file.
    datfn: str
    covfn: str
    winfn: str
    zfid:  float
    mcut:  float
    qcut:  float
    chiz_fid: float
    Hz_fid: float
    #
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        self.omh3    = 0.09633# For setting h if not otherwise given.
        self.old_slow= None
        #
    def get_requirements(self):
        zg  = np.linspace(0,self.zfid,8,endpoint=True)
        req = {\
                'Pk_interpolator': {'k_max': 30,'z': zg,\
                                   'nonlinear': False},\
               'sigma8_z': {'z': [0.0,self.zfid]},\
               'fsigma8':  {'z': [self.zfid]},\
               'Hubble':   {'z': [0.0,self.zfid]},\
               'comoving_radial_distance': {'z': [self.zfid]},\
               'omegam': None\
              }
        return(req)
    def logp(self,**params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""
        hub  = self.provider.get_Hubble(0)[0]/100.
        sig8 = self.provider.get_sigma8_z(0)[0]
        OmM  = params_values['omegam']
        b1   = params_values['b1']
        b2   = params_values['b2']
        bs   = params_values['bs']
        alp0 = params_values['alpha0']
        alp2 = params_values['alpha2']
        sn0  = params_values['SN0']
        sn2  = params_values['SN2']
        #
        slow = [OmM,hub,sig8]
        fast = [b1,b2,bs,alp0,alp2,sn0,sn2]
        thy  = self.predict(fast,slow)
        obs  = self.observe(thy)
        chi2 = np.dot(self.dd-obs,np.dot(self.cinv,self.dd-obs))
        #
        return(-0.5*chi2)
        #
    def loadData(self):
        """Loads the required data."""
        # First load the data.
        dd      = np.loadtxt(self.datfn)
        self.xx = dd[:,0]
        # Stack the data vector.
        self.dd  = dd[:,1:].T.flatten()
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        # We're only going to want some of the entries in computing chi^2.
        for i in np.nonzero(self.xx>self.mcut)[0]:       # Monopole.
            ii = i + 0*self.xx.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        for i in np.nonzero(self.xx>self.qcut)[0]:       # Quadrupole.
            ii = i + 1*self.xx.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        #for i in range(self.xx.size):                    # Hexadecapole.
        #    ii = i + 2*self.xx.size
        #    cov[ii, :] = 0
        #    cov[ :,ii] = 0
        #    cov[ii,ii] = 1e20
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        # Finally load the window function matrix.
        self.win = np.loadtxt(self.winfn)
        #
    def predict(self,fast,slow):
        """Predict P_ell(k) given the parameters."""
        OmM,hub,sig8                   = slow
        b1,b2,bs,alpha0,alpha2,sn0,sn2 = fast
        biases = [b1,b2,bs,0]      # Set b3=0
        cterms = [alpha0,alpha2,0] # Set alpha4=0 if no hexadecapole
        stoch  = [sn0,sn2]
        bpars  = biases + cterms + stoch
        # Make shorter names.
        pp   = self.provider 
        zfid = self.zfid
        #
        if self.old_slow is None:
            self.old_slow = np.array(slow) + 1
        if not np.allclose(slow,self.old_slow):
            # Get Plin.
            ki = np.logspace(-3.0,1.5,750)
            pi = pp.get_Pk_interpolator(nonlinear=False)
            pi = pi.P(self.zfid,ki*hub)*hub**3
            # Now set up the PT model.
            self.modPT = MomentExpansion(ki,pi,beyond_gauss=False,\
                            one_loop=True,shear=True,\
                            import_wisdom=False,\
                            kmin=1e-4,kmax=0.5,nk=200,cutoff=10,\
                            extrap_min=-4,extrap_max=3,N=2000,jn=10)
            # and update old_slow
            self.old_slow = slow.copy()
        # Compute the growth rate and work out the A-P scaling.
        s8   = pp.get_sigma8_z(self.zfid)[0]
        fs8  = pp.get_fsigma8(self.zfid)[0]
        ff   = fs8 / s8
        # Work out the A-P scaling to the fiducial cosmology.
        Hz   = pp.get_Hubble(self.zfid)[0]/pp.get_Hubble(0)[0]
        chiz = pp.get_comoving_radial_distance(self.zfid)[0]*hub
        apar,aperp = self.Hz_fid/Hz,chiz/self.chiz_fid
        # Call the PT model to get P_ell -- we'll grid it onto the
        # appropriate binning for the window function in observe.
        kv,p0,p2,p4=self.modPT.compute_redshift_space_power_multipoles(bpars,\
                         ff,apar=apar,aperp=aperp,reduced=True)
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([sn0,],p0)
        p2,p4 = np.append([0.,],p2),np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        return(tt)
        #
    def observe(self,tt):
        """Apply the window function matrix to get the binned prediction."""
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        thy =                     Spline(tt[:,0],tt[:,1])(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2])(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3])(kv)])
        # Now make the observed theory vector and compute chi^2.
        obs = np.dot(self.win,thy)
        return(obs)
        #
