import numpy as np

from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from classy import Class
from velocileptors.LPT.lpt_rsd_fftw          import LPT_RSD
from velocileptors.LPT.moment_expansion_fftw import MomentExpansion



class XiLikelihood(Likelihood):
    # From yaml file.
    datfn: str
    covfn: str
    zfid:  float
    mcut:  float
    qcut:  float
    #
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        # Set up a fiducial cosmology for distances.
        cc = Class()
        cc.set({'output':'mPk','P_k_max_h/Mpc':25.,'z_pk':self.zfid,\
                'A_s':2.11e-9, 'n_s':0.96824, 'h':0.6760,\
                'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                'z_reio': 7.0, 'omega_b':0.0224, 'omega_cdm':0.119})
        cc.compute()
        self.cc      = cc
        self.Hz_fid  = cc.Hubble(self.zfid)*2997.925/0.6760
        self.chiz_fid= cc.angular_distance(self.zfid)*(1+self.zfid)*0.6760
        self.omh3    = 0.09633# For setting h if not otherwise given.
        self.old_OmM = -100.0 # Just some unlikely value.
        self.old_hub = -100.0 # Just some unlikely value.
        self.old_sig8= -100.0 # Just some unlikely value.
    def logp(self,**params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""
        # Currently ns, omegab, etc. are hard-wired.  We could read
        # these from the params_values as well if we wanted and pass
        # them through to predict as e.g. self.ns.
        #
        #H0_theory = self.provider.get_param("H0")
        #cls = self.provider.get_Cl(ell_factor=True)
        OmM  = params_values.get('Omega_m',self.cc.Omega_m())
        if OmM<=0: OmM=0.3
        hub  = params_values.get('hub',(self.omh3/OmM)**0.3333)
        sig8 = params_values['sig8']
        b1   = params_values['b1']
        b2   = params_values['b2']
        bs   = params_values['bs']
        alp0 = params_values['alpha0']
        alp2 = params_values['alpha2']
        #
        pars = [OmM,hub,sig8,b1,b2,bs,alp0,alp2]
        thy  = self.predict(pars)
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
    def predict(self,pars):
        OmM,hub,sig8,b1,b2,bs,alpha0,alpha2 = pars
        bias = [b1,b2,bs,0.]
        cterm= [alpha0,alpha2,0,0]
        stoch= [0,0,0]
        bpars= bias + cterm + stoch
        #
        zfid = self.zfid
        if (np.abs(sig8-self.old_sig8)>0.001)|\
           (np.abs(hub -self.old_hub )>0.001)|\
           (np.abs(OmM -self.old_OmM )>0.001):
            wb = 0.0224
            wnu= 0.0006442013903673842
            ns = 0.96824
            wc = OmM*hub**2 - wb - wnu
            cc = Class()
            cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':zfid,\
                    'A_s':2e-9, 'n_s':ns, 'h':hub,\
                    'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                    'z_reio': 7.0, 'omega_b':wb, 'omega_cdm':wc})
            cc.compute()
            # Compute the growth rate.
            ff = cc.scale_independent_growth_factor_f(zfid)
            # and work out the A-P scaling to the fiducial cosmology.
            Hz   = self.cc.Hubble(zfid)*2997.925/hub
            chiz = self.cc.angular_distance(zfid)*(1+zfid)*hub
            apar,aperp = self.Hz_fid/Hz,chiz/self.chiz_fid
            # Need to rescale P(k) to match requested sig8 value.
            Af = (sig8/cc.sigma8())**2
            ki = np.logspace(-3.0,1.5,750)
            pi = np.array([cc.pk_cb(k*hub,zfid)*hub**3*Af for k in ki])
            # Now generate and save the PT model
            self.modPT = LPT_RSD(ki,pi,kIR=0.2,one_loop=True,shear=True)
            self.modPT.make_pltable(ff,apar=apar,aperp=aperp,\
                                    kmin=1e-3,kmax=0.8,nk=100,nmax=5)
            # and CLASS instance
            self.cc = cc
            # and update old_sig8, old_hub and old_OmM.
            self.old_sig8 = sig8
            self.old_hub  = hub
            self.old_OmM  = OmM
        #
        xi0,xi2,xi4 = self.modPT.combine_bias_terms_xiell(bpars)
        if np.isnan(xi0).any()|np.isnan(xi2).any()|np.isnan(xi4).any():
            raise RuntimeError("NaNs in xi computation.")
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
    #
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        # Set up a fiducial cosmology for distances.
        cc = Class()
        cc.set({'output':'mPk','P_k_max_h/Mpc':25.,'z_pk':self.zfid,\
                'A_s':2.11e-9, 'n_s':0.96824, 'h':0.6760,\
                'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                'z_reio': 7.0, 'omega_b':0.0224, 'omega_cdm':0.119})
        cc.compute()
        self.cc      = cc
        self.Hz_fid  = cc.Hubble(self.zfid)*2997.925/0.6760
        self.chiz_fid= cc.angular_distance(self.zfid)*(1+self.zfid)*0.6760
        self.omh3    = 0.09633# For setting h if not otherwise given.
        self.old_OmM = -100.0 # Just some unlikely value.
        self.old_hub = -100.0 # Just some unlikely value.
        self.old_sig8= -100.0 # Just some unlikely value.
        #
    def logp(self,**params_values):
        """Given a dictionary of nuisance parameter values params_values
        return a log-likelihood."""
        # Currently ns, omegab, etc. are hard-wired.  We could read
        # these from the params_values as well if we wanted and pass
        # them through to predict as e.g. self.ns.
        #
        #H0_theory = self.provider.get_param("H0")
        #cls = self.provider.get_Cl(ell_factor=True)
        OmM  = params_values.get('Omega_m',self.cc.Omega_m())
        if OmM<=0: OmM=0.3
        hub  = params_values.get('hub',(self.omh3/OmM)**0.3333)
        sig8 = params_values['sig8']
        b1   = params_values['b1']
        b2   = params_values['b2']
        bs   = params_values['bs']
        alp0 = params_values['alpha0']
        alp2 = params_values['alpha2']
        sn0  = params_values['SN0']
        sn2  = params_values['SN2']
        #
        pars = [OmM,hub,sig8,b1,b2,bs,alp0,alp2,sn0,sn2]
        thy  = self.predict(pars)
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
        for i in range(self.xx.size):                    # Dipole.
            ii = i + 1*self.xx.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        for i in np.nonzero(self.xx>self.self.qcut)[0]:  # Quadrupole.
            ii = i + 2*self.xx.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        for i in range(self.xx.size):                    # Octapole.
            ii = i + 3*self.xx.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        for i in range(self.xx.size):                    # Hexadecapole.
            ii = i + 4*self.xx.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        # Finally load the window function matrix.
        self.win = np.loadtxt(self.winfn)
        #
    def predict(self,pars):
        """Predict P_ell(k) given the parameters."""
        OmM,hub,sig8,b1,b2,bs,alpha0,alpha2,sn0,sn2 = pars
        biases = [b1,b2,bs,0]      # Set b3=0
        cterms = [alpha0,alpha2,0] # Set alpha4=0 if no hexadecapole
        stoch  = [sn0,sn2]
        bpars  = biases + cterms + stoch
        #
        zfid = self.zfid
        if (np.abs(sig8-self.old_sig8)>0.001)|\
           (np.abs(hub -self.old_hub )>0.001)|\
           (np.abs(OmM -self.old_OmM )>0.001):
            wb = 0.0224
            wnu= 0.0006442013903673842
            ns = 0.96824
            wc = OmM*hub**2 - wb - wnu
            cc = Class()
            cc.set({'output':'mPk','P_k_max_h/Mpc':100.,'z_pk':zfid,\
                    'A_s':2e-9, 'n_s':ns, 'h':hub,\
                    'N_ur':2.0328, 'N_ncdm':1,'m_ncdm':0.06,\
                    'z_reio': 7.0, 'omega_b':wb, 'omega_cdm':wc})
            cc.compute()
            # Need to rescale P(k) to match requested sig8 value.
            Af = (sig8/cc.sigma8())**2
            ki = np.logspace(-3.0,1.5,750)
            pi = np.array([cc.pk_cb(k*hub,zfid)*hub**3*Af for k in ki])
            # Now set up the PT model.
            self.modPT = MomentExpansion(ki,pi,beyond_gauss=False,\
                            one_loop=True,shear=True,\
                            import_wisdom=False,\
                            kmin=1e-4,kmax=0.5,nk=200,cutoff=10,\
                            extrap_min=-4,extrap_max=3,N=2000,jn=10)
            # and CLASS instance
            self.cc = cc
            # and update old_sig8, old_hub and old_OmM.
            self.old_sig8 = sig8
            self.old_hub  = hub
            self.old_OmM  = OmM
        # Compute the growth rate.
        ff = cc.scale_independent_growth_factor_f(zfid)
        # and work out the A-P scaling to the fiducial cosmology.
        Hz   = self.cc.Hubble(zfid)*2997.925/hub
        chiz = self.cc.angular_distance(zfid)*(1+zfid)*hub
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
        # Have to stack ell=0, 2 & 4 in bins of 0.02h/Mpc from 0.01-0.49h/Mpc.
        kv  = 0.01 + 0.02*np.arange(25)
        thy =                     Spline(tt[:,0],tt[:,1])(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2])(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3])(kv)])
        # Now make the observed theory vector and compute chi^2.
        obs = np.dot(self.win,thy)
        return(obs)
        #
