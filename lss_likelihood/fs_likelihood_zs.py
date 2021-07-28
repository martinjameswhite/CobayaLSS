import numpy as np

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from linear_theory import*

from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT

# Class to have a full-shape likelihood with multiple zs

class FSLikelihood(Likelihood):
    
    zfid: float
    Hz_fid: float
    chiz_fid: float
    sample_name: str
    
    fs_datfn: str
    covfn: str
    
    fs_kmin: float
    fs_mmax: float
    fs_qmax: float
    fs_matMfn: str
    fs_matWfn: str

    def initialize(self):
        """Sets up the class."""
        self.zstr = "%.2f" %(self.zfid)
        self.loadData()
        #

    def get_requirements(self):
        req = {'pt_pk_ell_mod': None,\
               'H0': None,\
               'sigma8': None,
               'bsig8_' + self.sample_name: None,\
               'b2_' + self.sample_name: None,\
               'bs_' + self.sample_name: None,\
               'alpha0_' + self.sample_name: None,\
               'alpha2_' + self.sample_name: None,\
               'SN0_' + self.sample_name: None,\
               'SN2_' + self.sample_name: None,\
              }
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        fs_thy  = self.fs_predict()
        fs_obs  = self.fs_observe(fs_thy)

        diff = self.dd - fs_obs
        
        chi2 = np.dot(diff,np.dot(self.cinv,diff))
        #print('diff', self.sample_name, diff[:20])
        #
        return(-0.5*chi2)
        #
        
    def loadData(self):
        """
        Loads the required data.
        
        Do this in two steps... first load full shape data then xirecon, concatenate after.
        
        The covariance is assumed to already be joint in the concatenated format.
        
        """
        # First load the data
        
        fs_dat = np.loadtxt(self.fs_datfn)
        self.kdat = fs_dat[:,0]
        self.p0dat = fs_dat[:,1]
        self.p2dat = fs_dat[:,2]
        
        # Make a list of indices for the monopole and quadrupole only
        yeses = self.kdat > 0
        nos   = self.kdat < 0
        self.fitiis = np.concatenate( (yeses, nos, yeses, nos, nos) )

        
        # Join the data vectors together
        self.dd = np.concatenate( (self.p0dat, self.p2dat))
        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        cov = cov[np.ix_(self.fitiis, self.fitiis)]

        # We're only going to want some of the entries in computing chi^2.
        kcut = (self.kdat > self.fs_mmax) | (self.kdat < self.fs_kmin)
        for i in np.nonzero(kcut)[0]:     # FS Monopole.
            ii = i + 0*self.kdat.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e25
        
        kcut = (self.kdat > self.fs_qmax) | (self.kdat < self.fs_kmin)
        for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
            ii = i + 1*self.kdat.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e25

        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #print(self.sample_name, np.diag(self.cinv)[:10])
        # Finally load the window function matrix.
        self.matM = np.loadtxt(self.fs_matMfn)
        self.matW = np.loadtxt(self.fs_matWfn)
        
        #
    def fs_predict(self):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        modPTs = pp.get_result('pt_pk_ell_mod')
        hub = pp.get_param('H0') / 100.
        sig8 = pp.get_param('sigma8')

        #
        b1   = pp.get_param('bsig8_' + self.sample_name)/sig8 - 1.0
        b2   = pp.get_param('b2_' + self.sample_name)
        bs   = pp.get_param('bs_' + self.sample_name)
        alp0 = pp.get_param('alpha0_' + self.sample_name)
        alp2 = pp.get_param('alpha2_' + self.sample_name)
        sn0  = pp.get_param('SN0_' + self.sample_name)
        sn2  = pp.get_param('SN2_' + self.sample_name)
        
        bias = [b1, b2, bs, 0.]
        cterm = [alp0,alp2,0,0]
        stoch = [sn0, sn2, 0]
        bvec = bias + cterm + stoch
        
        #print(self.zstr, b1, sig8)
        
        kv, p0, p2, p4 = modPTs[self.zstr].combine_bias_terms_pkell(bvec)
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2,p4 = np.append([0.,],p2),np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        return(tt)
        #
        
    def fs_observe(self,tt):
        """Apply the window function matrix to get the binned prediction."""
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        # wide angle
        expanded_model = np.matmul(self.matM, thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        convolved_model = np.matmul(self.matW, expanded_model )
        
        #np.savetxt('pobs_' + self.zstr + '_' + self.sample_name + '.txt',convolved_model)
        
        # keep only the monopole and quadrupole
        convolved_model = convolved_model[self.fitiis]
    
        return convolved_model
    

class PT_pk_theory_zs(Theory):
    """A class to return a PT P_ell module."""
    # From yaml file.
    zfids:    list
    chiz_fids: list
    Hz_fids:   list
    #
    def initialize(self):
        """Sets up the class."""
        # Don't need to do anything.
        pass
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zmax = max(self.zfids)
        zg  = np.linspace(0,zmax,100,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        #
        req = {\
               'omegam': None,\
               'omnuh2': None,\
               'H0': None,\
               'Pk_interpolator': {'k_max': 30, 'z':zg,\
                                   'nonlinear': False,\
                                   'vars_pairs': [['delta_nonu','delta_nonu']]},\
               'Hubble':   {'z': [0.0,] + self.zfids},\
               'sigma8_z': {'z': [0.0,] + self.zfids},\
               'fsigma8':  {'z': [0.0,] + self.zfids},\
               'comoving_radial_distance': {'z': [0,] + self.zfids}\
              }
        return(req)
    def get_can_provide(self):
        """What do we provide: a PT class that can compute xi_ell."""
        return ['pt_pk_ell_mod']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """Create and initialize the PT class."""
        # Make shorter names.
        pp   = self.provider
        
        # Get cosmological parameters
        OmM = pp.get_param('omegam')
        hub  = pp.get_Hubble(0)[0]/100.
        omnuh2 = pp.get_param('omnuh2')
        fnu =  omnuh2/ hub**2 /OmM
        
        modPTs = {}
        
        for zfid, chiz_fid, Hz_fid in zip(self.zfids,self.chiz_fids, self.Hz_fids):
            zstr = "%.2f"%(zfid)
            ff   = f_of_a(1/(1.+zfid), OmegaM=pp.get_param('omegam')) * (1 - 0.6 * fnu)

            ki   = np.logspace(-3.0,1.0,200)
            pi   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])
            pi   = pi.P(zfid,ki*hub)*hub**3
        
            # Work out the A-P scaling to the fiducial cosmology.        
            Hz   = pp.get_Hubble(zfid)[0]/pp.get_Hubble(0)[0]
            chiz = pp.get_comoving_radial_distance(zfid)[0]*hub
            #print(zfid, Hz, Hz_fid, chiz, chiz_fid)
            apar,aperp = Hz_fid/Hz,chiz/chiz_fid

            modPTs[zstr] = LPT_RSD(ki, pi, kIR=0.2,\
                             cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=8, jn=5)
        
            # this k vector is a custom-made monstrosity for our devilish aims
            kvec = np.concatenate( ([0.0005,],\
                                    np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                                    np.arange(0.03,0.51,0.01)) )
            modPTs[zstr].make_pltable(ff, kv=kvec, apar=apar, aperp=aperp, ngauss=2)
        #modPT.make_pltable(ff, kmin=1e-3, kmax=0.5, nk=200, apar=apar, aperp=aperp, ngauss=2)
        #
        state['pt_pk_ell_mod'] = modPTs
        #
