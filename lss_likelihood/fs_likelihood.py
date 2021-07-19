import numpy as np

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from linear_theory import*

import sys
sys.path.append('/global/homes/s/sfschen/Python/velocileptors/')

from velocileptors.LPT.lpt_rsd_fftw          import LPT_RSD
from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT

class FSLikelihood(Likelihood):
    
    zfid: float
    Hz_fid: float
    chiz_fid: float
    
    fs_datfn: str
    covfn: str
    
    fs_kmin: float
    fs_mmax: float
    fs_qmax: float
    fs_matMfn: str
    fs_matWfn: str
        
    #bao_templatefn: str
    #bao_templaters: float
    #bao_rmax: float
    #bao_rmin: float
        
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        #

    def get_requirements(self):
        req = {'pt_pk_ell_mod': None,\
               'bsig8': None,\
               'b2': None,\
               'bs': None,\
               'alpha0': None,\
               'alpha2': None,\
               'SN0': None,\
               'SN2': None,\
              }
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        fs_thy  = self.fs_predict()
        fs_obs  = self.fs_observe(fs_thy)

        obs = fs_obs
        
        chi2 = np.dot(self.dd-obs,np.dot(self.cinv,self.dd-obs))
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

        # Join the data vectors together
        self.dd = np.concatenate( (self.p0dat, self.p2dat))
        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        
        # We're only going to want some of the entries in computing chi^2.
        kcut = (self.kdat < self.fs_mmax) * (self.kdat > self.fs_kmin)
        for i in np.nonzero(kcut)[0]:     # FS Monopole.
            ii = i + 0*self.kdat.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
        
        kcut = (self.kdat < self.fs_mmax) * (self.kdat > self.fs_kmin)
        for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
            ii = i + 1*self.kdat.size
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20

        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        # Finally load the window function matrix.
        self.matM = np.loadtxt(self.fs_matMfn)
        self.matW = np.loadtxt(self.fs_matWfn)
        
        #
    def fs_predict(self):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        modPT= pp.get_result('pt_pk_ell_mod')
        hub  = pp.get_Hubble(0)[0]/100.
        sig8 = pp.get_sigma8_z(0)[0]
        #
        b1   = pp.get_param('bsig8')/sig8 - 1.0
        b2   = pp.get_param('b2')
        bs   = pp.get_param('bs')
        alp0 = pp.get_param('alpha0')
        alp2 = pp.get_param('alpha2')
        sn0  = pp.get_param('SN0')
        sn2  = pp.get_param('SN2')
        #
        bias = [b1,b2,bs,0.] # Set b3=0
        cterm= [alp0,alp2,0] # Set alpha4=0 if no hexadecapole
        stoch= [sn0,sn2]
        bpars= bias + cterm + stoch
        
        # Compute the growth rate and work out the A-P scaling.
        s8   = pp.get_sigma8_z(self.zfid)[0]
        fs8  = pp.get_fsigma8(self.zfid)[0]
        ff   = fs8 / s8
        print(ff, fs8, s8)
        #ff   = f_of_a(1/(1.+self.zfid), OmegaM=pp.get_param('omegam'))
        # Work out the A-P scaling to the fiducial cosmology.
        Hz   = pp.get_Hubble(self.zfid)[0]/pp.get_Hubble(0)[0]
        chiz = pp.get_comoving_radial_distance(self.zfid)[0]*hub
        apar,aperp = modPT.Hz_fid/Hz,chiz/modPT.chiz_fid
        # Call the PT model to get P_ell -- we'll grid it onto the
        # appropriate binning for the window function in observe.
        kv,p0,p2,p4=modPT.compute_redshift_space_power_multipoles(bpars,\
                         ff,apar=apar,aperp=aperp,reduced=True)
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([sn0,],p0)
        p2,p4 = np.append([0.,],p2),np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        return(tt)
        #
        
    def fs_observe(self,tt):
        """Apply the window function matrix to get the binned prediction."""
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        thy =                     Spline(tt[:,0],tt[:,1])(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2])(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3])(kv)])
        
        # wide angle
        expanded_model = np.matmul(self.matM, thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        convolved_model = np.matmul(self.matW, expanded_model )
    
        return convolved_model
    

class PT_pk_theory(Theory):
    """A class to return a PT P_ell module."""
    # From yaml file.
    zfid:     float
    chiz_fid: float
    Hz_fid:   float
    #
    def initialize(self):
        """Sets up the class."""
        # Don't need to do anything.
        pass
    def get_requirements(self):
        """What we need in order to provide P_ell."""
        zg  = np.linspace(0,self.zfid,8,endpoint=True)
        # Don't need sigma8_z, fsigma8 or radial distance
        # here, but want them up in likelihood and they
        # only depend on cosmological things (not biases).
        req = {\
               'omegam': None,\
               'Pk_interpolator': {'k_max': 30,'z': zg,\
                                   'nonlinear': False,\
                                   'vars_pairs': [['delta_nonu','delta_nonu']]},\
               'Hubble':   {'z': [0.0,self.zfid]},\
               'sigma8_z': {'z': [0.0,self.zfid]},\
               'fsigma8':  {'z': [self.zfid]},\
               'comoving_radial_distance': {'z': [self.zfid]}\
              }
        return(req)
    def get_can_provide(self):
        """What do we provide: a PT class that can compute xi_ell."""
        return ['pt_pk_ell_mod']
    def calculate(self, state, want_derived=True, **params_values_dict):
        """Create and initialize the PT class."""
        # Make shorter names.
        pp   = self.provider
        zfid = self.zfid
        # Get cosmological parameters
        hub  = pp.get_Hubble(0)[0]/100.
        #ff   = f_of_a(1/(1.+zfid), OmegaM=pp.get_param('omegam'))
        #print(ff)
        s8   = pp.get_sigma8_z(self.zfid)[0]
        fs8  = pp.get_fsigma8(self.zfid)[0]
        ff   = fs8 / s8
        print(self.zfid, pp.get_param('omegam'), fs8, s8)
        # and Plin.
        ki   = np.logspace(-3.0,1.5,750)
        pi   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])
        pi   = pi.P(self.zfid,ki*hub)*hub**3
        
        # Work out the A-P scaling to the fiducial cosmology.
        #Hz   = pp.get_Hubble(self.zfid)[0]/pp.get_Hubble(0)[0]
        #chiz = pp.get_comoving_radial_distance(self.zfid)[0]*hub
        #apar,aperp = self.Hz_fid/Hz,chiz/self.chiz_fid
        
        # Now generate and save the PT model
        modPT = MomentExpansion(ki,pi,beyond_gauss=False,\
                      one_loop=True,shear=True,\
                      import_wisdom=False,\
                      kmin=1e-4,kmax=0.5,nk=200,cutoff=10,\
                      extrap_min=-4,extrap_max=3,N=2000,jn=10)
        modPT.zfid     = self.zfid
        modPT.chiz_fid = self.chiz_fid
        modPT.Hz_fid   = self.Hz_fid
        #
        state['pt_pk_ell_mod'] = modPT
        #
