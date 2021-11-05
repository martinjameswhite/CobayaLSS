import numpy as np
import time
import json

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
from scipy.integrate import simps

from taylor_approximation import taylor_approximate
from compute_sigma8 import compute_sigma8

# Class to have a full-shape likelihood for a bunch of pieces of data from both galactic caps in the same z bin
# Currently assumes all data have the same fiducial cosmology etc.
# If not I suggest chaning the theory class so that instead of being labelled by "zstr" it gets labelled by sample name.
# And each sample name indexing the fiducial cosmology numbers (chi, Hz etc) in dictionaries. For another time...

class JointLikelihood(Likelihood):
    
    zfid: float
    Hz_fid: float
    chiz_fid: float
    
    fs_sample_names: list
    bao_sample_names: list
    
    fs_datfns: list
    bao_datfns: list

    covfn: str
    
    fs_kmins: list
    fs_mmaxs: list
    fs_qmaxs: list
    fs_matMfns: list
    fs_matWfns: list
    
    bao_rmaxs: list
    bao_rmins: list

    def initialize(self):
        """Sets up the class."""
        # Redshift Label for theory classes
        self.zstr = "%.2f" %(self.zfid)
        print(self.bao_sample_names,self.bao_datfns)
        print(self.fs_sample_names,self.fs_datfns)
        
        print("We are here!")
        
        self.pconv = {}
        self.xith = {}
        
        self.loadData()
        #

    def get_requirements(self):
        
        req = {'taylor_pk_ell_mod': None,\
               'taylor_xi_ell_mod': None,\
               'H0': None,\
               'sigma8': None,\
               'omegam': None,\
                'logA': None}
        
        for fs_sample_name in self.fs_sample_names:
            req_bias = { \
                   'bsig8_' + fs_sample_name: None,\
                   'b2_' + fs_sample_name: None,\
                   'bs_' + fs_sample_name: None,\
                   'alpha0_' + fs_sample_name: None,\
                   'alpha2_' + fs_sample_name: None,\
                   'SN0_' + fs_sample_name: None,\
                   'SN2_' + fs_sample_name: None\
                   }
            req = {**req, **req_bias}
        
        for bao_sample_name in self.bao_sample_names:
            req_bao = {\
                   'B1_' + bao_sample_name: None,\
                   'F_' +  bao_sample_name: None,\
                   'M0_' + bao_sample_name: None,\
                   'M1_' + bao_sample_name: None,\
                   'M2_' + bao_sample_name: None,\
                   'Q0_' + bao_sample_name: None,\
                   'Q1_' + bao_sample_name: None,\
                   'Q2_' + bao_sample_name: None,\
                    }
            req = {**req, **req_bao}
            
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        thy_obs = []
        
        for fs_sample_name in self.fs_sample_names:
            fs_thy  = self.fs_predict(fs_sample_name)
            fs_obs  = self.fs_observe(fs_thy, fs_sample_name)
            thy_obs = np.concatenate( (thy_obs,fs_obs) )
        
        for bao_sample_name in self.bao_sample_names:
            bao_thy = self.bao_predict(bao_sample_name)
            bao_obs = self.bao_observe(bao_thy,bao_sample_name)
            thy_obs = np.concatenate( (thy_obs, bao_obs) )
            
        diff = self.dd - thy_obs
        
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
        
        self.kdats = {}
        self.p0dats = {}
        self.p2dats = {}
        self.fitiis = {}
        
        for ii, fs_datfn in enumerate(self.fs_datfns):
            fs_sample_name = self.fs_sample_names[ii]
            fs_dat = np.loadtxt(fs_datfn)
            self.kdats[fs_sample_name] = fs_dat[:,0]
            self.p0dats[fs_sample_name] = fs_dat[:,1]
            self.p2dats[fs_sample_name] = fs_dat[:,2]
            
            # Make a list of indices for the monopole and quadrupole only in Fourier space
            # This is specified to each sample in case the k's are different.
            yeses = self.kdats[fs_sample_name] > 0
            nos   = self.kdats[fs_sample_name] < 0
            self.fitiis[fs_sample_name] = np.concatenate( (yeses, nos, yeses, nos, nos ) )
        
        self.rdats = {}
        self.xi0dats = {}
        self.xi2dats = {}
        
        for ii, bao_datfn in enumerate(self.bao_datfns):
            bao_sample_name = self.bao_sample_names[ii]
            bao_dat = np.loadtxt(bao_datfn)
            self.rdats[bao_sample_name] = bao_dat[:,0]
            self.xi0dats[bao_sample_name] = bao_dat[:,1]
            self.xi2dats[bao_sample_name] = bao_dat[:,2]
        
        # Join the data vectors together
        self.dd = []
        
        for fs_sample_name in self.fs_sample_names:
            self.dd = np.concatenate( (self.dd, self.p0dats[fs_sample_name], self.p2dats[fs_sample_name]) )
            
        for bao_sample_name in self.bao_sample_names:
            self.dd = np.concatenate( (self.dd, self.xi0dats[bao_sample_name], self.xi2dats[bao_sample_name]) )

        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        
        # We're only going to want some of the entries in computing chi^2.
        
        # this is going to tell us how many indices to skip to get to the nth multipole
        startii = 0
        
        for ss, fs_sample_name in enumerate(self.fs_sample_names):
            
            kcut = (self.kdats[fs_sample_name] > self.fs_mmaxs[ss])\
                          | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:     # FS Monopole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
            
            kcut = (self.kdats[fs_sample_name] > self.fs_qmaxs[ss])\
                       | (self.kdats[fs_sample_name] < self.fs_kmins[ss])
            
            for i in np.nonzero(kcut)[0]:       # FS Quadrupole.
                ii = i + startii
                cov[ii, :] = 0
                cov[ :,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.kdats[fs_sample_name].size
        
        for ss, bao_sample_name in enumerate(self.bao_sample_names):
            
            rcut = (self.rdats[bao_sample_name] < self.bao_rmins[ss])\
                              | (self.rdats[bao_sample_name] > self.bao_rmaxs[ss])
            
            for i in np.nonzero(rcut)[0]:
                ii = i + startii
                cov[ii,:] = 0
                cov[:,ii] = 0
                cov[ii,ii] = 1e25
                
            startii += self.rdats[bao_sample_name].size
            
            for i in np.nonzero(rcut)[0]:
                ii = i + startii
                cov[ii,:] = 0
                cov[:,ii] = 0
                cov[ii,ii] = 1e25
            
            startii += self.rdats[bao_sample_name].size
        
        
        # Copy it and save the inverse.
        self.cov  = cov
        self.cinv = np.linalg.inv(self.cov)
        #print(self.sample_name, np.diag(self.cinv)[:10])
        
        # Finally load the window function matrix.
        self.matMs = {}
        self.matWs = {}
        for ii, fs_sample_name in enumerate(self.fs_sample_names):
            self.matMs[fs_sample_name] = np.loadtxt(self.fs_matMfns[ii])
            self.matWs[fs_sample_name] = np.loadtxt(self.fs_matWfns[ii])
        
        #
        
    def combine_bias_terms_pkell(self,bvec, p0ktable, p2ktable, p4ktable):
        '''
        Same as function above but for the multipoles.
        
        Returns k, p0, p2, p4, assuming AP parameters from input p{ell}ktable
        '''
    
        b1,b2,bs,b3,alpha0,alpha2,alpha4,alpha6,sn,sn2,sn4 = bvec

        bias_monomials = np.array([1, b1, b1**2,\
                                   b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3,\
                                   alpha0, alpha2, alpha4,alpha6,sn,sn2,sn4])

        p0 = np.sum(p0ktable * bias_monomials,axis=1)
        p2 = np.sum(p2ktable * bias_monomials,axis=1)
        p4 = np.sum(p4ktable * bias_monomials,axis=1)
        
        return p0, p2, p4
    
    def fs_predict(self, fs_sample_name):
        """Use the PT model to compute P_ell, given biases etc."""
        
        pp   = self.provider
        
        taylorPTs = pp.get_result('taylor_pk_ell_mod')
        kv, p0ktable, p2ktable, p4ktable = taylorPTs[self.zstr]

        #
        sig8 = pp.get_param('sigma8')
        #sig8 = pp.get_result('sigma8')
        b1   = pp.get_param('bsig8_' + fs_sample_name)/sig8 - 1.0
        b2   = pp.get_param('b2_' + fs_sample_name)
        bs   = pp.get_param('bs_' + fs_sample_name)
        alp0 = pp.get_param('alpha0_' + fs_sample_name)
        alp2 = pp.get_param('alpha2_' + fs_sample_name)
        sn0  = pp.get_param('SN0_' + fs_sample_name)
        sn2  = pp.get_param('SN2_' + fs_sample_name)
        
        bias = [b1, b2, bs, 0.]
        cterm = [alp0,alp2,0,0]
        stoch = [sn0, sn2, 0]
        bvec = bias + cterm + stoch
        
        #print(self.zstr, b1, sig8)
        
        p0, p2, p4 = self.combine_bias_terms_pkell(bvec, p0ktable, p2ktable, p4ktable)
        
        #np.savetxt('pells_' + self.zstr + '_' + self.sample_name + '.txt',[kv,p0,p2,p4])
        
        # Put a point at k=0 to anchor the low-k part of the Spline.
        kv,p0 = np.append([0.,],kv),np.append([0.0,],p0)
        p2 = np.append([0.,],p2)
        p4 = np.append([0.0,],p4)
        tt    = np.array([kv,p0,p2,p4]).T
        
        if np.any(np.isnan(tt)):
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        
        return(tt)
        #
    
    def bao_predict(self, bao_sample_name):
        
        pp   = self.provider
        
        B1   = pp.get_param('B1_' + bao_sample_name)
        F   = pp.get_param('F_' + bao_sample_name)
        M0, M1, M2 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['M0','M1','M2']]
        Q0, Q1, Q2 = [pp.get_param(param_name + '_' + bao_sample_name) for param_name in ['Q0','Q1','Q2']]
        
        taylorPTs = pp.get_result('taylor_xi_ell_mod')
        rvec, xi0table, xi2table = taylorPTs[self.zstr]
        
        xi0t = xi0table[:,0] + B1*xi0table[:,1] + F*xi0table[:,2] \
             + B1**2 * xi0table[:,3] + F**2 * xi0table[:,4] + B1*F*xi0table[:,5]
        
        xi2t = xi2table[:,0] + B1*xi2table[:,1] + F*xi2table[:,2] \
             + B1**2 * xi2table[:,3] + F**2 * xi2table[:,4] + B1*F*xi2table[:,5]
        
        xi0t += polyval(1/rvec,[M0,M1,M2])
        xi2t += polyval(1/rvec,[Q0,Q1,Q2])
        
        return np.array([rvec,xi0t,xi2t]).T
    

        
    def fs_observe(self,tt,fs_sample_name):
        """Apply the window function matrix to get the binned prediction."""
        
        # Have to stack ell=0, 2 & 4 in bins of 0.001h/Mpc from 0-0.4h/Mpc.
        kv  = np.linspace(0.0,0.4,400,endpoint=False) + 0.0005
        thy =                     Spline(tt[:,0],tt[:,1],ext=3)(kv)
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,2],ext=3)(kv)])
        thy = np.concatenate([thy,Spline(tt[:,0],tt[:,3],ext=3)(kv)])
        
        if np.any(np.isnan(thy)) or np.max(thy) > 1e8:
            hub = self.provider.get_param('H0') / 100.
            sig8 = self.provider.get_param('sigma8')
            OmM = self.provider.get_param('omegam')
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        
        # wide angle
        expanded_model = np.matmul(self.matMs[fs_sample_name], thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        # Multiply by ad-hoc factor
        convolved_model = 0.89 * np.matmul(self.matWs[fs_sample_name], expanded_model )
        
        #np.savetxt('pobs_' + self.zstr + '_' + self.sample_name + '.txt',convolved_model)
        
        # keep only the monopole and quadrupole
        convolved_model = convolved_model[self.fitiis[fs_sample_name]]
        
        # Save the model:
        self.pconv[fs_sample_name] = convolved_model
    
        return convolved_model
    
    def bao_observe(self, tt, bao_sample_name):
        '''
        Bin the BAO results... probabaly should eventually use a matrix.
        '''
        
        rdat = self.rdats[bao_sample_name]
        
        thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
        thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
        #thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
        
        dr   = rdat[1]- rdat[0]
        
        tmp0 = np.zeros_like(rdat)
        tmp2 = np.zeros_like(rdat)
        
        for i in range(rdat.size):
            
            kl = rdat[i]-dr/2
            kr = rdat[i]+dr/2

            ss = np.linspace(kl, kr, 100)
            p0     = thy0(ss)
            tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
            p2     = thy2(ss)
            tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
            #p4     = thy4(ss)
            #tmp4[i]= np.trapz(ss**2*p4,x=ss)*3/(kr**3-kl**3)
            
        self.xith[bao_sample_name] = np.concatenate((tmp0,tmp2))
        
        return np.concatenate((tmp0,tmp2))
    

class Taylor_pk_theory_zs(Theory):
    """
    A class to return a set of derivatives for the Taylor series of Pkell.
    """
    zfids: list
    pk_filenames: list
    xi_filenames: list
    
    def initialize(self):
        """Sets up the class by loading the derivative matrices."""
        
        print("Loading Taylor series.")
        
        self.taylors_pk = {}
        self.taylors_xi = {}
        
        for zfid, pk_filename, xi_filename in zip(self.zfids, self.pk_filenames, self.xi_filenames):
            zstr = "%.2f"%(zfid)
            taylors_pk = {}
            taylors_xi = {}
            
            # Load the power spectrum derivatives
            json_file = open(pk_filename, 'r')
            emu = json.load( json_file )
            json_file.close()
            
            x0s = emu['x0']
            kvec = emu['kvec']
            derivs_p0 = [np.array(ll) for ll in emu['derivs0']]
            derivs_p2 = [np.array(ll) for ll in emu['derivs2']]
            derivs_p4 = [np.array(ll) for ll in emu['derivs4']]
            
            taylors_pk['x0'] = np.array(x0s)
            taylors_pk['kvec'] = np.array(kvec)
            taylors_pk['derivs_p0'] = derivs_p0
            taylors_pk['derivs_p2'] = derivs_p2
            taylors_pk['derivs_p4'] = derivs_p4
            
            # Load the correlation function derivatives
            json_file = open(xi_filename, 'r')
            emu = json.load( json_file )
            json_file.close()
            
            x0s = emu['x0']
            rvec = emu['rvec']
            derivs_x0 = [np.array(ll) for ll in emu['derivs0']]
            derivs_x2 = [np.array(ll) for ll in emu['derivs2']]
            
            taylors_xi['x0'] = np.array(x0s)
            taylors_xi['rvec'] = np.array(rvec)
            taylors_xi['derivs_xi0'] = derivs_x0
            taylors_xi['derivs_xi2'] = derivs_x2

            self.taylors_pk[zstr] = taylors_pk
            self.taylors_xi[zstr] = taylors_xi
            
            del emu
    
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
               'H0': None,\
               'logA': None,\
              }
        
        return(req)
    def get_can_provide(self):
        """What do we provide: a Taylor series class for pkells."""
        return ['taylor_pk_ell_mod','taylor_xi_ell_mod']
    
    def get_can_provide_params(self):
        return ['sigma8']
    
    def calculate(self, state, want_derived=True, **params_values_dict):
        """
        Just load up the derivatives and things.
        """
        pp = self.provider
        
        hub = pp.get_param('H0') / 100.
        logA = pp.get_param('logA')
        #sig8 = pp.get_param('sigma8')
        OmM = pp.get_param('omegam')
        sig8 = compute_sigma8(OmM,hub,logA)
        cosmopars = [OmM, hub, sig8]
        
        ptables = {}
        xitables = {}
        
        for zfid in self.zfids:
            zstr = "%.2f" %(zfid)
            
            # Load pktables
            x0s = self.taylors_pk[zstr]['x0']
            derivs0 = self.taylors_pk[zstr]['derivs_p0']
            derivs2 = self.taylors_pk[zstr]['derivs_p2']
            derivs4 = self.taylors_pk[zstr]['derivs_p4']
            
            kv = self.taylors_pk[zstr]['kvec']
            p0ktable = taylor_approximate(cosmopars, x0s, derivs0, order=3)
            p2ktable = taylor_approximate(cosmopars, x0s, derivs2, order=3)
            p4ktable = taylor_approximate(cosmopars, x0s, derivs4, order=3)
            
            ptables[zstr] = (kv, p0ktable, p2ktable, p4ktable)
            
            # Load xitables
            x0s = self.taylors_xi[zstr]['x0']
            derivs_xi0 = self.taylors_xi[zstr]['derivs_xi0']
            derivs_xi2 = self.taylors_xi[zstr]['derivs_xi2']
            
            rv = self.taylors_xi[zstr]['rvec']
            xi0table = taylor_approximate(cosmopars, x0s, derivs_xi0, order=3)
            xi2table = taylor_approximate(cosmopars, x0s, derivs_xi2, order=3)
            
            xitables[zstr] = (rv, xi0table, xi2table)
            
        #state['sigma8'] = sig8
        state['derived'] = {'sigma8': sig8}
        state['taylor_pk_ell_mod'] = ptables
        state['taylor_xi_ell_mod'] = xitables