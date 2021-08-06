import numpy as np
import time

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.signal import argrelmax
from scipy.interpolate import interp1d
from numpy.polynomial.polynomial import polyval

from scipy.special import spherical_jn
from scipy.integrate import simps

from linear_theory import*
from pnw_dst import pnw_dst

#from zeldovich_rsd_recon_fftw import Zeldovich_Recon
from velocileptors.LPT.lpt_rsd_fftw import LPT_RSD
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT

# Class to have a full-shape likelihood with multiple zs

class JointLikelihood(Likelihood):
    
    zfid: float
    Hz_fid: float
    chiz_fid: float
    fs_sample_name: str
    bao_sample_name: str
    
    fs_datfn: str
    bao_datfn: str

    covfn: str
    
    fs_kmin: float
    fs_mmax: float
    fs_qmax: float
    fs_matMfn: str
    fs_matWfn: str
    
    bao_rmax: float
    bao_rmin: float

    def initialize(self):
        """Sets up the class."""
        # Redshift Label for theory classes
        self.zstr = "%.2f" %(self.zfid)
        print(self.bao_datfn)
        
        # Spherical Bessel Transform objects for correlation function
        self.kint = np.logspace(-3, 2, 2000)
        self.sphr = SBT(self.kint,L=5,fourier=True,low_ring=False)
        
        self.loadData()
        #

    def get_requirements(self):
        req = {'pt_pk_ell_mod': None,\
               'pt_recon_zel_mod': None,\
               'H0': None,\
               'sigma8': None,\
               'omegam': None,\
               'bsig8_' + self.fs_sample_name: None,\
               'b2_' + self.fs_sample_name: None,\
               'bs_' + self.fs_sample_name: None,\
               'alpha0_' + self.fs_sample_name: None,\
               'alpha2_' + self.fs_sample_name: None,\
               'SN0_' + self.fs_sample_name: None,\
               'SN2_' + self.fs_sample_name: None,\
               'B1_' + self.bao_sample_name: None,\
               'F_' + self.bao_sample_name: None,\
               'M0_' + self.bao_sample_name: None,\
               'M1_' + self.bao_sample_name: None,\
               'M2_' + self.bao_sample_name: None,\
               'Q0_' + self.bao_sample_name: None,\
               'Q1_' + self.bao_sample_name: None,\
               'Q2_' + self.bao_sample_name: None,
              }
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        fs_thy  = self.fs_predict()
        fs_obs  = self.fs_observe(fs_thy)
        
        bao_thy = self.bao_predict()
        bao_obs = self.bao_observe(bao_thy)
        
        
        diff = self.dd - np.concatenate( (fs_obs,bao_obs) )
        
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
        
        bao_dat = np.loadtxt(self.bao_datfn)
        self.rdat = bao_dat[:,0]
        self.xi0dat = bao_dat[:,1]
        self.xi2dat = bao_dat[:,2]
        
        # Make a list of indices for the monopole and quadrupole only
        yeses = self.kdat > 0
        nos   = self.kdat < 0
        #yeses_bao = self.rdat > 0
        #nos_bao   = self.rdat < 0 # in fact we never use this because the xi files only have xi0, xi2
        #self.fitiis = np.concatenate( (yeses, nos, yeses, nos, nos ) + (yeses_bao, yeses_bao) )
        self.fitiis = np.concatenate( (yeses, nos, yeses, nos, nos ) )

        # Join the data vectors together
        #self.dd = np.concatenate( (self.p0dat, self.p2dat))
        
        # Now load the covariance matrix.
        #cov = np.loadtxt(self.fs_covfn)
        #cov = cov[np.ix_(self.fitiis, self.fitiis)]
        
        # Join the data vectors together
        self.dd = np.concatenate( (self.p0dat, self.p2dat, self.xi0dat, self.xi2dat))
        
        # Now load the covariance matrix.
        cov = np.loadtxt(self.covfn)
        
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
            
        rcut = (self.rdat < self.bao_rmin) | (self.rdat > self.bao_rmax)
        for i in np.nonzero(rcut)[0]:
            ii = i + 2*self.kdat.size + 0 * self.rdat.size
            cov[ii,:] = 0
            cov[:,ii] = 0
            cov[ii,ii] = 1e25
            
        for i in np.nonzero(rcut)[0]:
            ii = i + 2*self.kdat.size + 1 * self.rdat.size
            cov[ii,:] = 0
            cov[:,ii] = 0
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
        OmM = pp.get_param('omegam')

        #
        b1   = pp.get_param('bsig8_' + self.fs_sample_name)/sig8 - 1.0
        b2   = pp.get_param('b2_' + self.fs_sample_name)
        bs   = pp.get_param('bs_' + self.fs_sample_name)
        alp0 = pp.get_param('alpha0_' + self.fs_sample_name)
        alp2 = pp.get_param('alpha2_' + self.fs_sample_name)
        sn0  = pp.get_param('SN0_' + self.fs_sample_name)
        sn2  = pp.get_param('SN2_' + self.fs_sample_name)
        
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
        
        if np.any(np.isnan(tt)):
            print("NaN's encountered. Parameter values are: ", str(hub,sig8,OmM))
        
        return(tt)
        #
        
    def compute_bao_pkmu(self, mu_obs):
        '''
        Helper function to get P(k,mu) post-recon in RecIso.
        
        This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
        '''
    
        pp = self.provider
        B1   = pp.get_param('B1_' + self.bao_sample_name)
        F   = pp.get_param('F_' + self.bao_sample_name)
        
        Zels = pp.get_result('pt_recon_zel_mod')
        
        f0 = Zels[self.zstr]['fz']
        apar, aperp = Zels[self.zstr]['alphas']
        
        klin = Zels[self.zstr]['klin']
        pnw = Zels[self.zstr]['pnw']
        pw  = Zels[self.zstr]['pw']
        
        sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds = Zels[self.zstr]['sigmas']
        R = Zels[self.zstr]['R']

        Sk = np.exp(-0.5*(klin*R)**2)
        
        # Our philosophy here is to take kobs = klin
        # Then we predict P(ktrue) on the klin grid, so that the final answer is
        # Pobs(kobs,mu_obs) ~ Ptrue(ktrue, mu_true) = interp( klin, Ptrue(klin, mu_true) )(ktrue)
        # (Up to a normalization that we drop.)
        # Interpolating this way avoids having to interpolate Pw and Pnw separately.
        
        F_AP = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
        mu = mu_obs / F_AP / AP_fac
        ktrue = klin/aperp*AP_fac
        
        # First construct P_{dd,ss,ds} individually
        dampfac_dd = np.exp( -0.5 * klin**2 * sigmadd * (1 + f0*(2+f0)*mu**2) )
        pdd = ( (1 + F*mu**2)*(1-Sk) + B1 )**2 * (dampfac_dd * pw + pnw)
        
        # then Pss
        dampfac_ss = np.exp( -0.5 * klin**2 * sigmass )
        pss = Sk**2 * (dampfac_ss * pw + pnw)
        
        # Finally Pds
        dampfac_ds = np.exp(-0.5 * klin**2 * ( 0.5*sigmads_dd*(1+f0*(2+f0)*mu**2)\
                                             + 0.5*sigmads_ss \
                                             + (1+f0*mu**2)*sigmads_ds) )
        linfac = - Sk * ( (1+F*mu**2)*(1-Sk) + B1 )
        pds = linfac * (dampfac_ds * pw + pnw)
        
        # Sum it all up and interpolate?
        ptrue = pdd + pss - 2*pds
        pmodel = interp1d(klin, ptrue, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    
        return pmodel
    
    def bao_predict(self):
        
        pp = self.provider
        
        # Generate the sampling
        ngauss = 4
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        #L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        
        Zels = pp.get_result('pt_recon_zel_mod')
        klin = Zels[self.zstr]['klin']
        pknutable = np.zeros((len(nus),len(klin)))
    
        for ii, nu in enumerate(nus_calc):
            pknutable[ii,:] = self.compute_bao_pkmu(nu)
 
        pknutable[ngauss:,:] = np.flip(pknutable[0:ngauss],axis=0)
        
        p0 = 0.5 * np.sum((ws*L0)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[m0,m1,m2,m3,m4,m5]) / klin
        p2 = 2.5 * np.sum((ws*L2)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[q0,q1,q2,q3,q4,q5]) / klin
        #p4 = 4.5 * np.sum((ws*L4)[:,None]*pknutable,axis=0)

    
        p0t = interp1d(klin,p0, kind='cubic', bounds_error=False, fill_value=0)(self.kint)
        p2t = interp1d(klin,p2, kind='cubic', bounds_error=False, fill_value=0)(self.kint)
        #p4t = 0 * kint
        
        damping = np.exp(-(self.kint/10)**2)
        rr0, xi0t = self.sphr.sph(0,p0t * damping)
        rr2, xi2t = self.sphr.sph(2,p2t * damping); xi2t *= -1
        #rr2, xi4t = sphr.sph(4,p4t)
        #xi4t = 0 * rr0 # no hexadecapole to speed things up
                            
        M0, M1, M2 = [pp.get_param(param_name + '_' +self.bao_sample_name) for param_name in ['M0','M1','M2']]
        Q0, Q1, Q2 = [pp.get_param(param_name + '_' +self.bao_sample_name) for param_name in ['Q0','Q1','Q2']]
        
        xi0t += polyval(1/rr0,[M0,M1,M2])
        xi2t += polyval(1/rr0,[Q0,Q1,Q2])
        
        np.savetxt('xitest.dat', np.array([rr0,xi0t,xi2t]).T)
        
        return np.array([rr0,xi0t,xi2t]).T
        
    def fs_observe(self,tt):
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
        expanded_model = np.matmul(self.matM, thy )
        # Convolve with window (true) −> (conv) see eq. 2.18
        convolved_model = np.matmul(self.matW, expanded_model )
        
        #np.savetxt('pobs_' + self.zstr + '_' + self.sample_name + '.txt',convolved_model)
        
        # keep only the monopole and quadrupole
        convolved_model = convolved_model[self.fitiis]
    
        return convolved_model
    
    def bao_observe(self, tt):
        '''
        Bin the BAO results... probabaly should eventually use a matrix.
        '''
        
        thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
        thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
        #thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
        
        dr   = self.rdat[1]-self.rdat[0]
        
        tmp0 = np.zeros_like(self.rdat)
        tmp2 = np.zeros_like(self.rdat)
        
        for i in range(self.rdat.size):
            
            kl = self.rdat[i]-dr/2
            kr = self.rdat[i]+dr/2

            ss = np.linspace(kl, kr, 100)
            p0     = thy0(ss)
            tmp0[i]= np.trapz(ss**2*p0,x=ss)*3/(kr**3-kl**3)
            p2     = thy2(ss)
            tmp2[i]= np.trapz(ss**2*p2,x=ss)*3/(kr**3-kl**3)
            #p4     = thy4(ss)
            #tmp4[i]= np.trapz(ss**2*p4,x=ss)*3/(kr**3-kl**3)
        
        return np.concatenate((tmp0,tmp2))
    
    

class PT_pk_theory_zs(Theory):
    """A class to return PT modules for full shape and BAO."""
    # From yaml file.
    
    zfids:    list
    chiz_fids: list
    Hz_fids:   list
        
    bao_R: float
    
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
               'rdrag': None,\
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
        return ['pt_pk_ell_mod', 'pt_recon_zel_mod']
    
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
        Zels = {}
        
        for zfid, chiz_fid, Hz_fid in zip(self.zfids,self.chiz_fids, self.Hz_fids):
            
            
            zstr = "%.2f"%(zfid)
            ff   = f_of_a(1/(1.+zfid), OmegaM=pp.get_param('omegam')) * (1 - 0.6 * fnu)

            t1 = time.time()
            
            ki   = np.logspace(-3.0,2.0,300)
            pi   = pp.get_Pk_interpolator(nonlinear=False,var_pair=['delta_nonu','delta_nonu'])
            pi   = pi.P(zfid,ki*hub)*hub**3
        
            # Work out the A-P scaling to the fiducial cosmology.        
            Hz   = pp.get_Hubble(zfid)[0]/pp.get_Hubble(0)[0]
            chiz = pp.get_comoving_radial_distance(zfid)[0]*hub
            #print(zfid, Hz, Hz_fid, chiz, chiz_fid)
            apar,aperp = Hz_fid/Hz,chiz/chiz_fid
            
            t2 = time.time()
            print("Power spectrum and background time.", t2-t1)
            
            # Do the Full Shape Predictions
            t1 = time.time()
            modPTs[zstr] = LPT_RSD(ki, pi, kIR=0.2,\
                             cutoff=10, extrap_min = -4, extrap_max = 3, N = 2000, threads=8, jn=5)
        
            # this k vector is a custom-made monstrosity for our devilish aims
            kvec = np.concatenate( ([0.0005,],\
                                    np.logspace(np.log10(0.0015),np.log10(0.025),10, endpoint=True),\
                                    np.arange(0.03,0.51,0.01)) )
            modPTs[zstr].make_pltable(ff, kv=kvec, apar=apar, aperp=aperp, ngauss=2)
            #modPT.make_pltable(ff, kmin=1e-3, kmax=0.5, nk=200, apar=apar, aperp=aperp, ngauss=2)
            t2 = time.time()
            print("Full shape time", t2-t1)
            
            # Do the Zeldovich reconstruction predictions
            t2 = time.time()
            knw, pnw = pnw_dst(ki, pi)
            #knw, pnw = pnw_dst(ki,pi,ii_l=135,ii_r=225)
            pw = pi - pnw
            
            qbao   = pp.get_param('rdrag') * hub # want this in Mpc/h units
            
            t1 = time.time()

            j0 = spherical_jn(0,ki*qbao)
            Sk = np.exp(-0.5*(ki*15)**2)

            sigmadd = simps( 2./3 * pi * (1-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)
            sigmass = simps( 2./3 * pi * (-Sk)**2 * (1-j0), x = ki) / (2*np.pi**2)

            sigmads_dd = simps( 2./3 * pi * (1-Sk)**2, x = ki) / (2*np.pi**2)
            sigmads_ss = simps( 2./3 * pi * (-Sk)**2, x = ki) / (2*np.pi**2)
            sigmads_ds = -simps( 2./3 * pi * (1-Sk)*(-Sk)*j0, x = ki) / (2*np.pi**2) # this minus sign is because we subtract the cross term
        
            Zels[zstr] = {'R': self.bao_R,
                          'fz':ff,\
                          'alphas': (apar, aperp),\
                          'klin': ki, 'pnw': pnw, 'pw': pw,\
                          'sigmas': (sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds)}
            
            t2 = time.time()
            print("BAO time", t2-t1)

        #
        state['pt_pk_ell_mod'] = modPTs
        state['pt_recon_zel_mod'] = Zels
        #
