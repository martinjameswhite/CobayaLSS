import numpy as np

from cobaya.theory     import Theory
from cobaya.likelihood import Likelihood
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

from linear_theory import*

from velocileptors.LPT.lpt_rsd_fftw          import LPT_RSD
from velocileptors.LPT.moment_expansion_fftw import MomentExpansion
from velocileptors.Utils.spherical_bessel_transform import SphericalBesselTransform as SBT

class JointLikelihood(Likelihood):
    
    zfid: float
    Hz_fid: float
    chiz_fid: float
    
    fs_datfn: str
    bao_datfn: str
    covfn: str
    
    fs_kmin: float
    fs_mmax: float
    fs_qmax: float
    fs_matMfn: str
    fs_matWfn: str
        
    bao_templatefn: str
    bao_templaters: float
    bao_rmax: float
    bao_rmin: float
        
    def initialize(self):
        """Sets up the class."""
        self.loadData()
        #

    def get_requirements(self):
        req = {'pt_pk_ell_mod': None,\ # here begins the FS part
               'bsig8': None,\
               'b2': None,\
               'bs': None,\
               'alpha0': None,\
               'alpha2': None,\
               'SN0': None,\
               'SN2': None,\
               'template_f': None,\ # here begins the BAO section
               'template_klin': None,\
               'template_pnw': None,\
               'template_pw': None,\
               'template_sigmas': None,\
               'template_R': None
               'B1': None,\
               'F': None,\
               'M0': None,\
               'M1': None,\
               'M2': None,\
               'Q0': None,\
               'Q1': None,\
               'Q2': None,\
               'kint':None,\
               'sphr':None\
              }
        return(req)
    
    def logp(self,**params_values):
        """Return a log-likelihood."""
        
        fs_thy  = self.fs_predict()
        fs_obs  = self.fs_observe(fs_thy)
        
        bao_thy = self.bao_predict()
        bao_obs = self.bao_observe(bao_thy) # this will be for binning etc
        
        obs = np.concatenate( (fs_obs, bao_obs) )
        
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
        
        bao_dat = np.loadtxt(self.bao_datfn)
        self.rdat = bao_dat[:,0]
        self.xi0dat = bao_dat[:,1]
        self.xi2dat = bao_dat[:,2]
        
        # Join the data vectors together
        dd = np.concatenate( (self.p0dat, self.p2dat, self.xi0dat, self.xi2dat))
        
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
            
        rcut = (self.rdat < self.bao_rmax) * (self.rdat > self.bao_rmin)
        for i in np.nonzeros(rcut):
            ii = i + 2*self.kdat.size + 0 *self.rdat.size    #BAO Monopole
            cov[ii, :] = 0
            cov[ :,ii] = 0
            cov[ii,ii] = 1e20
            
        for i in np.nonzeros(rcut):
            ii = i + 2*self.kdat.size + 1 *self.rdat.size    #BAO Quadrupole
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
    
    def compute_bao_pkmu(self, B1, F, mu_obs, apar=1, aperp=1):
        '''
        Helper function to get P(k,mu) post-recon in RecIso.
        
        This is turned into Pkell and then Hankel transformed in the bao_predict funciton.
        '''
    
    
        f0 = self.provider.get_result('template_f')
        klin = self.provider.get_result('template_klin')
        pnw = self.provider.get_result('template_pnw')
        pw  = self.provider.get_result('template_pw')
        sigmadd, sigmass, sigmads_dd, sigmads_ss, sigmads_ds = self.provider.get_result('template_sigmas')
        R = self.provider.get_result('template_R')


        Sk = np.exp(-0.5*(klin*R)**2)
        
        F_AP = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F_AP**2 - 1) )
        mu = mu_obs / F_AP / AP_fac
        ktrue = klin/aperp*AP_fac
    
        # First shift the template:
        Gfid_shifted = interp1d(klin, pw, kind='cubic', fill_value=0,bounds_error=False)(ktrue)
    
        # First Pdd
        dampfac_dd = np.exp( -0.5 * ktrue**2 * sigmadd * (1 + f0*(2+f0)*mu**2) )
        pdd = ( (1 + F*mu**2)*(1-Sk) + B1 )**2 * dampfac_dd * Gfid_shifted
    
        # then Pss
        dampfac_ss = np.exp( -0.5 * ktrue**2 * sigmass )
        pss = Sk**2 * dampfac_ss * Gfid_shifted
    
        # Finally Pds
        dampfac_ds = np.exp(-0.5 * ktrue**2 * ( 0.5*sigmads_dd*(1+f0*(2+f0)*mu**2)\
                                          + 0.5*sigmads_ss \
                                          + (1+f0*mu**2)*sigmads_ds) )
        linfac = - Sk * ( (1+F*mu**2)*(1-Sk) + B1 )
        pds = linfac * dampfac_ds * Gfid_shifted
    
        pmodel = pdd + pss - 2*pds
    
        # Add broadband
        pmodel += (1 + B1 + (1-Sk)*F*mu_obs**2)**2 * pnw
    
        return pmodel
    
    def bao_predict(self)
        
        pp   = self.provider
        
        kint = pp.get_result('kint')
        sphr = pp.get_result('sphr')
        
        # Get f and b1... maybe there's a more efficient way to do this
        sig8 = pp.get_sigma8_z(0)[0]
        fs8  = pp.get_fsigma8(self.zfid)[0]
        ff   = fs8 / s8
        b1   = pp.get_param('bsig8')/sig8
        beta = ff/b1
        
        # get the BAO-flavored AP paramters
        rs_fid = self.bao_templaters
        rs   = pp.get_param('rs_drag')
        hub  = pp.get_Hubble(0)[0]/100.
        Hz   = pp.get_Hubble(self.zfid)[0]/pp.get_Hubble(0)[0]
        chiz = pp.get_comoving_radial_distance(self.zfid)[0]*hub
        
        apar,aperp = self.Hz_fid/Hz * rs_fid/rs, chiz/self.chiz_fid * rs_fid/rs
        
        # Generate the sampling
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        #L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
    
        pknutable = np.zeros((len(nus),len(plin)))
    
        for ii, nu in enumerate(nus_calc):
            pknutable[ii,:] = self.compute_bao_pkmu(B1,F, apar=apar,aperp=aperp)
 
                
        pknutable[ngauss:,:] = np.flip(pknutable[0:ngauss],axis=0)
        
        p0 = 0.5 * np.sum((ws*L0)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[m0,m1,m2,m3,m4,m5]) / klin
        p2 = 2.5 * np.sum((ws*L2)[:,None]*pknutable,axis=0) #+ 1000 * polyval(klin,[q0,q1,q2,q3,q4,q5]) / klin
        #p4 = 4.5 * np.sum((ws*L4)[:,None]*pknutable,axis=0)

    
        p0t = interp1d(klin,p0, kind='cubic', bounds_error=False, fill_value=0)(kint)
        p2t = interp1d(klin,p2, kind='cubic', bounds_error=False, fill_value=0)(kint)
        #p4t = 0 * kint
        
        damping = np.exp(-(kint/10)**2)
        rr0, xi0t = sphr.sph(0,p0t * damping)
        rr2, xi2t = sphr.sph(2,p2t * damping); xi2t *= -1
        #rr2, xi4t = sphr.sph(4,p4t)
        #xi4t = 0 * rr0 # no hexadecapole to speed things up
        
        xi0t += polyval(1/rr0,[pp.get_param('M0'),pp.get_param('M1'),pp.get_param('M2')])
        xi2t += polyval(1/rr0,[pp.get_param('Q0'),pp.get_param('Q1'),pp.get_param('Q2')])
        
        return np.array([rr0,xi0t,xi2t]).T
    
    def bao_observe(self, tt):
    '''
    Bin the BAO results... probabaly should eventually use a matrix.
    '''
        
        thy0 = Spline(tt[:,0],tt[:,1],ext='extrapolate')
        thy2 = Spline(tt[:,0],tt[:,2],ext='extrapolate')
        #thy4 = Spline(tt[:,0],tt[:,3],ext='extrapolate')
        
        dr   = self.rdat[1]-G.self.rdat[0]
        
        tmp0 = np.zeros_like(self.rdat)
        tmp2 = np.zeros_like(self.rdat)
        
        for i in range(G.ss.size):
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
    

class Zel_xirecon_theory(Theory):
    '''
    Maybe a class to store auxiliary functions for BAO fitting.
    '''
    zfid: float
    OmMfid: float # this is used to compute f_fid and D_fid to compute damping and template
        
    bao_R: float
    bao_templatefn: str
    bao_templatenwfn: str
    bao_templaters: float
        
    def initialize(self):
    '''
    Sets up the class. Don't do anything
    '''
        pass
    
    def get_requirements(self):
    '''
    We don't need anything to produce the damping forms etc.
    '''
        req = {}
        return req
    
    def get_can_provide(self):
        """
        We provide the wiggle/no-wiggle spectra, and damping parameters.
        """
        return ['template_f', 'template_k', 'template_pnw', 'template_pw', 'template_sigmas']
    
    def calculate(self):
    
        # Get the fiducial growth parameters
        self.D_fid = D_of_a(1/(1.+self.zfid),OmegaM=self.OmMfid)
        self.f_fid = f_of_a(1/(1.+self.zfid),OmegaM=self.OmMfid)
        
        self.klin, self.plin = np.loadtxt(self.bao_templatefn, unpack=True); self.plin *= self.D_fid**2
        self.knw,  self.pnw  = np.loadtxt(self.bao_templatenwfn, unpack=True); self.pnw *= self.D_fid**2
        
        self.pw = self.plin - self.pnw
        
        # Compute the various reconstruction correlators
        zelda = Zeldovich_Recon(self.klin, self.plin, R=self.bao_R)
        
        # Find the linear bao peak
        from scipy.signal import argrelmax
        qsearch = (zelda.qint > 80) * (zelda.qint < 120)
        ii = argrelmax( (zelda.qint**2 * zelda.corlins['mm'])[qsearch] )
        qbao = zelda.qint[qsearch][ii][0]

        sigmadd = np.interp(qbao, zelda.qint, zelda.Xlins['dd'] + zelda.Ylins['dd']/3)
        sigmass = np.interp(qbao, zelda.qint, zelda.Xlins['ss'] + zelda.Ylins['ss']/3)

        sigmads_dd = zelda.Xlins['dd'][-1]
        sigmads_ss = zelda.Xlins['ss'][-1]
        sigmads_ds = np.interp(qbao, zelda.qint, zelda.Xlins['ds'] + zelda.Ylins['ds']/3) - zelda.Xlins['ds'][-1]
        
        self.sigma_squared = (sigmadd, sigmass, 0.5*sigmads_dd + 0.5*sigmads_ss + sigmads_ds)

        state['template_f'] = self.f_fid
        state['template_k'] = self.klin
        state['template_pnw'] = self.pnw
        state['template_pw'] = self.pw
        state['template_sigmas'] = self.sigma_squared
        state['template_rs'] = self.bao_templaters
        state['template_R'] = self.bao_R
        
        state['kint'] = np.logspace(-3, 2, 2000)
        state['sphr'] = SBT(state['kint'],L=5,fourier=True,low_ring=False)
        
    

        
    
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
               'Pk_interpolator': {'k_max': 30,'z': zg,\
                                   'nonlinear': False},\
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
        #s8   = pp.get_sigma8_z(self.zfid)[0]
        #fs8  = pp.get_fsigma8(self.zfid)[0]
        #ff   = fs8 / s8
        # and Plin.
        ki   = np.logspace(-3.0,1.5,750)
        pi   = pp.get_Pk_interpolator(nonlinear=False)
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
