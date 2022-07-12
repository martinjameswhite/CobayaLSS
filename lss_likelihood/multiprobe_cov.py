import warnings
import numpy as np
from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline
from scipy.special import legendre
from sympy.physics.wigner import wigner_3j
from itertools import product, permutations
from velocileptors.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
from mcfit import P2xi, xi2P, Hankel

def get_ell(l_max):
    return np.arange(0, l_max+1, 2)        
   
def rescale_kk(k, power):
    return k[:, None]**power * k**power    
        
class MultiprobeCovariance():
    
    def __init__(self, spectrum_info, noise_info):
        
        #contains information about cosmological signals, window functions
        self.spectrum_info = spectrum_info
        self.noise_info = noise_info
        
        self.spectrum_types = list(self.spectrum_info.keys())
        self.n_spectrum_types = len(self.spectrum_types)
        self.n_zbin0_spectra = dict(zip(self.spectrum_types, [len(self.spectrum_types[s]['bins0']) for s in self.spectrum_types]))
        self.n_zbin1_spectra = dict(zip(self.spectrum_types, [len(self.spectrum_types[s]['bins1']) for s in self.spectrum_types]))
        self.n_bins_spectra = dict(zip(self.spectrum_types, [self.n_zbin0_spectra[s] * self.n_zbin1_spectra[s] for s in self.spectrum_types]))
        self.n_sep_spectra = dict(zip(self.spectrum_types, [self.spectrum_types[s]['n_dv_per_bin'] for s in self.spectrum_types]))
        
        self.k = np.logspace(-4,2,2000)
        self._pk2xi_sbt = SphericalBesselTransform(self.k)
        self.r = self._pk2xi_sbt.ydict[0]
        self._xi2pk_sbt = SphericalBesselTransform(self.r)        
        

    def compute_covariance(self):
        dt = np.dtype([('spectrum_type0','U10'), ('spectrum_type1','U10'), ('zbin00', np.int),
               ('zbin01', np.int), ('zbin10', np.int), ('zbin11', np.int), 
               ('separation0', np.float), ('separation1', np.float), ('value', np.float)])        
        
        spec_type_counter = []
        zbin_counter = {}
        cov_scale_mask = []
        
        for t0, t1 in product(self.spectrum_types, self.spectrum_types):
            if (t0, t1) in spec_type_counter:
                continue

            if (t0, t1) not in zbin_counter.keys():
                zbin_counter[(t0, t1)] = []
                
            sep0 = self.spectrum_info[t0]['separation']
            sep1 = self.spectrum_info[t1]['separation']

            n00, n01, n10, n11 = self.spectrum_info[t0]['bins0'], \
                self.spectrum_info[t0]['bins1'], \
                self.spectrum_info[t1]['bins0'], \
                self.spectrum_info[t1]['bins1']
                
            if (t0[0]=='p') & (t1[0]=='p'):
                cov_type = 'plk_plk'
            elif (t0[0]=='p') & (t1[0]=='c'):
                cov_type = 'plk_cell'
            elif (t0[0]=='c') & (t1[0]=='c'):
                cov_type = 'cell_cell'
            else:
                raise(ValueError('Covariance computation for spec types {}-{} not implemented'.format(t0, t1)))

            for zb00, zb01, zb10, zb11 in product(n00, n01, n10, n11):
                if ((zb00, zb01), (zb10, zb11)) in zbin_counter[(t0, t1)]:
                    continue
                
                #get model predictions to use in cov 
                models = []
                window_info = []
                for ti, t in enumerate([t0, t1]):
                    if ti==0:
                        zbins = (zb00, zb01)
                    else:
                        zbins = (zb10, zb11)
                    
                    if (t[0] == 'p') | (t[3] == 'd') | (t[3] == 'c'):
                        m = self.spectrum_info[t]['{}_model'.format(zbins[0])]
                        
                        models.append(m)
                    elif (t == 'c_cmbkcmbk'):
                        m = self.spectrum_info[t]['model']
                        models.append(m)
                    else:
                        m = self.spectrum_info[t]['{}_{}_model'.format(zbins[0], zbins[1])]
                        models.append(m)
                           
                    w = self.noise_info[t]['{}_{}'.format(zbins[0], zbins[1])]
                    window_info.append(w)
                    
                    if cov_type == 'plk_plk':
                        cov = self.get_Cllkk_auto(self.k)
                        
                    
                    
                zbin_counter[(t0, t1)].extend(
                    permutations(((zb00, zb01), (zb10, zb11))))

            spec_type_counter.extend(permutations((t0, t1)))        
            
            
def get_s(k, lowring=False):
    return P2xi(k, N=len(k), lowring=lowring).y            

def get_ell(l_max):
    return np.arange(0, l_max+1, 2)

def get_H(k, l_max=20, lowring=False):
    """Return Hankel circulant matrix for all (l, k, k')"""
    ell = get_ell(l_max)
    return np.stack([P2xi(k, l=l, N=len(k), lowring=lowring).matrix(full=False)[2] for l in ell], axis=0)

def get_Qlls(s, si, Q, l_max, lp_max, l_Q_max):
    """Return Q_ll(s) for all (l, l', s)"""
    
    ell = get_ell(l_max)
    ell_prime = get_ell(lp_max)
    Qls = CubicSpline(si, Q, axis=1)(s)
    ns = Qls.shape[1]
    
    Qlls = np.zeros((len(ell), len(ell_prime), ns))
    
    for i, l in enumerate(ell):
        for j, lp in enumerate(ell_prime):
            for l_Q in range(abs(l-lp), min(l+lp, l_Q_max) + 1, 2):
                w3sq = wigner_3j(l, lp, l_Q, 0, 0, 0)**2
                Qlls[i, j] += float(w3sq) * Qls[l_Q//2]
            Qlls[i, j] *= (-1)**(i+j)
            
    return Qlls

def get_Qllkk(k, sin, Q, H, l_max, lp_max, l_Q_max, lowring=False):
    """Return Q_ll(k, k) for all (l, l', k, k')"""
    s = get_s(k)
    ell = get_ell(l_max)
    ell_prime = get_ell(lp_max)
        
    Qlls = get_Qlls(s, sin, Q, l_max, lp_max, l_Q_max)
    
    Qllkk = H[:len(ell), None, ...] * Qlls[..., None, :]
    
    for j, lp in enumerate(ell_prime):
        for i, l in enumerate(ell):
            #Q[i, j] = Q[i, j] @ H[j]
            _, Qllkk[i, j] = P2xi(k, l=lp, N=len(k),
                              lowring=lowring)(Qllkk[i, j], axis=1,
                                               extrap=False, convonly=True)

    dlnk = np.diff(np.log(k)).mean()
    Qllkk *= 2 * np.pi**2 / dlnk / rescale_kk(k, 1.5)
    
    return Qllkk

def get_Plk(k, ki, Pi):
    """Return P_l(k) for all (l, k)"""
    # Pi should already contain shot noise for gg auto
    # interpolate
    P = CubicSpline(ki, Pi, axis=1)(k)
    # extrapolate as pow-law on the low-k side
    k_left = k[k < ki.min()]
    lnP_left = interp1d(np.log(ki[:2]), np.log(Pi[:, :2]), axis=1, kind='linear',
                        fill_value='extrapolate', assume_sorted=True)(np.log(k_left))
    P[:, k < ki.min()] = np.exp(lnP_left)
    # zero at the high-k end
    P[:, k > ki.max()] = 0
    
    return P

def get_Pllk(k, ki, Pi, l_max, lp_max, l_P_max):
    """Return P_ll(k) for all (l, l', k)"""
    Plk = get_Plk(k, ki, Pi)
    
    ell = get_ell(l_max)
    ell_prime = get_ell(lp_max)
    
    P = np.zeros((len(ell), len(ell_prime), len(k)))
    
    for i, l in enumerate(ell):
        for j, lp in enumerate(ell_prime):
            for l_P in range(abs(l-lp), min(l+lp, l_P_max) + 1, 2):
                w3sq = wigner_3j(l, l_P, lp, 0, 0, 0)**2
                P[i, j] += float(w3sq) * Plk[l_P//2]
            P[i, j] *= (2*lp+1)
            
    return P

def get_Cllkk_auto(k, ki, P, si, H, QW, QX, QS, W0, l_max=16, l_P_max=4, l_Q_max=10):
    """Return C_ll(k, k) for all (l, l', k, k')"""

    Pll = get_Pllk(k, ki, P, l_max, l_max+l_P_max, l_P_max)
    Qllkk_w = get_Qllkk(k, si, QW, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_x = get_Qllkk(k, si, QX, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_s = get_Qllkk(k, si, QS, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)

    ell = get_ell(l_max)
    C = np.zeros((len(ell), len(ell), len(k), len(k)))
    
    for i in range(QW.shape[0]):
        for j in range(QW.shape[1]):
            C += Pll[:, i][:, None, :, None] * Qllkk_w[[i], [j]] * Pll[:, j][None, :, None, :]

    for i in range(QX.shape[0]):
        CX = Pll[:, i][:, None, :, None] * Qllkk_x[[i]]
    CX += CX.swapaxes(0, 1).swapaxes(2, 3)
    
    C += CX + Qllkk_s
    
    C *= 2 * (2*ell+1)[:, None, None, None] * (2*ell+1)[None, :, None, None] \
        / W0**2
    
    return C

def get_Cllkk_cross(k, ki, P_ac, P_bd, P_ad, P_bc, si, H, Q_w_acbd, Q_w_adbc, Q_x_acbd, 
                    Q_x_bdac, Q_x_adbc, Q_x_bcad, Q_s_acbd, Q_s_adbc, W0ab, W0cd,
                    l_max=16, l_P_max=4, l_Q_max=10):
    
    Pll_ac = get_Pllk(k, ki, P_ac, l_max, l_max+l_P_max, l_P_max)
    Pll_bd = get_Pllk(k, ki, P_bd, l_max, l_max+l_P_max, l_P_max)
    Pll_ad = get_Pllk(k, ki, P_ad, l_max, l_max+l_P_max, l_P_max)
    Pll_bc = get_Pllk(k, ki, P_bc, l_max, l_max+l_P_max, l_P_max)
    
    
    Qllkk_w_acbd = get_Qllkk(k, si, Q_w_acbd, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_w_adbc = get_Qllkk(k, si, Q_w_adbc, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    
    Qllkk_x_acbd = get_Qllkk(k, si, Q_x_acbd, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_x_bdac = get_Qllkk(k, si, Q_x_bdac, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_x_adbc = get_Qllkk(k, si, Q_x_adbc, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_x_bcad = get_Qllkk(k, si, Q_x_bcad, H, l_max+l_P_max, l_max+l_P_max, l_Q_max) 
    
    Qllkk_s_acbd = get_Qllkk(k, si, Q_s_acbd, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)
    Qllkk_s_adbc = get_Qllkk(k, si, Q_s_adbc, H, l_max+l_P_max, l_max+l_P_max, l_Q_max)           
    
    ell = get_ell(l_P_max)
    C = np.zeros((len(ell), len(ell), len(k), len(k)))
    
    
    for i in range(Qllkk_w_acbd.shape[0]):
        for j in range(Qllkk_w_acbd.shape[1]):
            C += (Pll_ac[:, i][:, None, :, None] * Qllkk_w_acbd[[i], [j]] * Pll_bd[:, j][None, :, None, :] + 
                Pll_bd[:, i][:, None, :, None] * Qllkk_w_acbd[[i], [j]] * Pll_ac[:, j][None, :, None, :]) / 2
            C += (Pll_ad[:, i][:, None, :, None] * Qllkk_w_adbc[[i], [j]] * Pll_bc[:, j][None, :, None, :] + \
                Pll_bc[:, i][:, None, :, None] * Qllkk_w_adbc[[i], [j]] * Pll_ad[:, j][None, :, None, :]) / 2
            
    for i in range(Qllkk_x_acbd.shape[0]):
        C += (Pll_ac[:, i][:, None, :, None] * Qllkk_x_acbd[[i]] + Pll_ac[:, i][None, :, None, :] * Qllkk_x_acbd[[i]]) / 2
        C += (Pll_bd[:, i][:, None, :, None] * Qllkk_x_bdac[[i]] + Pll_bd[:, i][None, :, None, :] * Qllkk_x_bdac[[i]]) / 2
        C += (Pll_ad[:, i][:, None, :, None] * Qllkk_x_adbc[[i]] + Pll_ad[:, i][None, :, None, :] * Qllkk_x_adbc[[i]]) / 2
        C += (Pll_bc[:, i][:, None, :, None] * Qllkk_x_bcad[[i]] + Pll_bc[:, i][None, :, None, :] * Qllkk_x_bcad[[i]]) / 2
        
    C += Qllkk_s_acbd + Qllkk_s_adbc


    C *= (2*ell+1)[:, None, None, None] * (2*ell+1)[None, :, None, None] \
        / (W0ab * W0cd) #*2?
        
    return C