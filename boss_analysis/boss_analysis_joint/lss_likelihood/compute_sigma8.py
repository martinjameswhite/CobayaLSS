import numpy as np
import json
from taylor_approximation import taylor_approximate



s8_filename = '/global/cscratch1/sd/sfschen/finite_difference/emu/boss_s8.json'
json_file = open(s8_filename, 'r')
emu = json.load( json_file )
json_file.close()

lnA0 = emu['lnA0']
x0s = emu['x0']
derivs0 = [np.array(ll) for ll in emu['derivs0']]


def compute_sigma8(OmM, h, lnA):
    
    s8_emu = taylor_approximate([OmM,h], x0s, derivs0, order=3)[0]
    s8_emu *= np.exp(0.5*(lnA-lnA0))
    
    return s8_emu
    
    