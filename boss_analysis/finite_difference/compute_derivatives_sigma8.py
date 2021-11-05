import numpy as np
import sys
from taylor_approximation import compute_derivatives
import json

# Remake the data grid:
order = 4
Npoints = 2*order + 1
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68]; Nparams = len(x0s)
dxs = [0.01, 0.01]

output_shape = (1,)

center_ii = (order,)*Nparams
X0grid = np.zeros( (Npoints,)*Nparams+ output_shape)

# Load data
for ii in range(Npoints):
    for jj in range(Npoints):
        #print(ii,jj,kk)
        X0grid[ii,jj] = np.loadtxt('data/sigma8s/boss_s8_%d_%d.txt'%(ii,jj))
            
# Now compute the derivatives
derivs0 = compute_derivatives(X0grid, dxs, center_ii, 5)

# Now save:
outfile = 'emu/boss_s8.json'

list0 = [ dd.tolist() for dd in derivs0 ]

outdict = {'params': ['omegam', 'h'],\
           'x0': x0s,\
           'lnA0': 3.047,\
           'derivs0': list0,}

json_file = open(outfile, 'w')
json.dump(outdict, json_file)
json_file.close()