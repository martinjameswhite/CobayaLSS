import numpy as np
import sys
from taylor_approximation import compute_derivatives
import json

z = float(sys.argv[1])
rmin, rmax, dr = 50, 160, 0.5

# Remake the data grid:
order = 4
Npoints = 2*order + 1
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68, 0.73]; Nparams = len(x0s)
dxs = [0.01, 0.01, 0.05]

rr = np.arange(rmin, rmax, dr)
output_shape = (len(rr),6)

center_ii = (order,)*Nparams
X0grid = np.zeros( (Npoints,)*Nparams+ output_shape)
X2grid = np.zeros( (Npoints,)*Nparams+ output_shape)

# Load data
for ii in range(Npoints):
    for jj in range(Npoints):
        for kk in range(Npoints):
            #print(ii,jj,kk)
            X0grid[ii,jj,kk] = np.loadtxt('data/boss_z_%.2f/boss_xi0_%d_%d_%d.txt'%(z,ii,jj,kk))
            X2grid[ii,jj,kk] = np.loadtxt('data/boss_z_%.2f/boss_xi2_%d_%d_%d.txt'%(z,ii,jj,kk))
            
# Now compute the derivatives
derivs0 = compute_derivatives(X0grid, dxs, center_ii, 5)
derivs2 = compute_derivatives(X2grid, dxs, center_ii, 5)

# Now save:
outfile = 'emu/boss_z_%.2f_xiells.json'%(z)

list0 = [ dd.tolist() for dd in derivs0 ]
list2 = [ dd.tolist() for dd in derivs2 ]

outdict = {'params': ['omegam', 'h', 'sigma8'],\
           'x0': x0s,\
           'rvec': rr.tolist(),\
           'derivs0': list0,\
           'derivs2': list2,}

json_file = open(outfile, 'w')
json.dump(outdict, json_file)
json_file.close()