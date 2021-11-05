import numpy as np
from mpi4py import MPI
import sys

z = float(sys.argv[1])
fid_dists = np.loadtxt('fid_dists_z_%.2f.txt'%(z))

rmin, rmax, dr = 50, 160, 0.5

mpi_rank = MPI.COMM_WORLD.Get_rank()
mpi_size = MPI.COMM_WORLD.Get_size()
print( "Hello I am process %d of %d." %(mpi_rank, mpi_size) )

# Set up the output k vector:
from compute_xiell_tables import compute_xiell_tables


# First construct the grid
order = 4
# these are OmegaM, h, sigma8
x0s = [0.31, 0.68, 0.73]; Nparams = len(x0s) # these are chosen to be roughly at the BOSS best fit value
dxs = [0.01, 0.01, 0.05]

template = np.arange(-order,order+1,1)
Npoints = 2*order + 1
grid_axes = [ dx*template + x0 for x0, dx in zip(x0s,dxs)]

Inds   = np.meshgrid( * (np.arange(Npoints),)*Nparams, indexing='ij')
Inds = [ind.flatten() for ind in Inds]
center_ii = (order,)*Nparams
Coords = np.meshgrid( *grid_axes, indexing='ij')


for nn, iis in enumerate(zip(*Inds)):
    if nn%mpi_size == mpi_rank:
        coord = [Coords[i][iis] for i in range(Nparams)]
        print(coord,iis)
        xi0, xi2= compute_xiell_tables(coord,z=z,fid_dists=fid_dists, rmin=rmin, rmax=rmax, dr=dr)
        
        fb = 'data/boss_z_%.2f/'%(z)
        
        np.savetxt(fb + 'boss_xi0_%d_%d_%d.txt'%(iis),xi0)
        np.savetxt(fb + 'boss_xi2_%d_%d_%d.txt'%(iis),xi2)
        #np.savetxt(fb + 'boss_p4_%d_%d_%d.txt'%(iis),p4)

