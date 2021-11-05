#!/usr/bin/env python3
#
#
import numpy as np
import matplotlib.pyplot as plt

from cobaya.yaml import yaml_load_file
#
from getdist.mcsamples import MCSamplesFromCobaya
from getdist.mcsamples import loadMCSamples,MCSamples
import getdist.plots as gdplt
#
import os

info=yaml_load_file("/global/cscratch1/sd/mwhite/unWISE/chains/planck18.updated.yaml")
print(info['params'].keys())

planck18 = loadMCSamples("/global/cscratch1/sd/mwhite/unWISE/chains/planck18",\
                         settings={'ignore_rows':0.3})
p = planck18.getParams()
# Add S8 and "rename" H0 and Omega_m.
planck18.addDerived(p.sigma8*(p.Omega_m/0.3)**0.50,name='S8',label='S_8')
planck18.addDerived(p.sigma8*(p.Omega_m/0.3)**0.25,name='Sig8',label='\Sigma_8')
planck18.addDerived(p.H0/100.,name='hub',label='h')
planck18.addDerived(p.Omega_m,name='Omegam',label=r'\Omega_m')
#
# Just look at some statistics:
print("\nPlanck18:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( planck18.getInlineLatex(k) )


# The Planck18 lensing-only chains.
lbase = "/global/cscratch1/sd/mwhite/unWISE/chains/lensing_lenspriors/"
lbase = "/global/cscratch1/sd/mwhite/Planck/base/lensing_lenspriors_BAO_theta/"
plens = loadMCSamples(lbase+"base_lensing_lenspriors_BAO_theta",\
                      settings={'ignore_rows':0.3})
p = plens.getParams()
# Add S8 and "rename" H0 and Omega_m.
#plens.addDerived(p.sigma8*(p.omegam/0.3)**0.50,name='S8',label='S_8')
plens.addDerived(p.sigma8*(p.omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
plens.addDerived(p.H0/100.,name='hub',label='h')
plens.addDerived(p.omegam,name='Omegam',label=r'\Omega_m')
#
# Just look at some statistics:
print("\nPlanck lensing-only:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( plens.getInlineLatex(k) )


# DES-Y1
dbase = "/global/cscratch1/sd/akrolew/chains_unWISE_final/DES/despublic_y1a1_files_chains/"
des   = np.loadtxt(dbase+"d_l3.txt",usecols=[0,1,26,-1])
omh3  = des[:,0]*des[:,1]**3
chi2 = ((omh3-0.09633)/(5*0.00029))**2    # Chi^2 for Om.h^3 constraint, Eq. (12) PP18.
wt    = des[:, -1] * np.exp(-0.5*chi2)
des   = des[:,:-1]
des   = MCSamples(samples=des,names=['Omegam','hub','sigma8'],weights=wt,label='DES')
p = des.getParams()
des.addDerived(p.sigma8*(p.Omegam/0.3)**0.50,name='S8',label='S_8')
des.addDerived(p.sigma8*(p.Omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
#
# Just look at some statistics:
print("\nDES-Y1:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( des.getInlineLatex(k) )


# KiDS
dbase = "/global/cscratch1/sd/akrolew/chains_unWISE_final"+\
        "/KiDS/KiDS1000_3x2pt_fiducial_chains/cosmology/"
kids  = np.loadtxt(dbase+"samples_multinest_blindC_EE_nE_w.txt",\
                   usecols=[2,20,21,24,-1])
omh3  = kids[:,3]*kids[:,0]**3
chi2 = ((omh3-0.09633)/(5*0.00029))**2    # Chi^2 for Om.h^3 constraint, Eq. (12) PP18.
wt    = kids[:, -1] * np.exp(-0.5*chi2)
kids  = kids[:,:-1]
kids  = MCSamples(samples=kids,\
                  names=['hub','S8','sigma8','Omegam'],\
                  labels=['h','S_8','\sigma_8','\Omega_m'],\
                  weights=wt,label='KiDS')
p = kids.getParams()
kids.addDerived(p.sigma8*(p.Omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
#
# Just look at some statistics:
print("\nKiDS1000:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( kids.getInlineLatex(k) )


#
# Pull in the "comb"ined sample of unWISE blue+green.
#
dbase = "/global/cscratch1/sd/akrolew/chains_unWISE_final/Data"+\
        "/blugrn/fix_geometry_fix_omh3_dndz_sample/"
#
# Otherwise append all of the samples together into 1.
unwise = loadMCSamples(dbase+'green_blue_data_fix_omh3_'+'0',\
                     settings={'ignore_rows':0.3});
p = unwise.getParams()
cummap = np.exp(-np.min(0.5*p.chi2 + p.minuslogprior))
for i in range(1,20):
    newsamp = loadMCSamples(dbase+'green_blue_data_fix_omh3_'+str(i),\
                            settings={'ignore_rows':0.3});
    p = newsamp.getParams()
    curmap = np.exp(-np.min(0.5*p.chi2 + p.minuslogprior))
    unwise = unwise.getCombinedSamplesWithSamples(newsamp,sample_weights=(cummap,curmap))
    cummap += curmap
p = unwise.getParams()
unwise.addDerived(p.sigma8*(p.Omegam/0.3)**0.50,name='S8',label='S_8')
unwise.addDerived(p.sigma8*(p.Omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
unwise.addDerived(p.H0/100.,name='hub',label='h')
unwise.addDerived((p.sigma8/0.8228)**0.8*(p.Omegam/0.307)**0.6,name='Singh8',label='Singh_8')
#
# Just look at some statistics:
print("\nunWISE:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( unwise.getInlineLatex(k) )


# Our BOSS re-analysis.
dbase = "/global/cscratch1/sd/sfschen/boss_analysis_joint/chains/"
boss  = loadMCSamples(dbase+'bossz13_joint_lnA',\
                      settings={'ignore_rows':0.3});
p = boss.getParams()
boss.addDerived(p.omegam,name='Omegam',label='\Omega_m')
boss.addDerived(p.sigma8*(p.omegam/0.3)**0.50,name='S8',label='S_8')
boss.addDerived(p.sigma8*(p.omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
boss.addDerived(p.H0/100.,name='hub',label='h')
print("\nBOSS:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( boss.getInlineLatex(k) )


#
# And finally our DESI LRG x Planck constraints.
#
dbase = "/global/cscratch1/sd/mwhite/Fitting/CobayaLSS/chains/"
#
#desi = loadMCSamples(dbase+'lrg_s00_emu_clpt_lcdm',\
#                     settings={'ignore_rows':0.3});
desi = loadMCSamples(dbase+'lrg_s00_emu_clpt_wbao',\
                     settings={'ignore_rows':0.3});
p = desi.getParams()
desi.addDerived(p.omegam,name='Omegam',label='\Omega_m')
desi.addDerived(p.mysig8,name='sigma8',label='\sigma_8')
desi.addDerived(p.mysig8*(p.omegam/0.3)**0.50,name='S8',label='S_8')
desi.addDerived(p.mysig8*(p.omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
desi.addDerived(p.H0/100.,name='hub',label='h')
# Just look at some statistics:
print("\nDESI:")
for k in ['Omegam','hub','sigma8','S8','Sig8']:
    print( desi.getInlineLatex(k) )


# The DESI samples minus the first.
noone = loadMCSamples(dbase+'lrg_s99_emu_clpt_lcdm',\
                      settings={'ignore_rows':0.3});
p = noone.getParams()
noone.addDerived(p.omegam,name='Omegam',label='\Omega_m')
noone.addDerived(p.mysig8,name='sigma8',label='\sigma_8')
noone.addDerived(p.mysig8*(p.omegam/0.3)**0.50,name='S8',label='S_8')
noone.addDerived(p.mysig8*(p.omegam/0.3)**0.25,name='Sig8',label='\Sigma_8')
noone.addDerived(p.H0/100.,name='hub',label='h')



# Now let's look at S8 from a variety of different experiments.
g = gdplt.get_single_plotter(ratio=0.4,width_inch=9)
g.plots_1d([planck18,plens,kids,unwise,boss,desi,noone], ['Sig8'], nx=1, normalized=False,\
           colors=['C4','darkmagenta','maroon','sandybrown','cyan','teal','teal'],\
           ls=['-','-','-','-','-','-','--'],\
           legend_labels=["Planck","P-lens","KiDS","unWISE","BOSS","DESI","2,3,4"],\
           legend_ncol=7,\
           constrained_layout=True,xlims=[ [0.67,0.87] ]);
g.export("data_cmp_S8.pdf")

# And look at a simple corner plot.
g = gdplt.get_subplot_plotter()
g.triangle_plot([planck18,plens,kids,unwise,boss,desi],
                ["Omegam","sigma8"],filled=True,alphas=[1.0,0.6,0.5,0.4,0.3,0.5],\
                colors=['darkgrey','darkmagenta','maroon','sandybrown','cyan','teal'],\
                line_args=[{'color':'darkgrey'},\
                           {'color':'darkmagenta'},\
                           {'color':'maroon'},\
                           {'color':'sandybrown'},\
                           {'color':'cyan'},\
                           {'color':'teal'}],\
                constrained_layout=True,\
                legend_labels=["Planck","P-lens","KiDS","unWISE","BOSS","DESI"])
g.export("cmp_corner.pdf")
#



# A version for the BOSS paper.
g = gdplt.get_subplot_plotter()
g.triangle_plot([planck18,plens,kids,unwise,boss],
                ["Omegam","sigma8"],filled=True,alphas=[1.0,0.6,0.5,0.4,0.5],\
                colors=['darkgrey','darkmagenta','maroon','sandybrown','cyan'],\
                line_args=[{'color':'darkgrey'},\
                           {'color':'darkmagenta'},\
                           {'color':'maroon'},\
                           {'color':'sandybrown'},\
                           {'color':'cyan'}],\
                constrained_layout=True,\
                legend_labels=["Planck","P-lens","KiDS+","unWISE","BOSS"])
g.export("cmp_external.pdf")
#
