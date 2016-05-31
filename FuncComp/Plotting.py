# -*- coding: utf-8 -*-
"""
v1: Created on May 31, 2016
author: Daniel Garrett (dg622@cornell.edu)
"""

import numpy as np
import matplotlib.pyplot as plt
import FuncComp.Population as Population
import FuncComp.Functional as Functional
import FuncComp.MonteCarlo as MonteCarlo
import FuncComp.util as util
import time

"""This script will construct and plot the full single visit completeness joint
probability distribution and completeness. The planet population details are 
given in Population. Details on the functional approach and Monte Carlo 
approach are given in their respective modules.
"""

# number of points in s for functional approach
ns = 400
# number of points in dmag for functional approach
ndmag = 400
# number of sample planets for Monte Carlo trials
Nplanets = 1e9
# number of bins in s for Monte Carlo histograms
bins = 400
# number of bins in dmag for Monte Carlo histograms
bindmag = 400
# get analytical function joint pdf of s, dmag
ftic = time.time()
func = Functional.Functional(ns=ns,ndmag=ndmag)
ftoc = time.time()
ftime = ftoc-ftic
# get Monte CarLo trial joint pdf of s, dmag
mtic = time.time()
mc = MonteCarlo.MonteCarlo(Nplanets=Nplanets,bins=bins,bindmag=bindmag)
mtoc = time.time()
mctime = mtoc-mtic

# get population min and max values
pop = Population.Population()
amin = pop.arange.min().value
amax = pop.arange.max().value
emin = pop.erange.min()
emax = pop.erange.max()
rmin = pop.arange.min().value*(1. - pop.erange.max())
rmax = pop.arange.max().value*(1. + pop.erange.max())
Rmax = pop.Rrange.max().value
Rmin = pop.Rrange.min().value
pmax = pop.prange.max()
pmin = pop.prange.min()
x = pop.Rrange.unit.to(pop.arange.unit)
# Lambert phase function
PhiL = lambda b: (1./np.pi)*(np.sin(b) + (np.pi - b)*np.cos(b))
# maximum phase angle
bstar = 1.104728818644543
mindmag = -2.5*np.log10(pmax*(Rmax*x)**2/rmin**2)
maxdmag = -2.5*np.log10(pmin*(Rmin*x)**2/rmax**2*PhiL(np.pi - np.arcsin(0.0001/rmax)))
s = np.linspace(0., rmax, num=bins)
dmag = np.linspace(mindmag, maxdmag, num=bins)
ranges = (pmin, Rmin, rmax)
dmagmax = util.maxdmag(s, ranges, x)
ranges = (pmax, Rmax, rmin, rmax)
dmagmin = util.mindmag(s, ranges, x)

# set up plot values for functional approach
ssf, ddf = np.meshgrid(func.s,func.dmag)
f = np.ma.masked_array(func.pc, func.pc<1e-11)
fplotc = func.comp(ssf,rmax,mindmag,ddf)

# set up plot values for Monte Carlo approach
ssm, ddm = np.meshgrid(mc.s,mc.dmag)
m = np.ma.masked_array(mc.Hc, mc.Hc<1e-11)
mplotc = mc.comp(ssm,rmax,mindmag,ddm)

vmin = -5.
if np.log10(f.max()) > 0. or np.log10(m.max()) > 0.:
    if np.log10(f.max()) > np.log10(m.max()):
        vmax = np.log10(f.max())
    else:
        vmax = np.log10(m.max())
else:
    vmax = 0.
        
# use TeX fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# plot Functional pdf
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(111)
a1 = ax1.pcolormesh(ssf, ddf, np.log10(f), rasterized=True, vmin=vmin, vmax=vmax, edgecolor='none', cmap='jet')
cbar1 = fig1.colorbar(a1)
cbar1.set_label('$ \log_{10} (\mathrm{AU}^{-1} \mathrm{mag}^{-1}) $')
ax1.plot(s, dmagmax, 'k-')
ax1.plot(s, dmagmin, 'k-')
ax1.set_xlabel('s (AU)', fontsize=14)
ax1.set_ylabel('$ \Delta \mathrm{mag} $', fontsize=14)
ax1.set_title('Functional Approach Joint PDF', fontsize=14)
fig1.show()

# plot Monte Carlo pdf
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(111)
a2 = ax2.pcolormesh(ssm, ddm, np.log10(m), rasterized=True, vmin=vmin, vmax=vmax, edgecolor='none', cmap='jet', clim=(-5.,0.))
cbar2 = fig2.colorbar(a2)
cbar2.set_label('$ \log_{10} (\mathrm{AU}^{-1} \mathrm{mag}^{-1}) $')
ax2.plot(s, dmagmax, 'k-')
ax2.plot(s, dmagmin, 'k-')
ax2.set_xlabel('s (AU)', fontsize=14)
ax2.set_ylabel('$ \Delta \mathrm{mag} $', fontsize=14)
ax2.set_title('Monte Carlo Trials Joint PDF', fontsize=14)
fig2.show()

# plot Functional completeness
fig3 = plt.figure(3)
ax3 = fig3.add_subplot(111)
a3 = ax3.pcolormesh(ssf, ddf, fplotc, rasterized=True, vmin=0., vmax=1., edgecolor='none', cmap='jet')
cbar3 = fig3.colorbar(a3)
ax3.plot(s, dmagmax, 'k-')
ax3.plot(s, dmagmin, 'k-')
ax3.set_xlabel('s (AU)', fontsize=14)
ax3.set_ylabel('$ \Delta \mathrm{mag} $', fontsize=14)
ax3.set_title('Functional Approach Single-Visit Completeness')
fig3.show()

# plot Monte Carlo completeness
fig4 = plt.figure(4)
ax4 = fig4.add_subplot(111)
a4 = ax4.pcolormesh(ssm, ddm, mplotc, rasterized=True, vmin=0., vmax=1., edgecolor='none', cmap='jet')
cbar4 = fig4.colorbar(a4)
ax4.plot(s, dmagmax, 'k-')
ax4.plot(s, dmagmin, 'k-')
ax4.set_xlabel('s (AU)', fontsize=14)
ax4.set_ylabel('$ \Delta \mathrm{mag} $', fontsize=14)
ax4.set_title('Monte Carlo Trials Single-Visit Completeness')
fig4.show()

# computational time
print 'Functional Approach time: %r (s)' % ftime
print 'Monte Carlo Trials time:  %r (s)' % mctime