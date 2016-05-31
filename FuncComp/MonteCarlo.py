# -*- coding: utf-8 -*-
"""
v1: Created on May 31, 2016
author: Daniel Garrett (dg622@cornell.edu)
"""

import scipy.interpolate as interpolate
import numpy as np
import sys, copy_reg, types
# pp allows parallel computations
try:
    import pp
    ppLoad = True
except ImportError:
    ppLoad = False

import FuncComp.Population as Population
import FuncComp.statsFun as statsFun

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
        
def _function(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)

class MonteCarlo(object):
    """This class contains all attributes and methods necessary for performing
    single visit completeness calculations using the Monte Carlo trial approach.
    
    Args:
        Nplanets (float):
            Number of planetary samples
        bins (int):
            Number of bins for apparent separation
        bindmag (int)
            Number of bins for difference in magnitude
        smin (float):
            Minimum apparent separation value (AU)
        smax (float):
            Maximum apparent separation value (AU)
        dmagmin (float):
            Minimum difference in magnitude value
        dmagmax (float):
            Maximum difference in magnitude value
            
    Attributes:
        s (ndarray): 
            Points sampled in apparent separation (AU)
        dmag (ndarray):
            Points sampled in difference in magnitude
        Hc (ndarray):
            Normalized 2-D histogram representing the single visit completeness
            joint probability density function
        grid (callable(s,dmag)):
            Interpolant of single visit completeness joint pdf
        pdf (callable(s,dmag)):
            Vectorized interpolant of single visit completeness joint pdf
    
    """
    def __init__(self, Nplanets=1e6, bins=400, bindmag=400, smin=0., smax=None, dmagmin=None, dmagmax=None):
        print 'Monte Carlo approach to single visit completeness'        
        # get Population information
        pop = Population.Population()
        # min and max values for quantities
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
        # probability density functions
        f_R = pop.Radius
        f_p = pop.albedo
        f_e = pop.eccentricity
        f_a = pop.semi_axis
        # Lambert phase function
        PhiL = lambda b: (1./np.pi)*(np.sin(b) + (np.pi - b)*np.cos(b))
        # conversion factor for R values to a,r values
        x = pop.Rrange.unit.to(pop.arange.unit)
        # check defaults and replace if None
        if smax is None:
            smax = rmax
        if dmagmin is None:
            dmagmin = -2.5*np.log10(pmax*(Rmax*x)**2/rmin**2)
        if dmagmax is None:
            dmagmax = -2.5*np.log10(pmin*(Rmin*x)**2/rmax**2*PhiL(np.pi - np.arcsin(0.0001/rmax)))
        pdfs = (f_e, f_a, f_R, f_p)
        ranges = (emin, emax, amin, amax, Rmin, Rmax, pmin, pmax, smin, smax, dmagmin, dmagmax)
        binh = (bins,bindmag)
        Hc = np.zeros((bindmag,bins))
        steps = Nplanets/1e6
        if ppLoad:
            # set up job server for parallel computations to get histograms
            ppservers = ()
            if len(sys.argv) > 1:
                ncpus = int(sys.argv[1])
                # Creates jobserver with ncpus workers
                job_server = pp.Server(ncpus, ppservers=ppservers)
            else:
                job_server = pp.Server(ppservers=ppservers)
        if steps <= 1.:
            nplan = Nplanets
            print 'Samples: %r / %r' % (int(nplan),int(Nplanets))
            hc, dmagedges, sedges = oneMCsim(nplan, pdfs, ranges, binh, x)
            Hc += hc
        else:
            samples = 0
            while Nplanets-samples > 0.:
                if Nplanets-samples > 1e6:
                    nplan = 1e6
                    if ppLoad:
                        if int((Nplanets-samples)/nplan) > 100:
                            psteps = 100
                        else:
                            psteps = int((Nplanets-samples)/nplan)
                        jobs = [(job_server.submit(oneMCsim, (nplan, pdfs, ranges, binh, x), (), ('FuncComp.MonteCarlo', 'numpy as np', 'FuncComp.statsFun as statsFun'))) for step in xrange(psteps)]
                        for job in jobs:
                            hc, dmagedges, sedges = job()
                            Hc += hc
                            samples += nplan
                            print 'Samples: %r / %r' % (int(samples),int(Nplanets))
                    else:
                        hc, dmagedges, sedges = oneMCsim(nplan, pdfs, ranges, binh, x)
                        Hc += hc
                        samples += nplan
                        print 'Samples: %r / %r' % (int(samples),int(Nplanets))
                else:
                    nplan = Nplanets-samples
                    hc, dmagedges, sedges = oneMCsim(nplan, pdfs, ranges, binh, x)
                    Hc += hc
                    samples += nplan
                    print 'Samples: %r / %r' % (int(samples),int(Nplanets))
        if ppLoad:
            job_server.destroy()
        # Normalize histogram to get pdf
        self.Hc = Hc/(Nplanets*(dmagedges[1]-dmagedges[0])*(sedges[1]-sedges[0]))
        self.s = 0.5*(sedges[1:] + sedges[:-1])
        self.dmag = 0.5*(dmagedges[1:] + dmagedges[:-1])
        # interpolant for joint pdf of s, dmag
        self.grid = interpolate.RectBivariateSpline(self.s,self.dmag,self.Hc.T,kx=3,ky=3)
        # vectorized interpolant for joint pdf of s, dmag
        self.pdf = np.vectorize(self.grid)
        
    def comp(self, smin, smax, dmagmin, dmagmax):
        """Returns completeness value
        
        Args:
            smin (float or ndarray):
                Minimum planet-star separation
            smax (float or ndarray):
                Maximum planet-star separation
            dmagmin (float or ndarray):
                Minimum difference in brightness magnitude
            dmagmax (float or ndarray):
                Maximum difference in brightness magnitude
        
        Returns:
            f (ndarray):
                Array of completeness values
        
        """
        
        f = np.vectorize(self.grid.integral)
        
        return f(smin, smax, dmagmin, dmagmax)
        
def oneMCsim(nplan, pdfs, ranges, binh, x):
    """Performs Monte Carlo trial
    
    Args:
        nplan (int):
            Number of planets to sample
        pdfs (tuple):
            Probability density functions for eccentricity, semi-major axis,
            planetary radius, and geometric albedo
        ranges (tuple):
            emin (float): minimum eccentricity
            emax (float): maximum eccentricity
            amin (float): minimum semi-major axis (AU)
            amax (float): maximum semi-major axis (AU)
            Rmin (float): minimum planetary radius (km)
            Rmax (float): maximum planetary radius (km)
            pmin (float): minimum geometric albedo 
            pmax (float): maximum geometric albedo
            smin (float): minimum planet-star separation (AU)
            smax (float): maximum planet-star separation (AU)
            dmagmin (float): minimum difference in brightness magnitude
            dmagmax (float): maximum difference in brightness magnitude
        binh (tuple):
            Number of bins for s and dmag for 2-D histogram
        x (float):
            Conversion factor for AU to km
            
    Returns:
        hc (ndarray):
            2-D histogram values for planet-star separation and difference in 
            brightness magnitude
        sedges (ndarray):
            Edge values for 2-D histogram bins in planet-star separation (AU)
        dmagedges (ndarray):
            Edge values for 2-D histogram bins in difference in brightness
            magnitude
        """
        
    # unpack quantities needed            
    f_e, f_a, f_R, f_p = pdfs
    emin, emax, amin, amax, Rmin, Rmax, pmin, pmax, smin, smax, dmagmin, dmagmax = ranges
    bins, bindmag = binh
    # sample needed quantities
    MMC = np.random.rand(nplan)*2.*np.pi
    # eccentricity
    eMC = statsFun.simpSample(f_e, nplan, emin, emax)
    # semi-major axis
    aMC = statsFun.simpSample(f_a, nplan, amin, amax)
    # Radius and convert units
    RMC = statsFun.simpSample(f_R, nplan, Rmin, Rmax)*x
    # albedo
    pMC = statsFun.simpSample(f_p, nplan, pmin, pmax)
    # phase angle
    beta = np.arccos(2.*np.random.rand(nplan) - 1.)
    lMC = np.sin(beta)
    phiMC = (1./np.pi)*(np.sin(beta) + (np.pi - beta)*np.cos(beta))
    # Newton-Raphson to find eccentric anomaly (E)
    counter = 0
    err = 1.
    EMC = MMC + eMC
    minus1 = np.where((MMC < 0) & (MMC > -np.pi))[0]
    EMC[minus1] -= 2.*eMC[minus1]
    minus2 = np.where(MMC > np.pi)[0]
    EMC[minus2] -= 2.*eMC[minus2]
    while err > 1e-8 and counter < 1000:
        Enew = EMC + (MMC - EMC + eMC*np.sin(EMC))/(1. - eMC*np.cos(EMC))
        err = np.sum(np.abs(Enew - EMC))
        counter += 1
        EMC = Enew
    # find nu and make sure it is between 0 and 2pi
    nuMC = np.arctan2(np.sin(EMC)*np.sqrt(1.-eMC**2), np.cos(EMC)-eMC)
    inds = np.where(nuMC < 0.)[0]
    nuMC[inds] += 2.*np.pi
    # find r
    rMC = (aMC*(1. - eMC**2))/(1. + eMC*np.cos(nuMC))
    # find s
    sMC = rMC*lMC
    # find FR
    FRMC = pMC*phiMC*RMC**2/rMC**2
    # find dmag
    dmagMC = -2.5*np.log10(FRMC)
    # get pdf histogram of dMag, s
    # this is correct form of histogram2d
    hc, dmagedges, sedges = np.histogram2d(dmagMC, sMC, bins=[bindmag,bins], range=[[dmagmin, dmagmax], [smin, smax]])
    
    return (hc, dmagedges, sedges)