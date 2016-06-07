# -*- coding: utf-8 -*-
"""
v1: Created on May 31, 2016
author: Daniel Garrett (dg622@cornell.edu)
"""

import scipy.interpolate as interpolate
import numpy as np
import sys, copy_reg, types
import scipy.integrate as integrate
# pp allows parallel computations
try:
    import pp
    ppLoad = True
except ImportError:
    ppLoad = False

import FuncComp.Population as Population
import FuncComp.util as util

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

class Functional(object):
    """This class contains all attributes and methods necessary for performing
    single visit completeness calculations using an analytical function 
    approach.
    
    Args:
        smin (float): 
            Minimum apparent separation (AU)
        smax (float):
            Maximum apparent separation (AU)
        ns (int):
            Number of apparent separation points
        dmagmin (float):
            Minimum difference in magnitude
        dmagmax (float):
            Maximum difference in magnitude
        ndmag (int):
            Number of difference in magnitude points
    
    Attributes:
        s (ndarray):
            Points sampled in apparent separation (AU)
        dmag (ndarray):
            Points sampled in difference in magnitude
        pc (ndarray):
            Single visit completeness joint probability density function values
            at points sampled in s and dmag
        grid (callable(s,dmag)):
            Interpolant of single visit completeness joint pdf
        pdf (callable(s,dmag)):
            Vectorized interpolant of single visit completeness joint pdf
    
    """
    
    def __init__(self, smin=0., smax=None, ns=400, dmagmin=None, dmagmax=None, ndmag=400):
        print 'Analytical form of single visit completeness'        
        # import population information
        pop = Population.Population()
        # number of sample points for f_r and f_z
        n = 400
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
        zmin = pmin*Rmin**2
        zmax = pmax*Rmax**2
        # are any of these values constant?
        aconst = amin == amax
        econst = emin == emax
        Rconst = Rmin == Rmax
        pconst = pmin == pmax
        # conversion factor for R values to a,r values
        x = pop.Rrange.unit.to(pop.arange.unit)
        # Lambert phase function
        PhiL = lambda b: (1./np.pi)*(np.sin(b) + (np.pi - b)*np.cos(b))
        # maximum phase angle
        bstar = 1.104728818644543
        # check for defaults and replace if set to None
        if dmagmin is None:
            dmagmin = -2.5*np.log10(pmax*(Rmax*x)**2/rmin**2)
            print 'dmagmin: %r' % dmagmin
        if dmagmax is None:
            dmagmax = -2.5*np.log10(pmin*(Rmin*x)**2/rmax**2*PhiL(np.pi - np.arcsin(0.0001/rmax)))
        # probability density functions
        f_R = pop.Radius
        f_p = pop.albedo
        f_e = pop.eccentricity
        f_a = pop.semi_axis
        # linearly spaced arrays for range values
        r = np.linspace(rmin, rmax, num=n)
        z = np.linspace(zmin, zmax, num=n)
        # inverse 1 (<bstar) of sin(b)^2*Phi(b)
        b1 = np.linspace(0., bstar, num=50*n)
        binv1 = interpolate.InterpolatedUnivariateSpline(np.sin(b1)**2*PhiL(b1), b1, k=3, ext=1)
        # inverse 2 (>bstar) of sin(b)^2*Phi(b)
        b2 = np.linspace(bstar, np.pi, num=50*n)
        b2val = np.sin(b2)**2*PhiL(b2)
        binv2 = interpolate.InterpolatedUnivariateSpline(b2val[::-1], b2[::-1], k=3, ext=1)
        # if pp is loaded, set up jobserver
        if ppLoad:
            # set up job server for parallel computations to get f_r
            ppservers = ()
            if len(sys.argv) > 1:
                ncpus = int(sys.argv[1])
                # Creates jobserver with ncpus workers
                job_server = pp.Server(ncpus, ppservers=ppservers)
            else:
                job_server = pp.Server(ppservers=ppservers)
        # get pdf of r
        pdfr = np.array([])
        if aconst and econst:
            if ppLoad:
                jobs = [(job_server.submit(onef_r_aeconst, (ri, amin, emin), (), ('FuncComp.Functional', 'numpy as np'))) for ri in r]
                for job in jobs:
                    pdfr = np.hstack((pdfr, job()))
            else:
                for ri in r:
                    temp = onef_r_aeconst(ri, amin, emin)
                    pdfr = np.append(pdfr, temp)
        elif aconst:
            if ppLoad:
                jobs = [(job_server.submit(onef_r_aconst, (ri, amin, pop.erange, f_e), (), ('FuncComp.Functional', 'numpy as np', 'scipy.integrate as integrate'))) for ri in r]
                for job in jobs:
                    pdfr = np.hstack((pdfr, job()))
            else:                
                for ri in r:
                    temp = onef_r_aconst(ri, amin, pop.erange, f_e)
                    pdfr = np.append(pdfr, temp)
        elif econst:
            if ppLoad:
                jobs = [(job_server.submit(onef_r_econst, (ri, emin, pop.arange.to('AU').value, f_a), (), ('FuncComp.Functional', 'numpy as np', 'scipy.integrate as integrate'))) for ri in r]
                for job in jobs:
                    pdfr = np.hstack((pdfr, job()))
            else:
                for ri in r:
                    temp = onef_r_econst(ri, emin, pop.arange.to('AU').value, f_a)
                    pdfr = np.append(pdfr, temp)
        else:
            pdfs = (f_e, f_a)
            ranges = (amin, amax, emin, emax)
            print 'finding pdf of r'
            if ppLoad:
                jobs = [(job_server.submit(onef_r, (ri, pdfs, ranges), (), ('FuncComp.Functional', 'numpy as np'))) for ri in r]
                for job in jobs:
                    pdfr = np.hstack((pdfr, job()))            
            else:
                for ri in r:
                    temp = onef_r(ri,pdfs,ranges)
                    pdfr = np.append(pdfr,temp)
        # pdf of r
        f_r = interpolate.InterpolatedUnivariateSpline(r, pdfr, k=3, ext=1)
        # get pdf of zeta
        pdfs = (f_p, f_R)
        ranges = (pmin, pmax, Rmin, Rmax)
        print 'finding pdf of p*R^2'
        if ppLoad:
            jobs = [(job_server.submit(onef_zeta, (zetai, pdfs, ranges), (onef_R2,), ('FuncComp.Functional', 'numpy as np'))) for zetai in z]
            pdfz = np.array([])
            for job in jobs:
                pdfz = np.hstack((pdfz, job()))
        else:
            pdfz = np.array([])
            for zetai in z:
                temp = onef_zeta(zetai, pdfs, ranges)
                pdfz = np.append(pdfz,temp)
        # pdf of zeta = p*R^2
        f_z = interpolate.InterpolatedUnivariateSpline(z, pdfz, k=3, ext=1)
        # get joint pdf of s, dmag
        # check for defaults and replace if set to None
        if smax is None:
            self.s = np.linspace(smin,rmax,num=ns)
        else:
            self.s = np.linspace(smin,smax,num=ns)
        self.dmag = np.linspace(dmagmin,dmagmax,num=ndmag)
        ranges = (pmin, pmax, Rmin, Rmax, rmin, rmax, zmin, zmax)
        val = np.sin(bstar)**2*PhiL(bstar)
        pdfs = (f_z, f_r)
        funcs = (binv1, binv2)
        self.pc = np.zeros((len(self.dmag),len(self.s)))
        print 'finding joint pdf of s, dmag'
        if ppLoad:
            for i in xrange(len(self.s)):
                print 'Progress: %r / %r' % (i+1, len(self.s))
                if self.s[i] == 0.:
                    self.pc[:,i] = np.zeros((len(self.dmag),))
                else:
                    jobs = [(job_server.submit(onef_dmags, (self.dmag[j], self.s[i], ranges, val, pdfs, funcs, x), (onef_dmagsz, Jac, util.mindmag, util.maxdmag), ('FuncComp.Functional', 'FuncComp.util as util', 'numpy as np', 'scipy.integrate as integrate'))) for j in xrange(len(self.dmag))]
                for j in xrange(len(self.dmag)):
                    self.pc[j,i] = jobs[j]()
            job_server.destroy()
        else:
            for i in xrange(len(self.s)):
                print 'Progress: %r / %r' % (i+1, len(self.s))
                if self.s[i] == 0.:
                    self.pc[:,i] = np.zeros((len(self.dmag),))
                else:
                    for j in xrange(len(self.dmag)):
                        self.pc[j,i] = onef_dmags(self.dmag[j],self.s[i],ranges,val,pdfs,funcs,x)
        # interpolant for joint pdf of s, dmag
        self.grid = interpolate.RectBivariateSpline(self.s, self.dmag, self.pc.T,kx=3,ky=3)
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
        
def Jac(b):
    """Returns the determinant of the Jacobian matrix
    
    Args:
        b (float or ndarray):
            Phase angle
            
    Returns:
        J (float or ndarray):
            Determinant of Jacobian matrix
            
    """
            
    PhiL = lambda beta: (1./np.pi)*(np.sin(beta) + (np.pi-beta)*np.cos(beta))
    # derivative of PhiL with respect to beta
    PhiLp = lambda beta: (1./np.pi)*(beta-np.pi)*np.sin(beta)
    J = -2.5/(PhiL(b)*np.log(10.))*PhiLp(b)*np.sin(b) - 5./np.log(10.)*np.cos(b)
    
    return J

def onef_zeta(zetai, pdfs, ranges):
    """Determines probability density for zeta = p*R^2
    
    Args:
        zetai (float):
            Value of zeta = pR^2
        pdfs (tuple):
            Probability density functions f_p and f_R
        ranges (tuple):
            pmin (float): minimum value of geometric albedo
            pmax (float): maximum value of geometric albedo
            Rmin (float): minimum value of planetary radius (km)
            Rmax (float): maximum value of planetary radius (km)
            
    Returns:
        f (float):
            probability density of given zetai
            
    """
            
    f_p, f_R = pdfs
    pmin, pmax, Rmin, Rmax = ranges
    # Boole's rule
    # number of sample points
    nump = 1001
    p = np.linspace(pmin, pmax, nump)
    z = 7.*np.ones((nump,))
    z[1::2] = 32.
    z[2::4] = 12.
    z[4::4] = 14.
    z[-1] = 7.
    z[0] = 1.
    z[-1] = 1.
    pstep = p[1] - p[0]
    z = 2.*pstep/45.*z
    grand = np.zeros((nump,))
    arg = zetai/p
    good = np.where((arg >= Rmin**2) & (arg <= Rmax**2))[0]   
    grand[good] = (1./p[good])*onef_R2(arg[good], f_R)*f_p(p[good])
    f = np.dot(z,grand)

    return f
   
def onef_R2(R2, pdf):
    """Returns probability density of planetary radius squared (R^2)
    
    Args:
        R2 (float):
            Value of planetary radius squared
        pdf (callable(R)):
            Probability density function of planetary radius
    
    Returns:
        f (float):
            Probability density of planetary radius squared
            
    """
            
    f = 1./(2.*np.sqrt(R2))*pdf(np.sqrt(R2))
    
    return f

def onef_r(ri, pdfs, ranges):
    """Returns probability density of orbital radius r
    
    Args:
        ri (float):
            Value of orbital radius (AU)
        pdfs (tuple):
            Probability density functions for eccentricity and semi-major axis  
        ranges (tuple):
            amin (float): minimum semi-major axis (AU)
            amax (float): maximum semi-major axis (AU)
            emin (float): minimum eccentricity
            emax (float): maximum eccentricity
            
    Returns:
        f (float):
            Probability density of orbital radius
            
    """
            
    # Simpson's rule
    amin, amax, emin, emax = ranges
    if (ri == amin*(1.-emax)) or (ri == amax*(1.+emax)):
        f = 0.
    else:
        f_e, f_a = pdfs
        # must be odd
        na = 501
        ne = 9999
        a = np.linspace(amin, amax, na)
        z = 2.*np.ones((len(a),))
        z[::2] = 4.
        z[0] = 1.
        z[-1] = 1.
        astep = a[1] - a[0]
        z = astep/3.*z
        grand = np.zeros(np.shape(a))
        yi = 2.*np.ones((ne,))
        yi[::2] = 4.
        yi[0] = 1.
        yi[-1] = 1.
        for i in xrange(len(a)):
            emin1 = np.abs(1. - ri/a[i])
            if emin1 < emin:
                emin1 = emin
            if emin1 > emax:
                granda = 0.
            else:
                inte = lambda e: (1./np.pi)*ri/(a[i]*np.sqrt((a[i]*e)**2 - (a[i] - ri)**2))*f_e(e)*f_a(a[i])
                e = np.linspace(emin1+1.e-6,emax,ne)
                estep = e[1]-e[0]
                granda = np.dot(estep/3.*yi,inte(e))           
            grand[i] = granda
        f = np.dot(z,grand)
    
    return f
    
def onef_r_aeconst(r, a, e):
    """Returns probability density of orbital radius r for constant semi-major
    axis and eccentricity
    
    Args:
        r (float):
            Value of orbital radius (AU)
        a (float):
            Value of semi-major axis (AU)
        e (float):
            Value of eccentricity
            
    Returns:
        f (float):
            Probability density of orbital radius
    
    """
    
    if ((a*e)**2 - (a - r)**2) <= 0:
        f = 0.
    else:
        f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))
        
    return f
    
def onef_r_aconst(r, a, e, f_e):
    """Returns probability density of orbital radius r for constant semi-major
    axis and random eccentricity
    
    Args:
        r (float):
            Value of orbital radius (AU)
        a (float):
            Value of semi-major axis (AU)
        e (ndarray):
            Array containing minimum and maximum eccentricity
        f_e (callable(e)):
            Probability density function for eccentricity
            
    Returns:
        f (float):
            Probability density of orbital radius
    
    """
    
    emin = np.abs(1. - r/a) + 1e-6
    if emin > e.max():
        f = 0.
    else:
        grand = lambda ei: r/(np.pi*a*np.sqrt((a*ei)**2 - (a-r)**2))*f_e(ei)
        f = integrate.quad(grand, emin, e.max(), limit=100)[0]
        
    return f
    
def onef_r_econst(r, e, a, f_a):
    """Returns probability density of orbital radius r for random semi-major
    axis and constant eccentricity
    
    Args:
        r (float):
            Value of orbital radius (AU)
        e (float):
            Value of eccentricity
        a (ndarray):
            Array containing minimum and maximum semi-major axis (AU)
        f_a (callable(a)):
            Probability density function for semi-major axis
            
    Returns:
        f (float):
            Probability density of orbital radius
    
    """
    
    if (r == a.min()*(1.-e)) or (r == a.max()*(1.+e)):
        f = 0.
    else:
        amax = r/(1.-e)
        amin = r/(1.+e)
        if amax > a.max():
            amax = a.max()
        if amin < a.min():
            amin = a.min()
        grand = lambda ai: r/(np.pi*ai*np.sqrt((ai*e)**2 - (ai-r)**2))*f_a(ai)
        f = integrate.quad(grand, amin, amax, limit=500)[0]
        
    return f
    
def onef_dmags(dmag, s, ranges, val, pdfs, funcs, x):
    """Returns joint probability density of s and dmag
    
    Args:
        dmag (float):
            Value of difference in brightness magnitude
        s (float):
            Value of planet-star separation
        ranges (tuple):
            pmin (float): minimum value of geometric albedo
            pmax (float): maximum value of geometric albedo
            Rmin (float): minimum value of planetary radius (km)
            Rmax (float): maximum value of planetary radius (km)
            rmin (float): minimum value of orbital radius (AU)
            rmax (float): maximum value of orbital radius (AU)
            zmin (float): minimum value of pR^2 (km^2)
            zmax (float): maximum value of pR^2 (km^2)
        val (float):
            Value of sin(bstar)**2*Phi(bstar)
        pdfs (tuple):
            Probability density functions for pR^2 and orbital radius
        funcs (tuple):
            Inverse functions of sin(b)**2*Phi(b)
        x (float):
            Conversion factor between AU and km
            
    Returns:
        f (float):
            Joint probability density of s and dmag
            
    """
            
    pmin, pmax, Rmin, Rmax, rmin, rmax, zmin, zmax = ranges
    minrange = (pmax, Rmax, rmin, rmax)
    maxrange = (pmin, Rmin, rmax)
    
    if (dmag < util.mindmag(s, minrange, x)) or (dmag > util.maxdmag(s, maxrange, x)):
        f = 0.
    else:
        ztest = (s/x)**2*10.**(-0.4*dmag)/val
        ranges2 = (rmin, rmax)
        if ztest >= zmax:
            f = 0.
        else:
            if ztest < zmin:
                f = integrate.quad(onef_dmagsz, zmin, zmax, args=(dmag, s, val, pdfs, ranges2, funcs, x), full_output=1, limit=5000)[0]
            else:
                f = integrate.quad(onef_dmagsz, ztest, zmax, args=(dmag, s, val, pdfs, ranges2, funcs, x), full_output=1, limit=5000)[0]
            
    return f            

def onef_dmagsz(z, dmag, s, val, pdfs, ranges, funcs, x):
    """Returns joint probability density of s, dmag, and z
    
    Args:
        z (float):
            Value of pR^2
        dmag (float):
            Value of difference in brightness magnitude
        s (float):
            Value of planet-star separation (AU)
        val (float):
            Value of sin(bstar)**2*Phi(bstar)
        pdfs (tuple):
            Probability density functions for pR^2 and orbital radius
        funcs (tuple):
            Inverse functions of sin(b)**2*Phi(b)
        x (float):
            Conversion factor between AU and km
            
    Returns:
        f (float):
            Joint probability density of s, dmag, and z"""
            
    f_z, f_r = pdfs
    rmin, rmax = ranges
    binv1, binv2 = funcs
    f_b = lambda b: np.sin(b)/2.
    vals = (s/x)**2*10.**(-0.4*dmag)/z
    if type(z) == np.ndarray:
        f = np.zeros(np.shape(z))
        if vals.max() > val:
            b1 = binv1(vals[vals<val])
            b2 = binv2(vals[vals<val])
            r1 = s/np.sin(b1)
            r2 = s/np.sin(b2)
            f[vals<val] = f_z(z[vals<val])*f_b(b1)*f_r(r1)/np.abs(Jac(b1))
            f[vals<val] += f_z(z[vals<val])*f_b(b2)*f_r(r2)/np.abs(Jac(b2))
        else:
            b1 = binv1(vals)
            b2 = binv2(vals)
            r1 = s/np.sin(b1)
            r2 = s/np.sin(b2)
            f = f_z(z)*f_b(b1)*f_r(r1)/np.abs(Jac(b1))
            f += f_z(z)*f_b(b2)*f_r(r2)/np.abs(Jac(b2))
    else:
        if vals >= val:
            f = 0.
        else:
            b1 = binv1(vals)
            b2 = binv2(vals)
            r1 = s/np.sin(b1)
            r2 = s/np.sin(b2)
            if r1 < rmin or r1 > rmax:
                f1 = 0.
            else:
                f1 = f_z(z)*f_b(b1)*f_r(r1)/np.abs(Jac(b1))
            if r2 < rmin or r2 > rmax:
                f2 = 0.
            else:
                f2 = f_z(z)*f_b(b2)*f_r(r2)/np.abs(Jac(b2))
            f = f1 + f2
    
    return f