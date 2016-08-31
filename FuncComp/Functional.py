# -*- coding: utf-8 -*-
"""
Updated on August 30, 2016
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
        rmin = pop.arange.min().value*(1.0 - pop.erange.max())
        rmax = pop.arange.max().value*(1.0 + pop.erange.max())
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
        if dmagmax is None:
            dmagmax = -2.5*np.log10(pmin*(Rmin*x)**2/rmax**2*PhiL(np.pi - np.arcsin(0.0001/rmax)))
        # probability density functions
        f_R = pop.Radius
        f_p = pop.albedo
        f_e = pop.eccentricity
        f_a = pop.semi_axis
        # linearly spaced array for range values
        r = np.linspace(rmin, rmax, num=n)
        # inverse 1 (<bstar) of sin(b)^2*Phi(b)
        b1 = np.linspace(0.0, bstar, num=50*n)
        binv1 = interpolate.InterpolatedUnivariateSpline(np.sin(b1)**2*PhiL(b1), b1, k=3, ext=1)
        # inverse 2 (>bstar) of sin(b)^2*Phi(b)
        b2 = np.linspace(bstar, np.pi, num=50*n)
        b2val = np.sin(b2)**2*PhiL(b2)
        binv2 = interpolate.InterpolatedUnivariateSpline(b2val[::-1], b2[::-1], k=3, ext=1)
        # if pp is loaded, set up jobserver
        if ppLoad:
            ppservers = ()
            if len(sys.argv) > 1:
                ncpus = int(sys.argv[1])
                # Creates jobserver with ncpus workers
                job_server = pp.Server(ncpus, ppservers=ppservers)
            else:
                job_server = pp.Server(ppservers=ppservers)
        # get pdf of r
        print 'finding pdf of r'
        pdfr = np.array([])
        if aconst and econst:
            if ppLoad:
                jobs = [(job_server.submit(onef_r_aeconst, (ri, amin, emin, f_a), (), ('FuncComp.Functional', 'numpy as np'))) for ri in r]
                for job in jobs:
                    pdfr = np.hstack((pdfr, job()))
            else:
                for ri in r:
                    temp = onef_r_aeconst(ri, amin, emin, f_a)
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
            if ppLoad:
                jobs = [(job_server.submit(onef_r, (ri, pdfs, ranges), (grand2,grand1), ('FuncComp.Functional', 'numpy as np', 'scipy.integrate as integrate'))) for ri in r]
                for job in jobs:
                    pdfr = np.hstack((pdfr, job()))            
            else:
                for ri in r:
                    temp = onef_r(ri,pdfs,ranges)
                    pdfr = np.append(pdfr,temp)
        # pdf of r
        f_r = interpolate.InterpolatedUnivariateSpline(r, pdfr, k=3, ext=1)
        # get pdf of zeta = p*R^2
        print 'finding pdf of p*R^2'
        ranges = (pmin, pmax, Rmin, Rmax)
        pdfs = (f_p, f_R)
        pdfz = f_zeta(ranges, pdfs, n)
        # get joint pdf of s, dmag
        print 'finding joint pdf of s, dmag'
        # check for defaults and replace if set to None
        if smax is None:
            self.s = np.linspace(smin,rmax,num=ns)
        else:
            self.s = np.linspace(smin,smax,num=ns)
        self.dmag = np.linspace(dmagmin,dmagmax,num=ndmag)
        self.pc = np.zeros((len(self.dmag),len(self.s)))
        val = np.sin(bstar)**2*PhiL(bstar)
        if pconst and Rconst:
            f_z = pdfz.pRconst
        elif pconst:
            f_z = pdfz.pconst
        elif Rconst:
            f_z = pdfz.Rconst
        else:
            f_z = pdfz.f_z
        pdfs = (f_z, f_r)
        funcs = (binv1, binv2)
        if pconst and Rconst:
            ranges = (rmin, rmax)
            if ppLoad:
                for i in xrange(len(self.s)):
                    print 'Progress: %r / %r' % (i+1, len(self.s))
                    if self.s[i] == 0.:
                        self.pc[:,i] = np.zeros((len(self.dmag),))
                    else:
                        jobs = [(job_server.submit(onef_dmagsz, (zmin, self.dmag[j], self.s[i], val, pdfs, ranges, funcs, x), (Jac,), ('FuncComp.Functional', 'numpy as np'))) for j in xrange(len(self.dmag))]
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
                            self.pc[j,i] = onef_dmagsz(zmin, self.dmag[j], self.s[i], val, pdfs, ranges, funcs, x)
        else:
            ranges = (pmin, pmax, Rmin, Rmax, rmin, rmax, zmin, zmax)
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
        self.pdf = self.grid.ev
        # vectorized method to calculate completeness comp(smin,smax,dmagmin,dmagmax)
        self.comp = np.vectorize(self.grid.integral)
    
class f_zeta(object):
    """Determines the probability density function for zeta = p*R^2 and
    stores as instance method
        
    Args:
        ranges (tuple):
            pmin (float): minimum geometric albedo
            pmax (float): maximum geometric albedo
            Rmin (float): minimum planetary radius (km)
            Rmax (float): maximum planetary radius (km)
        pdfs (tuple):
            f_p (callable(p)): probability density function for albedo
            f_R (callable(R)): probability density function for planetary radius
        n (int):
            Number of points in zeta
                
    Attributes:
        pmin (float):
            Minimum value of geometric albedo
        pmax (float):
            Maximum value of geometric albedo
        Rmin (float):
            Minimum value of planetary radius (km)
        Rmax (float):
            Maximum value of planetary radius (km)
        f_p (callable(p)):
            Probability density function for albedo
        f_R (callable(R)):
            Probability density function for radius
        f_z (callable(z)):
            Probability density function for zeta = p*R^2
        """
        
    def __init__(self, ranges, pdfs, n):
        self.pmin, self.pmax, self.Rmin, self.Rmax = ranges
        self.f_p, self.f_R = pdfs
        pconst = self.pmin == self.pmax
        Rconst = self.Rmin == self.Rmax
        if not pconst and not Rconst:
            pdfs = (self.f_p, self.f_R)
            ranges = (self.pmin, self.pmax, self.Rmin, self.Rmax)
            z = np.linspace(self.pmin*self.Rmin**2, self.pmax*self.Rmax**2, num=n)
            pdfz = np.array([])
            if ppLoad:
                # set up job server for parallel computations
                ppservers = ()
                if len(sys.argv) > 1:
                    ncpus = int(sys.argv[1])
                    # Creates jobserver with ncpus workers
                    job_server = pp.Server(ncpus, ppservers=ppservers)
                else:
                    job_server = pp.Server(ppservers=ppservers)
                jobs = [(job_server.submit(onef_zeta, (zetai, pdfs, ranges), (pgrand,), ('FuncComp.Functional', 'numpy as np', 'scipy.integrate as integrate'))) for zetai in z]
                for job in jobs:
                    pdfz = np.hstack((pdfz, job()))
                job_server.destroy()
            else:
                for zetai in z:
                    temp = onef_zeta(zetai, pdfs, ranges)
                    pdfz = np.append(pdfz,temp)
            # pdf of zeta = p*R^2
            self.f_z = interpolate.InterpolatedUnivariateSpline(z, pdfz, k=3, ext=1)
            
    def pRconst(self,zi):
        """If geometric albedo and planetary radius are constant, this method 
        returns 1.
        
        """
        return 1.0
        
    def pconst(self, zi):
        """If geometric albedo is constant, this gives the probability density
        function of z = p*R^2
        
        """
        
        f = 1.0/(2.0*np.sqrt(self.pmin*zi))*self.f_R(np.sqrt(zi/self.pmin))
        
        return f
    
    def Rconst(self, zi):
        """If planetary radius is constant, this give the probability density
        function of z = p*R^2
        
        """
        
        f = 1.0/self.Rmin**2*self.f_p(zi/self.Rmin**2)
        
        return f
        
def Jac(b):
    """Returns the determinant of the Jacobian matrix
    
    Args:
        b (float or ndarray):
            Phase angle
            
    Returns:
        J (float or ndarray):
            Determinant of Jacobian matrix
            
    """
            
    PhiL = lambda beta: (1.0/np.pi)*(np.sin(beta) + (np.pi-beta)*np.cos(beta))
    # derivative of PhiL with respect to beta
    PhiLp = lambda beta: (1.0/np.pi)*(beta-np.pi)*np.sin(beta)
    J = -2.5/(PhiL(b)*np.log(10.0))*PhiLp(b)*np.sin(b) - 5.0/np.log(10.0)*np.cos(b)
    
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
    p1 = zetai/Rmax**2
    p2 = zetai/Rmin**2
    if p1 < pmin:
        p1 = pmin
    if p2 > pmax:
        p2 = pmax
    
    f = integrate.fixed_quad(pgrand,p1,p2,args=(zetai,f_p,f_R),n=200)[0]

    return f
    
def pgrand(p, z, f_p, f_R):
    """Integrand for determining probability density of zeta
    
    Args:
        p (float):
            Value of geometric albedo
        z (float):
            Value of zeta, p*R**2
        f_p (callable(p)):
            Probability density function for geometric albedo
        f_R (callable(R)):
            Probability density function for planetary radius
    
    Returns:
        f (float):
            Integrand
    
    """
    return 1.0/(2.0*np.sqrt(z*p))*f_R(np.sqrt(z/p))*f_p(p)
   
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
            
    amin, amax, emin, emax = ranges
    if (ri == amin*(1.0 - emax)) or (ri == amax*(1.0 + emax)):
        f = 0.0
    else:
        f_e, f_a = pdfs
        grand2v = np.vectorize(grand2)
        a1 = ri/(1.0+emax)
        a2 = ri/(1.0-emax)
        if a1 < amin:
            a1 = amin
        if a2 > amax:
            a2 = amax

        f = integrate.fixed_quad(grand2v, a1, a2, args=(ri,emin,emax,f_e,f_a), n=200)[0]
    
    return f
    
def grand1(e, a, r, f_e, f_a):
    """Returns first integrand for determining probability density of orbital
    radius
    
    Args:
        e (float):
            Value of eccentricity
        a (float):
            Value of semi-major axis (AU)
        r (float):
            Value of orbital radius (AU)
        f_e (callable(e)):
            Probability density function for eccentricity
        f_a (callable(a)):
            Probability density function for semi-major axis
            
    Returns:
        f (float):
            Integrand
    
    """
    
    f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*f_e(e)*f_a(a)
    
    return f
    
def grand2(a, r, emin, emax, f_e, f_a):
    """Returns second integrand for determining probability density of orbital
    radius
    
    Args:
        a (float):
            Value of semi-major axis (AU)
        r (float):
            Value of orbital radius (AU)
        emin (float):
            Minimum eccentricity
        emax (float):
            Maximum eccentricity
        f_e (callable(e)):
            Probability density function for eccentricity
        f_a (callable(a)):
            Probability density function for semi-major axis
    
    """
    
    emin1 = np.abs(1.0 - r/a)
    if emin1 < emin:
        emin1 = emin
    
    if emin1 > emax:
        f = 0.0
    else:
        f = integrate.fixed_quad(grand1, emin1, emax, args=(a,r,f_e,f_a), n=100)[0]

    return f
    
def onef_r_aeconst(r, a, e, f_a):
    """Returns probability density of orbital radius r for constant semi-major
    axis and eccentricity
    
    Args:
        r (float):
            Value of orbital radius (AU)
        a (float):
            Value of semi-major axis (AU)
        e (float):
            Value of eccentricity
        f_a (callable(a)):
            Probability density function for semi-major axis
            
    Returns:
        f (float):
            Probability density of orbital radius
    
    """
    
    if (r > a*(1.0-e)) & (r < a*(1.0+e)):
        f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))
    else:
        f = 0.0
        
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
    
    emin = np.abs(1.0 - r/a) 
    if emin > e.max():
        f = 0.0
    else:
        if emin < e.min():
            low = e.min()
        else:
            low = emin
                
        f = integrate.fixed_quad(grandac, low, e.max(), args=(a,r,f_e), n=200)[0]
        
    return f
    
def grandac(e, a, r, f_e):
    """Returns integrand for probability density of orbital radius where
    semi-major axis is a constant
    
    Args:
        e (float):
            Value of eccentricity
        a (float):
            Value of semi-major axis (AU)
        r (float):
            Value of orbital radius (AU)
        f_e (callable(e)):
            Probability density function for eccentricity
            
    Returns:
        f (float):
            Integrand for probability density of orbital radius where 
            semi-major axis is a constant
        
        """
    
    f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*f_e(e)
    
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
    
    a1 = r/(1.0-e)
    a2 = r/(1.0+e)
    if a.max() < a1:
        high = a.max()
    else:
        high = a1
    if a.min() < a2:
        low = a2
    else:
        low = a.min()
        
    f = integrate.fixed_quad(grandec, low, high, args=(e,r,f_a), n=200)[0]
    
    return f
    
def grandec(a, e, r, f_a):
    """Returns integrand for probability density of orbital radius where
    eccentricity is a constant
    
    Args:
        a (float):
            Value of semi-major axis (AU)
        e (float):
            Value of eccentricity
        r (float):
            Value of orbital radius (AU)
        f_a (callable(a)):
            Probability density function for semi-major axis
            
    Returns:
        f (float):
            Integrand for probability density of orbital radius where 
            eccentricity is a constant
    
    """
    
    f = r/(np.pi*a*np.sqrt((a*e)**2-(a-r)**2))*f_a(a)
    
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
    pconst = pmin == pmax
    Rconst = Rmin == Rmax
    if (dmag < util.mindmag(s, minrange, x)) or (dmag > util.maxdmag(s, maxrange, x)):
        f = 0.0
    elif (pconst & Rconst):
        ranges2 = (rmin, rmax)
        f = onef_dmagsz(pmin*Rmin**2, dmag, s, val, pdfs, ranges2, funcs, x)
    else:
        ztest = (s/x)**2*10.0**(-0.4*dmag)/val
        ranges2 = (rmin, rmax)
        if ztest >= zmax:
            f = 0.0
        else:
            if ztest < zmin:
                f = integrate.fixed_quad(onef_dmagsz, zmin, zmax, args=(dmag, s, val, pdfs, ranges2, funcs, x), n=61)[0]
            else:
                f = integrate.fixed_quad(onef_dmagsz, ztest, zmax, args=(dmag, s, val, pdfs, ranges2, funcs, x), n=61)[0]
            
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
    f_b = lambda b: np.sin(b)/2.0
    vals = (s/x)**2*10.0**(-0.4*dmag)/z
    if type(z) == np.ndarray:
        f = np.zeros(z.shape)
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
            f = 0.0
        else:
            b1 = binv1(vals)
            b2 = binv2(vals)
            r1 = s/np.sin(b1)
            r2 = s/np.sin(b2)
            if r1 < rmin or r1 > rmax:
                f1 = 0.0
            else:
                f1 = f_z(z)*f_b(b1)*f_r(r1)/np.abs(Jac(b1))
            if r2 < rmin or r2 > rmax:
                f2 = 0.0
            else:
                f2 = f_z(z)*f_b(b2)*f_r(r2)/np.abs(Jac(b2))
            f = f1 + f2
    
    return f