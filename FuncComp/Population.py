# -*- coding: utf-8 -*-
"""
v1: Created on May 31, 2016
author: Daniel Garrett (dg622@cornell.edu)
"""

import numpy as np
import astropy.units as u
import scipy.integrate as integrate

class Population(object):
    """This class contains all the planetary parameters necessary for sampling
    or finding probability distribution functions necessary for derived 
    quantities.
    
    Args:
        none
        
    Attributes:
        arange (Quantity):
            1D numpy ndarray containing minimum and maximum semi-major axis 
            (default units of AU)
        erange (ndarray):
            1D numpy ndarray containing minimum and maximum eccentricity
        Rrange (Quantity):
            1D numpy ndarray containing minimum and maximum planetary radius
            (default units of km)
        prange (ndarray):
            1D numpy ndarray containing minimum and maximum geometric albedo
        sigmar (float):
            Rayleigh distribution parameter
        ernorm (float):
            Rayleigh distribution normalization constant
    
    """
    
    def __init__(self):
        # minimum semi-major axis (AU)
        a_min = 0.5
        # maximum semi-major axis (AU)
        a_max = 5. 
        # semi-major axis range
        self.arange = np.array([a_min, a_max])*u.AU
        # minimum eccentricity
        e_min = np.finfo(float).eps*100.
        # maximum eccentricity
        e_max = 0.35
        # eccentricity range
        self.erange = np.array([e_min, e_max])
        # minimum planetary radius
        Rmin = 6000.
        # maximum planetary radius
        Rmax = 30000.
        self.Rrange = np.array([Rmin, Rmax])*u.km
        # minimum albedo
        p_min = 0.2
        # maximum albedo
        p_max = 0.3
        self.prange = np.array([p_min, p_max])
        # normalization constants for various distributions
        # Rayleigh distribution
        self.sigmar = 0.25
        # normalize Rayleigh distribution
        fun = lambda x: (x/self.sigmar**2)*np.exp(-x**2/(2.*self.sigmar**2))
        self.ernorm = integrate.quad(fun, self.erange.min(), self.erange.max())[0]
    
    def semi_axis(self, x):
        """Probability density function for semi-major axis in AU
        
        Args:
            x (float):
                Semi-major axis value in AU
                
        Returns:
            pdf (float):
                Probability density (units of 1/AU)
        
        """ 
        
        # log uniform distribution
        pdf = 1./(x*(np.log(self.arange.max().value) - np.log(self.arange.min().value)))
        
        return pdf
        
    def eccentricity(self, x):
        """Probability density function for eccentricity
        
        Args:
            x (float):
                eccentricity value
        
        Returns:
            pdf (float):
                probability density
        
        """
        
        # Rayleigh distribution
        pdf = (x/self.sigmar**2)*np.exp(-x**2/(2.*self.sigmar**2))/self.ernorm
        
        return pdf
        
    def Radius(self, x):
        """Probability density function for planet radius (km)
        
        Args:
            x (float):
                planet radius in km
        
        Returns:
            pdf (float):
                probability density function value
                
        """
        
        # log uniform distribution
        pdf = 1./(x*(np.log(self.Rrange.max().to(u.km).value) - np.log(self.Rrange.min().to(u.km).value)))
        
        return pdf
        
    def albedo(self, x):
        """Probability density function for albedo
        
        Args:
            x (float):
                albedo
        
        Returns:
            pdf (float):
                probability density function value
                
        """
        
        # log uniform distribution
        pdf = 1./(x*(np.log(self.prange.max()) - np.log(self.prange.min())))
        
        return pdf