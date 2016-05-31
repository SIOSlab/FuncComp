# -*- coding: utf-8 -*-
"""
v1: Created on May 31, 2016
author: Daniel Garrett (dg622@cornell.edu)
"""

import numpy as np

def maxdmag(s, ranges, x):
    """Calculates the maximum difference in magnitude for a given population
    and apparent separation value
    
    Args:
        s (ndarray):
            Apparent separation (AU)
        ranges (tuple):
            pmin (float): minimum geometric albedo
            Rmin (float): minimum planetary radius (km)
            rmax (float): maximum distance from star (AU)
        x (float):
            Conversion factor for AU to km
    
    Returns:
        maxdmag (ndarray):
            Maximum difference in magnitude for given population and separation
    
    """
    
    pmin, Rmin, rmax = ranges
    PhiL = lambda b: (1./np.pi)*(np.sin(b) + (np.pi - b)*np.cos(b))
    maxdmag = -2.5*np.log10(pmin*(Rmin*x/rmax)**2*PhiL(np.pi - np.arcsin(s/rmax)))

    return maxdmag
    
def mindmag(s, ranges, x):
    """Calculates the minimum difference in magnitude for a given population
    and apparent separation value
    
    Args:
        s (ndarray):
            Apparent separation (AU)
        ranges (tuple):
            pmax (float): maximum geometric albedo
            Rmax (float): maximum planetary radius (km)
            rmin (float): minimum distance from star (AU)
            rmax (float): maximum distance from star (AU)
        x (float):
            Conversion factor for AU to km
            
    Returns:
        mindmag (ndarray):
            Minimum difference in magnitude for given population and separation
    
    """
    pmax, Rmax, rmin, rmax = ranges
    bstar = 1.104728818644543
    PhiL = lambda b: (1./np.pi)*(np.sin(b) + (np.pi - b)*np.cos(b))
    if type(s) == np.ndarray:
        mindmag = -2.5*np.log10(pmax*(Rmax*x*np.sin(bstar)/s)**2*PhiL(bstar))
        mindmag[s < rmin*np.sin(bstar)] = -2.5*np.log10(pmax*(Rmax*x/rmin)**2*PhiL(np.arcsin(s[s < rmin*np.sin(bstar)]/rmin)))
        mindmag[s > rmax*np.sin(bstar)] = -2.5*np.log10(pmax*(Rmax*x/rmax)**2*PhiL(np.arcsin(s[s > rmax*np.sin(bstar)]/rmax)))
    else:
        if s < rmin*np.sin(bstar):
            mindmag = -2.5*np.log10(pmax*(Rmax*x/rmin)**2*PhiL(np.arcsin(s/rmin)))
        elif s > rmax*np.sin(bstar):
            mindmag = -2.5*np.log10(pmax*(Rmax*x/rmax)**2*PhiL(np.arcsin(s/rmax)))
        else:
            mindmag = -2.5*np.log10(pmax*(Rmax*x*np.sin(bstar)/s)**2*PhiL(bstar))
    
    return mindmag