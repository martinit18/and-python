#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:26:01 2020

@author: delande
"""

import copy
import numpy as np

"""
The class Geometry defines the geometry of the system, including the local Hilbert space on each site (currently limited to spin 1/2 systems) with no reference to the Hamiltonian
"""
class Geometry:
  def __init__(self, dimension, tab_dim ,tab_delta, spin_one_half=False):
    self.dimension = dimension
    self.tab_dim = tab_dim
#    self.tab_hs_dim = copy.deepcopy(tab_dim)
    self.tab_delta = tab_delta
 #    self.array_dim = np.array(tab_dim,dtype=np.intc)
    self.tab_dim_cumulative = np.zeros(dimension+1,dtype=int)
    ntot = 1
    self.tab_dim_cumulative[dimension] = 1
    self.delta_vol = 1.0
    for i in range(dimension-1,-1,-1):
      ntot *= tab_dim[i]
      self.tab_dim_cumulative[i] = ntot
      self.delta_vol *= tab_delta[i]
# Total number of sites
    self.ntot = ntot
    self.grid_position = list()
    for i in range(dimension):
      self.grid_position.append(0.5*tab_delta[i]*np.arange(1-tab_dim[i],tab_dim[i]+1,2))
    self.frequencies = []
    for i in range(dimension):
      self.frequencies.append(np.fft.fftshift(np.fft.fftfreq(tab_dim[i],d=tab_delta[i]/(2.0*np.pi))))
    self.tab_extended_dim = copy.deepcopy(tab_dim)
    self.spin_one_half = spin_one_half
    if spin_one_half:
# Local Hilbert space dimension
      self.lhs_dim = 2
      self.tab_extended_dim.append(self.lhs_dim)
    else:
      self.lhs_dim = 1
# Total Hilbert space dimension
    self.hs_dim = self.lhs_dim*self.ntot
    return
