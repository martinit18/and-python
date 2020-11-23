#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:26:01 2020

@author: delande
"""

import numpy as np

"""
The class Geometry defines the geometry of the system, with no reference to the Hamiltonian
"""
class Geometry:
  def __init__(self, dimension, tab_dim ,tab_delta):
    self.dimension = dimension
    self.tab_dim = tab_dim
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
    self.ntot = ntot
    self.tab_position = list()
    for i in range(dimension):
      self.tab_position.append(0.5*tab_delta[i]*np.arange(1-tab_dim[i],tab_dim[i]+1,2))
    self.frequencies = []
    for i in range(dimension):
      self.frequencies.append(np.fft.fftshift(np.fft.fftfreq(tab_dim[i],d=tab_delta[i]/(2.0*np.pi))))
    return
