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
  def __init__(self, dimension, tab_dim ,tab_delta, spin_one_half=False, reproducible_randomness=True, custom_seed=0, use_mkl_random=True, use_mkl_fft=True):
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
    self.use_mkl_random = use_mkl_random
    self.use_mkl_fft = use_mkl_fft
    self.reproducible_randomness = reproducible_randomness
    self.custom_seed = custom_seed
    self.define_random_number_generator()
    return

# If possible, use the MKL random number generator
# Otherwise, use Numpy random number generator  
  def define_random_number_generator(self): 
    if self.use_mkl_random:
      try:
        import mkl_random
      except ImportError:
        self.use_mkl_random=False
        print('No mkl_random found; Fallback to Numpy random')
    if self.use_mkl_random:
      self.rng = lambda seed: mkl_random.RandomState(seed, brng='SFMT19937')
    else:
      self.rng = lambda seed: np.random.default_rng(seed)
    return
  
"""
def mkl_rng(seed):
  import mkl_random
  return mkl_random.RandomState(seed, brng='SFMT19937')
  
def np_rng(seed):
  return np.random.default_rng(seed)
"""