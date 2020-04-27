#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:33:12 2019

@author: delande
"""

import math
import numpy as np
import timeit

def core_lyapounov(dim_x, loop_step, disorder, energy, inv_tunneling):
  psi_cur=1.0
  psi_old=0.0
  gamma=0.0
#  h=0.0
  for i in range(0, dim_x, loop_step):
    jmax=min(i+loop_step,dim_x)
    for j in range(i,jmax):
      psi_new=psi_cur*(inv_tunneling*(disorder[j]-energy))-psi_old
      psi_old=psi_cur
      psi_cur=psi_new
# The next line can be used to count the number of sign changes, directly related to integrated density of states
# disabled because the if has a large speed impact
#      if psi_cur*psi_old<0.0: h+=1.0
    gamma+=math.log(abs(psi_cur))
    psi_old/=psi_cur
    psi_cur=1.0
  gamma+=math.log(abs(psi_cur))
  return gamma

def compute_lyapounov(i, H, tab_energy):
  try:
    from anderson._lyapounov import ffi,lib
    use_cffi = True
#    print('Using CFFI version')
  except ImportError:
    use_cffi = False
    print("\n Warning, this uses the slow Python version, you should build the C version!\n")

  H.generate_disorder(seed=i+1234)
  dim_x = H.dim_x
  inv_tunneling = 1.0/H.tunneling

  start_lyapounov_time = timeit.default_timer()
  """
  What follows is a poor man's version of the recursion for transfer matrix
  for a single energy
  The routine core_lyapounov (Python version) or core_lyapounov_cffi (C version) are equivalelent, but much faster
  Note that the CFFI  version is more than 100 times faster than the Python version...
  r = 1.0
  gamma = 0.0
  h = 0.0
  for i in range(dim_x):
    r = inv_tunneling*(H.disorder[i]-energy)-1.0/r
    gamma += math.log(abs(r))
    if r<0.0: h+=1.0
  """
  loop_step=64
  number_of_energies = tab_energy.size
  tab_gamma = np.zeros(number_of_energies)
#  tab_h = np.zeros(number_of_energies)
  for i_energy in range(number_of_energies):
    if use_cffi:
      tab_gamma[i_energy]=lib.core_lyapounov_cffi(dim_x, loop_step,ffi.cast('double *',ffi.from_buffer(H.disorder)), tab_energy[i_energy], inv_tunneling)
    else:
      tab_gamma[i_energy]=core_lyapounov(dim_x, loop_step, H.disorder, tab_energy[i_energy], inv_tunneling)

  """
  loop_step=64
  psi_cur=1.0
  psi_old=0.0
  for i in range(0, dim_x, loop_step):
    jmax=min(i+loop_step,dim_x)
    for j in range(i,jmax):
      psi_new=psi_cur*(inv_tunneling*(H.disorder[j]-energy))-psi_old
      psi_old=psi_cur
      psi_cur=psi_new
    gamma+=math.log(abs(psi_cur))
    psi_old/=psi_cur
    psi_cur=1.0
  gamma+=math.log(abs(psi_cur))
  """
  """
  loop_step=64
  psi_cur=1.0;
  psi_old=0.0;
  for i in range(dim_x):
    psi_new=psi_cur*(inv_tunneling*(H.disorder[i]-energy))-psi_old
    psi_old=psi_cur
    psi_cur=psi_new
    if i%loop_step==0:
      gamma+=math.log(abs(psi_cur))
      psi_old/=psi_cur
      psi_cur=1.0
  gamma+=math.log(abs(psi_cur))
  """
  used_time = timeit.default_timer() - start_lyapounov_time
  number_of_ops = 5*dim_x*number_of_energies
#  lyapounov = gamma/(dim_x*H.delta_x)
#  integrated_dos = h/(dim_x*H.delta_x)
#  return (lyapounov,integrated_dos)
# The Lyapounov is here computed for the wavefunction (double for intensity)
  return tab_gamma/(dim_x*H.delta_x),used_time,number_of_ops

class Lyapounov:
  def __init__(self, e_min, e_max, number_of_e_steps):
    self.e_min = e_min
    self.e_max = e_max
    self.number_of_e_steps = number_of_e_steps
    self.tab_energy = np.zeros(number_of_e_steps+1)
    if number_of_e_steps==0:
      self.e_step = 0.0
    else:
      self.e_step = (e_max - e_min)/number_of_e_steps
    for i_e in range(number_of_e_steps+1):
      e = e_min + self.e_step*i_e
      self.tab_energy[i_e] = e
    return

  def compute_lyapounov(self, i, H):
    try:
      from anderson._lyapounov import ffi,lib
      use_cffi = True
  #    print('Using CFFI version')
    except ImportError:
      use_cffi = False
      print("\n Warning, this uses the slow Python version, you should build the C version!\n")

#   start_lyapounov_time = timeit.default_timer()
    H.generate_disorder(seed=i+1234)
    dim_x = H.dim_x
    inv_tunneling = 1.0/H.tunneling

    start_lyapounov_time = timeit.default_timer()
    """
    What follows is a poor man's version of the recursion for transfer matrix
    for a single energy
    The routine core_lyapounov (Python version) or core_lyapounov_cffi (C version) are equivalelent, but much faster
    Note that the CFFI  version is more than 100 times faster than the Python version...
    r = 1.0
    gamma = 0.0
    h = 0.0
    for i in range(dim_x):
      r = inv_tunneling*(H.disorder[i]-energy)-1.0/r
      gamma += math.log(abs(r))
      if r<0.0: h+=1.0
    """
    loop_step=64
    number_of_energies = self.tab_energy.size
    tab_gamma = np.zeros(number_of_energies)
  #  tab_h = np.zeros(number_of_energies)
    for i_energy in range(number_of_energies):
      if use_cffi:
        tab_gamma[i_energy]=lib.core_lyapounov_cffi(dim_x, loop_step,ffi.cast('double *',ffi.from_buffer(H.disorder)), self.tab_energy[i_energy], inv_tunneling)
      else:
        tab_gamma[i_energy]=core_lyapounov(dim_x, loop_step, H.disorder, self.tab_energy[i_energy], inv_tunneling)

    """
    loop_step=64
    psi_cur=1.0
    psi_old=0.0
    for i in range(0, dim_x, loop_step):
      jmax=min(i+loop_step,dim_x)
      for j in range(i,jmax):
        psi_new=psi_cur*(inv_tunneling*(H.disorder[j]-energy))-psi_old
        psi_old=psi_cur
        psi_cur=psi_new
      gamma+=math.log(abs(psi_cur))
      psi_old/=psi_cur
      psi_cur=1.0
    gamma+=math.log(abs(psi_cur))
    """
    """
    loop_step=64
    psi_cur=1.0;
    psi_old=0.0;
    for i in range(dim_x):
      psi_new=psi_cur*(inv_tunneling*(H.disorder[i]-energy))-psi_old
      psi_old=psi_cur
      psi_cur=psi_new
      if i%loop_step==0:
        gamma+=math.log(abs(psi_cur))
        psi_old/=psi_cur
        psi_cur=1.0
    gamma+=math.log(abs(psi_cur))
    """
    used_time = timeit.default_timer() - start_lyapounov_time
    number_of_ops = 5*dim_x*number_of_energies
  #  lyapounov = gamma/(dim_x*H.delta_x)
  #  integrated_dos = h/(dim_x*H.delta_x)
  #  return (lyapounov,integrated_dos)
  # The Lyapounov is here computed for the wavefunction (double for intensity)
    return tab_gamma/(dim_x*H.delta_x),used_time,number_of_ops
