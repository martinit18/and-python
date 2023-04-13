#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:33:12 2019

@author: delande
"""

import math
import mkl
import numpy as np
import timeit
import copy
import anderson
import ctypes
import numpy.ctypeslib as ctl
import scipy.linalg.lapack as lapack
#cimport scipy.linalg.cython_lapack as lapack
#import numpy.linalg.lapack_lite as lapack

def core_lyapounov(dim_x, loop_step, disorder, energy, inv_tunneling):
  psi_cur=1.0
  psi_old=math.pi/math.sqrt(13.0)
#  psi_old=0.0
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
#  gamma+=math.log(abs(psi_cur))
  return gamma

def new_core_lyapounov(H, energy, i0, nrescale, debug=False):
  if H.dimension==1:
    dim_x = H.tab_dim[0]
    inv_tunneling = 1.0/H.tab_tunneling[0]
    loop_step = 16
    psi_cur=1.0
    psi_old=math.pi/math.sqrt(13.0)
#  psi_old=0.0
    gamma=0.0
#  h=0.0
    for i in range(0, dim_x, loop_step):
      jmax=min(i+loop_step,dim_x)
      for j in range(i,jmax):
        psi_new=psi_cur*inv_tunneling*(H.disorder[j]-energy)-psi_old
        psi_old=psi_cur
        psi_cur=psi_new
  # The next line can be used to count the number of sign changes, directly related to integrated density of states
  # disabled because the if has a large speed impact
  #      if psi_cur*psi_old<0.0: h+=1.0
      gamma+=math.log(abs(psi_cur))
      psi_old/=psi_cur
      psi_cur=1.0
#  gamma+=math.log(abs(psi_cur))
    return gamma
  if H.dimension==2:
# Propagation is along the x direction
    dim_x = H.tab_dim[0]
    dim_y = H.tab_dim[1]
#    print('dim_y=',dim_y)
    if dim_y==1:
      energy += 4.0
    tunneling_x = H.tab_tunneling[0]
    tunneling_y = H.tab_tunneling[1]
    if dim_y==2:
      tunneling_y *= 0.5
#    gnn = np.identity(dim_y)
#    g1n = np.zeros((dim_y,dim_y))
#    gnn_old = np.zeros((dim_y,dim_y))
    if debug:
      tab_log_trans = np.zeros((dim_x-1-i0)//nrescale+1)
#    tab_log_trans = np.zeros(dim_x)
#    tab_log_trans_2 = np.zeros(dim_x)
#    x = math.log(dim_y)
    x = 0.0
# Here starts the main loop for propagation along the x direction
    """
    for i in range(dim_x):
      if i%nrescale == 0:
        b, piv, info = lapack.dgetrf(gnn)
#      print(b, piv, info)
        if i>i0:
          g1n, info = lapack.dgetrs(b, piv, g1n)
          small_b = np.linalg.norm(g1n)
#        print(i,g1n,small_b)
          x -= math.log(small_b)
          g1n *= 1.0/small_b
        if i==i0:
#          small_b = math.sqrt(dim_y)
#          x -= math.log(small_b)
          g1n = np.identity(dim_y)
        if debug:
          tab_log_trans[i//nrescale] = x
        gnn_old, info = lapack.dgetrs(b, piv, gnn_old)
        gnn = -gnn_old
        for j in range(dim_y):
          gnn[j,j] += energy-H.disorder[i,j]
          gnn[j,(j+1)%dim_y] -= tunneling_y
          gnn[j,(j-1+dim_y)%dim_y] -= tunneling_y
        gnn_old = np.identity(dim_y)
      else:
        continue
    """

    """
    gnn = np.zeros((dim_y,dim_y))
    g1n = np.identity(dim_y)
    gnn_old = np.identity(dim_y)
    for i in range(dim_x):
      if i%nrescale==0:
        for j in range(dim_y):
          gnn[j,j] += energy-H.disorder[i,j]
          gnn[j,(j+1)%dim_y] -= tunneling_y
          gnn[j,(j-1+dim_y)%dim_y] -= tunneling_y
        b, piv, info = lapack.dgetrf(gnn)
        g1n, info = lapack.dgetrs(b, piv, g1n)
        small_b = np.linalg.norm(g1n)
        g1n *= 1.0/small_b
#        print(i,g1n,small_b)
        x -= math.log(small_b)
        if debug:
          tab_log_trans[i//nrescale] = x
        gnn, info = lapack.dgetrs(b, piv, -gnn_old)
#        gnn_old = np.identity(dim_y)
      else:
        for j in range(dim_y):
#          x2 = energy-H.disorder[i,j]
#          jp1y = (j+1)%dim_y
#          jm1y = (j+dim_y-1)%dim_y
#          for k in range(dim_y):
          gnn_old[j,0:dim_y] = (energy-H.disorder[i,j])*gnn[j,0:dim_y] - gnn[(j+1)%dim_y,0:dim_y] - gnn[(j+dim_y-1)%dim_y,0:dim_y] - gnn_old[j,0:dim_y]
        gnn, gnn_old = gnn_old, gnn
    """
    Bn_old = np.zeros((dim_y,dim_y))
    g1n = np.identity(dim_y)
    Bn = np.identity(dim_y)
    for i in range(((dim_x-1)//nrescale)*nrescale+1):
      if i%nrescale==1:
        for j in range(dim_y):
          Bn_old[j,j] += (energy-H.disorder[i,j])
          Bn_old[j,(j+1)%dim_y] -= 1.0
          Bn_old[j,(j+dim_y-1)%dim_y] -= 1.0
      else:
        for j in range(dim_y):
          Bn_old[0:dim_y,j] += (energy-H.disorder[i,j])*Bn[0:dim_y,j] - Bn[0:dim_y,(j+1)%dim_y] - Bn[0:dim_y,(j+dim_y-1)%dim_y]
#        Bn_old[j,0:dim_y] = (energy-H.disorder[i,j])*Bn[j,0:dim_y] - Bn[(j+1)%dim_y,0:dim_y] - Bn[(j+dim_y-1)%dim_y,0:dim_y] - Bn_old[j,0:dim_y]
      Bn, Bn_old = Bn_old, -Bn
      if i%nrescale==0:
#        if i<=2:
#          print(i,'Bn',Bn)
#          print(i,'Bn_old',Bn_old)
        b, piv, info = lapack.dgetrf(Bn)
        if i>=i0:
          g1n, info = lapack.dgetrs(b, piv, g1n)
          small_b = np.linalg.norm(g1n)
          g1n *= 1.0/small_b
#        print(i,g1n,small_b)
        if i>i0:
          x -= math.log(small_b)
#        if i<=2:
#          print(i,'g1n',g1n)
        if debug:
          tab_log_trans[(i-i0)//nrescale] = x
        Bn_old, info = lapack.dgetrs(b, piv, Bn_old)
        Bn = np.identity(dim_y)
    x /= H.tab_delta[0]*((dim_x-i0-1)//nrescale)*nrescale
    if debug:
      return x, tab_log_trans
    else:
      return x


def core_lyapounov_non_diagonal_disorder(dim_x, loop_step, disorder, b, non_diagonal_disorder, energy, tunneling):
  if b==1:
#    psi_temp = np.zeros(loop_step+2)
    psi_cur=1.0
    psi_old=math.pi/math.sqrt(13.0)
    gamma=0.0
    for i in range(0, dim_x, loop_step):
      jmax=min(i+loop_step,dim_x)
      for j in range(i,jmax):
  #      print(j,i,j+b-i,)
        psi_new=(psi_cur*(disorder[j]-energy)+psi_old*(non_diagonal_disorder[j,0]-tunneling))/(tunneling-non_diagonal_disorder[j+1,0])
        psi_old=psi_cur
        psi_cur=psi_new
      gamma+=math.log(abs(psi_cur))
      psi_old/=psi_cur
      psi_cur=1.0
  else:
    psi_temp = np.zeros(loop_step+2*b)
    for i in range(2*b-1):
      psi_temp[i]=math.pi/math.sqrt(11.0+2*b-i)
    psi_temp[2*b-1]=1.0
    gamma=0.0
    for i in range(0, dim_x, loop_step):
      jmax=min(i+loop_step,dim_x)
      for j in range(i,jmax):
  #      print(j,i,j+b-i,)
        my_sum=psi_temp[j+b-i]*(disorder[j]-energy)+psi_temp[j+b-1-i]*(non_diagonal_disorder[j+b-1,0]-tunneling)
        if b>1: my_sum+=psi_temp[j+b+1-i]*(non_diagonal_disorder[j+b,0]-tunneling)
        for k in range(-b,-1):
          my_sum+=psi_temp[j+k+b-i]*non_diagonal_disorder[j+b+k,-k-1]
        for k in range(2,b):
          my_sum+=psi_temp[j+k+b-i]*non_diagonal_disorder[j+b,k-1]
        if b==1:
          psi_temp[j-i+2]=my_sum/(tunneling-non_diagonal_disorder[j+1,0])
        else:
          psi_temp[j-i+2*b]=-my_sum/non_diagonal_disorder[j+b,b-1]
  #    print(psi_temp)
      gamma+=math.log(abs(psi_temp[2*b+jmax-i-1]))
      print(gamma)
      psi_temp[0:2*b]=psi_temp[loop_step:loop_step+2*b]/psi_temp[2*b+jmax-i-1]
  return gamma

class Lyapounov:
  def __init__(self, e_min, e_max, number_of_e_steps, want_ctypes=True, i0=10, nrescale=10):
    self.e_min = e_min
    self.e_max = e_max
    self.number_of_e_steps = number_of_e_steps
    self.want_ctypes = want_ctypes
    self.use_ctypes = want_ctypes
    self.i0 = i0
    self.nrescale = nrescale
    self.tab_energy = np.zeros(number_of_e_steps+1)
    if number_of_e_steps==0:
      self.e_step = 0.0
    else:
      self.e_step = (e_max - e_min)/number_of_e_steps
    for i_e in range(number_of_e_steps+1):
      e = e_min + self.e_step*i_e
      self.tab_energy[i_e] = e
    return

  def compute_lyapounov(self, i_seed, H, timing, debug=False):
    """
    try:
      from anderson._lyapounov import ffi,lib
      use_cffi = True
  #    print('Using CFFI version')
    except ImportError:
      use_cffi = False
      print("\n Warning, this uses the slow Python version, you should build the C version!\n")
    """
    start_lyapounov_time = timeit.default_timer()

    H.generate_disorder(seed=i_seed+1234)
#    np.random.seed(i_seed+1234)
#    psi_cur=np.random.standard_normal(1)
#    psi_old=np.random.standard_normal(1)
#    print(psi_cur,psi_old)
    dim_x = H.tab_dim[0]
    tunneling = H.tab_tunneling[0]
    inv_tunneling = 1.0/H.tab_tunneling[0]
#    print(tunneling)
    if self.want_ctypes:
      try:
        lyapounov_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/lyapounov.so")
#        print(lyapounov_ctypes_lib)
        if H.disorder_type=='nice':
          self.use_ctypes =hasattr(lyapounov_ctypes_lib,'core_lyapounov_non_diagonal_disorder')
          if self.use_ctypes:
            lyapounov_ctypes_lib.core_lyapounov_non_diagonal_disorder.argtypes = [ctypes.c_int, ctypes.c_int, ctl.ndpointer(np.float64), ctypes.c_int, ctl.ndpointer(np.float64), ctypes.c_double, ctypes.c_double]
            lyapounov_ctypes_lib.core_lyapounov_non_diagonal_disorder.restype = ctypes.c_double
        else:
          self.use_ctypes =hasattr(lyapounov_ctypes_lib,'core_lyapounov')
          if self.use_ctypes:
            lyapounov_ctypes_lib.core_lyapounov.argtypes = [ctypes.c_int, ctypes.c_int, ctl.ndpointer(np.float64), ctypes.c_double, ctypes.c_double]
            lyapounov_ctypes_lib.core_lyapounov.restype = ctypes.c_double
        if self.use_ctypes == False:
          lyapounov_ctypes_lib = None
          if H.seed == 1234:
            print("\nWarning, lyapounov C library found, but without routine core_lyapounov or core_lyapounov_non_diagonal_disorder, this uses the slow Python version!\n")
      except:
        self.use_ctypes = False
        lyapounov_ctypes_lib = None
        if H.seed == 1234:
          print("\nWarning, no lyapounov C library found, this uses the slow Python version!\n")
    else:
      self.use_ctypes = False
      lyapounov_ctypes_lib = None

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
    loop_step=16
    number_of_energies = self.tab_energy.size
    tab_gamma = np.zeros(number_of_energies)
  #  tab_h = np.zeros(number_of_energies)
    for i_energy in range(number_of_energies):
      if self.use_ctypes:
        if H.disorder_type=='nice':
          tab_gamma[i_energy] = lyapounov_ctypes_lib.core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder, H.b, H.non_diagonal_disorder, self.tab_energy[i_energy], tunneling)
        else:
          tab_gamma[i_energy] = lyapounov_ctypes_lib.core_lyapounov(dim_x, loop_step, H.disorder, self.tab_energy[i_energy], inv_tunneling)
      else:
        if H.disorder_type=='nice':
          tab_gamma[i_energy] = core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder, H.b, H.non_diagonal_disorder, self.tab_energy[i_energy], tunneling)
#          tab_gamma[i_energy] = core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder,  H.non_diagonal_disorder, self.tab_energy[i_energy], tunneling)
        else:
#          tab_gamma[i_energy] = core_lyapounov(dim_x, loop_step, H.disorder, self.tab_energy[i_energy], inv_tunneling)
          if debug:
            tab_gamma[i_energy], tab_log_trans = new_core_lyapounov(H, self.tab_energy[i_energy], self.i0, self.nrescale, debug=debug)
          else:
            tab_gamma[i_energy] = new_core_lyapounov(H, self.tab_energy[i_energy], self.i0, self.nrescale, debug=debug)

#    print(2.0*new_core_lyapounov(H, 0.0)/(dim_x*H.tab_delta[0]))


    timing.LYAPOUNOV_TIME += timeit.default_timer() - start_lyapounov_time
    if H.dimension==1:
      if H.disorder_type=='nice':
        timing.LYAPOUNOV_NOPS += 10*dim_x*number_of_energies
      else:
        timing.LYAPOUNOV_NOPS += 5*dim_x*number_of_energies
#    if H.dimension==2:

  #  lyapounov = gamma/(dim_x*H.delta_x)
  #  integrated_dos = h/(dim_x*H.delta_x)
  #  return (lyapounov,integrated_dos)
  # The Lyapounov is here computed for the intensity (halve it for wavefunction), hence the multiplicative factor 2
  # The Lyapounov is now computed for the wavefunction (no factor 2)
    if H.dimension==1:
      return tab_gamma/(dim_x*H.tab_delta[0])
    if debug:
      return tab_gamma, tab_log_trans
    else:
      return tab_gamma
