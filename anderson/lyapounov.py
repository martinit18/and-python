#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:33:12 2019

@author: delande
"""

import math
import numpy as np
import timeit
import anderson
import scipy.linalg.lapack as lapack
import scipy.sparse
import scipy.sparse.linalg

def numba_decorator(x):
  try:
    import numba
    return numba.jit(fastmath=True,cache=True)(x)
  except:
    print('numba package not found, this will use the slower regular Python version')
    return(x)
  pass

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


def core_lyapounov_2d(H, energy, i0, nrescale, timing, use_ctypes, debug=False):
# Propagation is along the x direction
  dim_x = H.tab_dim[0]
  dim_y = H.tab_dim[1]
#    print('dim_y=',dim_y)
  if dim_y==1:
    energy += 4.0
  tunneling_x = H.tab_tunneling[0]
  inv_tunneling_x = 1.0/tunneling_x
  tunneling_y = H.tab_tunneling[1]
  if dim_y==2:
    tunneling_y *= 0.5
  if debug:
    tab_log_trans = np.zeros((dim_x-1-i0)//nrescale+1)
    tab_x = H.tab_delta[0]*nrescale*np.arange((dim_x-1-i0)//nrescale+1)
#  print('Disorder',H.disorder)
#  print(H.disorder.dtype,H.disorder.shape,H.disorder.flags)
  x = 0.0
  g1n = np.identity(dim_y)
  An = np.identity(dim_y)
  An_old = np.zeros((dim_y,dim_y))
  if use_ctypes:
    import ctypes
    import numpy.ctypeslib as ctl
    lyapounov_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/lyapounov.so")
    lyapounov_ctypes_lib.update_A_2d.argtypes = [ctypes.c_int, ctl.ndpointer(flags='C'), ctypes.c_double, ctypes.c_double, ctypes.c_double,ctypes.c_int, ctypes.c_int, ctl.ndpointer(flags='C'), ctl.ndpointer(flags='C')]
    lyapounov_ctypes_lib.update_A_2d.restype = None
  else:
    e_minus_H_local_over_tx = scipy.sparse.dia_array((dim_y,dim_y))
#    e_minus_H_local_over_tx.setdiag(0.0)
    non_diagonal_element = -tunneling_y*inv_tunneling_x
    e_minus_H_local_over_tx.setdiag(non_diagonal_element,-1)
    e_minus_H_local_over_tx.setdiag(non_diagonal_element,1)
    e_minus_H_local_over_tx.setdiag(non_diagonal_element,dim_y-1)
    e_minus_H_local_over_tx.setdiag(non_diagonal_element,1-dim_y)
#    print(e_minus_H_local_over_tx)
#    offsets = np.array([1-dim_y,-1,0,1,dim_y-1])
#    sub_diagonals = -tunneling_y*inv_tunneling_x*np.ones(dim_y)
#    data = np.array([sub_diagonals,sub_diagonals,sub_diagonals,sub_diagonals,sub_diagonals])
#    e_minus_H_local_over_tx = scipy.sparse.dia_array((data,offsets), shape=(dim_y,dim_y))
    e_minus_H_local_over_tx_linear_operator= scipy.sparse.linalg.aslinearoperator(e_minus_H_local_over_tx)
# Here starts the main loop for propagation along the x direction
  for i in range(((dim_x-1)//nrescale)*nrescale+1):
    start_scalar_time = timeit.default_timer()
    if use_ctypes:
      lyapounov_ctypes_lib.update_A_2d(dim_y, H.disorder, tunneling_x, tunneling_y, energy, nrescale, i, An, An_old)
    else:
      e_minus_H_local_over_tx.setdiag(inv_tunneling_x*(energy-H.disorder[i,:]))
      if i%nrescale==1:
        An_old += e_minus_H_local_over_tx
# The code in the previous line should be equivalent to the following 4 lines
#         for j in range(dim_y):
#           An_old[j,j] += inv_tunneling_x*(energy-H.disorder[i,j])
#           An_old[j,(j+1)%dim_y] -= tunneling_y*inv_tunneling_x
#           An_old[j,(j+dim_y-1)%dim_y] -= tunneling_y*inv_tunneling_x
      else:
        An_old += e_minus_H_local_over_tx_linear_operator.matmat(An)
# The code in the previous line should be equivalent to the following 4 lines
#        for j in range(dim_y):
#          An_old[j,0:dim_y] += inv_tunneling_x*((energy-H.disorder[i,j])*An[j,0:dim_y] - tunneling_y*(An[(j+1)%dim_y,0:dim_y] + An[(j+dim_y-1)%dim_y,0:dim_y]))
    An, An_old = An_old, -An
#      print('False ener i=',i,energy-H.disorder[i,:])
#      print('False Hamitonian i=',i,e_minus_H_local_over_tx)
#      print('False i=',i,An,An_old)
    timing.LYAPOUNOV_SCALAR_TIME+=(timeit.default_timer() - start_scalar_time)
    if i%nrescale==0:
      start_factorization_time = timeit.default_timer()
      b, piv, info = lapack.dgetrf(An)
      timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME+=(timeit.default_timer() - start_factorization_time)
      if i>=i0:
        start_solution_time = timeit.default_timer()
        g1n, info = lapack.dgetrs(b, piv, g1n, 1, 0)
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        one_over_small_b = 1.0/np.linalg.norm(g1n)
        g1n *= one_over_small_b
        if i>i0:
          x += math.log(one_over_small_b)
        if debug:
          tab_log_trans[(i-i0)//nrescale] = -2.0*x
        start_solution_time = timeit.default_timer()
        An_old = lapack.dgetrs(b, piv, An_old.T, 1, 0)[0].T
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        An = np.identity(dim_y)
  x /= H.tab_delta[0]*((dim_x-i0-1)//nrescale)*nrescale
  timing.LYAPOUNOV_MATRIX_FACTORIZATION_NOPS += 2*((dim_x-1)//nrescale)*dim_y**3/3
  timing.LYAPOUNOV_MATRIX_SOLUTION_NOPS += 4*((dim_x-1)//nrescale)*dim_y**3
  timing.LYAPOUNOV_SCALAR_NOPS += 4*dim_y*((dim_x-1)//nrescale) + 5*dim_y**2*((dim_x-1)//nrescale)*(nrescale-1) + dim_y**2*((dim_x-1)//nrescale)*nrescale
  if debug:
    return x, tab_x, tab_log_trans
  else:
    return x

def core_lyapounov_3d(H, energy, i0, nrescale, timing, use_ctypes, debug=False):
# Propagation is along the x direction
  dim_x = H.tab_dim[0]
  dim_y = H.tab_dim[1]
  dim_z = H.tab_dim[2]
  dim_trans = dim_y*dim_z
  tunneling_x = H.tab_tunneling[0]
  inv_tunneling_x = 1.0/tunneling_x
  tunneling_y = H.tab_tunneling[1]
  tunneling_z = H.tab_tunneling[2]
  if debug:
    tab_log_trans = np.zeros((dim_x-1-i0)//nrescale+1)
    tab_x = H.tab_delta[0]*nrescale*np.arange((dim_x-1-i0)//nrescale+1)
#  print('Disorder',H.disorder)
#  print(H.disorder.dtype,H.disorder.shape,H.disorder.flags)
  x = 0.0
  g1n = np.identity(dim_trans)
  An = np.identity(dim_trans)
  An_old = np.zeros((dim_trans,dim_trans))
  if use_ctypes:
    import ctypes
    import numpy.ctypeslib as ctl
    lyapounov_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/lyapounov.so")
    lyapounov_ctypes_lib.update_A_3d.argtypes = [ctypes.c_int, ctypes.c_int, ctl.ndpointer(flags='C'), ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctl.ndpointer(flags='C'), ctl.ndpointer(flags='C')]
    lyapounov_ctypes_lib.update_A_3d.restype = None
  else:
    e_minus_H_local_over_tx = scipy.sparse.dia_array((dim_trans,dim_trans))
#    e_minus_H_local_over_tx.setdiag(0.0)
    non_diagonal_element_y = -tunneling_y*inv_tunneling_x
    non_diagonal_element_z = -tunneling_z*inv_tunneling_x
    e_minus_H_local_over_tx.setdiag(non_diagonal_element_y,-dim_z)
    e_minus_H_local_over_tx.setdiag(non_diagonal_element_y,dim_z)
    e_minus_H_local_over_tx.setdiag(non_diagonal_element_y,-dim_z*(dim_y-1))
    e_minus_H_local_over_tx.setdiag(non_diagonal_element_y,dim_z*(dim_y-1))
    non_diagonal_vector = non_diagonal_element_z*np.ones(dim_z)
    non_diagonal_vector[dim_z-1] = 0.0
    non_diagonal_vector = np.tile(non_diagonal_vector,dim_y)
#    print('non_diagonal_vector_1',non_diagonal_vector)
    e_minus_H_local_over_tx.setdiag(non_diagonal_vector,1)
    e_minus_H_local_over_tx.setdiag(non_diagonal_vector,-1)
    non_diagonal_vector = np.zeros(dim_z)
    non_diagonal_vector[0] = non_diagonal_element_z
    non_diagonal_vector = np.tile(non_diagonal_vector,dim_y)
#    print('non_diagonal_vector_2',non_diagonal_vector)
    e_minus_H_local_over_tx.setdiag(non_diagonal_vector,dim_z-1)
    e_minus_H_local_over_tx.setdiag(non_diagonal_vector,1-dim_z)
#    print(e_minus_H_local_over_tx)
    e_minus_H_local_over_tx_linear_operator= scipy.sparse.linalg.aslinearoperator(e_minus_H_local_over_tx)
# Here starts the main loop for propagation along the x direction
  for i in range(((dim_x-1)//nrescale)*nrescale+1):
    start_scalar_time = timeit.default_timer()
    if use_ctypes:
      lyapounov_ctypes_lib.update_A_3d(dim_y,dim_z, H.disorder, tunneling_x, tunneling_y, tunneling_z, energy, nrescale, i, An, An_old)
    else:
      e_minus_H_local_over_tx.setdiag(inv_tunneling_x*(energy-H.disorder[i,:,:].reshape(dim_trans)))
      if i%nrescale==1:
        An_old += e_minus_H_local_over_tx
# The code in the previous line should be equivalent to the following 4 lines
#         for j in range(dim_y):
#           An_old[j,j] += inv_tunneling_x*(energy-H.disorder[i,j])
#           An_old[j,(j+1)%dim_y] -= tunneling_y*inv_tunneling_x
#           An_old[j,(j+dim_y-1)%dim_y] -= tunneling_y*inv_tunneling_x
      else:
        An_old += e_minus_H_local_over_tx_linear_operator.matmat(An)
# The code in the previous line should be equivalent to the following 4 lines
#        for j in range(dim_y):
#          An_old[j,0:dim_y] += inv_tunneling_x*((energy-H.disorder[i,j])*An[j,0:dim_y] - tunneling_y*(An[(j+1)%dim_y,0:dim_y] + An[(j+dim_y-1)%dim_y,0:dim_y]))
    An, An_old = An_old, -An
#      print('False ener i=',i,energy-H.disorder[i,:])
#      print('False Hamitonian i=',i,e_minus_H_local_over_tx)
#      print('False i=',i,An,An_old)
    timing.LYAPOUNOV_SCALAR_TIME+=(timeit.default_timer() - start_scalar_time)
    if i%nrescale==0:
      start_factorization_time = timeit.default_timer()
      b, piv, info = lapack.dgetrf(An)
      timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME+=(timeit.default_timer() - start_factorization_time)
      if i>=i0:
        start_solution_time = timeit.default_timer()
        g1n, info = lapack.dgetrs(b, piv, g1n, 1, 0)
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        one_over_small_b = 1.0/np.linalg.norm(g1n)
        g1n *= one_over_small_b
        if i>i0:
          x += math.log(one_over_small_b)
        if debug:
          tab_log_trans[(i-i0)//nrescale] = -2.0*x
        start_solution_time = timeit.default_timer()
        An_old = lapack.dgetrs(b, piv, An_old.T, 1, 0)[0].T
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        An = np.identity(dim_trans)
  x /= H.tab_delta[0]*((dim_x-i0-1)//nrescale)*nrescale
  timing.LYAPOUNOV_MATRIX_FACTORIZATION_NOPS += 2*((dim_x-1)//nrescale)*dim_trans**3/3
  timing.LYAPOUNOV_MATRIX_SOLUTION_NOPS += 4*((dim_x-1)//nrescale)*dim_trans**3
  timing.LYAPOUNOV_SCALAR_NOPS += 4*dim_trans*((dim_x-1)//nrescale) + 5*dim_trans**2*((dim_x-1)//nrescale)*(nrescale-1) + dim_trans**2*((dim_x-1)//nrescale)*nrescale
  if debug:
    return x, tab_x, tab_log_trans
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
  def __init__(self, energy, want_ctypes=True, i0=10, nrescale=10):
    self.energy = energy
    self.want_ctypes = want_ctypes
    self.use_ctypes = want_ctypes
    self.i0 = i0
    self.nrescale = nrescale
#    self.tab_energy = np.zeros(number_of_e_steps+1)
#    if number_of_e_steps==0:
#      self.e_step = 0.0
#    else:
#      self.e_step = (e_max - e_min)/number_of_e_steps
#    for i_e in range(number_of_e_steps+1):
#      e = e_min + self.e_step*i_e
#      self.tab_energy[i_e] = e
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
        import ctypes
        import numpy.ctypeslib as ctl
        lyapounov_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/lyapounov.so")
#        print(lyapounov_ctypes_lib)
        if H.dimension == 1:
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
        if H.dimension ==2:
          self.use_ctypes =hasattr(lyapounov_ctypes_lib,'update_A_2d')
          if not(self.use_ctypes):
            if H.seed == 1234:
              print("\nWarning, lyapounov C library found, but without routine update_A_2d, this uses the slow Python version!\n")
        if H.dimension ==3:
          self.use_ctypes =hasattr(lyapounov_ctypes_lib,'update_A_3d')
          if not(self.use_ctypes):
            if H.seed == 1234:
              print("\nWarning, lyapounov C library found, but without routine update_A_3d, this uses the slow Python version!\n")
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

    """
    loop_step=16
    number_of_energies = self.tab_energy.size
    tab_gamma = np.zeros(number_of_energies)
  #  tab_h = np.zeros(number_of_energies)
    for i_energy in range(number_of_energies):
      if H.dimension == 1:
        if self.use_ctypes:
          if H.disorder_type=='nice':
            tab_gamma[i_energy] = lyapounov_ctypes_lib.core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder, H.b, H.non_diagonal_disorder, self.tab_energy[i_energy], tunneling)
          else:
            tab_gamma[i_energy] = lyapounov_ctypes_lib.core_lyapounov(dim_x, loop_step, H.disorder, self.tab_energy[i_energy], inv_tunneling)
        else:
          if H.disorder_type=='nice':
            tab_gamma[i_energy] = core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder, H.b, H.non_diagonal_disorder, self.tab_energy[i_energy], tunneling)
      if H.dimension == 2:
        if debug:
          tab_gamma[i_energy], tab_x, tab_log_trans = core_lyapounov_2d(H, self.tab_energy[i_energy], self.i0, self.nrescale, timing, self.use_ctypes, debug=True)
        else:
          tab_gamma[i_energy] = core_lyapounov_2d(H, self.tab_energy[i_energy], self.i0, self.nrescale, timing, self.use_ctypes)
      if H.dimension == 3:
        if debug:
          tab_gamma[i_energy], tab_x, tab_log_trans = core_lyapounov_3d(H, self.tab_energy[i_energy], self.i0, self.nrescale, timing, self.use_ctypes, debug=True)
        else:
          tab_gamma[i_energy] = core_lyapounov_3d(H, self.tab_energy[i_energy], self.i0, self.nrescale, timing, self.use_ctypes)
    """

    loop_step=16
    if H.dimension == 1:
      if self.use_ctypes:
        if H.disorder_type=='nice':
          gamma= lyapounov_ctypes_lib.core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder, H.b, H.non_diagonal_disorder, self.energy, tunneling)
        else:
          gamma = lyapounov_ctypes_lib.core_lyapounov(dim_x, loop_step, H.disorder, self.energy, inv_tunneling)
      else:
        if H.disorder_type=='nice':
          gamma = core_lyapounov_non_diagonal_disorder(dim_x, loop_step, H.disorder, H.b, H.non_diagonal_disorder, self.energy, tunneling)
    if H.dimension == 2:
      if debug:
        gamma, tab_x, tab_log_trans = core_lyapounov_2d(H, self.energy, self.i0, self.nrescale, timing, self.use_ctypes, debug=True)
      else:
        gamma = core_lyapounov_2d(H, self.energy, self.i0, self.nrescale, timing, self.use_ctypes)
    if H.dimension == 3:
      if debug:
        gamma, tab_x, tab_log_trans = core_lyapounov_3d(H, self.energy, self.i0, self.nrescale, timing, self.use_ctypes, debug=True)
      else:
        gamma = core_lyapounov_3d(H, self.energy, self.i0, self.nrescale, timing, self.use_ctypes)

#    print(2.0*new_core_lyapounov(H, 0.0)/(dim_x*H.tab_delta[0]))


    timing.LYAPOUNOV_TIME += timeit.default_timer() - start_lyapounov_time
    if H.dimension==1:
      if H.disorder_type=='nice':
        timing.LYAPOUNOV_NOPS += 10*dim_x
      else:
        timing.LYAPOUNOV_NOPS += 5*dim_x
    if H.dimension==2:
      timing.LYAPOUNOV_NOPS += timing.LYAPOUNOV_MATRIX_FACTORIZATION_NOPS+timing.LYAPOUNOV_MATRIX_SOLUTION_NOPS+timing.LYAPOUNOV_SCALAR_NOPS
  #  lyapounov = gamma/(dim_x*H.delta_x)
  #  integrated_dos = h/(dim_x*H.delta_x)
  #  return (lyapounov,integrated_dos)
  # The Lyapounov is here computed for the intensity (halve it for wavefunction), hence the multiplicative factor 2
  # Change on May, 3, 2023, so that it is now for the wavefucntion
    if H.dimension==1:
      return 0.5*gamma/(dim_x*H.tab_delta[0])
    if H.dimension==2 or H.dimension==3:
      if debug:
        return gamma, tab_x, tab_log_trans
      else:
        return gamma

"""
Old stuff
def core_lyapounov_2d(dim_x, dim_y, disorder, energy, nrescale, i0, timing):
  use_c = True
  g1n = np.identity(dim_y)
  Bn = np.identity(dim_y)
  Bn_old = np.zeros((dim_y,dim_y))
  if use_c:
    lyapounov_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/lyapounov.so")
    lyapounov_ctypes_lib.update_B_c.argtypes = [ctypes.c_int, ctypes.c_int, ctl.ndpointer(np.float64), ctypes.c_double, ctypes.c_int, ctypes.c_int, ctl.ndpointer(np.float64), ctl.ndpointer(np.float64)]
    lyapounov_ctypes_lib.update_B_c.restype = None
  x = 0.0
# Here starts the main loop for propagation along the x direction
  for i in range(((dim_x-1)//nrescale)*nrescale+1):
    start_scalar_time = timeit.default_timer()
    print(use_c)
    if use_c:
      print('toto')
      lyapounov_ctypes_lib.update_B_c(dim_x, dim_y, disorder, energy, nrescale, i, Bn, Bn_old)
    else:
      if i%nrescale==1:
        for j in range(dim_y):
          Bn_old[j,j] += (energy-disorder[i,j])
          Bn_old[j,(j+1)%dim_y] -= 1.0
          Bn_old[j,(j+dim_y-1)%dim_y] -= 1.0
      else:
        for j in range(dim_y):
  #        Bn_old[j,1:dim_y-1] += (energy-disorder[i,1:dim_y-1])*Bn[j,1:dim_y-1] - Bn[j,2:dim_y] - Bn[j,0:dim_y-2]
  #        Bn_old[j,0] += (energy-disorder[i,0])*Bn[j,0] - Bn[j,dim_y-1] - Bn[j,1]
  #        Bn_old[j,dim_y-1] += (energy-disorder[i,dim_y-1])*Bn[j,dim_y-1] - Bn[j,dim_y-2] - Bn[j,0]
          Bn_old[0:dim_y,j] += (energy-disorder[i,j])*Bn[0:dim_y,j] - Bn[0:dim_y,(j+1)%dim_y] - Bn[0:dim_y,(j+dim_y-1)%dim_y]
    Bn, Bn_old = Bn_old, -Bn
#    print('i=',i,'Bn=',Bn,'Bn_old=',Bn_old)
    timing.LYAPOUNOV_SCALAR_TIME+=(timeit.default_timer() - start_scalar_time)
    if i%nrescale==0:
#        if i<=2:
#          print(i,'Bn',Bn)
#          print(i,'Bn_old',Bn_old)
      start_factorization_time = timeit.default_timer()
      b, piv, info = lapack.dgetrf(Bn)
#      print("i=",i,"piv=",piv,"b=",b)
      timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME+=(timeit.default_timer() - start_factorization_time)
      if i>=i0:
        start_solution_time = timeit.default_timer()
        g1n, info = lapack.dgetrs(b, piv, g1n)
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        small_b = np.linalg.norm(g1n)
#        print('i=',i,'g1n=',g1n)
#        print('i=',i,'small_b=',small_b)
        g1n *= 1.0/small_b
#        print(i,g1n,small_b)
      if i>i0:
        x -= math.log(small_b)
#        if i<=2:
#          print(i,'g1n',g1n)
      start_solution_time = timeit.default_timer()
      Bn_old, info = lapack.dgetrs(b, piv, Bn_old)
      timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
      Bn = np.identity(dim_y)
  return x

use_c = False
use_mkl = False
if use_mkl:
  mkl = ctypes.cdll.LoadLibrary("libmkl_rt.so")
  POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)
  POINTER_INT = ctypes.POINTER(ctypes.c_int)
  mkl.LAPACKE_dgetrf.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE, ctypes.c_int, POINTER_INT]
  mkl.LAPACKE_dgetrf.restype = ctypes.c_int
  mkl.LAPACKE_dgetrs.argtypes = [ctypes.c_int, ctypes.c_wchar, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE, ctypes.c_int, POINTER_INT, POINTER_DOUBLE, ctypes.c_int]
  mkl.LAPACKE_dgetrs.restype = ctypes.c_int
  Order = 101  # 101 for row-major, 102 for column major data structures
  char = 'n'
if use_c:
  lyapounov_ctypes_lib=ctypes.CDLL(anderson.__path__[0]+"/ctypes/lyapounov.so")
  POINTER_DOUBLE = ctypes.POINTER(ctypes.c_double)
  lyapounov_ctypes_lib.update_B_c.argtypes = [ctypes.c_int, ctypes.c_int, POINTER_DOUBLE, ctypes.c_double, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE, POINTER_DOUBLE]
  lyapounov_ctypes_lib.update_B_c.restype = None
  lyapounov_ctypes_lib.update_B_c_exp.argtypes = [ctypes.c_int, ctypes.c_int, POINTER_DOUBLE, ctypes.c_double, ctypes.c_int, ctypes.c_int, POINTER_DOUBLE, POINTER_DOUBLE]
  lyapounov_ctypes_lib.update_B_c_exp.restype = None
  x = 0.0
  g1n = np.identity(dim_y).reshape(dim_y*dim_y)
  Bn = np.identity(dim_y).reshape(dim_y*dim_y)
  Bn_old = np.zeros(dim_y*dim_y)
  if use_mkl:
     ipiv = np.zeros(dim_y, dtype=int)
# Here starts the main loop for propagation along the x direction
  for i in range(((dim_x-1)//nrescale)*nrescale+1):
    start_scalar_time = timeit.default_timer()
#      print('i=',i,'Bn_old before =',Bn_old)
#      print('i=',i,'Bn before =',Bn)
    lyapounov_ctypes_lib.update_B_c_exp(dim_x, dim_y, H.disorder.ctypes.data_as(POINTER_DOUBLE), energy, nrescale, i, Bn.ctypes.data_as(POINTER_DOUBLE), Bn_old.ctypes.data_as(POINTER_DOUBLE))
    Bn, Bn_old = Bn_old, -Bn
    timing.LYAPOUNOV_SCALAR_TIME+=(timeit.default_timer() - start_scalar_time)
    if i%nrescale==0:
 #        if i<=2:
 #          print(i,'Bn',Bn)
 #          print(i,'Bn_old',Bn_old)
      start_factorization_time = timeit.default_timer()
      if use_mkl:
        info = mkl.LAPACKE_dgetrf(Order, dim_y, dim_y, Bn.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT))
      else:
        b, piv, info = lapack.dgetrf(Bn.reshape(dim_y,dim_y))
#          print('Bn after:', Bn)
#          print('info=',info)
      timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME+=(timeit.default_timer() - start_factorization_time)
      if i>=i0:
        start_solution_time = timeit.default_timer()
        if use_mkl:
          info = mkl.LAPACKE_dgetrs(Order, char, dim_y, dim_y, Bn.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT), g1n.ctypes.data_as(POINTER_DOUBLE), dim_y)
        else:
          g1n = lapack.dgetrs(b, piv, g1n.reshape(dim_y,dim_y), 1)[0].reshape(-1)
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        small_b = np.linalg.norm(g1n)
        g1n *= 1.0/small_b
 #        print(i,g1n,small_b)
      if i>i0:
        x -= math.log(small_b)
 #        if i<=2:
 #          print(i,'g1n',g1n)
      if debug:
        tab_log_trans[(i-i0)//nrescale] = x
      start_solution_time = timeit.default_timer()
      if use_mkl:
        info = mkl.LAPACKE_dgetrs(Order, char, dim_y, dim_y, Bn.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT), Bn_old.ctypes.data_as(POINTER_DOUBLE), dim_y)
      else:
        Bn_old = lapack.dgetrs(b, piv, Bn_old.reshape(dim_y,dim_y), 1)[0].reshape(-1)
      timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
      Bn = np.identity(dim_y).reshape(dim_y*dim_y)
else:
  use_transpose = True
  if use_transpose:
    x = 0.0
    sparse_storage = True
    g1n = np.identity(dim_y)
    An = np.identity(dim_y)
    An_old = np.zeros((dim_y,dim_y))
    if use_mkl:
      ipiv = np.zeros(dim_y, dtype=int)
    if sparse_storage:
      offsets = np.array([1-dim_y,-1,0,1,dim_y-1])
      ex = -np.ones(dim_y)
      data = np.array([ex,ex,ex,ex,ex])
      a = sparse.dia_array((data,offsets), shape=(dim_y,dim_y))
      zz = aslinearoperator(a)
# Here starts the main loop for propagation along the x direction
    for i in range(((dim_x-1)//nrescale)*nrescale+1):
      start_scalar_time = timeit.default_timer()
      if i%nrescale==1:
        if sparse_storage:
          a.setdiag(energy-H.disorder[i,:])
          An_old += a
        else:
          for j in range(dim_y):
#            Bn_old[j,j] += (energy-H.disorder[i,j])
#            Bn_old[j,(j+1)%dim_y] -= 1.0
#            Bn_old[j,(j+dim_y-1)%dim_y] -= 1.0
            An_old[j,j] += (energy-H.disorder[i,j])
            An_old[j,(j+1)%dim_y] -= 1.0
            An_old[j,(j+dim_y-1)%dim_y] -= 1.0
#          print(Bn_old)
      else:
        if sparse_storage:
          a.setdiag(energy-H.disorder[i,:])
#            zz = aslinearoperator(a)
          An_old += (zz.matmat(An))
        else:
          for j in range(dim_y):
#              print(Bn_old[k*dim_y+j])
#              Bn_old[k*dim_y+j] = Bn_old[k*dim_y+j]+(energy-H.disorder[i,j])*Bn[k*dim_y+j] - Bn[k*dim_y+(j+1)%dim_y] - Bn[k*dim_y+(j+dim_y-1)%dim_y]
#              Bn_old[k,j] = Bn_old[k,j]+(energy-H.disorder[i,j])*Bn[k,j] - Bn[k,(j+1)%dim_y] - Bn[k,(j+dim_y-1)%dim_y]
#                Bn_old[0:dim_y,j] += (energy-H.disorder[i,j])*Bn[0:dim_y,j] - Bn[0:dim_y,(j+1)%dim_y] - Bn[0:dim_y,(j+dim_y-1)%dim_y]
            An_old[j,0:dim_y] += (energy-H.disorder[i,j])*An[j,0:dim_y] - An[(j+1)%dim_y,0:dim_y] - An[(j+dim_y-1)%dim_y,0:dim_y]
      An, An_old = An_old, -An
      timing.LYAPOUNOV_SCALAR_TIME+=(timeit.default_timer() - start_scalar_time)
      if i%nrescale==0:
#        if i<=2:
#          print(i,'Bn',Bn)
#          print(i,'Bn_old',Bn_old)
        start_factorization_time = timeit.default_timer()
        if use_mkl:
          info = mkl.LAPACKE_dgetrf(Order, dim_y, dim_y, An.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT))
        else:
          b, piv, info = lapack.dgetrf(An)
        timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME+=(timeit.default_timer() - start_factorization_time)
        if i>=i0:
          start_solution_time = timeit.default_timer()
          if use_mkl:
            mkl.LAPACKE_dgetrs(Order, 'n', dim_y, dim_y, An.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT), g1n.ctypes.data_as(POINTER_DOUBLE), dim_y)
          else:
#                print('g1n before:',g1n)
            g1n = lapack.dgetrs(b, piv, g1n, 1, 0)[0]
#                print('g1n after:',g1n)
          timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
          small_b = np.linalg.norm(g1n)
          g1n *= 1.0/small_b
#        print(i,g1n,small_b)
        if i>i0:
          x -= math.log(small_b)
#        if i<=2:
#          print(i,'g1n',g1n)
        if debug:
          tab_log_trans[(i-i0)//nrescale] = x
        start_solution_time = timeit.default_timer()
        if use_mkl:
          mkl.LAPACKE_dgetrs(Order, 'n', dim_y, dim_y, An.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT), (An_old.T).ctypes.data_as(POINTER_DOUBLE), dim_y)
        else:
          An_old = lapack.dgetrs(b, piv, An_old.T, 1, 0)[0].T
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        An = np.identity(dim_y)
  else:
    x = 0.0
    sparse_storage = True
    g1n = np.identity(dim_y)
    Bn = np.identity(dim_y)
    Bn_old = np.zeros((dim_y,dim_y))
    if use_mkl:
       ipiv = np.zeros(dim_y, dtype=int)
    if sparse_storage:
      offsets = np.array([1-dim_y,-1,0,1,dim_y-1])
      ex = -np.ones(dim_y)
      data = np.array([ex,ex,ex,ex,ex])
      a = sparse.dia_array((data,offsets), shape=(dim_y,dim_y))
      zz = aslinearoperator(a)
# Here starts the main loop for propagation along the x direction
    for i in range(((dim_x-1)//nrescale)*nrescale+1):
      start_scalar_time = timeit.default_timer()
      if i%nrescale==1:
        if sparse_storage:
          a.setdiag(energy-H.disorder[i,:])
          Bn_old += a
        else:
          for j in range(dim_y):
#            Bn_old[j,j] += (energy-H.disorder[i,j])
#            Bn_old[j,(j+1)%dim_y] -= 1.0
#            Bn_old[j,(j+dim_y-1)%dim_y] -= 1.0
            Bn_old[j,j] += (energy-H.disorder[i,j])
            Bn_old[j,(j+1)%dim_y] -= 1.0
            Bn_old[j,(j+dim_y-1)%dim_y] -= 1.0
#          print(Bn_old)
      else:
        if sparse_storage:
          a.setdiag(energy-H.disorder[i,:])
#            zz = aslinearoperator(a)
          Bn_old += (zz.matmat(Bn.T)).T
        else:
          for j in range(dim_y):
#              print(Bn_old[k*dim_y+j])
#              Bn_old[k*dim_y+j] = Bn_old[k*dim_y+j]+(energy-H.disorder[i,j])*Bn[k*dim_y+j] - Bn[k*dim_y+(j+1)%dim_y] - Bn[k*dim_y+(j+dim_y-1)%dim_y]
#              Bn_old[k,j] = Bn_old[k,j]+(energy-H.disorder[i,j])*Bn[k,j] - Bn[k,(j+1)%dim_y] - Bn[k,(j+dim_y-1)%dim_y]
            Bn_old[0:dim_y,j] += (energy-H.disorder[i,j])*Bn[0:dim_y,j] - Bn[0:dim_y,(j+1)%dim_y] - Bn[0:dim_y,(j+dim_y-1)%dim_y]
 #        Bn_old[j,0:dim_y] = (energy-H.disorder[i,j])*Bn[j,0:dim_y] - Bn[(j+1)%dim_y,0:dim_y] - Bn[(j+dim_y-1)%dim_y,0:dim_y] - Bn_old[j,0:dim_y]
      Bn, Bn_old = Bn_old, -Bn
      timing.LYAPOUNOV_SCALAR_TIME+=(timeit.default_timer() - start_scalar_time)
      if i%nrescale==0:
#        if i<=2:
#          print(i,'Bn',Bn)
#          print(i,'Bn_old',Bn_old)
        start_factorization_time = timeit.default_timer()
        if use_mkl:
          info = mkl.LAPACKE_dgetrf(Order, dim_y, dim_y, Bn.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT))
        else:
          b, piv, info = lapack.dgetrf(Bn)
        timing.LYAPOUNOV_MATRIX_FACTORIZATION_TIME+=(timeit.default_timer() - start_factorization_time)
        if i>=i0:
          start_solution_time = timeit.default_timer()
          if use_mkl:
            mkl.LAPACKE_dgetrs(Order, char, dim_y, dim_y, Bn.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT), g1n.ctypes.data_as(POINTER_DOUBLE), dim_y)
          else:
            g1n,_ = lapack.dgetrs(b, piv, g1n)
          timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
          small_b = np.linalg.norm(g1n)
          g1n *= 1.0/small_b
#        print(i,g1n,small_b)
        if i>i0:
          x -= math.log(small_b)
#        if i<=2:
#          print(i,'g1n',g1n)
        if debug:
          tab_log_trans[(i-i0)//nrescale] = x
        start_solution_time = timeit.default_timer()
        if use_mkl:
          mkl.LAPACKE_dgetrs(Order, char, dim_y, dim_y, Bn.ctypes.data_as(POINTER_DOUBLE), dim_y, ipiv.ctypes.data_as(POINTER_INT), Bn_old.ctypes.data_as(POINTER_DOUBLE), dim_y)
        else:
          Bn_old,_ = lapack.dgetrs(b, piv, Bn_old)
        timing.LYAPOUNOV_MATRIX_SOLUTION_TIME+=(timeit.default_timer() - start_solution_time)
        Bn = np.identity(dim_y)
"""
