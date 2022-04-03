#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

"""

import os
import sys
import math
import numpy as np
import time
from . import geometry, diag, io, lyapounov, propagation, hamiltonian, wavefunction, measurement, timing

__all__ = ["diag","io","lyapounov","propagation","geometry","hamiltonian","wavefunction","measurement","timing"]



#def expectation_value_local_operator(wfc, local_operator):
#  density = np.abs(wfc)**2
#  norm = np.sum(density)
#  x = np.sum(density*local_operator)
#  return x/norm

def compute_correlation(x,y,shift_center=False):
  z = np.fft.ifftn(np.fft.fftn(x)*np.conj(np.fft.fftn(y)))/x.size
  if shift_center:
    return np.fft.fftshift(z)
  else:
    return z


def determine_unique_postfix(fn):
  if not os.path.exists(fn):
    return ''
  path, name = os.path.split(fn)
  name, ext = os.path.splitext(name)
  make_fn = lambda i: os.path.join(path, '%s_%d%s' % (name, i, ext))
  i = 0
  while True:
    i = i+1
    uni_fn = make_fn(i)
    if not os.path.exists(uni_fn):
      return '_'+str(i)

def determine_if_launched_by_mpi():
# First try to detect if the python script is launched by mpiexec/mpirun
# It can be done by looking at an environment variable
# Unfortunaltely, this variable depends on the MPI implementation
# For MPICH and IntelMPI, MPI_LOCALNRANKS can be checked for existence
# For OpenMPI, it is OMPI_COMM_WORLD_SIZE
# In any case, when importing the module mpi4py, the MPI implementation for which
# the module was created is unknown. Thus, no portable way...
  if "MPI_LOCALNRANKS" in os.environ or "OMPI_COMM_WORLD_SIZE" in os.environ:
# The script has been launched by MPI,
# I must thus import the mpi4py module
    try: 
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      nprocs = comm.Get_size()
      mpi_version = True
      if "MPI_LOCALNRANKS" in os.environ:
        mpi_string = 'MPI version (MPICH) ran on '+str(nprocs)+' processes\n'
      if "OMPI_COMM_WORLD_SIZE" in os.environ:
        mpi_string = 'MPI version (OpenMPI) ran on '+str(nprocs)+' processes\n'        
    except ImportError:
# Launched by MPI, but no mpi4py module available. Abort the calculation.
      exit('mpi4py module not available! I stop!')      
  else:
# Not launched by MPI, use sequential code
    mpi_version = False
    comm = None
    nprocs = 1
    rank = 0
    mpi_string = 'Single processor version\n'
  mpi_string += '\nCalculation started on: {}'.format(time.asctime())
#  print("inside",mpi_string)

# In addition, ensure only one thread is used in MKL calculations
# Noop if no MKL library available
  try:
    import mkl
    mkl.set_num_threads(1)
    os.environ["MKL_NUM_THREADS"] = "1"
  except:
    pass
  return (mpi_version,comm,nprocs,rank,mpi_string)

