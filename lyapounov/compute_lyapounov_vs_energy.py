#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2020 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# compute_spectral_function.py
# Author: Dominique Delande
# Release date: April, 27, 2020
# License: GPL2 or later
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

"""

import os
import time
import numpy as np
import getpass
import sys
import timeit
import configparser
sys.path.append('../')
sys.path.append('/users/champ/delande/git/and-python/')
import anderson

#import matplotlib.pyplot as plt

if __name__ == "__main__":
  environment_string='Script ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +'Name of python script: {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Started on: {}'.format(time.asctime())+'\n'
  try:
# First try to detect if the python script is launched by mpiexec/mpirun
# It can be done by looking at an environment variable
# Unfortunaltely, this variable depends on the MPI implementation
# For MPICH and IntelMPI, MPI_LOCALNRANKS can be checked for existence
#   os.environ['MPI_LOCALNRANKS']
# For OpenMPI, it is OMPI_COMM_WORLD_SIZE
#   os.environ['OMPI_COMM_WORLD_SIZE']
# In any case, when importing the module mpi4py, the MPI implementation for which
# the module was created is unknown. Thus, no portable way...
# The following line is for OpenMPI
    os.environ['OMPI_COMM_WORLD_SIZE']
# If no KeyError raised, the script has been launched by MPI,
# I must thus import the mpi4py module
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mpi_version = True
    environment_string += 'MPI version ran on '+str(nprocs)+' processes\n\n'
  except KeyError:
# Not launched by MPI, use sequential code
    mpi_version = False
    nprocs = 1
    rank = 0
    environment_string += 'Single processor version\n\n'
  except ImportError:
# Launched by MPI, but no mpi4py module available. Abort the calculation.
    exit('mpi4py module not available! I stop!')

  if rank==0:
    initial_time=time.asctime()
    hostname = os.uname()[1].split('.')[0]
    print("Python script started on: {}".format(initial_time))
    print("{:>24}: {}".format('from',hostname))
    print("Name of python script: {}".format(os.path.abspath( __file__ )))

    config = configparser.ConfigParser()
    config.read('params.dat')

    Averaging = config['Averaging']
    n_config = Averaging.getint('n_config',1)
    n_config = (n_config+nprocs-1)//nprocs
    print("Total number of disorder realizations: {}".format(n_config*nprocs))
    print("Number of processes: {}".format(nprocs))

    System = config['System']
    system_size = System.getfloat('size')
    delta_x = System.getfloat('delta_x')
    boundary_condition = System.get('boundary_condition','periodic')

    Disorder = config['Disorder']
    disorder_type = Disorder.get('type','anderson gaussian')
    correlation_length = Disorder.getfloat('sigma',0.0)
    V0 = Disorder.getfloat('V0',0.0)
    disorder_strength = V0
    use_mkl_random = Disorder.getboolean('use_mkl_random',True)

    #Nonlinearity = config['Nonlinearity']
    #interaction_strength = Nonlinearity.getfloat('g',0.0)

    Lyapounov = config['Lyapounov']
    e_min = Lyapounov.getfloat('e_min',0.0)
    e_max = Lyapounov.getfloat('e_max',0.0)
    number_of_e_steps = Lyapounov.getint('number_of_e_steps',0)
    e_histogram = Lyapounov.getfloat('e_histogram',0.0)
    lyapounov_min = Lyapounov.getfloat('lyapounov_min',0.0)
    lyapounov_max = Lyapounov.getfloat('lyapounov_max',0.0)
    number_of_bins = Lyapounov.getint('number_of_bins',0)

  else:
    n_config = None
    system_size = None
    delta_x = None
    boundary_condition = None
    disorder_type = None
    correlation_length = None
    disorder_strength = None
    use_mkl_random = None
    e_min = None
    e_max = None
    number_of_e_steps = None
    e_histogram = None
    lyapounov_min = None
    lyapounov_max = None
    number_of_bins = None

  if mpi_version:
    n_config, system_size, delta_x,boundary_condition  = comm.bcast((n_config, system_size,delta_x,boundary_condition ))
    disorder_type, correlation_length, disorder_strength, use_mkl_random = comm.bcast((disorder_type, correlation_length, disorder_strength, use_mkl_random))
    e_min, e_max, number_of_e_steps, e_histogram, lyapounov_min, lyapounov_max, number_of_bins = comm.bcast((e_min, e_max, number_of_e_steps, e_histogram, lyapounov_min, lyapounov_max, number_of_bins))

  t1=time.perf_counter()
  timing=anderson.Timing()

  # Number of sites
  dim_x = int(system_size/delta_x+0.5)
  # Renormalize delta_x so that the system size is exactly what is wanted and split in an integer number of sites
  delta_x = system_size/dim_x
  try:
    import mkl
    mkl.set_num_threads(1)
    os.environ["MKL_NUM_THREADS"] = "1"
  except:
    pass

  assert boundary_condition in ['periodic','open'], "Boundary condition must be either 'periodic' or 'open'"
  # Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, use_mkl_random=use_mkl_random)

#  print(e_min,e_max,number_of_e_steps)

  lyapounov = anderson.lyapounov.Lyapounov(e_min,e_max,number_of_e_steps)
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,lyapounov=lyapounov)
#  tab_mean_integrated_dos = np.zeros(number_of_k_steps+1)
#  tab_std_integrated_dos = np.zeros(number_of_k_steps+1)
#  tab_mean_lyapounov = np.zeros(number_of_e_steps+1)
#  tab_std_lyapounov = np.zeros(number_of_e_steps+1)
  tab_lyapounov = np.zeros(number_of_e_steps+1)
  tab_global_lyapounov = np.zeros((2,number_of_e_steps+1))
#  tab_used_time = np.zeros(n_config)
#  tab_nops = np.zeros(n_config)
#  tab_integrated_dos= np.zeros((n_config,number_of_k_steps+1))


# The loop over configurations can be parallelized using multiprocessing.Pool
# But this is extremely inefficient especially for memory management
# Maybe is can be improved using e.g. starmap or starmap_async
# Currently disabled
#
#  number_of_cores = multiprocessing.cpu_count()
#  number_of_cores = 4
#  single_core = (number_of_cores==1)
#  if single_core:
#
# No parallelization, a simple loop makes debugging MUCH MUCH easier
# Here starts the loop over disorder configurations
  for i in range(n_config):
    tab_lyapounov, used_time, number_of_ops = lyapounov.compute_lyapounov(i+rank*n_config, H)
    tab_global_lyapounov[0] += tab_lyapounov
    tab_global_lyapounov[1] += tab_lyapounov**2
    timing.LYAPOUNOV_TIME+=used_time
    timing.LYAPOUNOV_NOPS+=number_of_ops
  if mpi_version:
    start_mpi_time = timeit.default_timer()
    tab_global_lyapounov_glob = np.empty_like(tab_global_lyapounov)
    comm.Reduce(tab_global_lyapounov,tab_global_lyapounov_glob)
    tab_global_lyapounov = np.copy(tab_global_lyapounov_glob)
    timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
  t2 = time.perf_counter()
  timing.TOTAL_TIME = t2-t1
  if mpi_version:
    timing.mpi_merge(comm)
#  else:
# The parallel case
#    pool = multiprocessing.Pool(number_of_cores)
# apply_sync returns immediately
#    results = [pool.apply_async(anderson.compute_lyapounov,(i, H, tab_energy)) for i in range(n_config)]
#    pool.close()
# Gather results using the get method (which blocks until the calculation is actually done)
#    for i in range(n_config):
#      tab_lyapounov[i], tab_used_time[i], tab_nops[i] = results[i].get()
#    timing.LYAPOUNOV_TIME=np.sum(tab_used_time)
#    timing.LYAPOUNOV_NOPS=np.sum(tab_nops)
#
# Compute mean value and standard deviation
  if rank==0:
    tab_global_lyapounov /= n_config*nprocs
    if n_config*nprocs>1:
      tab_global_lyapounov[1] = np.sqrt(np.abs(tab_global_lyapounov[1]-tab_global_lyapounov[0]**2)/(n_config*nprocs-1))
    else:
      tab_global_lyapounov[1] = 0.0
    anderson.io.output_density('lyapounov_vs_energy.dat',lyapounov.tab_energy,tab_global_lyapounov,header_string,print_type='lyapounov')

    """

# Other code to compute an histogram of the lyapounov exponent at a given energy
  lyapounov = anderson.lyapounov.Lyapounov(e_histogram,e_histogram,0)
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,lyapounov=lyapounov)
  tab_lyapounov = np.zeros(n_config)
  for i in range(n_config):
    tab_lyapounov[i], used_time, number_of_ops = lyapounov.compute_lyapounov(i, H)
    timing.LYAPOUNOV_TIME+=used_time
    timing.LYAPOUNOV_NOPS+=number_of_ops
  tab_histogram, bin_edges = np.histogram(tab_lyapounov, bins=number_of_bins, range=(lyapounov_min,lyapounov_max), density=True)
  anderson.io.output_density('histogram_lyapounov.dat',bin_edges[1:],tab_histogram,header_string,print_type='histogram_lyapounov')
    """


    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    print("Lyapounov time       = {0:.3f}".format(timing.LYAPOUNOV_TIME))
    print("Number of ops        = {0:.4e}".format(timing.LYAPOUNOV_NOPS))
    if mpi_version:
      print("MPI time             = {0:.3f}".format(timing.MPI_TIME))
    print("Total_time           = {0:.3f}".format(timing.TOTAL_TIME))
