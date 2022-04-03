#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

"""
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
# compute_IPR.py
# Author: Dominique Delande
# Release date: April, 27, 2020
# License: GPL2 or later

import os
import time
import math
import numpy as np
import getpass
import configparser
import timeit
import sys
sys.path.append('../')
sys.path.append('/users/champ/delande/git/and-python/')
import anderson




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

  if (rank==0):
    initial_time=time.asctime()
    hostname = os.uname()[1].split('.')[0]
    print("Python script started on: {}".format(initial_time))
    print("{:>24}: {}".format('from',hostname))
    print("Name of python script: {}".format(os.path.abspath( __file__ )))
#    print("Number of available threads: {}".format(multiprocessing.cpu_count()))
#    print("Number of disorder realizations: {}".format(n_config))

    timing=anderson.Timing()
    t1=time.perf_counter()

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

    Diagonalization = config['Diagonalization']
    diagonalization_method = Diagonalization.get('method','sparse')
    targeted_energy = Diagonalization.getfloat('targeted_energy')
    IPR_min = Diagonalization.getfloat('IPR_min',0.0)
    IPR_max = Diagonalization.getfloat('IPR_max')
    number_of_bins = Diagonalization.getint('number_of_bins')
  else:
    n_config = None
    system_size = None
    delta_x = None
    boundary_condition = None
    disorder_type = None
    correlation_length = None
    disorder_strength = None
    use_mkl_random = None
    diagonalization_method = None
    targeted_energy = None
 #  n_config = comm.bcast(n_config,root=0)

  if mpi_version:
    n_config, system_size, delta_x,boundary_condition  = comm.bcast((n_config, system_size,delta_x,boundary_condition ))
    disorder_type, correlation_length, use_mkl_random, disorder_strength = comm.bcast((disorder_type, correlation_length, use_mkl_random, disorder_strength))
    diagonalization_method, targeted_energy = comm.bcast((diagonalization_method, targeted_energy))

  timing=anderson.Timing()
  t1=time.perf_counter()

#  delta_x = comm.bcast(delta_x)
#  print(rank,n_config,system_size,delta_x)
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
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, use_mkl_random=use_mkl_random, interaction=0.0)
  diagonalization = anderson.diag.Diagonalization(targeted_energy,diagonalization_method)

  #comm.Bcast(H)
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,diagonalization=diagonalization)
  tab_r = np.zeros(H.dim_x-2)
  tab_energy = np.zeros(H.dim_x-2)
  emin=0.0
  emax=2.0
  nsteps=50
  estep = (emax-emin)/nsteps
  tab_num = np.zeros(nsteps,dtype=int)
  tab_hist_r = np.zeros(nsteps)
  tab_middle_energy = np.arange(start=emin,stop=emax,step=estep)+0.5*estep
  # Here starts the loop over disorder configurations
  for i in range(n_config):
    tab_energy, tab_r = diagonalization.compute_rbar(i+rank*n_config, H)
# accumulate r values in an energy-dependent array
    for j in range(H.dim_x-2):
      k = int((tab_energy[j]-emin)/estep)
      k = max(k,0)
      k = min(k,nsteps-1)
      tab_num[k]+=1
      tab_hist_r[k]+=tab_r[j]
  for k in range(nsteps):
    tab_hist_r[k]/=tab_num[k]

#    print(IPR)
  #  pool.apply_async(gpe_evolution, args)
  #  print(str(i), file=final_pf)
#  anderson.io.output_density('IPR'+str(rank)+'.dat',tab_IPR,tab_energy,header_string,print_type='IPR')

  if mpi_version:
    start_mpi_time = timeit.default_timer()
#    tab_energy_glob = np.zeros(n_config*nprocs)
#    tab_IPR_glob = np.zeros(n_config*nprocs)
#    comm.Gather(tab_energy,tab_energy_glob)
#    comm.Gather(tab_IPR,tab_IPR_glob)
    timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
  else:
    pass
#    tab_energy_glob = tab_energy
#    tab_IPR_glob = tab_IPR
  t2=time.perf_counter()
  timing.TOTAL_TIME = t2-t1
  if mpi_version:
    timing.mpi_merge(comm)
  if rank==0:
#    print(tab_IPR_glob.shape)
    anderson.io.output_density('tab_rbar.dat',tab_middle_energy,tab_hist_r,header_string,print_type='IPR')

    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Wallclock time {0:.3f} seconds".format(t2-t1))
    print()
    if mpi_version:
      print("MPI time             = {0:.3f}".format(timing.MPI_TIME))
    print("Total_CPU time       = {0:.3f}".format(timing.TOTAL_TIME))


