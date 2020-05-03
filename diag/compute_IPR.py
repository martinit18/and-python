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
import sys
sys.path.append('../')
sys.path.append('/users/champ/delande/git/and-python/')
import anderson
import mkl




if __name__ == "__main__":
  environment_string='Script ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +'Name of python script: {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Started on: {}'.format(time.asctime())+'\n'
  try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    mpi_version = True
    environment_string += 'MPI version ran on '+str(nprocs)+' processes\n\n'
  except ImportError:
    mpi_version = False
    nprocs = 1
    rank = 0
    environment_string += 'Single processor version\n\n'

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
    diagonalization_method = None
    targeted_energy = None
 #  n_config = comm.bcast(n_config,root=0)
  if mpi_version:
    n_config, system_size, delta_x,boundary_condition  = comm.bcast((n_config, system_size,delta_x,boundary_condition ))
    disorder_type, correlation_length, disorder_strength = comm.bcast((disorder_type, correlation_length, disorder_strength))
    diagonalization_method, targeted_energy = comm.bcast((diagonalization_method, targeted_energy))

#  delta_x = comm.bcast(delta_x)
#  print(rank,n_config,system_size,delta_x)
    # Number of sites
  dim_x = int(system_size/delta_x+0.5)
    # Renormalize delta_x so that the system size is exactly what is wanted and split in an integer number of sites
  delta_x = system_size/dim_x
    #V0=0.025
    #disorder_strength = np.sqrt(V0)
  mkl.set_num_threads(1)
  os.environ["MKL_NUM_THREADS"] = "1"

  assert boundary_condition in ['periodic','open'], "Boundary condition must be either 'periodic' or 'open'"


  # Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, interaction=0.0)
  diagonalization = anderson.diag.Diagonalization(targeted_energy,diagonalization_method)

  #comm.Bcast(H)
  header_string = environment_string+anderson.io.output_string(H,n_config,nprocs,diagonalization=diagonalization)

  tab_IPR = np.zeros(n_config)
  tab_energy = np.zeros(n_config)
  # Here starts the loop over disorder configurations
  for i in range(n_config):
    tab_energy[i], tab_IPR[i] = diagonalization.compute_IPR(i+rank*n_config, H)
#    print(IPR)
  #  pool.apply_async(gpe_evolution, args)
  #  print(str(i), file=final_pf)
#  anderson.io.output_density('IPR'+str(rank)+'.dat',tab_IPR,tab_energy,header_string,print_type='IPR')
  if mpi_version:
    tab_energy_glob = np.zeros(n_config*nprocs)
    tab_IPR_glob = np.zeros(n_config*nprocs)
    comm.Gather(tab_energy,tab_energy_glob)
    comm.Gather(tab_IPR,tab_IPR_glob)
  else:
    tab_energy_glob = tab_energy
    tab_IPR_glob = tab_IPR
  if rank==0:
#    print(tab_IPR_glob.shape)
    anderson.io.output_density('IPR.dat',tab_IPR_glob,tab_energy_glob,header_string,print_type='IPR')
    tab_histogram, bin_edges = np.histogram(tab_IPR_glob, bins=number_of_bins, range=(IPR_min,IPR_max), density=True)
#  print(tab_histogram)
#  print(bin_edges)
    anderson.io.output_density('histogram_IPR.dat',bin_edges[1:],tab_histogram,header_string,print_type='histogram_IPR')
    t2=time.perf_counter()

    final_time = time.asctime()
    print("Python script ended on: {}".format(final_time))
    print("Total execution time {0:.3f} seconds".format(t2-t1))
    print()
    print("Total_time           = {0:.3f}".format(t2-t1))


