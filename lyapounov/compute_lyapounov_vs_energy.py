#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

"""

import os
import time
import numpy as np
import getpass
import sys
import configparser
sys.path.append('../')
import anderson
import anderson.lyapounov
import anderson.io
import mkl

#import matplotlib.pyplot as plt

if __name__ == "__main__":
  environment_string='Script ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +'Name of python script: {}'.format(os.path.abspath( __file__ ))+'\n'\
             +'Started on: {}'.format(time.asctime())+'\n'
  mpi_version = False
  nprocs = 1
  rank = 0
  environment_string += 'Single processor version\n\n'

  config = configparser.ConfigParser()
  config.read('params.dat')

  Averaging = config['Averaging']
  n_config = Averaging.getint('n_config',1)

  System = config['System']
  system_size = System.getfloat('size')
  delta_x = System.getfloat('delta_x')
  boundary_condition = System.get('boundary_condition','periodic')

  Disorder = config['Disorder']
  disorder_type = Disorder.get('type','anderson gaussian')
  correlation_length = Disorder.getfloat('sigma',0.0)
  V0 = Disorder.getfloat('V0',0.0)
  disorder_strength = V0

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


  # Number of sites
  dim_x = int(system_size/delta_x+0.5)
  # Renormalize delta_x so that the system size is exactly what is wanted and split in an integer number of sites
  delta_x = system_size/dim_x

  assert boundary_condition in ['periodic','open'], "Boundary condition must be either 'periodic' or 'open'"
  # Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength)

  initial_time=time.asctime()
  hostname = os.uname()[1].split('.')[0]

  print("Python script started on: {}".format(initial_time))
  print("{:>24}: {}".format('from',hostname))
  print("Name of python script: {}".format(os.path.abspath( __file__ )))
  print("Total number of disorder realizations: {}".format(n_config*nprocs))
  print("Number of processes: {}".format(nprocs))

  timing=anderson.Timing()
  t1=time.perf_counter()
  mkl.set_num_threads(1)
  os.environ["MKL_NUM_THREADS"] = "1"

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
    tab_lyapounov, used_time, number_of_ops = lyapounov.compute_lyapounov(i, H)
    tab_global_lyapounov[0] += tab_lyapounov
    tab_global_lyapounov[1] += tab_lyapounov**2
    timing.LYAPOUNOV_TIME+=used_time
    timing.LYAPOUNOV_NOPS+=number_of_ops
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
  tab_global_lyapounov /= n_config
  if n_config>1:
    tab_global_lyapounov[1] = np.sqrt(np.abs(tab_global_lyapounov[1]-tab_global_lyapounov[0]**2)/(n_config-1))
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

  t2=time.perf_counter()

  final_time = time.asctime()
  print("Python script ended on: {}".format(final_time))
  print("Total execution time {0:.3f} seconds".format(t2-t1))
  print()
  print("Lyapounov time       = {0:.3f}".format(timing.LYAPOUNOV_TIME))
  print("Number of ops        = {0:.4e}".format(timing.LYAPOUNOV_NOPS))
  print("Total_time           = {0:.3f}".format(t2-t1))
