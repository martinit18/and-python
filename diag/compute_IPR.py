#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

"""


import os
import time
import math
import numpy as np
import getpass
import datetime
import copy
import configparser
import sys
sys.path.append('../')
import anderson
import anderson.diag
import anderson.io
import mkl


if __name__ == "__main__":
  environment_string='Script '+os.path.basename(__file__)+' ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
             +datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'\n'

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

  Diagonalization = config['Diagonalization']
  diagonalization_method = Diagonalization.get('method','sparse')
  targeted_energy = Diagonalization.getfloat('targeted_energy')
  IPR_min = Diagonalization.getfloat('IPR_min',0.0)
  IPR_max = Diagonalization.getfloat('IPR_max')
  number_of_bins = Diagonalization.getint('number_of_bins')

  # Number of sites
  dim_x = int(system_size/delta_x+0.5)
  # Renormalize delta_x so that the system size is exactly what is wanted and split in an integer number of sites
  delta_x = system_size/dim_x
  #V0=0.025
  #disorder_strength = np.sqrt(V0)

  assert boundary_condition in ['periodic','open'], "Boundary condition must be either 'periodic' or 'open'"

  initial_time=time.asctime()
  hostname = os.uname()[1].split('.')[0]
  print("Python script started on: {}".format(initial_time))
  print("{:>24}: {}".format('from',hostname))
  print("Name of python script: {}".format(os.path.abspath( __file__ )))
  #print("Number of available threads: {}".format(multiprocessing.cpu_count()))
  print("Number of disorder realizations: {}".format(n_config))

  timing=anderson.Timing()
  t1=time.perf_counter()
  mkl.set_num_threads(1)
  os.environ["MKL_NUM_THREADS"] = "1"

# Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, interaction=0.0)
  diagonalization = anderson.diag.Diagonalization(targeted_energy,diagonalization_method)

  header_string = environment_string+anderson.io.output_string(H,n_config,diagonalization=diagonalization)

  tab_IPR = np.zeros(n_config)
  tab_energy = np.zeros(n_config)
  # Here starts the loop over disorder configurations
  for i in range(n_config):
    tab_energy[i], tab_IPR[i] = diagonalization.compute_IPR(i, H)
#    print(IPR)
  #  pool.apply_async(gpe_evolution, args)
  #  print(str(i), file=final_pf)
  anderson.io.output_density('IPR.dat',tab_IPR,tab_energy,header_string,print_type='IPR')

  tab_histogram, bin_edges = np.histogram(tab_IPR, bins=number_of_bins, range=(IPR_min,IPR_max), density=True)
#  print(tab_histogram)
#  print(bin_edges)
  anderson.io.output_density('histogram_IPR.dat',bin_edges[1:],tab_histogram,header_string,print_type='histogram_IPR')

  t2=time.perf_counter()

  final_time = time.asctime()
  print("Python script ended on: {}".format(final_time))
  print("Total execution time {0:.3f} seconds".format(t2-t1))
  print()
  print("Total_time           = {0:.3f}".format(t2-t1))


