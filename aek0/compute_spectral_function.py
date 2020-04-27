#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 17:05:10 2019

@author: delande

Disordered one-dimensional system
Discretization in configuration space
3-point discretization of the Laplace operator

This program computes either the potential correlaction function or the spectral function (using Fourier transform of the temporal propagation)

V0 (=disorder_strength) is the square root of the variance of the disorder
sigma (=correlation_length) is the correlation length of the disorder
Various disorder types can be used, defined by the disorder_type variable:
  anderson gaussian: Usual Anderson model (spatially uncorrelated disorder) with Gaussian on-site distribution of the disorder
  regensburg: Gaussian on-site distribution with spatial correlation function <V(r)V(r+delta)>=V_0^2 exp(-0.5*delta^2/sigma^2)
  singapore: Gaussian on-site distribution with spatial correlation function <V(r)V(r+delta)>=V_0^2 sinc(delta/sigma)
  konstanz: Rayleigh on-site distribution (average V_0, variance V_0^2) with spatial correlation function <V(r)V(r+delta)>=V_0^2 exp(-0.5*delta^2/sigma^2)
  speckle: Rayleigh on-site distribution (average V_0, variance V_0^2) with spatial correlation function <V(r)V(r+delta)>=V_0^2 (1+sinc^2(delta/sigma))

There is addiitonally the possibility of adding a nonlinear term proportional to g in the GPE
Not tested, should not be used |-(.

  Internally, the wavefunction can be stored using two different layouts.
  This does NOT affect the storage of the wavefunctions used for measurements, which is always 'complex'
  This affecty only the vector used in the guts of the propagation algorithm.
  'real' is usually a bit faster.
  For data_layout == 'complex':
        wfc is assumed to be in format where
        wfc[0:2*dim_x:2] contains the real part of the wavefunction and
        wfc[1:2*dim_x:2] contains the imag part of the wavefunction.
      For data_layout == 'real':
        wfc is assumed to be in format where
        wfc[0:dim_x] contains the real part of the wavefunction and
        wfc[dim_x:2*dim_x] contains the imag part of the wavefunction.

"""

import os
import time
import math
import copy
import configparser
import numpy as np
import getpass
import sys
sys.path.append('../')
import anderson
import anderson.propagation
import anderson.io
import mkl



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

  Nonlinearity = config['Nonlinearity']
  interaction_strength = Nonlinearity.getfloat('g',0.0)

  Wavefunction = config['Wavefunction']
  my_initial_state = Wavefunction.get('initial_state')
  k_0 = 2.0*math.pi*Wavefunction.getfloat('k_0_over_2_pi')
  sigma_0 = Wavefunction.getfloat('sigma_0')

  Propagation = config['Propagation']
  method = Propagation.get('method','che')
  data_layout = Propagation.get('data_layout','real')
  i_tab_0 = 0

  Spectral = config['Spectral']
  e_range = Spectral.getfloat('range')
  e_resolution = Spectral.getfloat('resolution')

  # Number of sites
  dim_x = int(system_size/delta_x+0.5)
  # Renormalize delta_x so that the system size is exactly what is wanted and split in an integer number of sites
  delta_x = system_size/dim_x
  #V0=0.025
  #disorder_strength = np.sqrt(V0)

  spectral_function = anderson.propagation.Spectral_function(e_range,e_resolution)

  assert boundary_condition in ['periodic','open'], "Boundary condition must be either 'periodic' or 'open'"

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

# Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dim_x, delta_x, boundary_condition=boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, interaction=interaction_strength)

  # Define an initial state
  initial_state = anderson.Wavefunction(dim_x,delta_x)
  initial_state.type = my_initial_state
  assert initial_state.type in ["plane_wave","gaussian_wave_packet"], "Initial state is not properly defined"
  if (initial_state.type=='plane_wave'):
    anderson.Wavefunction.plane_wave(initial_state,k_0)
  if (initial_state.type=='gaussian_wave_packet'):
    anderson.Wavefunction.gaussian(initial_state,sigma_0,k_0)

# Define the structure of the temporal integration
  propagation = anderson.propagation.Temporal_Propagation(spectral_function.t_max,spectral_function.delta_t,method=method,data_layout=data_layout)

  measurement = anderson.propagation.Measurement(spectral_function.delta_t,  measure_autocorrelation=True)
  measurement_global = copy.deepcopy(measurement)
  measurement.prepare_measurement(propagation,delta_x,dim_x)
  measurement_global.prepare_measurement_global(propagation,delta_x,dim_x)

  header_string = environment_string+anderson.io.output_string(H,n_config,initial_state=initial_state,propagation=propagation,measurement=measurement_global,spectral_function=spectral_function)

# Here starts the loop over disorder configurations
  for i in range(n_config):
# Propagate from 0 to t_max
    anderson.propagation.gpe_evolution(i, initial_state, H, propagation, measurement, timing)
#   print(measurement.wfc_momentum[2128])
#   print(measurement.tab_autocorrelation[-1])
    measurement_global.merge_measurement(measurement)
  measurement_global.normalize(n_config)
  anderson.io.output_density('temporal_autocorrelation.dat',measurement_global.tab_t_measurement[i_tab_0:]-measurement_global.tab_t_measurement[i_tab_0],measurement_global.tab_autocorrelation,header_string,print_type='autocorrelation')

  tab_energies,tab_spectrum = spectral_function.compute_spectral_function(measurement_global.tab_autocorrelation)
  anderson.io.output_density('spectral_function.dat',tab_energies,tab_spectrum,header_string,print_type='spectral_function')

  t2=time.perf_counter()

  final_time = time.asctime()
  print("Python script ended on: {}".format(final_time))
  print("Total execution time {0:.3f} seconds".format(t2-t1))
  print()
  if (propagation.method=='gsl'):
    print("GSL time             = {0:.3f}".format(timing.GSL_TIME))
    print("  GPE time             = {0:.3f}".format(timing.GPE_TIME))
    print("Number of time steps =",timing.timing.N_SOLOUT)
  else:
    print("CHE time             = {0:.3f}".format(timing.CHE_TIME))
#  print("Nonlinear time       = {0:.3f}".format(nlin_time))
    print("Max order            =",timing.MAX_CHE_ORDER)
  print("Expect time          = {0:.3f}".format(timing.EXPECT_TIME))
  print("Dummy time           = {0:.3f}".format(timing.DUMMY_TIME))
  print("Number of ops        = {0:.4e}".format(timing.NUMBER_OF_OPS))
  print("Total_time           = {0:.3f}".format(t2-t1))

