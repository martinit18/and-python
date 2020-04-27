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
import numpy as np
import getpass
import copy
import configparser
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
#  environment_string='Script '+os.path.basename(__file__)+' ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
#             +datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'\n'

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
  initial_state_type = Wavefunction.get('initial_state')
  k_0 = 2.0*math.pi*Wavefunction.getfloat('k_0_over_2_pi')
  sigma_0 = Wavefunction.getfloat('sigma_0')

  Propagation = config['Propagation']
  method = Propagation.get('method','che')
  data_layout = Propagation.get('data_layout','real')
  t_max = Propagation.getfloat('t_max')
  delta_t = Propagation.getfloat('delta_t')
  i_tab_0 = 0

  Measurement = config['Measurement']
  delta_t_measurement = Measurement.getfloat('delta_t_measurement',delta_t)
  first_measurement_autocorr = Measurement.getint('first_measurement_autocorr',0)
  measure_density = Measurement.getboolean('density','False')
  measure_density_momentum = Measurement.getboolean('density_momentum','False')
  measure_autocorrelation = Measurement.getboolean('autocorrelation','False')
  measure_dispersion = Measurement.getboolean('dispersion','False')
  measure_dispersion_momentum = Measurement.getboolean('dispersion_momentum','False')
  if not measure_dispersion:
    measure_dispersion_momentum = False
  measure_wavefunction_final = Measurement.getboolean('wavefunction_final','False')
  measure_extended = Measurement.getboolean('extended','False')

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
#  print("Number of available threads: {}".format(multiprocessing.cpu_count()))
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
  initial_state.type = initial_state_type
  assert initial_state.type in ["plane_wave","gaussian_wave_packet"], "Initial state is not properly defined"
  if (initial_state.type=='plane_wave'):
    anderson.Wavefunction.plane_wave(initial_state,k_0)
  if (initial_state.type=='gaussian_wave_packet'):
    anderson.Wavefunction.gaussian(initial_state,sigma_0,k_0)

# Define the structure of the temporal integration
  propagation = anderson.propagation.Temporal_Propagation(t_max,delta_t,method=method,data_layout=data_layout)

  # Creates equally spaced points by delta_x covering the interval [-0.5*system_size,0.5*system_size]
  # The first (last) point is shifted right (left) by 0.5*delta_x
  #position = 0.5*delta_x*np.arange(1-dim_x,dim_x+1,2)

  # When computing an autocorrelation <psi(0)|psi(t)>, one can skip the first time steps, i.e. initialize
  # psi(0) with the wavefunction at some time tau. tau must be an integer multiple of delta_t_measurement.
  # args.first_step_autocorr is the integer tau/delta_t_measurement
  # In other words, the measurement of the autocorrelation function starts as time tau=delta_t_measurement*first_measurement_autocorr

  measurement = anderson.propagation.Measurement(delta_t_measurement, measure_density=measure_density, measure_density_momentum=measure_density_momentum, measure_autocorrelation=measure_autocorrelation, measure_dispersion=measure_dispersion, measure_dispersion_momentum=measure_dispersion_momentum, measure_wavefunction_final=measure_wavefunction_final,measure_extended=measure_extended)
  measurement_global = copy.deepcopy(measurement)
  measurement.prepare_measurement(propagation,delta_x,dim_x)
  measurement_global.prepare_measurement_global(propagation,delta_x,dim_x)
#  print(measurement.measure_density,measurement_global.measure_density)
#  print(measurement.density_final)
#  print(initial_state.sigma_0)
  header_string = environment_string+anderson.io.output_string(H,n_config,initial_state=initial_state,propagation=propagation,measurement=measurement_global)
#  print(header_string)
# Save the initial density in configuration space
#  np.savetxt('density_initial.dat',np.column_stack([initial_state.position,np.abs(initial_state.wfc)**2]),header=header_string+'     Position              Density')
#  anderson.io.output_density('density_initial.dat',initial_state.position,np.abs(initial_state.wfc)**2,header_string)

#  xx = anderson.propagation.prepare_measurement(propagation,delta_x,dim_x,delta_t_measurement,measure_density,measure_density_momentum,measure_autocorrelation,measure_dispersion,measure_final_wavefunction)
#  print(xx)

#  tab_autocorrelation = np.zeros((n_config,number_of_measurements),dtype=np.complex128)

  if (measurement_global.measure_density):
    anderson.io.output_density('density_initial.dat',initial_state.position,np.abs(initial_state.wfc)**2,header_string,print_type='density')
  if (measurement_global.measure_density_momentum):
    anderson.io.output_density('density_momentum_initial.dat',measurement.frequencies,np.abs(initial_state.convert_to_momentum_space())**2,header_string,print_type='density_momentum')
    anderson.io.output_density('wavefunction_momentum_initial.dat',measurement.frequencies,initial_state.convert_to_momentum_space(),header_string,print_type='wavefunction_momentum')

# Here starts the loop over disorder configurations
  for i in range(n_config):
# Propagate from 0 to t_max
    anderson.propagation.gpe_evolution(i, initial_state, H, propagation, measurement, timing)
#   print(measurement.wfc_momentum[2128])
#   print(measurement.tab_autocorrelation[-1])
    measurement_global.merge_measurement(measurement)
  tab_strings, tab_dispersion = measurement_global.normalize(n_config)

  if (measurement_global.measure_density):
    anderson.io.output_density('density_final.dat',initial_state.position,measurement_global.density_final,header_string,print_type='density')
  if (measurement_global.measure_density_momentum):
    anderson.io.output_density('density_momentum_final.dat',measurement.frequencies,measurement_global.density_momentum_final,header_string,print_type='density_momentum')
  if (measurement_global.measure_wavefunction_final):
    anderson.io.output_density('wavefunction_final.dat',initial_state.position,measurement_global.wfc,header_string,print_type='wavefunction')
  if (measurement_global.measure_wavefunction_momentum_final):
    anderson.io.output_density('wavefunction_momentum_final.dat',measurement.frequencies,measurement_global.wfc_momentum,header_string,print_type='wavefunction_momentum')
  if (measurement_global.measure_autocorrelation):
    anderson.io.output_density('temporal_autocorrelation.dat',measurement.tab_t_measurement[i_tab_0:]-measurement.tab_t_measurement[i_tab_0],measurement_global.tab_autocorrelation,header_string,print_type='autocorrelation')
  if (measurement_global.measure_dispersion):
    anderson.io.output_dispersion('dispersion.dat',tab_dispersion,tab_strings,header_string)


#  for i in range(n_config):
# Propagate from 0 to t_max
#    final_state, tab_x[i,:], tab_x2[i,:], tab_energy[i,:], tab_nonlinear_energy[i,:], tab_autocorrelation[i,:] = anderson.propagation.gpe_evolution(i, initial_state, H, propagation,timing)
#    tab_autocorrelation[i,:] = anderson.propagation.gpe_evolution(i, initial_state, H, propagation,timing)
# Accumulate density in configuration space
#    density_final += np.abs(final_state.wfc)**2
# Compute wavefunction in momentum space and accumulate the density
#    psic_momentum = delta_x*np.fft.fftshift(np.fft.fft(final_state.wfc))/np.sqrt((2.0*np.pi))
#    density_momentum_final += np.abs(psic_momentum)**2
#  args = (init_state, Npa, kick, deltax, tmax, dt, disorder_strength, interaction_strength, i)
#  pool.apply_async(gpe_evolution, args)
#  print(str(i), file=final_pf)
#  density_final/=n_config
#  density_momentum_final/=n_config
  t2=time.perf_counter()
  """
  i_tab_0 = propagation.first_measurement_autocorr

  header_string=environment_string\
             +params_string\
             +'Temporal autocorrelation function\n'\
             +'Column 1: Time\n'\
             +'Column 2: Real(<psi(0)|psi(t)>)\n'\
             +'Column 3: Imag(<psi(0)|psi(t)>)\n'\
             +'\n'
  np.savetxt('temporal_autocorrelation.dat',np.column_stack([propagation.tab_t_measurement[i_tab_0:]-propagation.tab_t_measurement[i_tab_0],np.mean(np.real(tab_autocorrelation[:,:number_of_measurements-i_tab_0]),0),np.mean(np.imag(tab_autocorrelation[:,:number_of_measurements-i_tab_0]),0)]),header=header_string)
  """

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

