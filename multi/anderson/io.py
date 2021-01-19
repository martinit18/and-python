#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:22:51 2020

@author: delande
"""

import numpy as np
import copy
import configparser
import sys
import math
import anderson
#from anderson import geometry

def my_abort(mpi_version,comm,message):
  if mpi_version:
    print(message)
    comm.Abort()
  else:
    sys.exit(message)


def parse_parameter_file(mpi_version,comm,nprocs,rank,parameter_file,my_list_of_sections):
  if rank==0:
    config = configparser.ConfigParser()
    config.read(parameter_file)

# Mandatory Averaging section
    if not config.has_section('Averaging'):
      my_abort(mpi_version,comm,'Parameter file does not have an Averaging section, I stop!\n')
    Averaging = config['Averaging']
    n_config = Averaging.getint('n_config',1)
    n_config = (n_config+nprocs-1)//nprocs
    print("Total number of disorder realizations: {}".format(n_config*nprocs))
    print("Number of processes: {}".format(nprocs))
    print()

# Mandatory System section
    if not config.has_section('System'):
      my_abort(mpi_version,comm,'Parameter file does not have a System section, I stop!\n')
    System = config['System']
    dimension = System.getint('dimension', 1)
    one_over_mass = System.getfloat('one_over_mass', 1.0)
    tab_size = list()
    tab_delta = list()
    tab_boundary_condition = list()
    all_options_ok = True
    for i in range(dimension):
      if not config.has_option('System','size_'+str(i+1)): all_options_ok=False
      if not config.has_option('System','delta_'+str(i+1)): all_options_ok=False
      if not config.has_option('System','boundary_condition_'+str(i+1)): all_options_ok=False
      tab_size.append(System.getfloat('size_'+str(i+1)))
      tab_delta.append(System.getfloat('delta_'+str(i+1)))
      tab_boundary_condition.append(System.get('boundary_condition_'+str(i+1),'periodic'))
      if not tab_boundary_condition[i] in ['periodic','open']: all_options_ok=False
    if not all_options_ok:
      my_abort(mpi_version,comm,'In the System section of the parameter file, each dimension must have a size, delta (discretization step) and boundary condition, I stop!\n')

#    print(dimension,tab_size,tab_delta,tab_boundary_condition)
# Number of sites
    tab_dim = list()
    for i in range(dimension):
      tab_dim.append(int(tab_size[i]/tab_delta[i]+0.5))
# Renormalize delta so that the system size is exactly what is wanted and split in an integer number of sites
      tab_delta[i] = tab_size[i]/tab_dim[i]
#  print(tab_dim)

# Mandatory Disorder section
    if not config.has_section('Disorder'):
      my_abort(mpi_version,comm,'Parameter file does not have a Disorder section, I stop!\n')
    Disorder = config['Disorder']
    disorder_type = Disorder.get('type','anderson_gaussian')
    if disorder_type=='nice' and dimension!=1:
      my_abort(mpi_version,comm,"The 'nice' disorder is implemented only in dimension 1, I stop!\n")
    correlation_length = Disorder.getfloat('sigma',0.0)
    V0 = Disorder.getfloat('V0',0.0)
    disorder_strength = V0
    non_diagonal_disorder_strength = Disorder.getfloat('non_diagonal_disorder_strength',0.0)
    b = Disorder.getint('b',1)
    use_mkl_random = Disorder.getboolean('use_mkl_random',True)

# Optional Spin section
    if 'Spin' in my_list_of_sections:
      if not config.has_section('Spin'):
        spin_one_half = False
#        my_abort(mpi_version,comm,'Parameter file does not have a Spin section, I stop!\n')
      else:
        Spin = config['Spin']
        spin_one_half = Spin.getboolean('spin_one_half',False)
        if spin_one_half and dimension != 1:
          my_abort(mpi_version,comm,'Spin 1/2 is supported only in dimension 1, I stop!\n')
        spin_orbit_interaction = Spin.getfloat('gamma',0.0)
        sigma_x = 0.5*Spin.getfloat('Omega',0.0)
        sigma_y = Spin.getfloat('beta',0.0)
        sigma_z = 0.5*Spin.getfloat('delta',0.0)
        alpha = Spin.getfloat('h',0.0)
    else:
      spin_one_half = False


# Optional Nonlinearity section
    if 'Nonlinearity' in my_list_of_sections:
      if not config.has_section('Nonlinearity'):
        my_abort(mpi_version,comm,'Parameter file does not have a Nonlinearity section, I stop!\n')
      Nonlinearity = config['Nonlinearity']
# First try to see if g_over_volume is defined
      interaction_strength = Nonlinearity.getfloat('g_over_volume')
# If not, try g
      if interaction_strength==None:
        interaction_strength = Nonlinearity.getfloat('g',0.0)
      else:
 # Multiply g_over_V by the volume of the system
        for i in range(dimension):
          interaction_strength *= tab_size[i]
#    print(interaction_strength)
    else:
      interaction_strength = 0.0

# Optional Wavefunction section
    if 'Wavefunction' in my_list_of_sections:
      if not config.has_section('Wavefunction'):
        my_abort(mpi_version,comm,'Parameter file does not have a Wavefuntion section, I stop!\n')
      Wavefunction = config['Wavefunction']
      all_options_ok=True
      initial_state_type = Wavefunction.get('initial_state')
      if initial_state_type not in ["plane_wave","gaussian_wave_packet","point"]: all_options_ok=False
#    assert initial_state_type in ["plane_wave","gaussian_wave_packet"], "Initial state is not properly defined"
      tab_k_0 = list()
      tab_sigma_0 = list()
      if initial_state_type in ["plane_wave","gaussian_wave_packet"]:
        for i in range(dimension):
          if not config.has_option('Wavefunction','k_0_over_2_pi_'+str(i+1)): all_options_ok=False
#        tab_k_0.append(2.0*math.pi*
          k_0_over_2_pi=Wavefunction.getfloat('k_0_over_2_pi_'+str(i+1))
# k_0_over_2_pi*size must be an integer is all dimensions
# Renormalize k_0_over_2_pi to ensure it
          tab_k_0.append(2.0*math.pi*int(k_0_over_2_pi*tab_size[i]+0.5)/tab_size[i])
          if initial_state_type != "plane_wave":
            if not config.has_option('Wavefunction','sigma_0_'+str(i+1)): all_options_ok=False
            tab_sigma_0.append(Wavefunction.getfloat('sigma_0_'+str(i+1)))
      if not all_options_ok:
        my_abort(mpi_version,comm,'In the Wavefunction section of the parameter file, each dimension must have a k0_over_2_pi value, and a sigma_0 value if not a plane wave, or be a single point, I stop!\n')

# Optional Propagation section
    if 'Propagation' in my_list_of_sections:
      if not config.has_section('Propagation'):
        my_abort(mpi_version,comm,'Parameter file does not have a Propagation section, I stop!\n')
      Propagation = config['Propagation']
      method = Propagation.get('method','che')
      accuracy = Propagation.getfloat('accuracy',1.e-6)
      accurate_bounds = Propagation.getboolean('accurate_bounds',False)
      want_ctypes = Propagation.getboolean('want_ctypes',True)
      data_layout = Propagation.get('data_layout','real')
      if 'Spectral' in my_list_of_sections:
        t_max=0.0
        delta_t=0.0
        i_tab_0 = 0
      else:
        all_options_ok = True
        if not config.has_option('Propagation','t_max'): all_options_ok = False
        t_max = Propagation.getfloat('t_max')
        if not config.has_option('Propagation','delta_t'): all_options_ok = False
        delta_t = Propagation.getfloat('delta_t')
        i_tab_0 = 0
        if not all_options_ok:
          my_abort(mpi_version,comm,'In the Propagation section of the parameter file, there must be a maximum propagation time t_max and a time step delta_t, I stop!\n')

# Optional Measurement section
    if 'Measurement' in my_list_of_sections:
      if not config.has_section('Measurement'):
        my_abort(mpi_version,comm,'Parameter file does not have a Measurement section, I stop!\n')

      Measurement = config['Measurement']
      delta_t_dispersion = Measurement.getfloat('delta_t_dispersion',delta_t)
      delta_t_density = Measurement.getfloat('delta_t_density',t_max)
      first_measurement_autocorr = Measurement.getint('first_measurement_autocorr',0)
      measure_density = Measurement.getboolean('density',False)
      measure_density_momentum = Measurement.getboolean('density_momentum',False)
      measure_autocorrelation = Measurement.getboolean('autocorrelation',False)
      measure_dispersion_position = Measurement.getboolean('dispersion_position',False)
      measure_dispersion_position2 = Measurement.getboolean('dispersion_position2',False)
      measure_dispersion_momentum = Measurement.getboolean('dispersion_momentum',False)
      measure_dispersion_energy = Measurement.getboolean('dispersion_energy',False)
      measure_wavefunction = Measurement.getboolean('wavefunction',False)
      measure_wavefunction_momentum = Measurement.getboolean('wavefunction_momentum',False)
      measure_extended = Measurement.getboolean('dispersion_variance',False)
      measure_g1 = Measurement.getboolean('g1',False)
      measure_overlap = Measurement.getboolean('overlap',False)
      use_mkl_fft = Measurement.getboolean('use_mkl_fft',True)
      remove_hot_pixel = Measurement.getboolean('remove_hot_pixel',False)

# Optional Diagonalization section
    if 'Diagonalization' in my_list_of_sections:
      if not config.has_section('Diagonalization'):
        my_abort(mpi_version,comm,'Parameter file does not have a Diagonalization section, I stop!\n')
      Diagonalization = config['Diagonalization']
      diagonalization_method = Diagonalization.get('method','sparse')
      targeted_energy = Diagonalization.getfloat('targeted_energy')
      IPR_min = Diagonalization.getfloat('IPR_min',0.0)
      IPR_max = Diagonalization.getfloat('IPR_max',1.0)
      number_of_bins = Diagonalization.getint('number_of_bins',1)
      number_of_eigenvalues = Diagonalization.getint('number_of_eigenvalues',1)

# Optional Spectral section
    if 'Spectral' in my_list_of_sections:
      if not config.has_section('Spectral'):
        my_abort(mpi_version,comm,'Parameter file does not have a Spectral section, I stop!\n')
      Spectral = config['Spectral']
      all_options_ok = True
      if not config.has_option('Spectral','range'): all_options_ok = False
      e_range = Spectral.getfloat('range')
      if not config.has_option('Spectral','resolution'): all_options_ok = False
      e_resolution = Spectral.getfloat('resolution')
      if not all_options_ok:
        my_abort(mpi_version,comm,'In the Spectral section of the parameter file, there must be a range and a resolution, I stop!\n')
      if not config.has_section('Propagation'):
        my_abort(mpi_version,comm,'Parameter file does not have a Propagation section, I stop!\n')

# Optional Lyapounov section
    if 'Lyapounov' in my_list_of_sections:
      if not config.has_section('Lyapounov'):
        my_abort(mpi_version,comm,'Parameter file does not have a Lyapounov section, I stop!\n')
      Lyapounov = config['Lyapounov']
      e_min = Lyapounov.getfloat('e_min',0.0)
      e_max = Lyapounov.getfloat('e_max',0.0)
      number_of_e_steps = Lyapounov.getint('number_of_e_steps',0)
      e_histogram = Lyapounov.getfloat('e_histogram',0.0)
      lyapounov_min = Lyapounov.getfloat('lyapounov_min',0.0)
      lyapounov_max = Lyapounov.getfloat('lyapounov_max',0.0)
      number_of_bins = Lyapounov.getint('number_of_bins',0)
      want_ctypes = Lyapounov.getboolean('want_ctypes',True)

  else:
    dimension = None
    one_over_mass = None
    n_config = None
    tab_size = None
    tab_delta = None
    tab_dim = None
    tab_boundary_condition = None
    disorder_type = None
    correlation_length = None
    disorder_strength = None
    non_diagonal_disorder_strength = None
    b = None
    use_mkl_random = None
    spin_one_half = None
    spin_orbit_interaction = None
    sigma_x = None
    sigma_y = None
    sigma_z = None
    alpha = None
    interaction_strength = None
    initial_state_type = None
    tab_k_0 = None
    tab_sigma_0 = None
    method = None
    accuracy = None
    accurate_bounds = None
    want_ctypes = None
    data_layout = None
    t_max = None
    delta_t = None
    i_tab_0 = None
    delta_t_dispersion = None
    delta_t_density = None
    first_measurement_autocorr = None
    measure_density = None
    measure_density_momentum = None
    measure_autocorrelation = None
    measure_dispersion_position = None
    measure_dispersion_position2 = None
    measure_dispersion_momentum = None
    measure_dispersion_energy = None
    measure_wavefunction = None
    measure_wavefunction_momentum = None
    measure_extended = None
    measure_g1 = None
    measure_overlap = None
    use_mkl_fft = None
    remove_hot_pixel = None
    diagonalization_method = None
    targeted_energy = None
    IPR_min = None
    IPR_max = None
    number_of_bins = None
    number_of_eigenvalues = None
    e_range = None
    e_resolution = None
    e_min = None
    e_max = None
    number_of_e_steps = None
    e_histogram = None
    lyapounov_min = None
    lyapounov_max = None


  if mpi_version:
    n_config, dimension, one_over_mass, tab_size, tab_delta, tab_dim, tab_boundary_condition  = comm.bcast((n_config, dimension, one_over_mass, tab_size,tab_delta, tab_dim, tab_boundary_condition))
    disorder_type, correlation_length, disorder_strength, non_diagonal_disorder_strength, b, use_mkl_random, interaction_strength = comm.bcast((disorder_type, correlation_length, disorder_strength, non_diagonal_disorder_strength, b, use_mkl_random, interaction_strength))
    if 'Spin' in my_list_of_sections:
      spin_one_half, spin_orbit_interaction, sigma_x, sigma_y, sigma_z, alpha = comm.bcast((spin_one_half, spin_orbit_interaction, sigma_x, sigma_y, sigma_z, alpha))
    if 'Wavefunction' in my_list_of_sections:
      initial_state_type, tab_k_0, tab_sigma_0 = comm.bcast((initial_state_type, tab_k_0, tab_sigma_0))
    if 'Propagation' in my_list_of_sections:
      method, accuracy, accurate_bounds, want_ctypes, data_layout, t_max, delta_t, i_tab_0 = comm.bcast((method, accuracy, accurate_bounds, want_ctypes, data_layout, t_max, delta_t, i_tab_0))
    if 'Measurement' in my_list_of_sections:
      delta_t_dispersion, delta_t_density, first_measurement_autocorr, measure_density, measure_density_momentum, measure_autocorrelation, measure_dispersion_position, measure_dispersion_position2, measure_dispersion_momentum, measure_dispersion_energy, measure_wavefunction, measure_wavefunction_momentum, measure_extended, measure_g1, measure_overlap, use_mkl_fft, remove_hot_pixel = comm.bcast((delta_t_dispersion, delta_t_density, first_measurement_autocorr, measure_density, measure_density_momentum, measure_autocorrelation, measure_dispersion_position,  measure_dispersion_position2, measure_dispersion_momentum, measure_dispersion_energy, measure_wavefunction, measure_wavefunction_momentum, measure_extended, measure_g1, measure_overlap, use_mkl_fft, remove_hot_pixel))
    if 'Diagonalization' in my_list_of_sections:
      diagonalization_method, targeted_energy, IPR_min, IPR_max, number_of_bins, number_of_eigenvalues  = comm.bcast((diagonalization_method, targeted_energy, IPR_min, IPR_max, number_of_bins, number_of_eigenvalues))
    if 'Spectral' in my_list_of_sections:
      e_range, e_resolution  = comm.bcast((e_range,e_resolution))
    if 'Lyapounov' in my_list_of_sections:
      e_min, e_max, number_of_e_steps, e_histogram, lyapounov_min, lyapounov_max, number_of_bins, want_ctypes = comm.bcast((e_min, e_max, number_of_e_steps, e_histogram, lyapounov_min, lyapounov_max, number_of_bins, want_ctypes))

  geometry = anderson.geometry.Geometry(dimension, tab_dim, tab_delta)
# Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.hamiltonian.Hamiltonian(geometry, tab_boundary_condition=tab_boundary_condition, one_over_mass=one_over_mass, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength,non_diagonal_disorder_strength=non_diagonal_disorder_strength, b=b, use_mkl_random=use_mkl_random, interaction=interaction_strength)
  if spin_one_half:
    H.add_spin_one_half(spin_orbit_interaction=spin_orbit_interaction, sigma_x=sigma_x, sigma_y=sigma_y, sigma_z=sigma_z, alpha=alpha)
  else:
    H.spin_one_half = False
  return_list = [geometry, H]

# Define an initial state
  if 'Wavefunction' in my_list_of_sections:
    initial_state = anderson.wavefunction.Wavefunction(geometry)
    initial_state.type = initial_state_type
    if (initial_state.type=='plane_wave'):
      anderson.wavefunction.Wavefunction.plane_wave(initial_state,tab_k_0)
    if (initial_state.type=='gaussian_wave_packet'):
      anderson.wavefunction.Wavefunction.gaussian(initial_state,tab_k_0,tab_sigma_0)
    if (initial_state.type=='point'):
      anderson.wavefunction.Wavefunction.point(initial_state)
    return_list.append(initial_state)

# Define the structure of spectral_function
  if 'Spectral' in my_list_of_sections:
    spectral_function = anderson.propagation.Spectral_function(e_range,e_resolution)
    t_max = spectral_function.t_max
    delta_t = spectral_function.delta_t
    delta_t_dispersion = delta_t
    delta_t_density = delta_t
    measure_density = False
    measure_density_momentum = False
    measure_autocorrelation = True
    measure_dispersion_position = False
    measure_dispersion_position2 = False
    measure_dispersion_momentum = False
    measure_dispersion_energy = False
    measure_wavefunction = False
    measure_wavefunction_momentum = False
    measure_extended = False
    measure_g1 = False
    measure_overlap = False
    return_list.append(spectral_function)

# Define the structure of the temporal integration
  if 'Propagation' in my_list_of_sections:
    propagation = anderson.propagation.Temporal_Propagation(t_max,delta_t,method=method, accuracy=accuracy, accurate_bounds=accurate_bounds, data_layout=data_layout,want_ctypes=want_ctypes)
    return_list.append(propagation)

# Define the structure of measurements
  if 'Measurement' in my_list_of_sections:
    measurement = anderson.measurement.Measurement(geometry, delta_t_dispersion, delta_t_density, measure_density=measure_density, measure_density_momentum=measure_density_momentum, measure_autocorrelation=measure_autocorrelation, measure_dispersion_position=measure_dispersion_position, measure_dispersion_position2=measure_dispersion_position2, measure_dispersion_momentum=measure_dispersion_momentum, measure_dispersion_energy=measure_dispersion_energy, measure_wavefunction=measure_wavefunction, measure_wavefunction_momentum=measure_wavefunction_momentum, measure_extended=measure_extended,measure_g1=measure_g1, measure_overlap=measure_overlap, use_mkl_fft=use_mkl_fft, remove_hot_pixel=remove_hot_pixel)
    measurement_global = copy.deepcopy(measurement)
#  print(measurement.measure_density,measurement.measure_autocorrelation,measurement.measure_dispersion,measurement.measure_dispersion_momentum)
    measurement.prepare_measurement(propagation)
#  print(measurement.density_final.shape)
    measurement_global.prepare_measurement(propagation,global_measurement=True)
    return_list.append(measurement)
    return_list.append(measurement_global)

# Define the structure of diagonalization
  if 'Diagonalization' in my_list_of_sections:
    diagonalization = anderson.diag.Diagonalization(targeted_energy,method=diagonalization_method, IPR_min= IPR_min, IPR_max=IPR_max, number_of_bins=number_of_bins, number_of_eigenvalues=number_of_eigenvalues)
    return_list.append(diagonalization)

# Define the structure of lyapounov
  if 'Lyapounov' in my_list_of_sections:
    lyapounov = anderson.lyapounov.Lyapounov(e_min,e_max,number_of_e_steps,want_ctypes=want_ctypes)
    return_list.append(lyapounov)

  return_list.append(n_config)
#  print(return_list)
  return(return_list)

def output_string(H,n_config,nprocs=1,propagation=None,initial_state=None,measurement=None,spectral_function=None,diagonalization=None,lyapounov=None,timing=None):
  params_string = 'Disorder type                        = '+H.disorder_type+'\n'\
                 +'Correlation length                   = '+str(H.correlation_length)+'\n'\
                 +'Dimension                            = '+str(H.dimension)+'\n'\
                 +'use MKL random number generator      = '+str(H.use_mkl_random)+'\n'
  volume = 1.0
  for i in range(H.dimension):
    volume *= H.tab_dim[i]*H.tab_delta[i]
    params_string += \
                  'Size_'+str(i+1)+'                               = '+str(H.tab_dim[i]*H.tab_delta[i])+'\n'\
                 +'delta_'+str(i+1)+'                              = '+str(H.tab_delta[i])+'\n'\
                 +'N_'+str(i+1)+'                                  = '+str(H.tab_dim[i])+'\n'\
                 +'Boundary_Condition_'+str(i+1)+'                 = '+H.tab_boundary_condition[i]+'\n'\
                 +'1/mass                               = '+str(H.one_over_mass)+'\n'
  params_string += \
                  'Volume                               = '+str(volume)+'\n'
  params_string += \
                  'V0                                   = '+str(H.disorder_strength)+'\n'
  if H.disorder_type=='nice':
    params_string += \
                  'Non diagonal disorder strength       = '+str(H.non_diagonal_disorder_strength)+'\n'\
                 +'Number of non diagonal channels      = '+str(H.b)+'\n'
  if H.spin_one_half:
    params_string += \
                  'Spin 1/2                             = True\n'\
                 +'gamma (p sigma_z)                    = '+str(H.spin_orbit_interaction)+'\n'\
                 +'Omega (sigma_x/2)                    = '+str(2.0*H.sigma_x)+'\n'\
                 +'beta (sigma_y)                       = '+str(H.sigma_y)+'\n'\
                 +'delta (sigma_z/2)                    = '+str(2.0*H.sigma_z)+'\n'\
                 +'h (p^2 sigma_z)                      = '+str(H.alpha)+'\n'
  params_string += \
                  'g                                    = '+str(H.interaction)+'\n'\
                 +'g_over_volume                        = '+str(H.interaction/volume)+'\n'\
                 +'Number of disorder realizations      = '+str(n_config*nprocs)+'\n'\
                 +'Number of processes                  = '+str(nprocs)+'\n'\
                 +'Number of realizations per proc      = '+str(n_config)+'\n'
  if not initial_state == None:
    params_string += \
                  'Initial state                        = '+initial_state.type+'\n'
    if initial_state.type != 'point':
      for i in range(H.dimension):
        params_string += \
                  'k_0_'+str(i+1)+'                                = '+str(initial_state.tab_k_0[i])+'\n'
        if initial_state.type == 'gaussian_wave_packet':
          params_string += \
                 'sigma_0_'+str(i+1)+'                             = '+str(initial_state.tab_sigma_0[i])+'\n'
  if not propagation == None:
    params_string += \
                  'Integration Method                   = '+propagation.method+'\n'\
                 +'accuracy                             = '+str(propagation.accuracy)+'\n'
    if propagation.method=='che':
      params_string += \
                  'accurate spectrum bounds             = '+str(propagation.accurate_bounds)+'\n'\
                 +'use ctypes implementation            = '+str(propagation.use_ctypes)+'\n'
      if not timing==None:
        params_string += \
                  'maximum Chebyshev order              = '+str(timing.MAX_CHE_ORDER)+'\n'\
                 +'maximum non-linear phase             = '+str(timing.MAX_NONLINEAR_PHASE)+'\n'
    params_string += \
                  'data layout                          = '+propagation.data_layout+'\n'\
                 +'time step                            = '+str(propagation.delta_t)+'\n'\
                 +'total time                           = '+str(propagation.t_max)+'\n'
  if not measurement == None:
    params_string += \
                  'time step for dispersion measurement = '+str(measurement.delta_t_dispersion)+'\n'\
                 +'time step for density measurement    = '+str(measurement.delta_t_density)+'\n'
    if initial_state.type=='plane_wave':
      params_string += \
                  'remove hot pixel in momentum density = '+str(measurement.remove_hot_pixel)+'\n'
    if measurement.measure_overlap:
      params_string += \
                  '|overlap|**2 with initial state      = '+str(abs(measurement.overlap)**2)+'\n'
  if not spectral_function == None:
    params_string += \
                  'energy range                         = '+str(spectral_function.e_range)+'\n'\
                 +'energy resolution                    = '+str(spectral_function.e_resolution)+'\n'
  if not diagonalization == None:
    params_string += \
                  'targeted_energy                      = '+str(diagonalization.targeted_energy)+'\n'\
                 +'diagonalization method               = '+diagonalization.method+'\n'\
                 +'number of computed eigenvalues       = '+str(diagonalization.number_of_eigenvalues)+'\n'
  if not lyapounov == None:
    params_string += \
                  'minimum energy                       = '+str(lyapounov.e_min)+'\n'\
                 +'maximum energy                       = '+str(lyapounov.e_max)+'\n'\
                 +'energy step                          = '+str(lyapounov.e_step)+'\n'\
                 +'number of energy steps               = '+str(lyapounov.number_of_e_steps)+'\n'\
                 +'use ctypes implementation            = '+str(lyapounov.use_ctypes)+'\n'
  params_string += '\n'
#  print(params_string)
#                 +'first measurement step          = '+str(my_first_measurement_autocorr)+'\n'\
#                  +'total measurement time          = '+str(my_t_max-my_first_measurement_autocorr*my_delta_t_measurement)+'\n'\
  return params_string



def output_density(file,data,geometry,header_string='Origin of data not specified',data_type='density',tab_abscissa=[],file_type='savetxt'):
#  print(data.shape)
#  print(tab_abscissa)
  if file_type=='savetxt':
    if data_type=='density':
      column_1='Position'
      column_2='Density'
      column_3='Std. deviation of density'
      specific_string='Spatial density\n'
    if data_type=='density_momentum':
      column_1='Momentum'
      column_2='Density'
      column_3='Std. deviation of density'
      specific_string='Density in momentum space\n'
    if data_type=='wavefunction':
      column_1='Position'
      column_2='Re(wavefunction)'
      column_3='Im(wavefunction)'
      specific_string='Wavefunction in configuration space\n'
    if data_type=='wavefunction_momentum':
      column_1='Momentum'
      column_2='Re(wavefunction)'
      column_3='Im(wavefunction)'
      specific_string='Wavefunction in momentum space\n'
    if data_type=='autocorrelation':
      column_1='Time'
      column_2='Re(<psi(0)|psi(t)>)'
      column_3='Im(<psi(0)|psi(t)>)'
      specific_string='Temporal autocorrelation function\n'
    if data_type=='spectral_function':
      column_1='Energy'
      column_2='Spectral function'
      specific_string='Spectral function\n'
    if data_type=='density_of_states':
      column_1='Energy'
      column_2='Density of states'
      specific_string='Density of states\n'
    if data_type=='g1':
      column_1='Relative position'
      column_2='Re(g1)'
      column_3='Im(g1)'
      specific_string='g1 correlation function\n'
    if data_type=='IPR':
      column_1='IPR'
      specific_string='Inverse participation ratio for the eigenstate closer to the targeted energy\n'\
                   +'Average IPR           = '+str(np.mean(data))+'\n'\
                   +'Std. deviation of IPR = '+str(np.std(data))+'\n'
    if data_type=='histogram_IPR':
      column_1 = 'Right bin edge'
      column_2 = 'Normalized distribution'
      specific_string='Distribution of the inverse participation ratio for the eigenstates closer to the targeted energy\n'
    if data_type=='rbar':
      column_1 = 'Energy'
      column_2 = 'rbar value'
      specific_string='rbar value\n'
    if data_type=='histogram_r':
      column_1 = 'Right bin edge'
      column_2 = 'Normalized distribution'
      specific_string='Distribution of the r value for the eigenstates closer to the targeted energy\n'
    if data_type=='lyapounov':
      column_1='Energy'
      column_2='Lyapounov for the intensity (halve it for wavefunction)'
      column_3='Std. deviation of Lyapounov'
      column_4='Localization length for intensity (double it for wavefunction'
      column_5='Std. deviation of localization length'
      specific_string='Lyapounov and localization length vs. energy\n'
    list_of_columns = []
    tab_strings = []
#    print(specific_string)
    next_column = 1
    dimension = geometry.dimension
    if data_type in ['density','density_momentum']:
#      print(data.ndim,data.shape)
# The simple case where there is only 1d data
      if dimension==1:
        if data_type=='density':
          header_string=str(geometry.tab_dim[0])+' '+str(geometry.tab_delta[0])+'\n'+header_string
        if data_type=='density_momentum':
          header_string=str(geometry.tab_dim[0])+' '+str(2.0*np.pi/(geometry.tab_dim[0]*geometry.tab_delta[0]))+'\n'+header_string
        if data.ndim==1:
          if tab_abscissa!=[] and data.size==tab_abscissa[0].size:
            list_of_columns.append(tab_abscissa[0])
            tab_strings.append('Column '+str(next_column)+': '+column_1)
            next_column += 1
          list_of_columns.append(data)
          tab_strings.append('Column '+str(next_column)+': '+column_2)
          next_column += 1
          array_to_print=np.column_stack(list_of_columns)
        if data.ndim==2:
#          print(data.shape[1],tab_abscissa[0].size)
          if tab_abscissa!=[] and data.shape[1]==tab_abscissa[0].size:
            list_of_columns.append(tab_abscissa[0])
            tab_strings.append('Column '+str(next_column)+': '+column_1)
            next_column += 1
          list_of_columns.append(data[0,:])
          tab_strings.append('Column '+str(next_column)+': '+column_2)
          next_column += 1
          if data.shape[0]==2:
            list_of_columns.append(data[1,:])
            tab_strings.append('Column '+str(next_column)+': '+column_3)
            next_column += 1
          array_to_print=np.column_stack(list_of_columns)
      if dimension==2:
 # Add at the beginning of the file minimal info describing the data
        if data_type=='density':
          header_string=str(geometry.tab_dim[0])+' '+str(geometry.tab_delta[0])+'\n'\
                       +str(geometry.tab_dim[1])+' '+str(geometry.tab_delta[1])+'\n'\
                       +header_string
        if data_type=='density_momentum':
          header_string=str(geometry.tab_dim[0])+' '+str(2.0*np.pi/(geometry.tab_dim[0]*geometry.tab_delta[0]))+'\n'\
                       +str(geometry.tab_dim[1])+' '+str(2.0*np.pi/(geometry.tab_dim[1]*geometry.tab_delta[1]))+'\n'\
                       +header_string
        if data.ndim==2:
          array_to_print = data[:,:]
        if data.ndim==3:
          array_to_print=data[0,:,:]
    if data_type in ['wavefunction','wavefunction_momentum','g1']:
      if dimension==1:
        if data_type=='wavefunction' or data_type=='g1':
          header_string=str(geometry.tab_dim[0])+' '+str(geometry.tab_delta[0])+'\n'+header_string
        if data_type=='wavefunction_momentum':
          header_string=str(geometry.tab_dim[0])+' '+str(2.0*np.pi/(geometry.tab_dim[0]*geometry.tab_delta[0]))+'\n'+header_string
#        print(data.size,tab_abscissa[0].size)
        if tab_abscissa!=[] and data.size==tab_abscissa[0].size:
          if data_type=='g1':
            list_of_columns.append(tab_abscissa[0]-0.5*geometry.tab_delta[0])
          else:
            list_of_columns.append(tab_abscissa[0])
          tab_strings.append('Column '+str(next_column)+': '+column_1)
          next_column += 1
        list_of_columns.append(np.real(data))
        tab_strings.append('Column '+str(next_column)+': '+column_2)
        next_column += 1
        list_of_columns.append(np.imag(data))
        tab_strings.append('Column '+str(next_column)+': '+column_3)
        next_column += 1
        array_to_print=np.column_stack(list_of_columns)
      if dimension==2:
        if data_type=='wavefunction' or data_type=='g1':
          header_string=str(geometry.tab_dim[0])+' '+str(geometry.tab_delta[0])+'\n'\
                       +str(geometry.tab_dim[1])+' '+str(geometry.tab_delta[1])+'\n'\
                       +header_string
#          print(header_string)
        if data_type=='wavefunction_momentum':
          header_string=str(geometry.tab_dim[0])+' '+str(2.0*np.pi/(geometry.tab_dim[0]*geometry.tab_delta[0]))+'\n'\
                       +str(geometry.tab_dim[1])+' '+str(2.0*np.pi/(geometry.tab_dim[1]*geometry.tab_delta[1]))+'\n'\
                       +header_string
        array_to_print=data[:,:]
    if data_type in ['autocorrelation']:
      list_of_columns.append(tab_abscissa)
      tab_strings.append('Column '+str(next_column)+': '+column_1)
      next_column += 1
      list_of_columns.append(np.real(data))
      tab_strings.append('Column '+str(next_column)+': '+column_2)
      next_column += 1
      list_of_columns.append(np.imag(data))
      tab_strings.append('Column '+str(next_column)+': '+column_3)
      next_column += 1
      array_to_print=np.column_stack(list_of_columns)
    if data_type in ['spectral_function','density_of_states','histogram_IPR','rbar','histogram_r']:
      list_of_columns.append(tab_abscissa)
      tab_strings.append('Column '+str(next_column)+': '+column_1)
      next_column += 1
      list_of_columns.append(data)
      tab_strings.append('Column '+str(next_column)+': '+column_2)
      next_column += 1
      array_to_print=np.column_stack(list_of_columns)
    if data_type in ['IPR']:
      list_of_columns.append(data)
      tab_strings.append('Column '+str(next_column)+': '+column_1)
      next_column += 1
      array_to_print=np.column_stack(list_of_columns)
    if data_type in ['lyapounov']:
      list_of_columns.append(tab_abscissa)
      tab_strings.append('Column '+str(next_column)+': '+column_1)
      next_column += 1
      list_of_columns.append(data[0])
      tab_strings.append('Column '+str(next_column)+': '+column_2)
      next_column += 1
      list_of_columns.append(data[1])
      tab_strings.append('Column '+str(next_column)+': '+column_3)
      next_column += 1
      list_of_columns.append(1.0/data[0])
      tab_strings.append('Column '+str(next_column)+': '+column_4)
      next_column += 1
      list_of_columns.append(data[1]/data[0]**2)
      tab_strings.append('Column '+str(next_column)+': '+column_5)
      next_column += 1

      array_to_print=np.column_stack(list_of_columns)
#   print(list_of_columns,len(list_of_columns))
 #   if len(list_of_columns) == 1:
 #     array_to_print = list_of_columns
 #   else:
#    print(list_of_columns)
#    array_to_print=np.column_stack(list_of_columns)
#    print(list_of_columns)
#    print(tab_strings)
    np.savetxt(file,array_to_print,header=header_string+specific_string+'\n'.join(tab_strings)+'\n')
  return

def output_dispersion(file,tab_data,tab_strings,general_string='Origin of data not specified'):
#  print('\n'.join(tab_strings))
  np.savetxt(file,tab_data,header=general_string+'\n'.join(tab_strings)+'\n')
  return

def print_measurements_final(measurement,header_string='Origin of data not specified'):
  if (measurement.measure_density):
#      print(measurement.grid_position)
    anderson.io.output_density('density_final.dat',measurement.density_final,measurement,header_string=header_string,tab_abscissa=measurement.grid_position,data_type='density')
    for i in range(measurement.tab_t_measurement_density.size):
      anderson.io.output_density('density_intermediate_'+str(i+1)+'.dat',measurement.density_intermediate[i],measurement,header_string=header_string+'Time = '+str(measurement.tab_t_measurement_density[i])+' \n',tab_abscissa=measurement.grid_position,data_type='density')
  if (measurement.measure_density_momentum):
    anderson.io.output_density('density_momentum_final.dat',measurement.density_momentum_final,measurement,header_string=header_string,tab_abscissa=measurement.frequencies,data_type='density_momentum')
    for i in range(measurement.tab_t_measurement_density.size):
      anderson.io.output_density('density_momentum_intermediate_'+str(i+1)+'.dat',measurement.density_momentum_intermediate[i],measurement,header_string=header_string+'Time = '+str(measurement.tab_t_measurement_density[i])+' \n',tab_abscissa=measurement.frequencies,data_type='density_momentum')
  if (measurement.measure_wavefunction):
    anderson.io.output_density('wavefunction_final.dat',measurement.wfc,measurement,header_string=header_string,tab_abscissa=measurement.grid_position,data_type='wavefunction')
  if (measurement.measure_wavefunction_momentum):
    anderson.io.output_density('wavefunction_momentum_final.dat',measurement.wfc_momentum,measurement,header_string=header_string,tab_abscissa=measurement.frequencies,data_type='wavefunction_momentum')
  if (measurement.measure_autocorrelation):
    anderson.io.output_density('temporal_autocorrelation.dat',measurement.tab_autocorrelation,measurement,tab_abscissa=measurement.tab_t_measurement_dispersion,header_string=header_string,data_type='autocorrelation')
  if (measurement.measure_dispersion_position or measurement.measure_dispersion_momentum or measurement.measure_dispersion_energy):
    anderson.io.output_dispersion('dispersion.dat',measurement.tab_dispersion,measurement.tab_strings,header_string)
  if (measurement.measure_g1):
    anderson.io.output_density('g1_final.dat',measurement.g1,measurement,header_string=header_string,tab_abscissa=measurement.grid_position,data_type='g1')
    for i in range(measurement.tab_t_measurement_density.size):
      anderson.io.output_density('g1_intermediate_'+str(i+1)+'.dat',measurement.g1_intermediate[i],measurement,header_string=header_string+'Time = '+str(measurement.tab_t_measurement_density[i])+' \n',tab_abscissa=measurement.grid_position,data_type='g1')
#    print("c'est fini")
  return

def print_measurements_initial(measurement,initial_state,header_string='Origin of data not specified'):
  if (measurement.measure_density):
#      print(measurement.grid_position)
    anderson.io.output_density('density_initial.dat',np.abs(initial_state.wfc)**2,measurement,header_string=header_string,tab_abscissa=measurement.grid_position,data_type='density')
  if (measurement.measure_density_momentum):
    anderson.io.output_density('density_momentum_initial.dat',np.abs(initial_state.convert_to_momentum_space())**2,measurement,header_string=header_string,tab_abscissa=measurement.frequencies,data_type='density_momentum')
  if (measurement.measure_wavefunction):
    anderson.io.output_density('wavefunction_initial.dat',initial_state.wfc,measurement,header_string=header_string,tab_abscissa=measurement.grid_position,data_type='wavefunction')
  if (measurement.measure_wavefunction_momentum):
    anderson.io.output_density('wavefunction_momentum_initial.dat',initial_state.convert_to_momentum_space(),measurement,header_string=header_string,tab_abscissa=measurement.frequencies,data_type='wavefunction_momentum')
  return
