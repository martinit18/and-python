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
    disorder_type = Disorder.get('type','anderson gaussian')
    correlation_length = Disorder.getfloat('sigma',0.0)
    V0 = Disorder.getfloat('V0',0.0)
    disorder_strength = V0
    use_mkl_random = Disorder.getboolean('use_mkl_random',True)

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
      if initial_state_type not in ["plane_wave","gaussian_wave_packet"]: all_options_ok=False
#    assert initial_state_type in ["plane_wave","gaussian_wave_packet"], "Initial state is not properly defined"
      tab_k_0 = list()
      tab_sigma_0 = list()
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
        my_abort(mpi_version,comm,'In the Wavefunction section of the parameter file, each dimension must have a k0_over_2_pi value, and a sigma_0 value if not a plane wave, I stop!\n')

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
      delta_t_measurement = Measurement.getfloat('delta_t_measurement',delta_t)
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

  else:
    dimension = None
    n_config = None
    tab_size = None
    tab_delta = None
    tab_dim = None
    tab_boundary_condition = None
    disorder_type = None
    correlation_length = None
    disorder_strength = None
    use_mkl_random = None
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
    delta_t_measurement = None
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
    diagonalization_method = None
    targeted_energy = None
    IPR_min = None
    IPR_max = None
    number_of_bins = None
    number_of_eigenvalues = None

  if mpi_version:
    n_config, dimension,tab_size,tab_delta,tab_dim, tab_boundary_condition  = comm.bcast((n_config,dimension,tab_size,tab_delta, tab_dim, tab_boundary_condition))
    disorder_type, correlation_length, disorder_strength, use_mkl_random, interaction_strength = comm.bcast((disorder_type, correlation_length, disorder_strength, use_mkl_random, interaction_strength))
    if 'Wavefunction' in my_list_of_sections:
      initial_state_type, tab_k_0, tab_sigma_0 = comm.bcast((initial_state_type, tab_k_0, tab_sigma_0))
    if 'Propagation' in my_list_of_sections:
      method, accuracy, accurate_bounds, want_ctypes, data_layout, t_max, delta_t, i_tab_0 = comm.bcast((method, accuracy, accurate_bounds, want_ctypes, data_layout, t_max, delta_t, i_tab_0))
    if 'Measurement' in my_list_of_sections:
      delta_t_measurement, first_measurement_autocorr, measure_density, measure_density_momentum, measure_autocorrelation, measure_dispersion_position, measure_dispersion_position2, measure_dispersion_momentum, measure_dispersion_energy, measure_wavefunction, measure_wavefunction_momentum, measure_extended, measure_g1, measure_overlap, use_mkl_fft = comm.bcast((delta_t_measurement, first_measurement_autocorr, measure_density, measure_density_momentum, measure_autocorrelation, measure_dispersion_position,  measure_dispersion_position2, measure_dispersion_momentum, measure_dispersion_energy, measure_wavefunction, measure_wavefunction_momentum, measure_extended, measure_g1, measure_overlap, use_mkl_fft))
    if 'Diagonalization' in my_list_of_sections:
      diagonalization_method, targeted_energy, IPR_min, IPR_max, number_of_bins, number_of_eigenvalues  = comm.bcast((diagonalization_method, targeted_energy, IPR_min, IPR_max, number_of_bins, number_of_eigenvalues))

# Prepare Hamiltonian structure (the disorder is NOT computed, as it is specific to each realization)
  H = anderson.Hamiltonian(dimension,tab_dim,tab_delta, tab_boundary_condition=tab_boundary_condition, disorder_type=disorder_type, correlation_length=correlation_length, disorder_strength=disorder_strength, use_mkl_random=use_mkl_random, interaction=interaction_strength)
  return_list = [H]

# Define an initial state
  if 'Wavefunction' in my_list_of_sections:
    initial_state = anderson.Wavefunction(tab_dim,tab_delta)
    initial_state.type = initial_state_type
    if (initial_state.type=='plane_wave'):
      anderson.Wavefunction.plane_wave(initial_state,tab_k_0)
    if (initial_state.type=='gaussian_wave_packet'):
      anderson.Wavefunction.gaussian(initial_state,tab_k_0,tab_sigma_0)
    return_list.append(initial_state)

# Define the structure of the temporal integration
  if 'Propagation' in my_list_of_sections:
    propagation = anderson.propagation.Temporal_Propagation(t_max,delta_t,method=method, accuracy=accuracy, accurate_bounds=accurate_bounds, data_layout=data_layout,want_ctypes=want_ctypes)
    return_list.append(propagation)

# Define the structure of measurements
  if 'Measurement' in my_list_of_sections:
    measurement = anderson.propagation.Measurement(delta_t_measurement, measure_density=measure_density, measure_density_momentum=measure_density_momentum, measure_autocorrelation=measure_autocorrelation, measure_dispersion_position=measure_dispersion_position, measure_dispersion_position2=measure_dispersion_position2, measure_dispersion_momentum=measure_dispersion_momentum, measure_dispersion_energy=measure_dispersion_energy, measure_wavefunction=measure_wavefunction, measure_wavefunction_momentum=measure_wavefunction_momentum, measure_extended=measure_extended,measure_g1=measure_g1, measure_overlap=measure_overlap, use_mkl_fft=use_mkl_fft)
    measurement_global = copy.deepcopy(measurement)
#  print(measurement.measure_density,measurement.measure_autocorrelation,measurement.measure_dispersion,measurement.measure_dispersion_momentum)
    measurement.prepare_measurement(propagation,tab_delta,tab_dim)
#  print(measurement.density_final.shape)
    measurement_global.prepare_measurement_global(propagation,tab_delta,tab_dim)
    return_list.append(measurement)
    return_list.append(measurement_global)

# Define the structure of diagonalization
  if 'Diagonalization' in my_list_of_sections:
    diagonalization = anderson.diag.Diagonalization(targeted_energy,method=diagonalization_method, IPR_min= IPR_min, IPR_max=IPR_max, number_of_bins=number_of_bins, number_of_eigenvalues=number_of_eigenvalues)
    return_list.append(diagonalization)


  return_list.append(n_config)
#  print(return_list)
#  return (H, initial_state, propagation, measurement, measurement_global, n_config)
  return(return_list)

def output_string(H,n_config,nprocs=1,propagation=None,initial_state=None,measurement=None,spectral_function=None,diagonalization=None,lyapounov=None,timing=None):
  params_string = 'Disorder type                   = '+H.disorder_type+'\n'\
                 +'Correlation length              = '+str(H.correlation_length)+'\n'\
                 +'Dimension                       = '+str(H.dimension)+'\n'\
                 +'use MKL random number generator = '+str(H.use_mkl_random)+'\n'
  volume = 1.0
  for i in range(H.dimension):
    volume *= H.tab_dim[i]*H.tab_delta[i]
    params_string += \
                  'Size_'+str(i+1)+'                          = '+str(H.tab_dim[i]*H.tab_delta[i])+'\n'\
                 +'delta_'+str(i+1)+'                         = '+str(H.tab_delta[i])+'\n'\
                 +'N_'+str(i+1)+'                             = '+str(H.tab_dim[i])+'\n'\
                 +'Boundary_Condition_'+str(i+1)+'            = '+H.tab_boundary_condition[i]+'\n'

  params_string += \
                  'Volume                          = '+str(volume)+'\n'
  params_string += \
                  'V0                              = '+str(H.disorder_strength)+'\n'\
                 +'g                               = '+str(H.interaction)+'\n'\
                 +'g_over_volume                   = '+str(H.interaction/volume)+'\n'\
                 +'Number of disorder realizations = '+str(n_config*nprocs)+'\n'\
                 +'Number of processes             = '+str(nprocs)+'\n'\
                 +'Number of realizations per proc = '+str(n_config)+'\n'
  if not initial_state == None:
    params_string += \
                  'Initial state                   = '+initial_state.type+'\n'
    for i in range(H.dimension):
      params_string += \
                  'k_0_'+str(i+1)+'                           = '+str(initial_state.tab_k_0[i])+'\n'
      if initial_state.type == 'gaussian_wave_packet':
        params_string += \
                 'sigma_0_'+str(i+1)+'                       = '+str(initial_state.tab_sigma_0[i])+'\n'
  if not propagation == None:
    params_string += \
                  'Integration Method              = '+propagation.method+'\n'\
                 +'accuracy                        = '+str(propagation.accuracy)+'\n'
    if propagation.method=='che':
      params_string += \
                  'accurate spectrum bounds        = '+str(propagation.accurate_bounds)+'\n'\
                 +'use ctypes implementation       = '+str(propagation.use_ctypes)+'\n'
      if not timing==None:
        params_string += \
                  'maximum Chebyshev order         = '+str(timing.MAX_CHE_ORDER)+'\n'\
                 +'maximum non-linear phase        = '+str(timing.MAX_NONLINEAR_PHASE)+'\n'
    params_string += \
                  'data layout                     = '+propagation.data_layout+'\n'\
                 +'time step                       = '+str(propagation.delta_t)+'\n'\
                 +'total time                      = '+str(propagation.t_max)+'\n'
  if not measurement == None:
    params_string += \
                  'time step for measurement       = '+str(measurement.delta_t_measurement)+'\n'
    if measurement.measure_overlap:
      params_string += \
                  '|overlap|**2 with initial state = '+str(abs(measurement.overlap)**2)+'\n'
  if not spectral_function == None:
    params_string += \
                  'energy range                    = '+str(spectral_function.e_range)+'\n'\
                 +'energy resolution               = '+str(spectral_function.e_resolution)+'\n'
  if not diagonalization == None:
    params_string += \
                  'targeted_energy                 = '+str(diagonalization.targeted_energy)+'\n'\
                 +'diagonalization method          = '+diagonalization.method+'\n'\
                 +'number of computed eigenvalues  = '+str(diagonalization.number_of_eigenvalues)+'\n'
  if not lyapounov == None:
    params_string += \
                  'minimum energy                  = '+str(lyapounov.e_min)+'\n'\
                 +'maximum energy                  = '+str(lyapounov.e_max)+'\n'\
                 +'energy step                     = '+str(lyapounov.e_step)+'\n'\
                 +'number of energy steps          = '+str(lyapounov.number_of_e_steps)+'\n'
  params_string += '\n'
#  print(params_string)
#                 +'first measurement step          = '+str(my_first_measurement_autocorr)+'\n'\
#                  +'total measurement time          = '+str(my_t_max-my_first_measurement_autocorr*my_delta_t_measurement)+'\n'\
  return params_string

"""
def output_density(file,position,density,general_string,print_type='density'):
#  print(position)
#  print(density.shape)
  if print_type in ['density','density_momentum','spectral_function','IPR','histogram_IPR','lyapounov','histogram_lyapounov','potential']:
    if density.ndim==1:
      array_to_print=np.column_stack((position,density))
      number_of_arrays=1
    else:
      number_of_arrays=density.shape[0]
      array_to_print=np.hstack((position.reshape((position.size,1)),np.transpose(density)))
#  array_to_print = np.concatenate((position.reshape((position.size,1)),np.stack((density[i,:] for i in range(number_of_arrays)),axis=-1)),axis=1)
    if print_type=='density':
      specific_string='Spatial density\n'\
                   +'Column 1: Position\n'\
                   +'Column 2: Density\n'
    if print_type=='density_momentum':
      specific_string='Density in momentum space\n'\
                   +'Column 1: Momentum\n'\
                   +'Column 2: Density\n'
    if print_type=='spectral_function':
      specific_string='Spectral function\n'\
                   +'Column 1: Energy\n'\
                   +'Column 2: Spectral function\n'
    if print_type=='IPR':
      specific_string='Inverse participation ratio for the eigenstate closer to the targeted energy\n'\
                   +'Average IPR           = '+str(np.mean(position))+'\n'\
                   +'Std. deviation of IPR = '+str(np.std(position))+'\n'\
                   +'Column 1: IPR\n'\
                   +'Column 2: Energy of the state (should be very close to the targeted energy)\n'
    if print_type=='histogram_IPR':
      specific_string='Distribution of the inverse participation ratio for the eigenstate closer to the targeted energy\n'\
                   +'Column 1: Right bin edge\n'\
                   +'Column 2: Normalized distribution\n'
    if print_type=='lyapounov':
      specific_string='Lyapounov exponent for the wavefunction (double this value for the Lyapounov exponent for intensity)\n'\
                   +'Column 1: Energy\n'\
                   +'Column 2: Mean value  of the Lyapounov exponent\n'
    if print_type=='histogram_lyapounov':
      specific_string='Distribution of the Lyapounov exponent for the wavefunction (double this value for the Lyapounov exponent for intensity)\n'\
                   +'Column 1: Right bin edge\n'\
                   +'Column 2: Normalized distribution\n'
    if print_type=='potential':
      specific_string='Disordered potential\n'\
                   +'Column 1: Position\n'\
                   +'Column 2: Potential\n'
    if number_of_arrays == 2:
      specific_string+='Column 3: Standard deviation\n'
  if print_type in ['wavefunction','wavefunction_momentum','autocorrelation']:
    array_to_print=np.column_stack((position,np.real(density),np.imag(density)))
#    print(array_to_print.shape)
    if print_type=='wavefunction':
      specific_string='Wavefunction in configuration space\n'\
                     +'Column 1: Position\n'\
                     +'Column 2: Re(Wavefunction)\n'\
                     +'Column 3: Im(Wavefunction)\n'
    if print_type=='wavefunction_momentum':
      specific_string='Wavefunction in momenum space\n'\
                     +'Column 1: Momentum\n'\
                     +'Column 2: Re(Wavefunction)\n'\
                     +'Column 3: Im(Wavefunction)\n'
    if print_type=='autocorrelation':
      specific_string='Temporal autocorrelation function\n'\
                     +'Column 1: Time\n'\
                     +'Column 2: Real(<psi(0)|psi(t)>)\n'\
                     +'Column 3: Imag(<psi(0)|psi(t)>)\n'
  if print_type in ['wavefunction_eigenstate']:
    array_to_print=np.column_stack((position,density))
#    print(array_to_print.shape)
    specific_string='Wavefunction in configuration space\n'\
                     +'Column 1: Position\n'\
                     +'From Column 2: Wavefunction for the various eigenstates\n'
  np.savetxt(file,array_to_print,header=general_string+specific_string)
  return
"""

def output_density(file,data,H,header_string='Origin of data not specified',data_type='density',tab_abscissa=[],file_type='savetxt'):
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
    list_of_columns = []
    tab_strings = []
#    print(specific_string)
    next_column = 1
    dimension = H.dimension
    if data_type in ['density','density_momentum']:
#      print(data.ndim,data.shape)
# The simple case where there is only 1d data
      if dimension==1:
        if data_type=='density':
          header_string=str(H.tab_dim[0])+' '+str(H.tab_delta[0])+'\n'+header_string
        if data_type=='density_momentum':
          header_string=str(H.tab_dim[0])+' '+str(2.0*np.pi/(H.tab_dim[0]*H.tab_delta[0]))+'\n'+header_string
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
          header_string=str(H.tab_dim[0])+' '+str(H.tab_delta[0])+'\n'\
                       +str(H.tab_dim[1])+' '+str(H.tab_delta[1])+'\n'\
                       +header_string
        if data_type=='density_momentum':
          header_string=str(H.tab_dim[0])+' '+str(2.0*np.pi/(H.tab_dim[0]*H.tab_delta[0]))+'\n'\
                       +str(H.tab_dim[1])+' '+str(2.0*np.pi/(H.tab_dim[1]*H.tab_delta[1]))+'\n'\
                       +header_string
        if data.ndim==2:
          array_to_print = data[:,:]
        if data.ndim==3:
          array_to_print=data[0,:,:]
    if data_type in ['wavefunction','wavefunction_momentum','g1']:
      if dimension==1:
        if data_type=='wavefunction' or data_type=='g1':
          header_string=str(H.tab_dim[0])+' '+str(H.tab_delta[0])+'\n'+header_string
        if data_type=='wavefunction_momentum':
          header_string=str(H.tab_dim[0])+' '+str(2.0*np.pi/(H.tab_dim[0]*H.tab_delta[0]))+'\n'+header_string
#        print(data.size,tab_abscissa[0].size)
        if tab_abscissa!=[] and data.size==tab_abscissa[0].size:
          if data_type=='g1':
            list_of_columns.append(tab_abscissa[0]-0.5*H.tab_delta[0])
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
          header_string=str(H.tab_dim[0])+' '+str(H.tab_delta[0])+'\n'\
                       +str(H.tab_dim[1])+' '+str(H.tab_delta[1])+'\n'\
                       +header_string
#          print(header_string)
        if data_type=='wavefunction_momentum':
          header_string=str(H.tab_dim[0])+' '+str(2.0*np.pi/(H.tab_dim[0]*H.tab_delta[0]))+'\n'\
                       +str(H.tab_dim[1])+' '+str(2.0*np.pi/(H.tab_dim[1]*H.tab_delta[1]))+'\n'\
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
    if data_type in ['spectral_function','histogram_IPR','rbar','histogram_r']:
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

