#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 11:22:51 2020

@author: delande
"""

import numpy as np

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
                 +'use CFFI implementation         = '+str(propagation.use_cffi)+'\n'
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
  if not spectral_function == None:
    params_string += \
                  'energy range                    = '+str(spectral_function.e_range)+'\n'\
                 +'energy resolution               = '+str(spectral_function.e_resolution)+'\n'
  if not diagonalization == None:
    params_string += \
                  'targeted_energy                 = '+str(diagonalization.targeted_energy)+'\n'\
                 +'diagonalization method          = '+diagonalization.method+'\n'
  if not lyapounov == None:
    params_string += \
                  'minimum energy                  = '+str(lyapounov.e_min)+'\n'\
                 +'maximum energy                  = '+str(lyapounov.e_max)+'\n'\
                 +'energy step                     = '+str(lyapounov.e_step)+'\n'\
                 +'number of energy steps          = '+str(lyapounov.number_of_e_steps)+'\n'
  params_string += '\n'
#    print(params_string)
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
    list_of_columns = []
    tab_strings = []
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
    if data_type in ['spectral_function']:
      list_of_columns.append(tab_abscissa)
      tab_strings.append('Column '+str(next_column)+': '+column_1)
      next_column += 1
      list_of_columns.append(data)
      tab_strings.append('Column '+str(next_column)+': '+column_2)
      next_column += 1
      array_to_print=np.column_stack(list_of_columns)
#    print(list_of_columns)
#    print(tab_strings)
    np.savetxt(file,array_to_print,header=header_string+specific_string+'\n'.join(tab_strings)+'\n')
  return

def output_dispersion(file,tab_data,tab_strings,general_string='Origin of data not specified'):
#  print('\n'.join(tab_strings))
  np.savetxt(file,tab_data,header=general_string+'\n'.join(tab_strings)+'\n')
  return

