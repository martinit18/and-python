#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 11:45:37 2020

@author: delande
"""

import numpy as np
import timeit
import copy
from anderson.geometry import Geometry

class Measurement(Geometry):
  def __init__(self, geometry, delta_t_dispersion, delta_t_density, delta_t_spectral_function, i_tab_0=0, teta_measurement=0.0,  measure_density=False, measure_density_momentum=False, measure_autocorrelation=False, measure_dispersion_position=False, measure_dispersion_position2=False, measure_dispersion_momentum=False, measure_dispersion_energy=False,measure_wavefunction=False, measure_wavefunction_momentum=False, measure_extended=False, measure_g1=False, measure_overlap=False, measure_spectral_function=False, use_mkl_fft=True, remove_hot_pixel=False):
    super().__init__(geometry.dimension,geometry.tab_dim,geometry.tab_delta, spin_one_half=geometry.spin_one_half)
    self.delta_t_dispersion = delta_t_dispersion
    self.delta_t_density = delta_t_density
    self.delta_t_spectral_function = delta_t_spectral_function
    self.i_tab_0 = i_tab_0
    self.teta_measurement = teta_measurement
    self.measure_density = measure_density
    self.measure_density_momentum = measure_density_momentum
    self.measure_autocorrelation = measure_autocorrelation
    self.measure_dispersion_position = measure_dispersion_position
    self.measure_dispersion_position2 = measure_dispersion_position2
    self.measure_dispersion_momentum = measure_dispersion_momentum
    self.measure_dispersion_energy = measure_dispersion_energy
    self.measure_wavefunction = measure_wavefunction
    self.measure_wavefunction_momentum = measure_wavefunction_momentum
    self.extended = measure_extended
    self.measure_g1 = measure_g1
    self.measure_overlap = measure_overlap
    self.measure_spectral_function = measure_spectral_function
    self.use_mkl_fft = use_mkl_fft
    self.remove_hot_pixel = remove_hot_pixel
    return

  def prepare_measurement(self,propagation,is_spectral_function=False, is_inner_spectral_function=False,spectral_function=None, global_measurement=False):
    delta_t = propagation.delta_t
    t_max = propagation.t_max
    """
    how_often_to_measure = int(self.delta_t_measurement/delta_t+0.5)
    propagation.delta_t = self.delta_t_measurement/how_often_to_measure
    number_of_measurements = int(t_max/self.delta_t_measurement+1.99999)
    number_of_time_steps = int(t_max/delta_t+0.99999)
    self.tab_i_measurement = np.arange(start=0,stop=number_of_measurements*how_often_to_measure,step=how_often_to_measure,dtype=int)
    self.tab_t_measurement = delta_t*self.tab_i_measurement
# correct the last time
    self.tab_i_measurement[number_of_measurements-1]=number_of_time_steps
    self.tab_t_measurement[number_of_measurements-1]=t_max
    """
    if is_spectral_function:
      self.tab_time = np.zeros((spectral_function.n_pts_autocorr+1,4))
      self.tab_time[:,0] = delta_t*np.arange(spectral_function.n_pts_autocorr+1)
      self.tab_time[:,1] = 1.0
      self.tab_t_measurement_dispersion = delta_t*np.arange(spectral_function.n_pts_autocorr+1)
      self.tab_autocorrelation = np.zeros(spectral_function.n_pts_autocorr+1,dtype=np.complex128)
      self.tab_energies = np.fft.fftshift(np.fft.fftfreq(2*spectral_function.n_pts_autocorr+1,d=spectral_function.delta_t/(2.0*np.pi)))+0.5*(spectral_function.e_max+spectral_function.e_min)
      self.tab_t_measurement_spectral_function = np.array([0.0])
      if not is_inner_spectral_function:
        self.tab_spectrum = np.zeros(2*spectral_function.n_pts_autocorr+1)
#    tab_time[0:dim_tab_time_propagation,1] = 1
    else:
      dim_tab_time_propagation = int(t_max/delta_t+0.999)
      dim_tab_time_dispersion = int(t_max/self.delta_t_dispersion+0.999)
      number_of_measurements_dispersion = dim_tab_time_dispersion+1
      dim_tab_time_density = int(t_max/self.delta_t_density+0.999)
      number_of_measurements_density = dim_tab_time_density+1
      dim_tab_time_spectral_function = int(t_max/self.delta_t_spectral_function+0.999)
      number_of_measurements_spectral_function = dim_tab_time_spectral_function+1
#      print(t_max,self.delta_t_spectral_function)
#      print('Number of measurements of the spectral function = ',number_of_measurements_spectral_function)
      tab_time = np.zeros((1+dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function,4))
      tab_time[0:dim_tab_time_propagation,0] = delta_t*np.arange(dim_tab_time_propagation)
  #    tab_time[0:dim_tab_time_propagation,1] = 1
      tab_time[dim_tab_time_propagation:dim_tab_time_propagation+dim_tab_time_dispersion,0] = self.delta_t_dispersion*np.arange(dim_tab_time_dispersion)
      tab_time[dim_tab_time_propagation:dim_tab_time_propagation+dim_tab_time_dispersion,1] = 1
      tab_time[dim_tab_time_propagation+dim_tab_time_dispersion:dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density,0] = self.delta_t_density*np.arange(dim_tab_time_density)
      tab_time[dim_tab_time_propagation+dim_tab_time_dispersion:dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density,2] = 1
      tab_time[dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density:dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function,0] = self.delta_t_spectral_function*np.arange(dim_tab_time_spectral_function)
      tab_time[dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density:dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function,3] = 1
      tab_time[dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function,0] = t_max
      tab_time[dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function,1:4] = 1
  #    print(tab_time)
  # sort by increasing time
      tab_time = tab_time[np.argsort(tab_time[:, 0])]
  #    print(tab_time)
  # merge when several times (quasi-)coincide
      j=0
      tiny =1.e-12
      tab_time_new = np.zeros((1+dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function,4))
      tab_time_new[0,:] = tab_time[0,:]
      for i in range(dim_tab_time_propagation+dim_tab_time_dispersion+dim_tab_time_density+dim_tab_time_spectral_function):
        if tab_time[i+1,0]-tab_time[i,0]>tiny:
          tab_time_new[j+1,:] = tab_time[i+1,:]
          j+=1
        else:
          tab_time_new[j,1:] += tab_time[i+1,1:]
      self.tab_time = tab_time_new[:j+1,:]
  # For the first and last time, remove the computation of densities
  #    self.tab_time[0,2]=0.0
  #    self.tab_time[-1,2]=0.0
  #    print(tab_time)
  # Select only events where dispersion is measured
      self.tab_t_measurement_dispersion = self.tab_time[self.tab_time[:,1]==1.0,0]
  #    print (self.tab_t_measurement)
      dim_density = self.tab_dim[:]
      dim_dispersion = [number_of_measurements_dispersion]
      dim_dispersion_vec = [self.dimension,number_of_measurements_dispersion]
      if global_measurement:
        if self.extended:
          dim_density.insert(0,2)
          dim_dispersion.insert(0,2)
          dim_dispersion_vec.insert(0,2)
        else:
          dim_density.insert(0,1)
          dim_dispersion.insert(0,1)
          dim_dispersion_vec.insert(0,1)
      if self.measure_density:
        self.density_final = np.zeros(dim_density)
  #      print(self.density_final.shape,tab_dim)
      if self.measure_autocorrelation:
        self.tab_autocorrelation = np.zeros(number_of_measurements_dispersion,dtype=np.complex128)
      if self.measure_density_momentum:
        self.density_momentum_final = np.zeros(dim_density)
      if self.measure_dispersion_position:
        self.tab_position = np.zeros(dim_dispersion_vec)
      if self.measure_dispersion_position2:
        self.tab_position2 = np.zeros(dim_dispersion_vec)
      if self.measure_dispersion_momentum:
        self.tab_momentum = np.zeros(dim_dispersion_vec)
      if self.measure_dispersion_energy:
        self.tab_energy = np.zeros(dim_dispersion)
        self.tab_nonlinear_energy = np.zeros(dim_dispersion)
      if self.measure_wavefunction:
        self.wfc =  np.zeros(self.tab_hs_dim,dtype=np.complex128)
#      if self.measure_wavefunction_momentum:
#        self.wfc_momentum =  np.zeros(self.tab_dim,dtype=np.complex128)
      if self.measure_g1:
        self.g1 =  np.zeros(self.tab_dim,dtype=np.complex128)
        dim_g1 = self.tab_dim[:]
        dim_g1.insert(0,number_of_measurements_density)
        self.g1_intermediate = np.zeros(dim_g1,dtype=np.complex128)
      if self.measure_overlap:
        self.overlap = 0.0
#      print('Measure spectral function = ',self.measure_spectral_function)
      if self.measure_spectral_function:
        self.tab_energies = np.fft.fftshift(np.fft.fftfreq(2*spectral_function.n_pts_autocorr+1,d=spectral_function.delta_t/(2.0*np.pi)))+0.5*(spectral_function.e_max+spectral_function.e_min)
 #       print(number_of_measurements_spectral_function)
        self.tab_spectrum = np.zeros((2*spectral_function.n_pts_autocorr+1,number_of_measurements_spectral_function))
      else:
        self.tab_time[:,3]=0.0
      self.tab_t_measurement_spectral_function = self.tab_time[self.tab_time[:,3]==1.0,0]
  # What follows is the code for the intermediate times
      self.tab_t_measurement_density = self.tab_time[self.tab_time[:,2]==1.0,0]
  #    print(self.tab_t_measurement_density)
      dim_density.insert(0,number_of_measurements_density)
      if self.measure_density:
        self.density_intermediate = np.zeros(dim_density)
  #      print(dim_density)
      if self.measure_density_momentum:
        self.density_momentum_intermediate = np.zeros(dim_density)
    return


  def merge_measurement(self,measurement):
    if self.measure_density:
#      print(measurement.density_final.shape)
#      print(self.density_final.shape)
      self.density_final[0] += measurement.density_final
      self.density_intermediate[:,0] += measurement.density_intermediate
#      print(self.density_intermediate.shape,measurement.density_intermediate.shape)
#      print('merge 1',self.density_intermediate[:,0]**2)
      if self.extended:
        self.density_final[1] += measurement.density_final**2
        self.density_intermediate[:,1] += measurement.density_intermediate**2
#        print('merge 2',self.density_intermediate[:,1])
    if self.measure_density_momentum:
      self.density_momentum_final[0] += measurement.density_momentum_final
      self.density_momentum_intermediate[:,0] += measurement.density_momentum_intermediate
    if self.extended:
        self.density_momentum_final[1] += measurement.density_momentum_final**2
        self.density_momentum_intermediate[:,1] += measurement.density_momentum_intermediate**2
    if self.measure_autocorrelation:
      self.tab_autocorrelation += measurement.tab_autocorrelation
    if self.measure_dispersion_position:
      self.tab_position[0] += measurement.tab_position
      if self.extended:
        self.tab_position[1] += measurement.tab_position**2
    if self.measure_dispersion_position2:
      self.tab_position2[0] += measurement.tab_position2
      if self.extended:
        self.tab_position2[1] += measurement.tab_position2**2
    if self.measure_dispersion_momentum:
      self.tab_momentum[0] += measurement.tab_momentum
      if self.extended:
        self.tab_momentum[1] += measurement.tab_momentum**2
    if self.measure_dispersion_energy:
      self.tab_energy[0] += measurement.tab_energy
      self.tab_nonlinear_energy[0] += measurement.tab_nonlinear_energy
      if self.extended:
        self.tab_energy[1] += measurement.tab_energy**2
        self.tab_nonlinear_energy[1] += measurement.tab_nonlinear_energy**2
    if self.measure_wavefunction:
      self.wfc += measurement.wfc
#    if self.measure_wavefunction_momentum:
#      self.wfc_momentum += measurement.wfc_momentum
    if self.measure_g1:
      self.g1 +=  measurement.g1
      self.g1_intermediate += measurement.g1_intermediate
    if self.measure_overlap:
      self.overlap += measurement.overlap
    if self.measure_spectral_function:
      self.tab_spectrum += measurement.tab_spectrum
    return

  def mpi_merge_measurement(self,comm,timing):
    start_mpi_time = timeit.default_timer()
    try:
      from mpi4py import MPI
    except ImportError:
      print("mpi4py is not found!")
      return
    if self.measure_density:
      toto = np.empty_like(self.density_final)
      comm.Reduce(self.density_final,toto)
      self.density_final = np.copy(toto)
      toto = np.empty_like(self.density_intermediate)
      comm.Reduce(self.density_intermediate,toto)
      self.density_intermediate = np.copy(toto)
    if self.measure_density_momentum:
      toto = np.empty_like(self.density_momentum_final)
      comm.Reduce(self.density_momentum_final,toto)
      self.density_momentum_final = np.copy(toto)
      toto = np.empty_like(self.density_momentum_intermediate)
      comm.Reduce(self.density_momentum_intermediate,toto)
      self.density_momentum_intermediate = np.copy(toto)
    if self.measure_autocorrelation:
      toto = np.empty_like(self.tab_autocorrelation)
      comm.Reduce(self.tab_autocorrelation,toto)
      self.tab_autocorrelation = np.copy(toto)
    if self.measure_dispersion_position:
      toto = np.empty_like(self.tab_position)
      comm.Reduce(self.tab_position,toto)
      self.tab_position = np.copy(toto)
    if self.measure_dispersion_position2:
      toto = np.empty_like(self.tab_position2)
      comm.Reduce(self.tab_position2,toto)
      self.tab_position2 =  np.copy(toto)
    if self.measure_dispersion_momentum:
      toto = np.empty_like(self.tab_momentum)
      comm.Reduce(self.tab_momentum,toto)
      self.tab_momentum = np.copy(toto)
    if self.measure_dispersion_energy:
      toto = np.empty_like(self.tab_energy)
      comm.Reduce(self.tab_energy,toto)
      self.tab_energy =  np.copy(toto)
      comm.Reduce(self.tab_nonlinear_energy,toto)
      self.tab_nonlinear_energy = np.copy(toto)
    if self.measure_wavefunction:
      toto = np.empty_like(self.wfc)
      comm.Reduce(self.wfc,toto)
      self.wfc = np.copy(toto)
#    if self.measure_wavefunction_momentum:
#      toto = np.empty_like(self.wfc_momentum)
#      comm.Reduce(self.wfc_momentum,toto)
#      self.wfc_momentum = np.copy(toto)
    if self.measure_g1:
      toto = np.empty_like(self.g1)
      comm.Reduce(self.g1,toto)
      self.g1 = np.copy(toto)
      toto = np.empty_like(self.g1_intermediate)
      comm.Reduce(self.g1_intermediate,toto)
      self.g1_intermediate = np.copy(toto)
    if self.measure_overlap:
      toto = np.zeros(1,dtype=np.complex128)
      comm.Reduce(self.overlap,toto)
      self.overlap = toto[0]
    if self.measure_spectral_function:
      toto = np.empty_like(self.tab_spectrum)
      comm.Reduce(self.tab_spectrum,toto)
      self.tab_spectrum = np.copy(toto)
    timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
    return

  def normalize(self,n_config):
    if self.measure_density:
      self.density_final /= n_config
      self.density_intermediate /= n_config
      if self.extended:
        self.density_final[1] = np.sqrt(np.abs(self.density_final[1]-self.density_final[0]**2)/n_config)
        self.density_intermediate[:,1] = np.sqrt(np.abs(self.density_intermediate[:,1]-self.density_intermediate[:,0]**2)/n_config)
    if self.measure_density_momentum:
      self.density_momentum_final /= n_config
      self.density_momentum_intermediate /= n_config
      if self.extended:
        self.density_momentum_final[1] = np.sqrt(np.abs(self.density_momentum_final[1]-self.density_momentum_final[0]**2)/n_config)
        self.density_momentum_intermediate[:,1] = np.sqrt(np.abs(self.density_momentum_intermediate[:,1]-self.density_momentum_intermediate[:,0]**2)/n_config)
    if self.measure_autocorrelation:
      self.tab_autocorrelation /= n_config
    list_of_columns = [self.tab_t_measurement_dispersion]
#    print(list_of_columns)
    tab_strings=['Column 1: Time']
    next_column = 2
    if self.measure_dispersion_position:
      self.tab_position /= n_config
      if self.tab_position.shape[0]==2:
        self.tab_position[1] = np.sqrt(np.abs(self.tab_position[1]-self.tab_position[0]**2)/n_config)
    if self.measure_dispersion_position2:
      self.tab_position2 /= n_config
      if self.tab_position2.shape[0]==2:
        self.tab_position2[1] = np.sqrt(np.abs(self.tab_position2[1]-self.tab_position2[0]**2)/n_config)
    if self.measure_dispersion_position or self.measure_dispersion_position2:
      for i in range(self.dimension):
        if self.measure_dispersion_position:
          list_of_columns.append(self.tab_position[0,i])
          tab_strings.append('Column '+str(next_column)+': <r_'+str(i+1)+'>')
          next_column += 1
          if self.tab_position.shape[0]==2:
            list_of_columns.append(self.tab_position[1,i])
            tab_strings.append('Column '+str(next_column)+': Standard deviation of <r_'+str(i+1)+'>')
            next_column += 1
        if self.measure_dispersion_position2:
          list_of_columns.append(self.tab_position2[0,i])
          tab_strings.append('Column '+str(next_column)+': <r_'+str(i+1)+'^2>')
          next_column += 1
          if self.tab_position2.shape[0]==2:
            list_of_columns.append(self.tab_position2[1,i])
            tab_strings.append('Column '+str(next_column)+': Standard deviation of <r_'+str(i+1)+'^2>')
            next_column += 1
    if self.measure_dispersion_momentum:
      self.tab_momentum /= n_config
      if self.tab_momentum.shape[0]==2:
        self.tab_momentum[1] = np.sqrt(np.abs(self.tab_momentum[1]-self.tab_momentum[0]**2)/n_config)
      for i in range(self.dimension):
        list_of_columns.append(self.tab_momentum[0,i])
        tab_strings.append('Column '+str(next_column)+': <p_'+str(i+1)+'>')
        next_column += 1
        if self.tab_momentum.shape[0]==2:
          list_of_columns.append(self.tab_momentum[1,i])
          tab_strings.append('Column '+str(next_column)+': Standard deviation of <p_'+str(i+1)+'>')
          next_column += 1
    if self.measure_dispersion_energy:
      self.tab_energy /= n_config
      self.tab_nonlinear_energy /= n_config
      list_of_columns.append(self.tab_energy[0])
      tab_strings.append('Column '+str(next_column)+': Total energy')
      next_column += 1
      if self.tab_energy.shape[0]==2:
        self.tab_energy[1] = np.sqrt(np.abs(self.tab_energy[1]-self.tab_energy[0]**2)/n_config)
        list_of_columns.append(self.tab_energy[1])
        tab_strings.append('Column '+str(next_column)+': Standard deviation of total energy')
        next_column += 1
      list_of_columns.append(self.tab_nonlinear_energy[0])
      tab_strings.append('Column '+str(next_column)+': Nonlinear energy')
      next_column += 1
      if self.tab_nonlinear_energy.shape[0]==2:
        self.tab_nonlinear_energy[1] = np.sqrt(np.abs(self.tab_nonlinear_energy[1]-self.tab_nonlinear_energy[0]**2)/n_config)
        list_of_columns.append(self.tab_nonlinear_energy[1])
        tab_strings.append('Column '+str(next_column)+': Standard deviation of nonlinear energy')
        next_column += 1
    if self.measure_wavefunction:
      self.wfc /= n_config
#    if self.measure_wavefunction_momentum:
#      self.wfc_momentum /= n_config
    if self.measure_g1:
      self.g1 /= n_config
      self.g1_intermediate /= n_config
    if self.measure_overlap:
      self.overlap /= n_config
    if self.measure_spectral_function:
      self.tab_spectrum /= n_config
    self.tab_strings = tab_strings
    self.tab_dispersion = np.column_stack(list_of_columns)
#    print(tab_strings)
#    print(list_of_columns)
    return
#    return tab_strings, np.column_stack(list_of_columns)



  def perform_measurement_dispersion(self, i_tab, H, psi, init_state_autocorr):
#    print('teta_measurement = ',self.teta_measurement)
    if self.spin_one_half:
# Select only a single component
      local_psi = copy.deepcopy(psi)
      local_psi.wfc = np.zeros(self.tab_dim,dtype=np.complex128)
      local_psi.wfc[:] = np.cos(self.teta_measurement)*psi.wfc[0::2]+np.sin(self.teta_measurement)*psi.wfc[1::2]
    else:
      local_psi=psi
    if self.measure_dispersion_position or self.measure_dispersion_position2:
      density = local_psi.wfc.real**2+local_psi.wfc.imag**2
      if self.spin_one_half:
        norm = 1.0/H.delta_vol
      else:
        norm = np.sum(density)
      for i in range(psi.dimension):
        local_density = np.sum(density, axis = tuple(j for j in range(local_psi.dimension) if j!=i))
#    print(dim,local_density.shape,local_density)
#        np.savetxt('toto_dispersion.dat',local_density)
        if self.measure_dispersion_position:
          self.tab_position[i,i_tab] = np.sum(local_psi.grid_position[i]*local_density)/norm
        if self.measure_dispersion_position2:
          self.tab_position2[i,i_tab] = np.sum(local_psi.grid_position[i]**2*local_density)/norm
    if self.measure_dispersion_energy:
      self.tab_energy[i_tab], self.tab_nonlinear_energy[i_tab] = psi.energy(H)
    if (self.measure_dispersion_momentum):
      psi_momentum = local_psi.convert_to_momentum_space(self.use_mkl_fft)
      density = psi_momentum.real**2+psi_momentum.imag**2
      if self.spin_one_half:
        norm = 1.0/H.delta_vol
      else:
        norm = np.sum(density)
      for i in range(local_psi.dimension):
        local_density = np.sum(density, axis = tuple(j for j in range(local_psi.dimension) if j!=i))
        self.tab_momentum[i,i_tab] = np.sum(self.frequencies[i]*local_density)/norm
    if self.measure_autocorrelation and i_tab>=self.i_tab_0:
# Inlining the overlap method is slighlty faster
#          measurement.tab_autocorrelation[i_tab-i_tab_0] = psi.overlap(init_state_autocorr)
      self.tab_autocorrelation[i_tab-self.i_tab_0] = np.vdot(init_state_autocorr.wfc,psi.wfc)*H.delta_vol
    return

  def perform_measurement_density(self, i_tab, psi):
#    print(self.spin_one_half,self.teta_measurement)
    if self.spin_one_half:
# Select only a single component
      local_psi = copy.deepcopy(psi)
      local_psi.wfc = np.zeros(self.tab_dim,dtype=np.complex128)
      local_psi.wfc[:] = np.cos(self.teta_measurement)*psi.wfc[0::2]+np.sin(self.teta_measurement)*psi.wfc[1::2]
    else:
      local_psi=psi
    if self.measure_density:
#      print(self.density_intermediate.shape)
      self.density_intermediate[i_tab,:] = local_psi.wfc.real**2+local_psi.wfc.imag**2
#      np.savetxt('toto_density.dat',self.density_intermediate[i_tab])
    if self.measure_density_momentum:
      psi_momentum = local_psi.convert_to_momentum_space(self.use_mkl_fft)
      self.density_momentum_intermediate[i_tab,:] = psi_momentum.real**2+psi_momentum.imag**2
    if self.measure_g1:
      self.g1_intermediate[i_tab,:] = np.fft.fftshift(np.fft.ifftn(np.fft.fftn(local_psi.wfc)*np.conj(np.fft.fftn(local_psi.wfc))))*local_psi.delta_vol
    return

  def perform_measurement_final(self, psi, init_state_autocorr):
#    if self.spin_one_half:
# Select only a single component
#      local_psi = copy.deepcopy(psi)
#      local_psi.wfc = np.zeros(self.tab_dim,dtype=np.complex128)
#      local_psi.wfc[:] = np.cos(self.teta_measurement)*psi.wfc[0::2]+np.sin(self.teta_measurement)*psi.wfc[1::2]
#    else:
#      local_psi=psi
#    if self.measure_density:
#      self.density_final = local_psi.wfc.real**2+local_psi.wfc.imag**2
    if self.measure_wavefunction:
      self.wfc = psi.wfc
#    if self.measure_wavefunction_momentum:
#      self.wfc_momentum = local_psi.convert_to_momentum_space(self.use_mkl_fft)
#    if self.measure_density_momentum:
#      if self.measure_wavefunction_momentum:
#        self.density_momentum_final = self.wfc_momentum.real**2+self.wfc_momentum.imag**2
#      else:
#        psi_momentum = local_psi.convert_to_momentum_space(self.use_mkl_fft)
#        self.density_momentum_final = psi_momentum.real**2+psi_momentum.imag**2
#    if self.measure_g1:
#      self.g1 = #np.fft.fftshift(np.fft.ifftn(np.fft.fftn(local_psi.wfc)*np.conj(np.fft.fftn(local_psi.wfc))))*local_psi.delta_vol
    if self.measure_overlap:
      self.overlap = np.vdot(init_state_autocorr.wfc,psi.wfc)*psi.delta_vol
#      print(self.overlap)
    return

