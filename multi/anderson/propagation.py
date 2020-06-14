#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:33:12 2019

@author: delande
"""

import math
import numpy as np
from scipy.integrate import ode
import scipy.special as sp
import anderson
import timeit

class Temporal_Propagation:
  def __init__(self, t_max, delta_t, method='che', data_layout='real'):
    self.t_max = t_max
    self.method = method
    self.data_layout = data_layout
    self.delta_t = delta_t
    self.script_delta_t = 0.
    self.accuracy = 0.0
    return

  def compute_chebyshev_coefficients(self,accuracy,timing):
    assert self.script_delta_t!=0.,"Rescaled time step for Chebyshev propagation is zero!"
    i = int(2.0*self.script_delta_t)+10
    while (abs(sp.jv(i,self.script_delta_t))<accuracy or i==0):
      i -= 1
# max order must be an even number
    max_order = 2*((i+1)//2)
    if (max_order>timing.MAX_CHE_ORDER):
      timing.MAX_CHE_ORDER=max_order
#  print(max_order)
#  tab_coef=np.zeros(max_order+1)
    z=np.arange(max_order+1)
    self.tab_coef=-4.0*sp.jv(z,-self.script_delta_t)*(((z%4)>1)-0.5)
    self.tab_coef[0] *=  0.5
    self.accuracy = accuracy
# The three previous lines are equivalent to the following loop
#  tab_coef=np.zeros(max_order+1)
#  for order in range(1,max_order):
#    tab_coef[order] = -4.0*sp.jv(order,-delta_t)*(((order%4)>1)-0.5)
# The following uses mpmath but is awfully slow
#   tab_coef[order] = -4.0*mpmath.besselj(order,-delta_t)*(((order%4)>1)-0.5)
#  tab_coef[0] = sp.jv(0,-delta_t)
    return

def apply_minus_i_h_gpe_complex(wfc, H, rhs):
  dim_x = H.dim_x
  if (H.interaction == 0.0):
    H.diagonal_term=H.disorder
  else:
    H.diagonal_term=H.disorder+H.interaction*(np.abs(wfc)**2)
  if H.boundary_condition=='periodic':
    rhs[0]       = 1j * (H.tunneling * (wfc[dim_x-1] + wfc[1]) - H.diagonal_term[0] * wfc[0])
    rhs[dim_x-1] = 1j * (H.tunneling * (wfc[dim_x-2] + wfc[0]) - H.diagonal_term[dim_x-1] * wfc[dim_x-1])
  else:
    rhs[0]       = 1j * (H.tunneling * wfc[1]       - H.diagonal_term[0]       * wfc[0])
    rhs[dim_x-1] = 1j * (H.tunneling * wfc[dim_x-2] - H.diagonal_term[dim_x-1] * wfc[dim_x-1])
  rhs[1:dim_x-1] = 1j * (H.tunneling * (wfc[0:dim_x-2] + wfc[2:dim_x]) - H.diagonal_term[1:dim_x-1] * wfc[1:dim_x-1])
  return rhs

def apply_minus_i_h_gpe_real(wfc, H, rhs):
  dim_x = H.dim_x
  if (H.interaction == 0.0):
    H.diagonal_term=H.disorder
  else:
    H.diagonal_term=H.disorder+H.interaction*(wfc[0:dim_x]**2+wfc[dim_x:2*dim_x]**2)
  if H.boundary_condition=='periodic':
    rhs[0]         = -H.tunneling * (wfc[2*dim_x-1] + wfc[dim_x+1]) + H.diagonal_term[0]       * wfc[dim_x]
    rhs[dim_x]     =  H.tunneling * (wfc[dim_x-1]   + wfc[1])       - H.diagonal_term[0]       * wfc[0]
    rhs[dim_x-1]   = -H.tunneling * (wfc[2*dim_x-2] + wfc[dim_x])   + H.diagonal_term[dim_x-1] * wfc[2*dim_x-1]
    rhs[2*dim_x-1] =  H.tunneling * (wfc[dim_x-2]   + wfc[0])       - H.diagonal_term[dim_x-1] * wfc[dim_x-1]
  else:
    rhs[0]         = -H.tunneling *  wfc[dim_x+1]   + H.diagonal_term[0]       * wfc[dim_x]
    rhs[dim_x]     =  H.tunneling *  wfc[1]         - H.diagonal_term[0]       * wfc[0]
    rhs[dim_x-1]   = -H.tunneling *  wfc[2*dim_x-2] + H.diagonal_term[dim_x-1] * wfc[2*dim_x-1]
    rhs[2*dim_x-1] =  H.tunneling *  wfc[dim_x-2]   - H.diagonal_term[dim_x-1] * wfc[dim_x-1]
  rhs[1:dim_x-1]         = -H.tunneling * (wfc[dim_x:2*dim_x-2] + wfc[dim_x+2:2*dim_x]) + H.diagonal_term[1:dim_x-1]*wfc[dim_x+1:2*dim_x-1]
  rhs[dim_x+1:2*dim_x-1] =  H.tunneling * (wfc[0:dim_x-2]       + wfc[2:dim_x])         - H.diagonal_term[1:dim_x-1]*wfc[1:dim_x-1]
  return rhs

def elementary_clenshaw_step_complex(wfc, H, psi, psi_old, c_coef, one_or_two, add_real):
  dim_x = H.dim_x
  if (add_real):
    if H.boundary_condition=='periodic':
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*(psi[1]+psi[dim_x-1]))+c_coef*wfc[0]-psi_old[0]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*(psi[0]+psi[dim_x-2]))+c_coef*wfc[dim_x-1]-psi_old[dim_x-1]
    else:
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*psi[1])+c_coef*wfc[0]-psi_old[0]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*psi[dim_x-2])+c_coef*wfc[dim_x-1]-psi_old[dim_x-1]
    psi_old[1:dim_x-1]=one_or_two*(H.script_disorder[1:dim_x-1]*psi[1:dim_x-1]-H.script_tunneling*(psi[2:dim_x]+psi[0:dim_x-2]))+c_coef*wfc[1:dim_x-1]-psi_old[1:dim_x-1]
  else:
    if H.boundary_condition=='periodic':
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*(psi[1]+psi[dim_x-1]))+1j*c_coef*wfc[0]-psi_old[0]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*(psi[0]+psi[dim_x-2]))+1j*c_coef*wfc[dim_x-1]-psi_old[dim_x-1]
    else:
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*psi[1])+1j*c_coef*wfc[0]-psi_old[0]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*psi[dim_x-2])+1j*c_coef*wfc[dim_x-1]-psi_old[dim_x-1]
    psi_old[1:dim_x-1]=one_or_two*(H.script_disorder[1:dim_x-1]*psi[1:dim_x-1]-H.script_tunneling*(psi[2:dim_x]+psi[0:dim_x-2]))+1j*c_coef*wfc[1:dim_x-1]-psi_old[1:dim_x-1]
  return

def elementary_clenshaw_step_real(wfc, H, psi, psi_old, c_coef, one_or_two, add_real):
  dim_x = H.dim_x
  if (add_real):
    if H.boundary_condition=='periodic':
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*(psi[1]+psi[dim_x-1]))+c_coef*wfc[0]-psi_old[0]
      psi_old[dim_x]=one_or_two*(H.script_disorder[0]*psi[dim_x]-H.script_tunneling*(psi[dim_x+1]+psi[2*dim_x-1]))+c_coef*wfc[dim_x]-psi_old[dim_x]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*(psi[0]+psi[dim_x-2]))+c_coef*wfc[dim_x-1]-psi_old[dim_x-1]
      psi_old[2*dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[2*dim_x-1]-H.script_tunneling*(psi[dim_x]+psi[2*dim_x-2]))+c_coef*wfc[2*dim_x-1]-psi_old[2*dim_x-1]
    else:
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*psi[1])+c_coef*wfc[0]-psi_old[0]
      psi_old[dim_x]=one_or_two*(H.script_disorder[0]*psi[dim_x]-H.script_tunneling*psi[dim_x+1])+c_coef*wfc[dim_x]-psi_old[dim_x]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*psi[dim_x-2])+c_coef*wfc[dim_x-1]-psi_old[dim_x-1]
      psi_old[2*dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[2*dim_x-1]-H.script_tunneling*psi[2*dim_x-2])+c_coef*wfc[2*dim_x-1]-psi_old[2*dim_x-1]
    psi_old[1:dim_x-1]=one_or_two*(H.script_disorder[1:dim_x-1]*psi[1:dim_x-1]-H.script_tunneling*(psi[2:dim_x]+psi[0:dim_x-2]))+c_coef*wfc[1:dim_x-1]-psi_old[1:dim_x-1]
    psi_old[dim_x+1:2*dim_x-1]=one_or_two*(H.script_disorder[1:dim_x-1]*psi[dim_x+1:2*dim_x-1]-H.script_tunneling*(psi[dim_x+2:2*dim_x]+psi[dim_x:2*dim_x-2]))+c_coef*wfc[dim_x+1:2*dim_x-1]-psi_old[dim_x+1:2*dim_x-1]
  else:
    if H.boundary_condition=='periodic':
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*(psi[1]+psi[dim_x-1]))-c_coef*wfc[dim_x]-psi_old[0]
      psi_old[dim_x]=one_or_two*(H.script_disorder[0]*psi[dim_x]-H.script_tunneling*(psi[dim_x+1]+psi[2*dim_x-1]))+c_coef*wfc[0]-psi_old[dim_x]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*(psi[0]+psi[dim_x-2]))-c_coef*wfc[2*dim_x-1]-psi_old[dim_x-1]
      psi_old[2*dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[2*dim_x-1]-H.script_tunneling*(psi[dim_x]+psi[2*dim_x-2]))+c_coef*wfc[dim_x-1]-psi_old[2*dim_x-1]
    else:
      psi_old[0]=one_or_two*(H.script_disorder[0]*psi[0]-H.script_tunneling*psi[1])-c_coef*wfc[dim_x]-psi_old[0]
      psi_old[dim_x]=one_or_two*(H.script_disorder[0]*psi[dim_x]-H.script_tunneling*psi[dim_x+1])+c_coef*wfc[0]-psi_old[dim_x]
      psi_old[dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[dim_x-1]-H.script_tunneling*psi[dim_x-2])-c_coef*wfc[2*dim_x-1]-psi_old[dim_x-1]
      psi_old[2*dim_x-1]=one_or_two*(H.script_disorder[dim_x-1]*psi[2*dim_x-1]-H.script_tunneling*psi[2*dim_x-2])+c_coef*wfc[dim_x-1]-psi_old[2*dim_x-1]
    psi_old[1:dim_x-1]=one_or_two*(H.script_disorder[1:dim_x-1]*psi[1:dim_x-1]-H.script_tunneling*(psi[2:dim_x]+psi[0:dim_x-2]))-c_coef*wfc[dim_x+1:2*dim_x-1]-psi_old[1:dim_x-1]
    psi_old[dim_x+1:2*dim_x-1]=one_or_two*(H.script_disorder[1:dim_x-1]*psi[dim_x+1:2*dim_x-1]-H.script_tunneling*(psi[dim_x+2:2*dim_x]+psi[dim_x:2*dim_x-2]))+c_coef*wfc[1:dim_x-1]-psi_old[dim_x+1:2*dim_x-1]
  return

"""
The two routines chebyshev_step_clenshaw_python and chebyshev_step_clenshaw_cffi should be completely equivalent
The first use pure Python, the second one uses a C code and cffi (roughly 10 times faster)
"""

def chebyshev_step_clenshaw_python(wfc, H, propagation,timing):
  dim_x = H.dim_x
  max_order = propagation.tab_coef.size-1
  assert max_order%2==0,"Max order {} must be an even number".format(max_order)
  if propagation.data_layout == 'real':
    elementary_clenshaw_step_routine = elementary_clenshaw_step_real
    psi_old = np.zeros(2*dim_x)
  else:
    elementary_clenshaw_step_routine = elementary_clenshaw_step_complex
    psi_old = np.zeros(dim_x,dtype=np.complex128)
  psi = propagation.tab_coef[-1] * wfc
  elementary_clenshaw_step_routine(wfc, H, psi, psi_old, propagation.tab_coef[-2], 2.0, 0)
  for order in range(propagation.tab_coef.size-3,0,-2):
    elementary_clenshaw_step_routine(wfc, H, psi_old, psi, propagation.tab_coef[order], 2.0, 1)
    elementary_clenshaw_step_routine(wfc, H, psi, psi_old, propagation.tab_coef[order-1], 2.0, 0)
  elementary_clenshaw_step_routine(wfc, H, psi_old, psi, propagation.tab_coef[0], 1.0, 1)
  if H.interaction==0.0:
    phase = propagation.delta_t*H.medium_energy
    cos_phase = math.cos(phase)
    sin_phase = math.sin(phase)
  else:
    if propagation.data_layout == 'real':
      nonlinear_phase = propagation.delta_t*H.interaction*(psi[0:dim_x]**2+psi[dim_x:2*dim_x]**2)
    else:
      nonlinear_phase = propagation.delta_t*H.interaction*(np.real(psi)**2+np.imag(psi)**2)
    timing.MAX_NONLINEAR_PHASE = max(timing.MAX_NONLINEAR_PHASE,np.amax(nonlinear_phase))
    phase=propagation.delta_t*H.medium_energy+nonlinear_phase
    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
  if propagation.data_layout == 'real':
      wfc[0:dim_x] = psi[0:dim_x]*cos_phase+psi[dim_x:2*dim_x]*sin_phase
      wfc[dim_x:2*dim_x] = psi[dim_x:2*dim_x]*cos_phase-psi[0:dim_x]*sin_phase
  else:
      wfc[:] = psi[:] * (cos_phase-1j*sin_phase)
#  print(psi[2000],wfc[2000])
  return

def chebyshev_step_clenshaw_cffi(wfc, H, propagation,timing):
  from anderson._chebyshev import ffi,lib
  dim_x = H.dim_x
  max_order = propagation.tab_coef.size-1
  nonlinear_phase = ffi.new("double *", timing.MAX_NONLINEAR_PHASE)
  assert max_order%2==0,"Max order {} must be an even number".format(max_order)
  if propagation.data_layout=='real':
    psi_old = np.zeros(2*dim_x)
    psi = np.zeros(2*dim_x)
    lib.chebyshev_clenshaw_real(dim_x,max_order,str.encode(H.boundary_condition),ffi.cast('double *',ffi.from_buffer(wfc)),ffi.cast('double *',ffi.from_buffer(psi)),ffi.cast('double *',ffi.from_buffer(psi_old)),H.script_tunneling,ffi.cast('double *',ffi.from_buffer(H.script_disorder)),ffi.cast('double *',ffi.from_buffer(propagation.tab_coef)),H.interaction*propagation.delta_t,H.medium_energy*propagation.delta_t,nonlinear_phase)
  else:
    psi_old = np.zeros(dim_x,dtype=np.complex128)
    psi = np.zeros(dim_x,dtype=np.complex128)
#    print('Entering toto_che_complex')
    lib.chebyshev_clenshaw_complex(dim_x,max_order,str.encode(H.boundary_condition),ffi.cast('double _Complex *',ffi.from_buffer(wfc)),ffi.cast('double _Complex *',ffi.from_buffer(psi)),ffi.cast('double _Complex *',ffi.from_buffer(psi_old)),H.script_tunneling,ffi.cast('double *',ffi.from_buffer(H.script_disorder)),ffi.cast('double *',ffi.from_buffer(propagation.tab_coef)),H.interaction*propagation.delta_t,H.medium_energy*propagation.delta_t,nonlinear_phase)
  timing.MAX_NONLINEAR_PHASE = nonlinear_phase[0]
  return

def gross_pitaevskii(t, wfc, H, data_layout, rhs, timing):
    """Returns rhs of Gross-Pitaevskii equation with discretized space
    For data_layout == 'complex':
      wfc is assumed to be in format where
      wfc[0:2*ntot:2] contains the real part of the wavefunction and
      wfc[1:2*ntot:2] contains the imag part of the wavefunction.
    For data_layout == 'real':
      wfc is assumed to be in format where
      wfc[0:ntot] contains the real part of the wavefunction and
      wfc[ntot:2*ntot] contains the imag part of the wavefunction.
    """

    start_time = timeit.default_timer()
    if (data_layout=='real'):
      apply_minus_i_h_gpe_real(wfc, H, rhs)
    else:
      apply_minus_i_h_gpe_complex(wfc.view(np.complex128), H, rhs.view(np.complex128))
    timing.GPE_TIME+=(timeit.default_timer() - start_time)
    timing.NUMBER_OF_OPS+=16.0*H.tab_cumulative_dim[0]
    return rhs

class Measurement:
  def __init__(self, delta_t_measurement, measure_density=False, measure_density_momentum=False, measure_autocorrelation=False, measure_dispersion_position=False, measure_dispersion_momentum=False, measure_dispersion_energy=False,measure_wavefunction_final=False,measure_extended=False,use_mkl_fft=True):
    self.delta_t_measurement = delta_t_measurement
    self.measure_density = measure_density
    self.measure_density_momentum = measure_density_momentum
    self.measure_autocorrelation = measure_autocorrelation
    self.measure_dispersion_position = measure_dispersion_position
    self.measure_dispersion_momentum = measure_dispersion_momentum
    self.measure_dispersion_energy = measure_dispersion_energy
    self.measure_wavefunction_final = measure_wavefunction_final
    self.measure_wavefunction_momentum_final = measure_wavefunction_final
    self.extended = measure_extended
    self.use_mkl_fft = use_mkl_fft
    return

  def prepare_measurement(self,propagation,tab_delta,tab_dim):
    delta_t = propagation.delta_t
    t_max = propagation.t_max
    how_often_to_measure = int(self.delta_t_measurement/delta_t+0.5)
    propagation.delta_t = self.delta_t_measurement/how_often_to_measure
    number_of_measurements = int(t_max/self.delta_t_measurement+1.99999)
    number_of_time_steps = int(t_max/delta_t+0.99999)
    self.dimension = len(tab_dim)
    self.tab_i_measurement = np.arange(start=0,stop=number_of_measurements*how_often_to_measure,step=how_often_to_measure,dtype=int)
    self.tab_t_measurement = delta_t*self.tab_i_measurement
# correct the last time
    self.tab_i_measurement[number_of_measurements-1]=number_of_time_steps
    self.tab_t_measurement[number_of_measurements-1]=t_max
#    self.density_final = np.zeros(0)
#    self.tab_autocorrelation = np.zeros(0,dtype=np.complex128)
#    self.frequencies = np.zeros(0)
#    self.density_momentum_final = np.zeros(0)
#    self.tab_position = np.zeros(0)
#    self.tab_position2 = np.zeros(0)
#    self.tab_energy = np.zeros(0)
#    self.tab_nonlinear_energy = np.zeros(0)
#    self.tab_momentum = np.zeros(0)
    dim_dispersion = [number_of_measurements]
    self.wfc =  np.zeros(tab_dim,dtype=np.complex128)
    self.wfc_momentum =  np.zeros(tab_dim,dtype=np.complex128)
    if (self.measure_density):
      self.density_final = np.zeros(tab_dim)
    if (self.measure_autocorrelation):
      self.tab_autocorrelation = np.zeros(number_of_measurements,dtype=np.complex128)
    if (self.measure_density_momentum):
      self.density_momentum_final = np.zeros(tab_dim)
    if (self.measure_density_momentum or self.measure_dispersion_momentum):
      self.frequencies = []
      for i in range(self.dimension):
        self.frequencies.append(np.fft.fftshift(np.fft.fftfreq(tab_dim[i],d=tab_delta[i]/(2.0*np.pi))))
    if (self.measure_dispersion_position):
      self.tab_position = np.zeros(dim_dispersion)
      self.tab_position2 = np.zeros(dim_dispersion)
    if (self.measure_dispersion_momentum):
      self.tab_momentum = np.zeros(dim_dispersion)
    if (self.measure_dispersion_energy):
      self.tab_energy = np.zeros(dim_dispersion)
      self.tab_nonlinear_energy = np.zeros(dim_dispersion)
    if (self.measure_wavefunction_final):
      self.wfc =  np.zeros(tab_dim,dtype=np.complex128)
    if (self.measure_wavefunction_momentum_final):
      self.wfc_momentum =  np.zeros(tab_dim,dtype=np.complex128)
    return

  def prepare_measurement_global(self,propagation,tab_delta,tab_dim):
    delta_t = propagation.delta_t
    t_max = propagation.t_max
    how_often_to_measure = int(self.delta_t_measurement/delta_t+0.5)
    propagation.delta_t = self.delta_t_measurement/how_often_to_measure
    number_of_measurements = int(t_max/self.delta_t_measurement+1.99999)
    number_of_time_steps = int(t_max/delta_t+0.99999)
    self.dimension = len(tab_dim)
    self.tab_i_measurement = np.arange(start=0,stop=number_of_measurements*how_often_to_measure,step=how_often_to_measure,dtype=int)
    self.tab_t_measurement = delta_t*self.tab_i_measurement
# correct the last time
    self.tab_i_measurement[number_of_measurements-1]=number_of_time_steps
    self.tab_t_measurement[number_of_measurements-1]=t_max
#    self.density_final = np.zeros((1,0))
#    self.tab_autocorrelation = np.zeros(0,dtype=np.complex128)
#    self.frequencies = np.zeros(0)
#    self.density_momentum_final = np.zeros(0)
#    self.tab_position = np.zeros(0)
#    self.tab_position2 = np.zeros(0)
#    self.tab_energy = np.zeros(0)
#    self.tab_nonlinear_energy = np.zeros(0)
#    self.tab_momentum = np.zeros(0)
    dim_density = tab_dim[:]
    dim_dispersion = [number_of_measurements]
    if self.extended:
      dim_density.insert(0,2)
      dim_dispersion.insert(0,2)
    else:
      dim_density.insert(0,1)
      dim_dispersion.insert(0,1)
#    print(dim_density)
#    print(dim_dispersion)
    self.wfc =  np.zeros(tab_dim,dtype=np.complex128)
    self.wfc_momentum =  np.zeros(tab_dim,dtype=np.complex128)
    if (self.measure_density):
      self.density_final = np.zeros(dim_density)
    if (self.measure_autocorrelation):
      self.tab_autocorrelation = np.zeros(number_of_measurements,dtype=np.complex128)
    if self.measure_density_momentum or self.measure_dispersion_momentum:
      self.density_momentum_final = np.zeros(dim_density)
      self.frequencies = []
      for i in range(self.dimension):
        self.frequencies.append(np.fft.fftshift(np.fft.fftfreq(tab_dim[i],d=tab_delta[i]/(2.0*np.pi))))
    if (self.measure_dispersion_position):
      self.tab_position = np.zeros(dim_dispersion)
      self.tab_position2 = np.zeros(dim_dispersion)
    if (self.measure_dispersion_momentum):
        self.tab_momentum = np.zeros(dim_dispersion)
    if (self.measure_dispersion_energy):
      self.tab_energy = np.zeros(dim_dispersion)
      self.tab_nonlinear_energy = np.zeros(dim_dispersion)
    if (self.measure_wavefunction_final):
      self.wfc =  np.zeros(tab_dim,dtype=np.complex128)
    if (self.measure_wavefunction_momentum_final):
      self.wfc_momentum =  np.zeros(tab_dim,dtype=np.complex128)
    return

  def merge_measurement(self,measurement):
    if self.measure_density:
      self.density_final[0] += measurement.density_final
      if self.extended:
        self.density_final[1] += measurement.density_final**2
    if self.measure_density_momentum:
      self.density_momentum_final[0] += measurement.density_momentum_final
      if self.extended:
        self.density_momentum_final[1] += measurement.density_momentum_final**2
    if self.measure_autocorrelation:
      self.tab_autocorrelation += measurement.tab_autocorrelation
    if self.measure_dispersion_position:
      self.tab_position[0] += measurement.tab_position
      self.tab_position2[0] += measurement.tab_position2
      if self.extended:
        self.tab_position[1] += measurement.tab_position**2
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
    if self.measure_wavefunction_final:
      self.wfc += measurement.wfc
    if self.measure_wavefunction_momentum_final:
      self.wfc_momentum += measurement.wfc_momentum
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
    if self.measure_density_momentum:
      toto = np.empty_like(self.density_momentum_final)
      comm.Reduce(self.density_momentum_final,toto)
      self.density_momentum_final = np.copy(toto)
    if self.measure_autocorrelation:
      toto = np.empty_like(self.tab_autocorrelation)
      comm.Reduce(self.tab_autocorrelation,toto)
      self.tab_autocorrelation = np.copy(toto)
    if self.measure_dispersion_position:
      toto = np.empty_like(self.tab_position)
      comm.Reduce(self.tab_position,toto)
      self.tab_position = np.copy(toto)
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
    if self.measure_wavefunction_final:
      toto = np.empty_like(self.wfc)
      comm.Reduce(self.wfc,toto)
      self.wfc = np.copy(toto)
    if self.measure_wavefunction_momentum_final:
      toto = np.empty_like(self.wfc_momentum)
      comm.Reduce(self.wfc_momentum,toto)
      self.wfc_momentum = np.copy(toto)
    timing.MPI_TIME+=(timeit.default_timer() - start_mpi_time)
    return

  def normalize(self,n_config):
    if self.measure_density:
      self.density_final /= n_config
      if self.extended:
        self.density_final[1] = np.sqrt(np.abs(self.density_final[1]-self.density_final[0]**2)/n_config)
    if self.measure_density_momentum:
      self.density_momentum_final /= n_config
      if self.extended:
        self.density_momentum_final[1] = np.sqrt(np.abs(self.density_momentum_final[1]-self.density_momentum_final[0]**2)/n_config)
    if self.measure_autocorrelation:
      self.tab_autocorrelation /= n_config
    list_of_columns = [self.tab_t_measurement]
    tab_strings=['Column 1: Time']
    next_column = 2
    if self.measure_dispersion_position:
      self.tab_position /= n_config
      self.tab_position2 /= n_config
      list_of_columns.append(self.tab_position[0])
      tab_strings.append('Column '+str(next_column)+': <x>')
      next_column += 1
      if self.tab_position.shape[0]==2:
        self.tab_position[1] = np.sqrt(np.abs(self.tab_position[1]-self.tab_position[0]**2)/n_config)
        list_of_columns.append(self.tab_position[1])
        tab_strings.append('Column '+str(next_column)+': Standard deviation of <x>')
        next_column += 1
      list_of_columns.append(self.tab_position2[0])
      tab_strings.append('Column '+str(next_column)+': <x^2>')
      next_column += 1
      if self.tab_position2.shape[0]==2:
        self.tab_position2[1] = np.sqrt(np.abs(self.tab_position2[1]-self.tab_position2[0]**2)/n_config)
        list_of_columns.append(self.tab_position2[1])
        tab_strings.append('Column '+str(next_column)+': Standard deviation of <x^2>')
        next_column += 1
    if self.measure_dispersion_momentum:
      self.tab_momentum /= n_config
      list_of_columns.append(self.tab_momentum[0])
      tab_strings.append('Column '+str(next_column)+': <p>')
      next_column += 1
      if self.tab_momentum.shape[0]==2:
        self.tab_momentum[1] = np.sqrt(np.abs(self.tab_momentum[1]-self.tab_momentum[0]**2)/n_config)
        list_of_columns.append(self.tab_momentum[1])
        tab_strings.append('Column '+str(next_column)+': Standard deviation of <p>')
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
    if self.measure_wavefunction_final:
      self.wfc /= n_config
    if self.measure_wavefunction_momentum_final:
      self.wfc_momentum /= n_config
  #  print(tab_strings)
  #  print(list_of_columns)

    return tab_strings, np.column_stack(list_of_columns)

  """
  def output_string(self, H, propagation,initial_state,n_config):
#    environment_string='Script '+os.path.basename(__file__)+' ran by '+getpass.getuser()+' on machine '+os.uname()[1]+'\n'\
#                      +datetime.datetime.now().strftime("%Y-%m-%d %H:%M")+'\n'
    params_string='Disorder type                   = '+H.disorder_type+'\n'\
                 +'Lx                              = '+str(H.dim_x*H.delta_x)+'\n'\
                 +'dx                              = '+str(H.delta_x)+'\n'\
                 +'Nx                              = '+str(H.dim_x)+'\n'\
                 +'V0                              = '+str(H.disorder_strength)+'\n'\
                 +'g                               = '+str(H.interaction)+'\n'\
                 +'Boundary Condition              = '+H.boundary_condition+'\n'\
                 +'Initial state                   = '+initial_state.type+'\n'\
                 +'k_0                             = '+str(initial_state.k_0)+'\n'\
                 +'sigma_0                         = '+str(initial_state.sigma_0)+'\n'\
                 +'Integration Method              = '+str(propagation.method)+'\n'\
                 +'time step                       = '+str(propagation.delta_t)+'\n'\
                 +'time step for measurement       = '+str(self.delta_t_measurement)+'\n'\
                 +'total time                      = '+str(propagation.t_max)+'\n'\
                 +'Number of disorder realizations = '+str(n_config)+'\n'\
                 +'\n'
#    print(params_string)
#                 +'first measurement step          = '+str(first_measurement_autocorr)+'\n'\
#                  +'total measurement time          = '+str(t_max-first_measurement_autocorr*delta_t_measurement)+'\n'\

    return params_string
  """

class Spectral_function:
  def __init__(self,e_range,e_resolution):
    self.n_pts_autocorr = int(0.5*e_range/e_resolution+0.5)
# In case e_range/e_resolution is not an integer, I keep e_resolution and rescale e_range
    self.e_range = 2.0*e_resolution*self.n_pts_autocorr
    self.delta_t = 2.0*np.pi/(self.e_range*(1.0+0.5/self.n_pts_autocorr))
    self.t_max = self.delta_t*self.n_pts_autocorr
    self.e_resolution = e_resolution
    return

  def compute_spectral_function(self,tab_autocorrelation):
# The autocorrelation function for negative time is simply the complex conjugate of the one at positive time
# We will use an inverse FFT, so that positive time must be first, followed by negative times (all increasing)
# see manual of numpy.fft.ifft for explanations
# The number of points in tab_autocorrelation is n_pts_autocorr+1
    tab_autocorrelation_symmetrized=np.concatenate((tab_autocorrelation,np.conj(tab_autocorrelation[:0:-1])))
# Make the inverse Fourier transform which is by construction real, so keep only real part
# Note that it is surely possible to improve using Hermitian FFT (useless as it uses very few resources)
# Both the spectrum and the energies are reordered in ascending order
    tab_spectrum=np.fft.fftshift(np.real(np.fft.ifft(tab_autocorrelation_symmetrized)))/self.e_resolution
    tab_energies=np.fft.fftshift(np.fft.fftfreq(2*self.n_pts_autocorr+1,d=self.delta_t/(2.0*np.pi)))
    return tab_energies,tab_spectrum



def gpe_evolution(i, initial_state, H, propagation, measurement, timing, debug=False):
  assert propagation.data_layout in ["real","complex"]
  assert H.boundary_condition in ["periodic","open"]
  assert propagation.method in ["ode","che"]
  def solout(t,y):
    timing.N_SOLOUT+=1
    return None
  """
  Determines whether the cffi version is present
  If not, use Python version
  """
  try:
    from anderson._chebyshev import ffi,lib
    chebyshev_step = chebyshev_step_clenshaw_cffi
    if debug: print('Using CFFI version')
  except ImportError:
    chebyshev_step = chebyshev_step_clenshaw_python
    print("\n Warning, this uses the slow Python version, you should build the C version!\n")



  dim_x = H.dim_x
  delta_x = H.delta_x
#  i_tab_0 = propagation.first_measurement_autocorr
  i_tab_0 = 0
#  print('start gen disorder',timeit.default_timer())
  H.generate_disorder(seed=i+1234)
#  print('end   gen disorder',timeit.default_timer())
#  tab_x = np.zeros_like(propagation.tab_i_measurement,dtype=np.float64)
#  tab_x2 = np.zeros_like(propagation.tab_i_measurement,dtype=np.float64)
#  tab_energy = np.zeros_like(propagation.tab_i_measurement,dtype=np.float64)
#  tab_nonlinear_energy = np.zeros_like(propagation.tab_i_measurement,dtype=np.float64)
#  tab_autocorrelation = np.zeros_like(propagation.tab_i_measurement,dtype=np.complex128)
#  tab_x[0] = initial_state.expectation_value_local_operator(initial_state.position)
#  tab_x2[0] = initial_state.expectation_value_local_operator(initial_state.position**2)
#  tab_energy[0], tab_nonlinear_energy[0] = initial_state.energy(H)
#  init_state_autocorr = anderson.Wavefunction(dim_x,delta_x)
  if (measurement.measure_autocorrelation):
    init_state_autocorr = anderson.Wavefunction(dim_x,delta_x)
    if (i_tab_0==0):
      measurement.tab_autocorrelation[0] = initial_state.overlap(initial_state)
      init_state_autocorr.wfc[:] = initial_state.wfc[:]
  if (measurement.measure_dispersion_position):
    measurement.tab_position[0] = initial_state.expectation_value_local_operator(initial_state.position)
    measurement.tab_position2[0] = initial_state.expectation_value_local_operator(initial_state.position**2)
  if (measurement.measure_dispersion_energy):
    measurement.tab_energy[0], measurement.tab_nonlinear_energy[0] = initial_state.energy(H)
  if (measurement.measure_dispersion_momentum):
    initial_state.wfc_momentum = initial_state.convert_to_momentum_space()
#      initial_state.wfc_momentum = delta_x*np.fft.fftshift(np.fft.fft(initial_state.wfc))/np.sqrt((2.0*np.pi))
    measurement.tab_momentum[0] = initial_state.expectation_value_local_momentum_operator(measurement.frequencies)

  if propagation.data_layout=='real':
    y = np.concatenate((np.real(initial_state.wfc),np.imag(initial_state.wfc)))
  else:
    y = np.copy(initial_state.wfc)
#  print(timeit.default_timer())
  if (propagation.method=='ode'):
    rhs = np.zeros(2*dim_x)
    solver = ode(f=lambda t,y: gross_pitaevskii(t,y,H,propagation.data_layout,rhs,timing)).set_integrator('dop853', atol=1e-5, rtol=1e-4)
    solver.set_solout(solout)
    solver.set_initial_value(y)
  else:
#    start_dummy_time=timeit.default_timer()
    e_min, e_max = H.energy_range()
#    timing.DUMMY_TIME+=(timeit.default_timer() - start_dummy_time)
    H.medium_energy = 0.5*(e_min+e_max)
#    print(e_min,e_max)
    H.script_tunneling, H.script_disorder = H.script_h(e_min,e_max)
    propagation.script_delta_t = 0.5*propagation.delta_t*(e_max-e_min)
    accuracy = 1.e-6
    propagation.compute_chebyshev_coefficients(accuracy,timing)


#time evolution
  i_tab = 1
  psi = anderson.Wavefunction(dim_x,delta_x)
#  print(timeit.default_timer())
  for i_prop in range(1,measurement.tab_i_measurement[-1]+1):
    t_next=min(i_prop*propagation.delta_t,propagation.t_max)
#     print(i_prop,t_next)
    if (propagation.method == 'ode'):
      start_ode_time = timeit.default_timer()
      solver.integrate(t_next)
      timing.ODE_TIME+=(timeit.default_timer() - start_ode_time)
    else:
      start_che_time = timeit.default_timer()
#      print(i_prop,start_che_time)
#      print(timing.MAX_NONLINEAR_PHASE)
      chebyshev_step(y, H, propagation,timing)
#      print(timing.MAX_NONLINEAR_PHASE)
#      print(y[2000])
      timing.CHE_TIME+=(timeit.default_timer() - start_che_time)
      timing.NUMBER_OF_OPS+=16.0*dim_x*propagation.tab_coef.size
    if i_prop==measurement.tab_i_measurement[i_tab]:
      start_dummy_time=timeit.default_timer()
      if (propagation.method == 'ode'): y=solver.y
      if propagation.data_layout=='real':
# The following two lines are faster than the natural implementation psi.wfc= y[0:dim_x]+1j*y[dim_x:2*dim_x]
        psi.wfc.real=y[0:dim_x]
        psi.wfc.imag=y[dim_x:2*dim_x]
      else:
        psi.wfc = y.view(np.complex128)
      timing.DUMMY_TIME+=(timeit.default_timer() - start_dummy_time)
      start_expect_time = timeit.default_timer()
#      print(start_expect_time)
      if (measurement.measure_dispersion_position):
        measurement.tab_position[i_tab] = psi.expectation_value_local_operator(psi.position)
        measurement.tab_position2[i_tab] = psi.expectation_value_local_operator(psi.position**2)
      if measurement.measure_dispersion_energy:
        measurement.tab_energy[i_tab], measurement.tab_nonlinear_energy[i_tab] = psi.energy(H)
      if (measurement.measure_dispersion_momentum):
        psi.wfc_momentum = psi.convert_to_momentum_space(measurement.use_mkl_fft)
        measurement.tab_momentum[i_tab] = psi.expectation_value_local_momentum_operator(measurement.frequencies)
      if (measurement.measure_autocorrelation):
        if (i_tab==i_tab_0):
          init_state_autocorr.wfc[:] = psi.wfc[:]
        if (i_tab>=i_tab_0):
# Inlining the overlap method is slighlty faster
#          measurement.tab_autocorrelation[i_tab-i_tab_0] = psi.overlap(init_state_autocorr)
          measurement.tab_autocorrelation[i_tab-i_tab_0] = np.vdot(init_state_autocorr.wfc,psi.wfc)*delta_x
      timing.EXPECT_TIME+=(timeit.default_timer() - start_expect_time)
      i_tab+=1
  start_expect_time = timeit.default_timer()
  if (measurement.measure_density):
    measurement.density_final = np.abs(psi.wfc)**2
  if (measurement.measure_density_momentum):
    psic_momentum = psi.convert_to_momentum_space(measurement.use_mkl_fft)
    measurement.density_momentum_final = np.abs(psic_momentum)**2
  if (measurement.measure_wavefunction_final):
    measurement.wfc = psi.wfc
  if (measurement.measure_wavefunction_momentum_final):
    if not measurement.measure_density_momentum:
      psic_momentum = psi.convert_to_momentum_space(measurement.use_mkl_fft)
    measurement.wfc_momentum = psic_momentum
  timing.EXPECT_TIME+=(timeit.default_timer() - start_expect_time)
#  print(timeit.default_timer())
  return
