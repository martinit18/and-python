#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:30:03 2020

@author: delande
"""
import numpy as np
#import math
import sys
import itertools
from anderson.geometry import Geometry

class Wavefunction(Geometry):
  def __init__(self, geometry):
    super().__init__(geometry.dimension,geometry.tab_dim,geometry.tab_delta,spin_one_half=geometry.spin_one_half)
    self.wfc = np.zeros(self.tab_extended_dim,dtype=np.complex128)
    return

  def gaussian(self):

    grid_position = np.meshgrid(*self.grid_position,indexing='ij')
    tab_phase = np.zeros(self.tab_dim)
    tab_amplitude = np.zeros(self.tab_dim)
#    print(tab_position[0].shape)
#    print(tab_position[1].shape)
    for i in range(len(grid_position)):
      tab_phase += self.tab_k_0[i]*grid_position[i]
      tab_amplitude += (grid_position[i]/self.tab_sigma_0[i])**2
# The next two lines are to avoid too small values of abs(psi[i])
# which slow down the calculation
    threshold = 100.
    tab_amplitude =np.where(tab_amplitude>threshold,threshold,tab_amplitude)
    psi = np.exp(-0.5*tab_amplitude+1j*tab_phase)
    psi = psi/(np.linalg.norm(psi)*np.sqrt(self.delta_vol))
    if self.spin_one_half:
      self.wfc = np.multiply.outer(psi,self.lhs_state)
    else:
      self.wfc = psi
    return

  def chirped(self):
    grid_position = np.meshgrid(*self.grid_position,indexing='ij')
    tab_phase = np.zeros(self.tab_dim)
    tab_amplitude = np.zeros(self.tab_dim)
#    print(tab_position[0].shape)
#    print(tab_position[1].shape)
    for i in range(len(grid_position)):
      tab_phase += self.tab_k_0[i]*grid_position[i]+self.tab_chirp[i]*grid_position[i]**2
      tab_amplitude += (grid_position[i]/self.tab_sigma_0[i])**2
# The next two lines are to avoid too small values of abs(psi[i])
# which slow down the calculation
    threshold = 100.
    tab_amplitude =np.where(tab_amplitude>threshold,threshold,tab_amplitude)
    psi = np.exp(-0.5*tab_amplitude+1j*tab_phase)
    psi = psi/(np.linalg.norm(psi)*np.sqrt(self.delta_vol))
    if self.spin_one_half:
      self.wfc = np.multiply.outer(psi,self.lhs_state)
    else:
      self.wfc = psi
    return

  def plane_wave(self):
#    self.type = 'Plane wave'
    grid_position = np.meshgrid(*self.grid_position,indexing='ij')
#    print(grid_position)
    tab_phase = np.zeros(self.tab_dim)
    for i in range(len(grid_position)):
      tab_phase += self.tab_k_0[i]*grid_position[i]
    psi =  np.exp(1j*tab_phase)/np.sqrt(self.ntot*self.delta_vol)
    if self.spin_one_half:
      if self.lhs_state is None:
        self.lhs_state=np.ones(self.lhs_dim)/np.sqrt(self.lhs_dim)
      self.wfc = np.multiply.outer(psi,self.lhs_state)
    else:
      self.wfc = psi
#    print(self.wfc.shape)
#    print(self.wfc)
    return

  def point(self):
#    point = list()
#    for i in range(self.dimension): point.append(0)
#    point.append(':')
#    print(point)
#    print(tuple(point))
    if self.spin_one_half:
      self.wfc.ravel()[0:self.lhs_state.size] = self.lhs_state[:]/self.delta_vol
    else:
      self.wfc.ravel()[0] = 1.0/self.delta_vol
 #   print(self.wfc.shape)
 #   print(self.wfc)
    return

  def multi_point(self,seed=2345):
# This creates an initial state with several delta peaks in configuration space
# These points are locatedon a multidimensional regular rectangular array
# so that there is at least distance "minimum_distance" (in length units, not numer of sites)
# between points
# Currently works only when there is no spin-orbit
# Array A is an array of integer sequences
# A[i] will contain the indices along durection i where delta-peaks are put
# This routine should not be called with spin-orbit coupling
    if self.spin_one_half:
      sys.exit("muti_point initial state not yet implemented for spin-orbit systems, I stop!")
    self.seed = seed
    if self.use_mkl_random:
      try:
        import mkl_random
      except ImportError:
        self.use_mkl_random=False
        print('No mkl_random found; Fallback to Numpy random')
    if self.use_mkl_random:
      mkl_random.RandomState(77777, brng='SFMT19937')
      mkl_random.seed(seed,brng='SFMT19937')
      my_random_normal = mkl_random.standard_normal
    else:
      np.random.seed(seed)
      my_random_normal = np.random.standard_normal
#    print(self.use_mkl_random,self.seed)
    A = np.empty(self.dimension,dtype=object)
    for i in range(self.dimension):
# If the minimum distance is too small (=0 when not set), only a single point is used
      if (self.minimum_distance<self.tab_delta[i]):
        A[i] = [0]
      else:
# Number of points along axis i where a delta peak is put, always smaller than the total number of sites
        num_steps = np.int(np.floor(self.tab_dim[i]*self.tab_delta[i]/self.minimum_distance))
        if num_steps>0:
# The integer step in indexing along axis i
          step = np.int(np.ceil(self.tab_dim[i]/num_steps))
#          print(num_steps,step)
# The sequence of indices along axis i
          A[i] = range(0,step*num_steps,step)
        else:
          A[i]=[0]
#      print(i,A[i])
# This creates the tuple of point indices where a delta-peak is put
    aa = tuple(prod for prod in itertools.product(*A))
    my_sum = 0.0
    my_random_sequence = my_random_normal(2*len(aa))
    my_sum = np.sum(my_random_sequence**2)
    normalization_factor = 1.0/(self.delta_vol*np.sqrt(my_sum))
    j=0
#    print(aa)
# Normalize so that the norm of the initial state is unity
#    normalization_factor = 1.0/(self.delta_vol*np.sqrt(len(aa)))
# Write the delta-peaks in the wavefunction
    for x in aa:
#      print(x)
#      self.wfc[x] = normalization_factor
      self.wfc[x] = my_random_sequence[2*j:2*j+2].view(np.complex128)*normalization_factor
      j += 1
#      my_sum += np.abs(self.wfc[x])**2
#    normalization_factor = 1.0/(self.delta_vol*np.sqrt(my_sum))
#    for x in aa:
#      self.wfc[x] *= normalization_factor
#    print(self.wfc.ravel()[0])
    return

  def prepare_initial_state(self,seed=2345):
    if (self.type=='plane_wave'):
      self.plane_wave()
    if (self.type=='gaussian_wave_packet'):
      self.gaussian()
    if (self.type=='chirped_wave_packet'):
      self.chirped()
    if (self.type=='point'):
      self.point()
    if (self.type=='multi_point'):
      self.multi_point(seed=seed)
    return  

  def overlap(self, other_wavefunction):
#    return np.sum(self.wfc*np.conj(other_wavefunction.wfc))*self.delta_x
# The following line is 5 times faster!
    return np.vdot(self.wfc,other_wavefunction.wfc)*self.delta_vol

  def convert_to_momentum_space(self,use_mkl_fft=True):
    if self.dimension==1:
# The following three lines are for the simple 1D case
#      wfc_momentum = self.tab_delta[0]*np.fft.fft(self.wfc)/np.sqrt(2.0*np.pi)
#      wfc_momentum *= np.exp(-1j*np.arange(self.tab_dim[0])*np.pi*(1.0/self.tab_dim[0]-1.0))
#      return np.fft.fftshift(wfc_momentum)
# which can be combined in one line
      return np.fft.fftshift(self.tab_delta[0]*np.fft.fft(self.wfc)*np.exp(-1j*np.arange(self.tab_dim[0])*np.pi*(1.0/self.tab_dim[0]-1.0))/np.sqrt(2.0*np.pi))
# The conversion space->momentum is a bit tricky, here are the explanations:
# Position x_i is discretized as x_k=(k+1/2-dim_x/2)delta_x for 0\leq k \lt dim_x
# Momentum is discretized as p_l = l delta_p for 0 \leq l \lt dim_x and delta_p=2\pi/(dim_x*delta_x)
# (more on negative momenta below)
# The position->momentum transformation is defined by \psi(p) = 1/\sqrt{2\pi} \int_0^L \psi(x) \exp(-ipx) dx
# where L=dim_x*delta_x is the system size.
# An elementary calculation shows that, after discretization:
# \psi(p_l) = delta_x/\sqrt{2\pi} \exp(-i\pi l(1/dim_x-1)) \sum_{k=0..dim_x-1} \psi(x_k) \exp{-2i\pi kl/dim_x}
# The last sum is directly given by the routine np.fft.fft (or mkl_fft.fft) applied on self.wfc
# After that, remains the multiplication by delta_x/\sqrt{2\pi} \exp(-i\pi l(1/dim_x-1))
# Because of FFT, the momentum is also periodic, meaning that the l values above are not 0..dim_x-1, but rather
# -dim_x/2..dim_x/2-1. At the exit of np.fft.fft, they are in the order 0,1,...dim_x/2-1,-dim_x/2,..,-1.
# They are put back in the natural order -dim_x/2...dim_x/2-1 using the np.fft.fftshift routine.
    if use_mkl_fft:
      try:
        import mkl_fft
        my_fft = mkl_fft.fftn
      except ImportError:
        my_fft = np.fft.fftn
    else:
       my_fft = np.fft.fftn
    wfc_momentum = self.delta_vol*my_fft(self.wfc)/(np.sqrt(2.0*np.pi)**self.dimension)
# Limited to dimension 10
    if (self.dimension>10):
      print('too large dimension, phase factor for the wavefunction in momentum space is not computed')
      return np.fft.fftshift(wfc_momentum)
    else:
# Multiplication by the proper phase factor is done consecutively along each direction
# using the flexible einsum routine of Numpy
      string = 'ijklmnopqr'
# In 2d, the following two lines are a bit more efficient
#      wfc_momentum = (wfc_momentum.T*np.exp(-1j*np.arange(self.tab_dim[0])*np.pi*(1.0/self.tab_dim[0]-1.0))).T
#      wfc_momentum *= np.exp(-1j*np.arange(self.tab_dim[1])*np.pi*(1.0/self.tab_dim[1]-1.0))
# This general case works in any dimension
      for i in range(self.dimension):
        phase_factor = np.exp(-1j*np.arange(self.tab_dim[i])*np.pi*(1.0/self.tab_dim[i]-1.0))
        wfc_momentum = np.einsum(string[0:self.dimension]+','+string[i]+'->'+string[0:self.dimension],wfc_momentum,phase_factor)
      return np.fft.fftshift(wfc_momentum)

  def convert_from_momentum_space_to_configuration_space(self,use_mkl_fft=True):
    if self.dimension == 1:
# The following three lines are for the simple 1D case
#     wfc = np.fft.ifftshift(self.wfc_momentum)*np.exp(1j*np.arange(self.tab_dim[0])*np.pi*(1.0/self.tab_dim[0]-1.0))
#      wfc = np.fft.ifft(wfc)*np.sqrt(2.0*np.pi)/self.tab_delta[0]
#      return wfc
# which can be combined in one line
      return np.fft.ifft(np.fft.ifftshift(self.wfc_momentum)*np.exp(1j*np.arange(self.tab_dim[0])*np.pi*(1.0/self.tab_dim[0]-1.0)))*np.sqrt(2.0*np.pi)/self.tab_delta[0]
# The conversion from momentum to configuration space follows the same steps than from configuration to momentum space, in reversed order. See explanations in the configuration->momentum routine
    if use_mkl_fft:
      try:
        import mkl_fft
        my_ifft = mkl_fft.ifftn
      except ImportError:
        my_ifft = np.fft.ifftn
    else:
       my_ifft = np.fft.ifftn
 # Shift the momentum component in proper order
    wfc = np.fft.ifftshift(self.wfc_momentum)
# Limited to dimension 10
    if (self.dimension>10):
      print('too large dimension, I cannot compute the phase factors, wavefunction in config space will be badly wrong')
    else:
# Multiplication by the proper phase factor is done consecutively along each direction
# using the flexible einsum routine of Numpy
      string = 'ijklmnopqr'
# This general case works in any dimension
      for i in range(self.dimension):
        phase_factor = np.exp(1j*np.arange(self.tab_dim[i])*np.pi*(1.0/self.tab_dim[i]-1.0))
        wfc = np.einsum(string[0:self.dimension]+','+string[i]+'->'+string[0:self.dimension],wfc,phase_factor)
# A simple inverse Fourier transform and a proper scaling gives the wavefunction in configuration space
    return my_ifft(wfc)*(np.sqrt(2.0*np.pi)**self.dimension)/self.delta_vol


  def energy(self, H):
#    print(self.wfc.shape)
#    rhs = H.apply_h(self.wfc)
#    print(H.spin_one_half)
    if H.interaction==0.0:
      non_linear_energy=0.0
    else:
#      non_linear_energy = 0.5*H.interaction*np.sum(np.abs(self.wfc)**4)*self.delta_vol
      non_linear_energy = 0.5*H.interaction*np.sum((self.wfc.real**2+self.wfc.imag**2)**2)*self.delta_vol
#    energy = np.sum(np.real(self.wfc.ravel()*np.conjugate(H.apply_h(self.wfc))))*self.delta_vol + non_linear_energy
#    print(self.wfc.ravel().real.dtype,self.wfc.real.dtype,H.apply_h(self.wfc.real).dtype)
    if H.is_real:
      energy = np.sum(self.wfc.ravel().real*H.apply_h(self.wfc.ravel().real)+self.wfc.ravel().imag*H.apply_h(self.wfc.ravel().imag))*self.delta_vol + non_linear_energy
#      print('Complex energy = ',np.sum(np.conj(self.wfc)*H.apply_h(self.wfc))*self.delta_vol)
#      print(self.wfc.shape)
#      print((H.apply_h(self.wfc).shape))
    else:
      energy = np.sum(np.real(np.conj(self.wfc.ravel())*H.apply_h(self.wfc.ravel())))*self.delta_vol+non_linear_energy
    norm = np.linalg.norm(self.wfc)**2*self.delta_vol
#    print('norm=',norm,energy,non_linear_energy)
    return energy/norm,non_linear_energy/norm
