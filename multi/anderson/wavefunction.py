#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:30:03 2020

@author: delande
"""
import numpy as np
from anderson.geometry import Geometry

class Wavefunction(Geometry):
  def __init__(self, geometry):
    super().__init__(geometry.dimension,geometry.tab_dim,geometry.tab_delta)
    self.wfc = np.zeros(self.tab_dim,dtype=np.complex128)
    return

  def gaussian(self,tab_k_0,tab_sigma_0):
    self.tab_k_0 = tab_k_0
    self.tab_sigma_0 = tab_sigma_0
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
    self.wfc = psi/(np.linalg.norm(psi)*np.sqrt(self.delta_vol))
    return

  def plane_wave(self,tab_k_0):
#    self.type = 'Plane wave'
    self.tab_k_0 = tab_k_0
    grid_position = np.meshgrid(*self.grid_position,indexing='ij')
    tab_phase = np.zeros(self.tab_dim)
    for i in range(len(grid_position)):
      tab_phase += self.tab_k_0[i]*grid_position[i]
    self.wfc = np.exp(1j*tab_phase)/np.sqrt(self.ntot*self.delta_vol)
    return

  def overlap(self, other_wavefunction):
#    return np.sum(self.wfc*np.conj(other_wavefunction.wfc))*self.delta_x
# The following line is 5 times faster!
    return np.vdot(self.wfc,other_wavefunction.wfc)*self.delta_vol

  def convert_to_momentum_space(self,use_mkl_fft=True):
#    psic_momentum = self.delta_x*np.fft.fft(self.wfc)/np.sqrt(2.0*np.pi)
#   psic_momentum *= np.exp(-1j*np.arange(self.dim_x)*np.pi*(1.0/self.dim_x-1.0))
#    return np.fft.fftshift(self.delta_x*np.fft.fft(self.wfc)*np.exp(-1j*np.arange(self.dim_x)*np.pi*(1.0/self.dim_x-1.0))/np.sqrt(2.0*np.pi))
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


  def energy(self, H):
#    rhs = H.apply_h(self.wfc)
    if H.interaction==0.0:
      non_linear_energy=0.0
    else:
#      non_linear_energy = 0.5*H.interaction*np.sum(np.abs(self.wfc)**4)*self.delta_vol
      non_linear_energy = 0.5*H.interaction*np.sum((self.wfc.real**2+self.wfc.imag**2)**2)*self.delta_vol
#    energy = np.sum(np.real(self.wfc.ravel()*np.conjugate(H.apply_h(self.wfc))))*self.delta_vol + non_linear_energy
#    print(self.wfc.ravel().real.dtype,self.wfc.real.dtype,H.apply_h(self.wfc.real).dtype)
    energy = np.sum(self.wfc.ravel().real*H.apply_h(self.wfc.real)+self.wfc.ravel().imag*H.apply_h(self.wfc.imag))*self.delta_vol + non_linear_energy
    norm = np.linalg.norm(self.wfc)**2*self.delta_vol
#    print('norm=',norm,energy,non_linear_energy)
    return energy/norm,non_linear_energy/norm