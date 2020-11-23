#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 19:27:50 2020

@author: delande
"""

import numpy as np
import scipy.sparse as ssparse
from anderson.geometry import Geometry

"""
The class Hamiltonian contains all properties that define a disordered Hamiltonian discretized spatially
interaction if g for Gross-Pitaeskii
script_tunneling and script_disorder are just rescaled variables used when the temporal propagation if performed using Chebyshev polynomials
They are used in the rescaling of the Hamiltonian to bring its spectrum between -1 and +1
"""

class Hamiltonian(Geometry):
  def __init__(self, geometry, tab_boundary_condition, disorder_type='anderson_gaussian', one_over_mass=1.0,  correlation_length=0.0, disorder_strength=0.0, non_diagonal_disorder_strength=0.0, b=1, use_mkl_random=False, interaction=0.0):
    super().__init__(geometry.dimension,geometry.tab_dim,geometry.tab_delta)
    dimension = self.dimension
    tab_dim = self.tab_dim
    tab_delta = self.tab_delta
    self.disorder_strength = disorder_strength
    self.non_diagonal_disorder_strength = non_diagonal_disorder_strength
    self.one_over_mass = one_over_mass
    self.tab_tunneling = list()
    self.delta_vol = 1.0
    self.diagonal = 0.0
    self.array_boundary_condition = np.zeros(dimension,dtype=np.intc)
    for i in range(dimension):
      self.tab_tunneling.append(0.5*one_over_mass/tab_delta[i]**2)
      self.delta_vol *= tab_delta[i]
      self.diagonal += one_over_mass/tab_delta[i]**2
      if tab_boundary_condition[i]=='periodic':
        self.array_boundary_condition[i] = 1
    self.interaction = interaction
    self.tab_boundary_condition = tab_boundary_condition
    self.disorder_type = disorder_type
    self.correlation_length = correlation_length
    self.use_mkl_random = use_mkl_random
#    self.diagonal_term = np.zeros(tab_dim)
#    self.script_tunneling = 0.
#    self.script_disorder = np.zeros(tab_dim)
#    self.medium_energy = 0.
    if disorder_type=='nice':
      self.b = b
      self.non_diagonal_disorder = np.zeros((self.ntot+self.b,self.b))
    self.generate=''
    if (disorder_type in ['anderson_gaussian','anderson_uniform','anderson_cauchy','nice']):
      self.generate='direct'
      return
# Build mask for correlated potentials
    if disorder_type=='regensburg':
      self.generate = 'simple mask'
      self.mask = np.zeros(tab_dim)
      tab_k = list()
      for i in range(dimension):
        toto = np.zeros(tab_dim[i])
        half_size = tab_dim[i]//2+1
        toto[0:half_size] = -0.25*(np.arange(half_size)*2.0*np.pi*self.correlation_length/(self.tab_dim[i]*self.tab_delta[i]))**2
        toto[tab_dim[i]+1-half_size:tab_dim[i]] = toto[half_size-1:0:-1]
        tab_k.append(toto)
      tab_distance = np.meshgrid(*tab_k,indexing='ij')
      for i in range(dimension):
         self.mask += tab_distance[i]
      self.mask = np.exp(self.mask)
      self.mask *= np.sqrt(self.ntot/np.sum(self.mask**2))
#      print(self.mask)
      return
    if disorder_type=='konstanz':
      self.generate = 'field mask'
      self.mask = np.zeros(tab_dim)
      tab_k = list()
      for i in range(dimension):
        toto = np.zeros(tab_dim[i])
        half_size = tab_dim[i]//2+1
        toto[0:half_size] = -0.5*(np.arange(half_size)*2.0*np.pi*self.correlation_length/(self.tab_dim[i]*self.tab_delta[i]))**2
        toto[tab_dim[i]+1-half_size:tab_dim[i]] = toto[half_size-1:0:-1]
        tab_k.append(toto)
      tab_distance = np.meshgrid(*tab_k,indexing='ij')
      for i in range(dimension):
         self.mask += tab_distance[i]
      self.mask = np.exp(self.mask)
      self.mask *= np.sqrt(self.ntot/np.sum(self.mask**2))
      return
    """
    if disorder_type=='singapore':
      self.generate = 'simple mask'
      self.mask = np.zeros(dim_x)
      inv_k_length = 2.0*np.pi*self.correlation_length/(self.dim_x*self.delta_x)
      if inv_k_length<2.0/dim_x:
        mask_size = dim_x//2+1
      else:
        mask_size = int(1.0/inv_k_length+0.5)
#      self.mask = np.zeros(dim_x)
      self.mask[0:mask_size] = np.ones(mask_size)
      self.mask[dim_x-mask_size+1:dim_x]=self.mask[mask_size-1:0:-1]
      self.mask*=np.sqrt(dim_x/np.sum(self.mask**2))
      return
    if disorder_type=='speckle':
      self.generate = 'field mask'
      self.mask = np.zeros(dim_x)
      inv_k_length = 2.0*np.pi*self.correlation_length/(self.dim_x*self.delta_x)
      if inv_k_length<2.0/dim_x:
        mask_size = dim_x//2+1
      else:
        mask_size = int(1.0/inv_k_length+0.5)
#      self.mask = np.zeros(dim_x)
      self.mask[0:mask_size] = np.ones(mask_size)
      self.mask[dim_x-mask_size+1:dim_x]=self.mask[mask_size-1:0:-1]
      self.mask*=np.sqrt(dim_x/np.sum(self.mask**2))
      return
#      print(dim_x,mask_size)
#      print(self.mask)
#      self.mask[0]=dim_x
    """



  """
  Generate a specific disorder configuration
  """
  def generate_disorder(self,seed):
#    print(self.use_mkl_random)
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
      my_random_uniform = mkl_random.uniform
    else:
      np.random.seed(seed)
      my_random_normal = np.random.standard_normal
      my_random_uniform = np.random.uniform
#    print('seed=',seed)
#    print(self.tab_dim_cumulative)
    if self.disorder_type=='anderson_uniform':
      self.disorder = self.diagonal + self.disorder_strength*my_random_uniform(-0.5,0.5,self.ntot).reshape(self.tab_dim)/np.sqrt(self.delta_vol)
#      print(self.disorder)
      return
    if self.disorder_type=='anderson_gaussian':
      self.disorder = self.diagonal + self.disorder_strength*my_random_normal(self.ntot).reshape(self.tab_dim)/np.sqrt(self.delta_vol)
#      print(self.disorder.shape,self.disorder.dtype)
      return
    if self.disorder_type=='nice':
# Only in dimension 1
      self.disorder = self.diagonal + self.disorder_strength*my_random_uniform(-0.5,0.5,self.ntot)/np.sqrt(self.delta_vol)
      self.non_diagonal_disorder = self.non_diagonal_disorder_strength*my_random_uniform(-1.0,1.0,self.b*(self.ntot+self.b)).reshape((self.ntot+self.b,self.b))
#      print(self.disorder)
      return
    if self.generate=='simple mask':
      self.disorder =  self.diagonal + self.disorder_strength*np.real(np.fft.ifftn(self.mask*np.fft.fftn(my_random_normal(self.ntot).reshape(self.tab_dim))))
#      self.print_potential()
      return
    if self.generate=='field mask':
# When a field_mask is used, the initial data is a complex uncorrelated set of Gaussian distributed random numbers in configuration space
# The Fourier transform in momentum space is also a complex uncorrelated set of Gaussian distributed random numbers
# Thus the first FT is useless and can be short circuited
      self.disorder =  self.diagonal + 0.5*self.disorder_strength*self.ntot*np.abs(np.fft.ifftn(self.mask*my_random_normal(2*self.ntot).view(np.complex128).reshape(self.tab_dim)))**2
# Alternatively (slower)
#      self.disorder =  self.diagonal + 0.5*self.disorder_strength*np.abs(np.fft.ifftn(self.mask*np.fft.fftn(my_random_normal(2*self.ntot).view(np.complex128).reshape(self.tab_dim))))**2
#      self.print_potential()
#      print(self.disorder.shape,self.disorder.dtype)
      return
    sys.exit('Disorder '+self.disorder_type+' not yet implemented!')
    return

#  def print_potential(self,filename='potential.dat'):
#    np.savetxt(filename,self.disorder-2.0*self.tunneling)
#    return

  """
  Converts Hamiltonian to a full matrix for Lapack diagonalization
  """

  def generate_full_matrix(self):
    tab_index = np.zeros(self.dimension,dtype=int)
    matrix = np.diag(self.disorder.ravel())
    ntot = self.ntot
    sub_diagonal= np.zeros((2*self.dimension,ntot))
    tab_offset = np.zeros(2*self.dimension,dtype=int)
    for j in range(self.dimension):
# Offsets for the regulat hoppings
      tab_offset[j] = self.tab_dim_cumulative[j+1]
# Offsets for the periodic hoppings
      tab_offset[j+self.dimension] = self.tab_dim_cumulative[j+1]*(self.tab_dim[j]-1)
#    print(tab_offset)
    for i in range(ntot):
      index = i
      for j in range(self.dimension-1,-1,-1):
        index,tab_index[j] = divmod(index,self.tab_dim[j])
#      print(i,tab_index)
# Hopping along the various dimensions
      for j in range(self.dimension):
# Regular hopping
        if tab_index[j]<self.tab_dim[j]-1:
          sub_diagonal[j,i] = -self.tab_tunneling[j]
        else:
# Only if periodic boundary condition
          if self.tab_boundary_condition[j]=='periodic':
            sub_diagonal[j+self.dimension,i-(self.tab_dim[j]-1)*self.tab_dim_cumulative[j+1]] = -self.tab_tunneling[j]
    for j in range(self.dimension):
      np.fill_diagonal(matrix[tab_offset[j]:,:],sub_diagonal[j,:])
      np.fill_diagonal(matrix[:,tab_offset[j]:],sub_diagonal[j,:])
      if self.tab_boundary_condition[j]=='periodic':
        np.fill_diagonal(matrix[tab_offset[j+self.dimension]:,:],sub_diagonal[j+self.dimension,:])
        np.fill_diagonal(matrix[:,tab_offset[j+self.dimension]:],sub_diagonal[j+self.dimension,:])
#    print(matrix)
    return matrix
    """
    if (self.dimension==1):
      matrix = np.diag(self.disorder)
      ntot = self.tab_dim_cumulative[0]
      np.fill_diagonal(matrix[1:,:],-self.tab_tunneling[0])
      np.fill_diagonal(matrix[:,1:],-self.tab_tunneling[0])
      if self.tab_boundary_condition[0]=='periodic':
        matrix[0,ntot-1] = -self.tab_tunneling[0]
        matrix[ntot-1,0] = -self.tab_tunneling[0]
#    print(matrix)
      return matrix
    if (self.dimension==2):
    """
    """
      matrix = np.diag(self.disorder.ravel())
      ntot = self.tab_dim_cumulative[0]
      sub_diagonal= np.zeros(ntot)
      nx = self.tab_dim[0]
      ny = self.tab_dim[1]
# Hopping along x
      for i in range(0,nx-1):
        sub_diagonal[i*ny:(i+1)*ny] = -self.tab_tunneling[0]
      np.fill_diagonal(matrix[ny:,:],sub_diagonal)
      np.fill_diagonal(matrix[:,ny:],sub_diagonal)
      if self.tab_boundary_condition[0]=='periodic':
        np.fill_diagonal(matrix[ntot-ny:,:],sub_diagonal)
        np.fill_diagonal(matrix[:,ntot-ny:],sub_diagonal)
# Hopping along y
      sub_diagonal[:]=0.0
      for i in range(ny-1):
        sub_diagonal[i:ntot:ny] = -self.tab_tunneling[1]
      np.fill_diagonal(matrix[1:,:],sub_diagonal)
      np.fill_diagonal(matrix[:,1:],sub_diagonal)
      if self.tab_boundary_condition[1]=='periodic':
        sub_diagonal[:]=0.0
        sub_diagonal[0:ntot:ny] = -self.tab_tunneling[1]
        np.fill_diagonal(matrix[ny-1:,:],sub_diagonal)
        np.fill_diagonal(matrix[:,ny-1:],sub_diagonal)
#      print(matrix)
    """
    """
#      print(self.disorder.ravel())
      matrix = np.diag(self.disorder.ravel())
#      ntot = self.tab_dim_cumulative[0]
      sub_diagonal= np.zeros(self.tab_dim_cumulative[0])
# Hopping along x
      for i in range(self.tab_dim[0]-1):
        sub_diagonal[i*self.tab_dim_cumulative[1]:(i+1)*self.tab_dim_cumulative[1]] = -self.tab_tunneling[0]
      np.fill_diagonal(matrix[self.tab_dim_cumulative[1]:,:],sub_diagonal)
      np.fill_diagonal(matrix[:,self.tab_dim_cumulative[1]:],sub_diagonal)
      if self.tab_boundary_condition[0]=='periodic':
        np.fill_diagonal(matrix[self.tab_dim_cumulative[0]-self.tab_dim_cumulative[1]:,:],sub_diagonal)
        np.fill_diagonal(matrix[:,self.tab_dim_cumulative[0]-self.tab_dim_cumulative[1]:],sub_diagonal)
# Hopping along y
      sub_diagonal[:]=0.0
      for i in range(self.tab_dim_cumulative[1]-1):
        sub_diagonal[i:self.tab_dim_cumulative[0]:self.tab_dim_cumulative[1]] = -self.tab_tunneling[1]
      np.fill_diagonal(matrix[1:,:],sub_diagonal)
      np.fill_diagonal(matrix[:,1:],sub_diagonal)
      if self.tab_boundary_condition[1]=='periodic':
        sub_diagonal[:]=0.0
        sub_diagonal[0:self.tab_dim_cumulative[0]:self.tab_dim_cumulative[1]] = -self.tab_tunneling[1]
        np.fill_diagonal(matrix[self.tab_dim_cumulative[1]-1:,:],sub_diagonal)
        np.fill_diagonal(matrix[:,self.tab_dim_cumulative[1]-1:],sub_diagonal)
      print(matrix)
      return matrix
    """
    """
# loop over all indices of the big matrix
    for i in range(self.tab_dim_cumulative[0]):
# generate in tab_index the physical indices along the various directions
      index = i
      for j in range(self.dimension-1,-1,-1):
        index,tab_index[j] = divmod(index,self.tab_dim[j])
# Compute neighbors along the current direction
        if tab_index[j]<self.tab_dim[j] and tab_index[j]>0:
          neighbor = tab_index[j]+1

      print(i,tab_index)
#      matrix[i,i+1] = -self.tunneling
#      matrix[i+1,i] = -self.tunneling
#    if self.boundary_condition=='periodic':
#      matrix[0,n-1] = -self.tunneling
#      matrix[n-1,0] = -self.tunneling
#    print(matrix)
    return matrix
    """

  def generate_full_complex_matrix(self,pivot):
    matrix = self.generate_full_matrix().astype(np.complex128)
    np.fill_diagonal(matrix,matrix.diagonal()-pivot)
    return matrix

  """
  Converts Hamiltonian to a sparse matrix for sparse diagonalization
  """
  def generate_sparse_matrix(self):
    tab_index = np.zeros(self.dimension,dtype=int)
    diagonal = self.disorder.ravel()
    ntot = self.ntot
    sub_diagonal= np.zeros((2*self.dimension,ntot))
    tab_offset = np.zeros(2*self.dimension,dtype=int)
    for j in range(self.dimension):
# Offsets for the regulat hoppings
      tab_offset[j] = self.tab_dim_cumulative[j+1]
# Offsets for the periodic hoppings
      tab_offset[j+self.dimension] = self.tab_dim_cumulative[j+1]*(self.tab_dim[j]-1)
#    print(tab_offset)
    for i in range(ntot):
      index = i
      for j in range(self.dimension-1,-1,-1):
        index,tab_index[j] = divmod(index,self.tab_dim[j])
#      print(i,tab_index)
# Hopping along the various dimensions
      for j in range(self.dimension):
# Regular hopping
        if tab_index[j]<self.tab_dim[j]-1:
          sub_diagonal[j,i] = -self.tab_tunneling[j]
        else:
# Only if periodic boundary condition
          if self.tab_boundary_condition[j]=='periodic':
            sub_diagonal[j+self.dimension,i-(self.tab_dim[j]-1)*self.tab_dim_cumulative[j+1]] = -self.tab_tunneling[j]
    diagonals = [diagonal]
    offsets = [0]
    for j in range(self.dimension):
      diagonals.extend([sub_diagonal[j,:],sub_diagonal[j,:]])
      offsets.extend([tab_offset[j],-tab_offset[j]])
      if self.tab_boundary_condition[j]=='periodic':
        diagonals.extend([sub_diagonal[j+self.dimension,:],sub_diagonal[j+self.dimension,:]])
        offsets.extend([tab_offset[j+self.dimension],-tab_offset[j+self.dimension]])
#    print(diagonals)
#    print(offsets)
    self.sparse_matrix = ssparse.diags(diagonals,offsets,format='csr')
#    print('Sparse matrix computed',self.sparse_matrix.dtype)
#    print(matrix.toarray())
    return

  """
  Converts Hamiltonian to a complex sparse matrix for sparse diagonalization
  """
  def generate_sparse_complex_matrix(self,pivot):
    matrix = self.generate_sparse_matrix().astype(np.complex128)
    matrix.setdiag(matrix.diagonal()-pivot)
#    print(matrix.toarray())
    return matrix

  """
  Apply Hamiltonian on a wavefunction
  """

  def apply_h(self, wfc):
    if self.dimension==1:
      dim_x = self.tab_dim[0]
      tunneling = self.tab_tunneling[0]
      if wfc.dtype==np.float64:
        rhs = np.empty(dim_x,dtype=np.float64)
      else:
        rhs = np.empty(dim_x,dtype=np.complex128)
      if self.tab_boundary_condition[0]=='periodic':
        rhs[0]       = -tunneling * (wfc[dim_x-1] + wfc[1]) + self.disorder[0] * wfc[0]
        rhs[dim_x-1] = -tunneling * (wfc[dim_x-2] + wfc[0]) + self.disorder[dim_x-1] * wfc[dim_x-1]
      else:
        rhs[0]       = -tunneling * wfc[1]       + self.disorder[0]       * wfc[0]
        rhs[dim_x-1] = -tunneling * wfc[dim_x-2] + self.disorder[dim_x-1] * wfc[dim_x-1]
      rhs[1:dim_x-1] = -tunneling * (wfc[0:dim_x-2] + wfc[2:dim_x]) + self.disorder[1:dim_x-1] * wfc[1:dim_x-1]
      return rhs
    else:
      return self.sparse_matrix.dot(wfc.ravel())

  """
  Try to estimate bounds of the spectrum of The Hamiltonian
  This is needed for the temporal propagation using the Chebyshev method
  """
  def energy_range(self, accurate=False):
# The accurate determination should be used for strong disorder
    if (accurate):
  # rough estimate of the maximum energy (no disorder taken into account)
      e_max_0=2.0*self.diagonal
#      for i in range(self.dimension):
#        e_max_0 += 2.0/(self.tab_delta[i]**2)

      n_iterations_hamiltonian_bounds=min(10,int(50./np.log10(e_max_0)))
  # rough estimate of the minimum energy (no disorder taken into account)
      e_min_0 = 0.0
  # First determine the lower bound
  # Start with plane wave k=0 (minimum energy state)
      psic = Wavefunction(self.tab_dim, self.tab_delta)
      tab_k= []
      for i in range(self.dimension):
        tab_k.append(0.0)
      psic.plane_wave(tab_k)
  #    print(psic.wfc.shape)
  #    psic2 = np.empty(self.dim_x,dtype=np.complex128)
      finished = False
      estimated_bound = 0.0
      while not finished:
        for i in range(n_iterations_hamiltonian_bounds-1):
          psic.wfc = self.apply_h(psic.wfc).reshape(self.tab_dim)-e_max_0*psic.wfc
        norm = np.linalg.norm(psic.wfc)**2*self.delta_vol
        psic.wfc = self.apply_h(psic.wfc).reshape(self.tab_dim)-e_max_0*psic.wfc
        new_norm = np.linalg.norm(psic.wfc)**2*self.delta_vol
        new_estimated_bound = np.sqrt(new_norm/norm)
        norm = new_norm
        if (norm>1.e100):
          psic.wfc *= 1.0/np.sqrt(norm)
          norm = 1.0
  #    print(norm,new_estimated_bound)
        if (new_estimated_bound < 1.0001*estimated_bound):
          finished = True
        estimated_bound=new_estimated_bound
      e_max = e_min_0 + 1.01 * estimated_bound
  #    print(e_max)
  # Then determine the lower bound
  # Start with plane wave k=pi (maximum energy state)
      tab_k= []
      for i in range(self.dimension):
        tab_k.append(np.pi/self.tab_delta[i])
      psic.plane_wave(tab_k)
      finished = False
      estimated_bound = 0.0
      while not finished:
        for i in range(n_iterations_hamiltonian_bounds-1):
          psic.wfc = self.apply_h(psic.wfc).reshape(self.tab_dim)-e_min_0*psic.wfc
        norm = np.linalg.norm(psic.wfc)**2*self.delta_vol
        psic.wfc = self.apply_h(psic.wfc).reshape(self.tab_dim)-e_min_0*psic.wfc
        new_norm = np.linalg.norm(psic.wfc)**2*self.delta_vol
        new_estimated_bound = np.sqrt(new_norm/norm)
        norm = new_norm
        if (norm>1.e100):
          psic.wfc *= 1.0/np.sqrt(norm)
          norm = 1.0
  #    print(norm,new_estimated_bound)
        if (new_estimated_bound < 1.0001*estimated_bound):
          finished = True
        estimated_bound=new_estimated_bound
      e_min = e_max_0 - 1.01 * estimated_bound
    else:
# Very basic bounds using the min/max of the potential
      e_max_0=self.diagonal
#      for i in range(self.dimension):
#        e_max_0 += 1.0/(self.tab_delta[i]**2)
      e_min=np.amin(self.disorder)-e_max_0
      e_max=np.amax(self.disorder)+e_max_0
#    print(e_min,e_max)
    self.e_min = e_min
    self.e_max = e_max
    self.medium_energy = 0.5*(e_max+e_min)
    self.two_over_delta_e = 2.0/(e_max-e_min)
    self.two_e0_over_delta_e = self.medium_energy*self.two_over_delta_e
    return
  """
  Computes the rescaling of Hamiltonian for the Chebyshev method
  """
  def script_h(self,e_min,e_max):
    one_over_delta_e = 1.0/(e_max-e_min)
    #script_tunneling = 2.0*self.tunneling*one_over_delta_e
    #script_disorder = (2.0*self.disorder-(e_min+e_max))*one_over_delta_e
    self.two_over_delta_e = 2.0*one_over_delta_e
    self.two_e0_over_delta_e = (e_min+e_max)*one_over_delta_e
#  print(script_tunneling,script_disorder[0:10])
    #return script_tunneling, script_disorder
    return