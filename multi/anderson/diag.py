#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 16:33:12 2019

@author: delande
"""

import numpy as np
import scipy.sparse.linalg as sparse_linalg
# import scipy.sparse

class Diagonalization:
  def __init__(self,targeted_energy,method='sparse',number_of_eigenvalues=1,IPR_min=0.0,IPR_max=1.0,number_of_bins=1):
    self.targeted_energy = targeted_energy
    self.method = method
    self.IPR_min = IPR_min
    self.IPR_max = IPR_max
    self.number_of_bins = number_of_bins
    self.number_of_eigenvalues = number_of_eigenvalues

  def compute_IPR(self, i, H):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_matrix()
#    print(matrix)
      w, v = np.linalg.eigh(matrix)
#    print(w)
#      index = np.abs(w-self.targeted_energy).argmin()
      index_array = np.argsort(abs(w-self.targeted_energy))
#    print(index_array[0:self.number_of_eigenvalues])
      imin = np.min(index_array[0:self.number_of_eigenvalues])
      imax = np.max(index_array[0:self.number_of_eigenvalues])+1
    if self.method=='sparse':
      H.generate_sparse_matrix()
      w, v = sparse_linalg.eigsh(H.sparse_matrix,k=self.number_of_eigenvalues,sigma=self.targeted_energy,mode='normal')
      imin = 0
      imax = self.number_of_eigenvalues
# The sparse_linalg.eigsh is complete bullshit for complex hermitian matrices, it requires to sort the eigenvalues
      if (H.spin_one_half):
        index_array = np.argsort(w)
        w = w[index_array]
        v = v[:,index_array]
  # The normalization (division by delta_vol) ensures that IPR is roughly the inverse of the localization length
    if H.spin_one_half:
      IPR = np.sum(np.abs(v[:,imin:imax])**4,axis=0)/H.delta_vol
    else:
      IPR = np.sum(v[:,imin:imax]**4,axis=0)/H.delta_vol
    return w[imin:imax], IPR

  def compute_tab_r(self, i, H):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_matrix()
#    print(matrix)
      w = np.linalg.eigvalsh(matrix)
      index_array = np.argsort(abs(w-self.targeted_energy))
#    print(index_array[0:self.number_of_eigenvalues])
      imin = np.min(index_array[0:self.number_of_eigenvalues])
      imax = np.max(index_array[0:self.number_of_eigenvalues])+1
    if self.method=='sparse':
      H.generate_sparse_matrix()
#      print(H.sparse_matrix.dtype)
#      matrix2 = H.generate_sparse_complex_matrix(1j)
      w = sparse_linalg.eigsh(H.sparse_matrix,k=self.number_of_eigenvalues,sigma=self.targeted_energy,mode='normal',return_eigenvectors=False)
# The sparse_linalg.eigsh is complete bullshit for complex hermitian matrices, it requires to sort the eigenvalues
      if (H.spin_one_half):
        w = np.sort(w)
      imin = 0
      imax = self.number_of_eigenvalues
    tab_r = np.zeros(imax-2-imin)
    for j in range(imin,imax-2):
      r = (w[j+2]-w[j+1])/(w[j+1]-w[j])
      if r>1.0: r=1.0/r
      tab_r[j-imin] = r
    return (w[imin+1:imax-1],tab_r)

  def compute_wavefunction(self,i,H):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_matrix()
#    print(matrix)
      w, v = np.linalg.eigh(matrix)
      index_array = np.argsort(abs(w-self.targeted_energy))
#    print(index_array[0:self.number_of_eigenvalues])
      imin = np.min(index_array[0:self.number_of_eigenvalues])
      imax = np.max(index_array[0:self.number_of_eigenvalues])+1
    if self.method=='sparse':
      H.generate_sparse_matrix()
      w, v = sparse_linalg.eigsh(H.sparse_matrix,k=self.number_of_eigenvalues,sigma=self.targeted_energy,mode='normal')
      imin = 0
      imax = self.number_of_eigenvalues
# The sparse_linalg.eigsh is complete bullshit for complex hermitian matrices, it requires to sort the eigenvalues
      if (H.spin_one_half):
        index_array = np.argsort(w)
        w = w[index_array]
        v = v[:,index_array]
    return w[imin:imax],v[:,imin:imax]/np.sqrt(H.delta_vol)

  def compute_full_spectrum(self,i,H):
    H.generate_disorder(seed=i+1234)
    matrix = H.generate_full_matrix()
#    print(matrix)
    w = np.linalg.eigvalsh(matrix)
    return w

  def compute_spectrum(self,i,H):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_matrix()
#    print(matrix)
      w = np.linalg.eigvalsh(matrix)
      index_array = np.argsort(abs(w-self.targeted_energy))
#    print(index_array[0:self.number_of_eigenvalues])
      imin = np.min(index_array[0:self.number_of_eigenvalues])
      imax = np.max(index_array[0:self.number_of_eigenvalues])+1
      return w[imin:imax]
# Old, less efficient, code
      """
# identify the closest eigenvalue
      index = np.abs(w-self.targeted_energy).argmin()
      index_right = index+1
      index_left = index-1
# search the next ones either on the left of right sides
      for i in range(k-1):
        if abs(w[index_right]-self.targeted_energy)>abs(w[index_left]-self.targeted_energy):
          index_left -=1
        else:
          index_right += 1
      return w[index_left+1:index_right]
      """
    if self.method=='sparse':
      H.generate_sparse_matrix()
      w = sparse_linalg.eigsh(H.sparse_matrix,k=self.number_of_eigenvalues,sigma=self.targeted_energy,mode='normal',return_eigenvectors=False)
# The sparse_linalg.eigsh is complete bullshit for complex hermitian matrices, it requires to sort the eigenvalues
      if (H.spin_one_half):
        w = np.sort(w)
      return w


  def compute_landscape(self,i,H,initial_state,pivot):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_complex_matrix(pivot)
      print(matrix[0,0])
      landscape = 1.0/np.abs(np.linalg.solve(matrix,initial_state.wfc))+pivot.real
    if self.method=='sparse':
      matrix = H.generate_sparse_matrix()
# The following line is obviously less efficient
#    matrix = H.generate_full_matrix()
# The following line uses a CSR storage for the sparse matrix and is slightly less efficient
#    matrix = scipy.sparse.csr_matrix(matrix)
      landscape = 1.0/np.abs(sparse_linalg.spsolve(matrix,initial_state.wfc))+self.targeted_energy
    return landscape

  def compute_landscape_2(self,i,H,pivot):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_complex_matrix(pivot)
      landscape = np.zeros(H.dim_x,dtype=np.complex128)
#      print(matrix[0,0])
      for i in range(H.dim_x):
        init = np.zeros(H.dim_x,dtype=np.complex128)
        init[i] = 1.0
        init2 = np.linalg.solve(matrix,init)
        landscape[i] = np.linalg.solve(np.conj(matrix),init2)[i]
#        print(landscape[i])
      landscape = np.sqrt(np.real(landscape))
    if self.method=='sparse':
      matrix = H.generate_sparse_complex_matrix(pivot)
      landscape = np.zeros(H.dim_x,dtype=np.complex128)
      for i in range(H.dim_x):
        init = np.zeros(H.dim_x,dtype=np.complex128)
        init[i] = 1.0
        init2 = sparse_linalg.spsolve(matrix,init)
        landscape[i] = sparse_linalg.spsolve(np.conj(matrix),init2)[i]
#        print(landscape[i])
      landscape = np.sqrt(np.real(landscape))
# The following line is obviously less efficient
#    matrix = H.generate_full_matrix()
# The following line uses a CSR storage for the sparse matrix and is slightly less efficient
#    matrix = scipy.sparse.csr_matrix(matrix)

    return landscape