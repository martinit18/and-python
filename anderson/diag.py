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
  def __init__(self,targeted_energy,method='sparse'):
    self.targeted_energy = targeted_energy
    self.method = method

  def compute_IPR(self, i, H):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_matrix()
#    print(matrix)
      w, v = np.linalg.eigh(matrix)
#    print(w)
      index = np.abs(w-self.targeted_energy).argmin()
    if self.method=='sparse':
      matrix = H.generate_sparse_matrix()
# The following line is obviously less efficient
#    matrix = H.generate_full_matrix()
# The following line uses a CSR storage for the sparse matrix and is slightly less efficient
#    matrix = scipy.sparse.csr_matrix(matrix)
      w, v = sparse_linalg.eigsh(matrix,k=1,sigma=self.targeted_energy,mode='normal')
      index = 0
# The normalization (division by delta_x) ensures that IPR is roughly the inverse of the localization length
    IPR = np.sum(v[:,index]**4)/(H.delta_x)
#  print('Energy=',w[index])
    return (w[index],IPR)

  def compute_wavefunction(self,i,H,k=4):
    H.generate_disorder(seed=i+1234)
    if self.method=='lapack':
      matrix = H.generate_full_matrix()
#    print(matrix)
      w, v = np.linalg.eigh(matrix)
#    print(w)
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
      return w[index_left+1:index_right],v[:,index_left+1:index_right]/np.sqrt(H.delta_x)

    if self.method=='sparse':
      matrix = H.generate_sparse_matrix()
# The following line is obviously less efficient
#    matrix = H.generate_full_matrix()
# The following line uses a CSR storage for the sparse matrix and is slightly less efficient
#    matrix = scipy.sparse.csr_matrix(matrix)
      w, v = sparse_linalg.eigsh(matrix,k=k,sigma=self.targeted_energy,mode='normal')
#      print(w)
#  print('Energy=',w[index])
      return w,v/np.sqrt(H.delta_x)

  def compute_full_spectrum(self,i,H):
    H.generate_disorder(seed=i+1234)
    matrix = H.generate_full_matrix()
#    print(matrix)
    w, v = np.linalg.eigh(matrix)
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
      landscape = 1.0/np.abs(sparse_linalg.spsolve(matrix,initial_state.wfc))
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