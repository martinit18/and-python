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

