#!/usr/bin/env python3
import sys
import math
import string
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

#for arg in sys.argv:
#  print arg
if len(sys.argv)<2 or len(sys.argv)>3:
  print('Usage: view_3d.py file [argument]')
  print('       file: a real or complex 3d file')
  print('       argument for complex data:')
  print('         0: real part [default]')
  print('         1: imaginary part')
  print('         2: square modulus')
  print('         3: modulus')
  print('       argument for real data:')
  print('         0: data [default]')
  print('         2: data**2')
  sys.exit()

my_choice=0
if len(sys.argv)==3:
  my_choice=int(sys.argv[2])

with open(sys.argv[1]) as f:
  datafile = f.readlines()
  for line in datafile:
    if "# N_1" in line:
      N_1 = int(line.split()[-1])
    if "# N_2" in line:
      N_2 = int(line.split()[-1])
    if "# N_3" in line:
      N_3 = int(line.split()[-1])

try:
  arr=np.loadtxt(sys.argv[1],dtype=np.float64)
  if my_choice==2:
    Z=np.abs(arr)**2
  else:
    Z=arr
except:
  arr=np.loadtxt(sys.argv[1],dtype=np.complex128)
  if my_choice==3:
    Z=np.abs(arr)
  if my_choice==2:
    Z=np.abs(arr)**2
  if my_choice==1:
    Z=np.imag(arr)
  if my_choice==0:
    Z=np.real(arr)
Z = Z.reshape((N_1,N_2,N_3))
print(Z.shape)
print('Maximum value = ',Z.max())
print('Minimum value = ',Z.min())
im = plt.imshow(Z[N_1//2,:,:],origin='lower',interpolation='nearest')

plt.show()

im = plt.imshow(Z[:,N_2//2,:],origin='lower',interpolation='nearest')

plt.show()

im = plt.imshow(Z[:,:,N_3//2],origin='lower',interpolation='nearest')

plt.show()
