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
  print('Usage: view_2d.py file [argument]')
  print('       file: a real or complex 2d numpy.savetxt file') 
  print('       argument for complex data:')
  print('         0: real part [default]')
  print('         1: imaginary part')
  print('         2: square modulus')
  print('       argument for real data:')
  print('         0: data [default]')
  print('         2: data**2')
  sys.exit()

my_choice=0
if len(sys.argv)==3:
  my_choice=int(sys.argv[2])

try:
  arr=np.loadtxt(sys.argv[1],dtype=np.float64)
  if my_choice==2:
    Z=np.abs(arr)**2
  else:
    Z=arr
except:
  arr=np.loadtxt(sys.argv[1],dtype=np.complex128)
  if my_choice==2:
    Z=np.abs(arr)**2
  if my_choice==1:
    Z=np.imag(arr)
  if my_choice==0:
    Z=np.real(arr)
print('Maximum value = ',Z.max())
print('Minimum value = ',Z.min())
im = plt.imshow(Z,origin='lower',interpolation='nearest')

plt.show()

