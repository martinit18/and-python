#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "Dominique Delande"
__copyright__ = "Copyright (C) 2020 Dominique Delande"
__license__ = "GPL version 2 or later"
__version__ = "1.0"
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, see <http://www.gnu.org/licenses/>.
# ____________________________________________________________________
#
# my_extract_np.py
# Author: Dominique Delande
# Release date: Nov, 18, 2020
# License: GPL2 or later

"""
Created on Mon 18 Nov 2013

@author: Dominique Delande

This scripts reads data in a 2d file (for exemple created by np.savetxt) and
performs cuts along the x and y axis, assuming that the origin is in the middle
of the data, that is at positions n1//2, n2//2.
It prints the results in two 1d files, with a "_cut_{x,y}" suffix.
It also projects the data along the x and y axes, by summing over the transverse direction and printing the results in two 1d files, with a "_{x,y}" suffix.
"""

import sys
import os
import math
import numpy as np

def determine_unique_postfix(fn):
  if not os.path.exists(fn):
    return ''
  path, name = os.path.split(fn)
  name, ext = os.path.splitext(name)
  make_fn = lambda i: os.path.join(path, '%s_%d%s' % (name, i, ext))
  for i in range(1, sys.maxsize):
    uni_fn = make_fn(i)
    if not os.path.exists(uni_fn):
      return '_'+str(i)
  return '_XXX'

#for arg in sys.argv:
#  print arg
#for arg in sys.argv:
#  print arg
if len(sys.argv)<2 or len(sys.argv)>3:
  print('Usage: my_extract_np.py file [argument]')
  print('       file: a real or complex 2d numpy.savetxt file')
  print('       argument for complex data:')
  print('         0: real part [default]')
  print('         1: imaginary part')
  print('         2: square modulus')
  print('         3: modulus')
  print('       argument for real data:')
  print('         0: data [default]')
  print('         2: data**2')
  sys.exit()

file_name=sys.argv[1]


my_choice=0
if len(sys.argv)==3:
  my_choice=int(sys.argv[2])

try:
  arr=np.loadtxt(file_name,dtype=np.float64)
  if my_choice==2:
    Z=np.abs(arr)**2
  else:
    Z=arr
except:
  arr=np.loadtxt(file_name,dtype=np.complex128)
  if my_choice==3:
    Z=np.abs(arr)
  if my_choice==2:
    Z=np.abs(arr)**2
  if my_choice==1:
    Z=np.imag(arr)
  if my_choice==0:
    Z=np.real(arr)

# Try to extract information from the first two lines
try:
  f = open(file_name,'r')
  line=(f.readline().lstrip('#')).split()
  n1=int(line[0])
  if len(line)>1:
    delta1=float(line[-1])
  else:
    delta1=1.0
  line=(f.readline().lstrip('#')).split()
  n2=int(line[0])
  if len(line)>1:
    delta2=float(line[-1])
  else:
    delta2=1.0
except:
  n1,n2 = Z.shape
  delta1=1.0
  delta2=1.0

#print(n1,n2)
#X, Y = np.meshgrid(x, y)
#Z = np.exp(-(X*X+Y*Y))
print('Maximum value = ',Z.max())
print('Minimum value = ',Z.min())
name, ext = os.path.splitext(file_name)
density_cut_x_file_name=name+'_cut_x'+ext
g = open(density_cut_x_file_name,'w')
print('#',n1, file=g)
for i in range(n1):
  print((i-n1//2)*delta1,Z[i,n2//2], file=g)
g.close()
density_cut_y_file_name=name+'_cut_y'+ext
g = open(density_cut_y_file_name,'w')
print('#',n2, file=g)
for i in range(n2):
  print((i-n2//2)*delta2,Z[n1//2,i], file=g)
g.close()
density_x_file_name=name+'_x'+ext
g = open(density_x_file_name,'w')
print('#',n1, file=g)
for i in range(n1):
  print((i-n1//2)*delta1,np.sum(Z[i,:])*delta2, file=g)
g.close()
density_y_file_name=name+'_y'+ext
g = open(density_y_file_name,'w')
print('#',n2, file=g)
for i in range(n2):
  print((i-n2//2)*delta2,np.sum(Z[:,i])*delta2, file=g)
g.close()

#print density_cut_x_file_name

#im = plt.imshow(Z,origin='lower',interpolation='nearest')
#plt.show()

