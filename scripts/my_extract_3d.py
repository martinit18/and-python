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

This scripts reads data in a 3d file and
performs cuts along the x, y and z axes, assuming that the origin is in the middle
of the data, that is at positions n1//2, n2//2, n3//2.
It prints the results in two 1d files, with a "_cut_{x,y,z}" suffix.
"""

import sys
import os
#import math
import numpy as np
import argparse

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

def main():
  parser = argparse.ArgumentParser(description='Extract data from a 3D file, and outputs 1D cuts and projections along the x, y and z axes',usage='use "%(prog)s -h" for more information',formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument('filename', type=argparse.FileType('r'), help='name of the file containing the 2D data')
  parser.add_argument('-c', '--column', type=int, default=0, \
  help='''column to read:
  For complex data:
    0: Re(data) [default]
    1: Im(data)
    2: |data|**2
    3: |data|
  For real data:
    0: data [default]
    2: |data|**2
    3: |data|''')
  args = parser.parse_args()
  file_name = args.filename.name
  my_choice = args.column
  with open(sys.argv[1]) as f:
    datafile = f.readlines()
    for line in datafile:
      if "# N_1" in line:
        N_1 = int(line.split()[-1])
      if "# N_2" in line:
        N_2 = int(line.split()[-1])
      if "# N_3" in line:
        N_3 = int(line.split()[-1])
      if "# delta_1" in line:
        delta_1 = float(line.split()[-1])
      if "# delta_2" in line:
        delta_2 = float(line.split()[-1])
      if "# delta_3" in line:
        delta_3 = float(line.split()[-1])

  try:
    arr=np.loadtxt(file_name,dtype=np.float64)
    if my_choice==3:
      Z=np.abs(arr)
      specific_string='_modulus'
    if my_choice==2:
      Z=np.abs(arr)**2
      specific_string='_modulus_square'
    if my_choice==1:
      sys.exit('Illegal choice "1" (imaginary part) for real data')
    if my_choice==0:
      Z=arr
      specific_string=''
    if my_choice>3 or my_choice<0:
      sys.exit('Illegal choice "'+str(my_choice)+'" for -c argument')
  except ValueError:
    arr=np.loadtxt(file_name,dtype=np.complex128)
    if my_choice==3:
      Z=np.abs(arr)
      specific_string='_modulus'
    if my_choice==2:
      Z=np.abs(arr)**2
      specific_string='_modulus_square'
    if my_choice==1:
      Z=np.imag(arr)
      specific_string='_Im'
    if my_choice==0:
      Z=np.real(arr)
      specific_string='_Re'
    if my_choice>3 or my_choice<0:
      sys.exit('Illegal choice "'+str(my_choice)+'" for -c argument')

  Z = Z.reshape((N_1,N_2,N_3))
  print(Z.shape)
  #print(n1,n2)
  #X, Y = np.meshgrid(x, y)
  #Z = np.exp(-(X*X+Y*Y))
  print('Maximum value = ',Z.max())
  print('Minimum value = ',Z.min())
  name, ext = os.path.splitext(file_name)
  name = name+specific_string
  density_cut_x_file_name=name+'_cut_x'+ext
  g = open(density_cut_x_file_name,'w')
  print('#',N_1, file=g)
  for i in range(N_1):
    print((i-N_1//2)*delta_1,Z[i,N_2//2,N_3//2], file=g)
  g.close()
  density_cut_y_file_name=name+'_cut_y'+ext
  g = open(density_cut_y_file_name,'w')
  print('#',N_2, file=g)
  for i in range(N_2):
    print((i-N_2//2)*delta_2,Z[N_1//2,i,N_3//2], file=g)
  g.close()
  density_cut_z_file_name=name+'_cut_z'+ext
  g = open(density_cut_z_file_name,'w')
  print('#',N_3, file=g)
  for i in range(N_3):
    print((i-N_3//2)*delta_3,Z[N_1//2,N_2//2,i], file=g)
  g.close()

  #print density_cut_x_file_name

  #im = plt.imshow(Z,origin='lower',interpolation='nearest')
  #plt.show()

if __name__ == "__main__":
  main()

