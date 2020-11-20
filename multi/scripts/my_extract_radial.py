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
# my_extract_radial.py
# Author: Dominique Delande
# Release date: Nov, 18, 2020
# License: GPL2 or later

"""
Created on Mon 16 Nov 2020

@author: Dominique Delande

This scripts reads data in a 2d file (for exemple created by np.savetxt) and assumes that it has a rotational symmetry around the center (in the middle of each axis). It then gathers all points as a function of the radial distance, and
performs a spline fit of these points, in order to create a smooth function
of the radius. It then prints the smoothed radial function in a 1d file
"""

import sys
import os
#import math
import numpy as np
import scipy.interpolate
import scipy.signal
#import statsmodels.api as sm
#import pylab as plt


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
  print('Usage: my_extract_radial.py file [argument]')
  print('       file: a real or complex 2d numpy.savetxt file')
  print('       argument for complex data:')
  print('         0: real part [default]')
  print('         1: imaginary part')
  print('         2: square modulus')
  print('         3: modulus')
  print('       argument for real data:')
  print('         0: data [default]')
  print('         2: data**2')
  print('         3: |data|')
  sys.exit()

file_name=sys.argv[1]

# Here are the parameters that may need to be modified
# and adapted to the specific data
# Number of points in the radial direction
number_of_points = 2000
# Divide the [0,rmax] interval in intervals of size interval_width*r_min
interval_width = 40.0
# This must be a typical over which the function varies
distance_between_knots = 1.0
# Order of the spline function
# k=3 is usually a good compromise
# smaller k generates less smooth data
# larger k may produce spikes in the output data
k=3

# Default is to read the first column
my_choice=0
if len(sys.argv)==3:
  my_choice=int(sys.argv[2])

try:
  arr=np.loadtxt(file_name,dtype=np.float64)
  if my_choice==1:
    sys.exit("Illegal choice '1' for the argument with real data")
  if my_choice==2:
    Z=np.abs(arr)**2
  if my_choice==3:
    Z=np.abs(arr)
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
density_radial_file_name=name+'_radial'+ext
#g = open(density_radial_file_name,'w')
tab_r = np.zeros((n1,n2))
#print(tab_r.shape,Z.shape)
# Compute the radial distance r and sort the function by increasing r
for i in range(n1):
  for j in range(n2):
    tab_r[i,j] = np.sqrt(((i-n1//2)*delta1)**2+((j-n2//2)*delta2)**2)
idx=np.argsort(tab_r,axis=None)
r = tab_r.ravel()[idx]
func = Z.ravel()[idx]
#y = scipy.signal.savgol_filter(func,501,3)
#np.savetxt('toto.dat',np.column_stack((r,func)))
# Determine the maximum r (which is the min of the radial distances along x and y)
r_max = min(0.5*n1*delta1,0.5*n2*delta2)
# Minimum distance
r_min = min(delta1,delta2)

# Compute the number of knots
number_of_knots = int(interval_width/distance_between_knots+0.5)
delta_r = interval_width*r_min
# Round to integers
number_of_intervals = int (r_max/delta_r+0.5)
xnew = np.linspace(0.0,r_max,num=number_of_points+1)
ynew = np.zeros(number_of_points+1)
# No interpolation for the first point at r=0
ynew[0] = func[0]

delta_r = r_max/number_of_intervals
tab_position = np.zeros(number_of_intervals+1,dtype=int)
tab_position[0] = 0
tab_evaluation = np.zeros(number_of_intervals+1,dtype=int)
tab_evaluation[0] = 0
for i in range(number_of_intervals):
  r_sup = delta_r*(i+1)
#  print(r_sup)
  tab_position[i+1] = np.argmax(r>r_sup)
  tab_evaluation[i+1] = np.argmax(xnew>r_sup)
tab_evaluation[number_of_intervals]=number_of_points+1
#print('tab_position',tab_position)
#print('tab_evaluation',tab_evaluation)
xnew = np.linspace(0.0,r_max,num=number_of_points+1)
ynew = np.zeros(number_of_points+1)
# No interpolation for the first point at r=0
ynew[0] = func[0]
for i in range(0,number_of_intervals):
#  print(i,xnew[i*number_of_points_per_interval+2],xnew[(i+1)*number_of_points_per_interval-1],r[tab_position[i]],r[tab_position[i+1]-1],tab_position[i],tab_position[i+1])
#  print(i,delta_r*i,delta_r*(i+1))
#  print('r',r[tab_position[i]:tab_position[i+1]])
#  print('func',func[tab_position[i]:tab_position[i+1]])
  tab_knots = np.linspace(delta_r*i+delta_r/(number_of_knots+1),delta_r*i+delta_r*number_of_knots/(number_of_knots+1),num=number_of_knots)
#  print('tab_knots',tab_knots)
#  print('xnew',xnew[i*number_of_points_per_interval+2:(i+1)*number_of_points_per_interval])
  s = scipy.interpolate.LSQUnivariateSpline(r[tab_position[i]:tab_position[i+1]],func[tab_position[i]:tab_position[i+1]], tab_knots, k=k, check_finite=False)
#  print('s ok')
  ynew[tab_evaluation[i]:tab_evaluation[i+1]] = s(xnew[tab_evaluation[i]:tab_evaluation[i+1]])
#lowess = sm.nonparametric.lowess(func[10000:20000], r[10000:20000], is_sorted=True, frac=0.1, xvals=xvals)
#plt.plot(r,func)
#plt.plot(xvals,lowess)
#plt.show()
np.savetxt(density_radial_file_name,np.column_stack((xnew,ynew)))

"""
rmax = min(0.5*n1*delta1,0.5*n2*delta2)
number_of_intervals = int(0.05*min(0.5*n1,0.5*n2))
number_of_points_per_interval = 50
number_of_points = number_of_points_per_interval*number_of_intervals
xnew = np.zeros(number_of_points)
ynew = np.zeros(number_of_points)
delta_r = rmax/number_of_intervals
tab_position = np.zeros(number_of_intervals+2,dtype=int)
tab_position[0] = 0
for i in range(number_of_intervals+1):
  r_sup = delta_r*(i+1)
  tab_position[i+1] = np.argmax(r>r_sup)
#print(tab_position)
for i in range(1,number_of_intervals):
  tck = scipy.interpolate.splrep(r[tab_position[i-1]:tab_position[i+2]], func[tab_position[i-1]:tab_position[i+2]], k=5, s=0.3)
  xnew[i*number_of_points_per_interval:(i+1)*number_of_points_per_interval] = np.linspace(delta_r*i,delta_r*(i+1),num=number_of_points_per_interval)
  ynew[i*number_of_points_per_interval:(i+1)*number_of_points_per_interval] = scipy.interpolate.splev(xnew[i*number_of_points_per_interval:(i+1)*number_of_points_per_interval],tck)
#np.savetxt('toto.dat',np.column_stack((r,func)))
np.savetxt('toto2.dat',np.column_stack((xnew,ynew)))

#print(Z.ravel()[idx])

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
"""
