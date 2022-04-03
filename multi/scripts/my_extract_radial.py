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
import argparse
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


def main():
  parser = argparse.ArgumentParser(description='Extract data from a 2D (numpy.savetxt format) file, and prints a 1D radial average, after proper smoothing',usage='use "%(prog)s -h" for more information',formatter_class=argparse.RawTextHelpFormatter)
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
  parser.add_argument('-k','--k', type=int, default=3, help='order of the spline [default=3]')
  parser.add_argument('-n','--num_points', type=int, default=2000, help='number of points for the radial data [default=2000]')
  parser.add_argument('-d','--knots_spacing', type=float, default=None, help='distance between knots [default=1.5*min(discretization step_x,discretization_step_y)]')
  parser.add_argument('-m','--r_min', type=float, default=0.0, help='minimum radius [default=0.0]')
  parser.add_argument('-M','--r_max', type=float, default=None, help='maximum radius [default=min(size_x,size_y)/2]')
  parser.add_argument('-i','--interval_width', type=float, default=None, help='width of elementary interval for spline [default=min(size_x,size_y)/2]')
  args = parser.parse_args()
  file_name = args.filename.name
  my_choice = args.column
  k = args.k
  if k<1 or k>5:
    sys.exit('k should 1 <= k <= 5')
  number_of_points = args.num_points
  distance_between_knots = args.knots_spacing
  r_min = args.r_min
  r_max = args.r_max
  interval_width = args.interval_width

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

  # Try to extract information from the first four lines
  try:
    f = open(file_name,'r')
    line=(f.readline().lstrip('#')).split()
    n1=int(line[0])
    found_delta = False
    if len(line)>1:
      delta1=float(line[-1])
      found_delta = True
    else:
      delta1=1.0
    line=(f.readline().lstrip('#')).split()
    n2=int(line[0])
    if len(line)>1:
      delta2=float(line[-1])
    else:
      delta2=1.0
    if not found_delta:
      line=(f.readline().lstrip('#')).split()
      delta1=float(line[-1])
      line=(f.readline().lstrip('#')).split()
      delta2=float(line[-1])
  except:
    n1,n2 = Z.shape
    delta1=1.0
    delta2=1.0

  if r_max == None:
    r_max = min(0.5*n1*delta1,0.5*n2*delta2)
  if interval_width == None:
    interval_width = r_max
  if distance_between_knots== None:
    distance_between_knots = 1.5*min(delta1,delta2)
  if distance_between_knots<min(delta1,delta2):
    print('WARNING: distance_between_knots is smaller than the spatial discretization!\n         This will certainly fail if r_min=0\n         Hope you understand what you are doing...')
  #print(n1,n2)
  #X, Y = np.meshgrid(x, y)
  #Z = np.exp(-(X*X+Y*Y))
  print('Maximum value = ',Z.max())
  print('Minimum value = ',Z.min())
  name, ext = os.path.splitext(file_name)
  name = name+specific_string
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


  # Compute the number of knots
  number_of_knots = int(interval_width/distance_between_knots+0.5)
  # Round to integers
  number_of_intervals = int((r_max-r_min)/interval_width+0.5)
  delta_r = (r_max-r_min)/number_of_intervals
# r_excess is the additional guard on each interval r, which ensures smoothness at
# the boundary between intervals
  r_excess = 2.0*min(delta1,delta2)
# If r_min-r_excess<0.0, the fit will try to access negative values of r
# Thus, it is needed to slightly extend r and func towards negative values
  if r_min<r_excess:
    extension = np.argmax(r>(r_excess-r_min))
    extended_r = np.zeros(n1*n2+extension)
    extended_r[extension:n1*n2+extension] = r[0:n1*n2]
    extended_r[0:extension] = - r[extension:0:-1]
    r = extended_r
    extended_func = np.zeros(n1*n2+extension)
    extended_func[extension:n1*n2+extension] = func[0:n1*n2]
    extended_func[0:extension] = func[extension:0:-1]
    func = extended_func
#    print(extended_r[0:2*extension])

  xnew = np.linspace(r_min,r_max,num=number_of_points+1)
  ynew = np.zeros(number_of_points+1)

  tab_position_up = np.zeros(number_of_intervals+1,dtype=int)
  tab_position_down = np.zeros(number_of_intervals+1,dtype=int)
  tab_evaluation = np.zeros(number_of_intervals+1,dtype=int)
  tab_position_down[0] = np.argmax(r>(r_min-r_excess))
  tab_evaluation[0] = 0
  for i in range(number_of_intervals):
    r_sup = r_min+delta_r*(i+1)
  #  print(r_sup)
    tab_position_up[i+1] = np.argmax(r>(r_sup+r_excess))
    tab_position_down[i+1] = np.argmax(r>(r_sup-r_excess))
    tab_evaluation[i+1] = np.argmax(xnew>r_sup)
  tab_evaluation[number_of_intervals]=number_of_points+1

  for i in range(number_of_intervals):
    tab_knots = np.linspace(r_min+delta_r*i-r_excess+delta_r/(number_of_knots+1),r_min+delta_r*(i+1)+r_excess-delta_r/(number_of_knots+1),num=number_of_knots)
 #   print('tab_knots',tab_knots)
 #   print('r',r[tab_position_down[i]],r[tab_position_up[i+1]])
 #   print('xnew',xnew[tab_evaluation[i]],xnew[tab_evaluation[i+1]-1])
  #  print('xnew',xnew[i*number_of_points_per_interval+2:(i+1)*number_of_points_per_interval])
    s = scipy.interpolate.LSQUnivariateSpline(r[tab_position_down[i]:tab_position_up[i+1]],func[tab_position_down[i]:tab_position_up[i+1]], tab_knots, k=k, check_finite=False)
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

if __name__ == "__main__":
  main()

