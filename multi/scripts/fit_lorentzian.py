#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:32:27 2020

@author: delande
"""

import math
import numpy as np
import scipy.optimize

def lorentzian(x, center, half_width, maximum):
  return maximum*half_width**2/((x-center)**2+half_width**2)

def simple_fit(x,y,p0):
  popt, pcov = scipy.optimize.curve_fit(lorentzian, x, y, p0)
  return popt,pcov

# fancy_fit is an attempt of self-adjustment of the fitting parameters (especially half-width)
# in order to iteratively converge to a better fit.
# Does not really work well in conditions where it could be useful, that is far from a Lorentzian
def fancy_fit(x,y,p0):
# Copy first guess in plocal
  plocal=p0[:]
# Width of the fitting interval in units of the half-width of the Lorentzian
  coef=1.0
# Number of iterations
  jmax = 1
  for j in range(jmax):
# Adjust the interval for fitting
# The next line forces the center and can be commented out
    plocal[0]=p0[0]
    imin = np.argmax(x>plocal[0]-coef*plocal[1])
    imax = np.argmax(x>plocal[0]+coef*plocal[1])
#  print(imin,imax,x[imin],x[imax])
    plocal, pcov = scipy.optimize.curve_fit(lorentzian, x[imin:imax], y[imin:imax], plocal)
#    print(j,plocal[0],plocal[1])
  return plocal,pcov

i_min=1
i_max=25
# Half-width of the fitting interval
half_width=0.4
# check_error_bars prints errors in the least-square adjustments
# Makes it possible to check that error bars are correctly computed
check_error_bars = True
# next parameter can be 'Python' or 'C' depending on which program generated the data
origin_of_data = 'Python'
#origin_of_data = 'C'

result = np.zeros((i_max+1-i_min,8))
for i in range(i_max+1-i_min):
  result[i,0]=i+i_min
  if origin_of_data=='C':
# Data produced by the an2d_propXX C program
    filename='density_momentum_'+"{:03d}".format(i+i_min)+'_radial.dat'
  else:
# Data produced by the Python compute_prop program
    if i<i_max-i_min:
      filename='density_momentum_intermediate_'+str(i+i_min)+'_radial.dat'
    else:
      filename='density_momentum_final_radial.dat'
  data=np.loadtxt(filename)
  x=data[:,0]
  y=data[:,1]
#  print(x)
#  print(y)
# Initial guess
  p0 = [0.628, 0.02, 2.0]
  imin = np.argmax(x>p0[0]-half_width)
  imax = np.argmax(x>p0[0]+half_width)
  popt, pcov = scipy.optimize.curve_fit(lorentzian, x[imin:imax], y[imin:imax], p0)
#  print(pcov)
# Correct the bug (?) is curve_fit, so that the uncertainty on the parameters is correct
  pcov*=(imax-imin)
#  popt, pcov = fancy_fit(x, y, p0)
#  popt, pcov = simple_fit(x, y, p0)
  result[i,1] = popt[1]
  result[i,2] = math.sqrt(pcov[1,1])
  result[i,3] = popt[0]
  result[i,4] = math.sqrt(pcov[0,0])
  result[i,5] = popt[2]
  result[i,6] = math.sqrt(pcov[2,2])
# The next line computes the total weight of the fitted Lorentzian, should be close to 1
  result[i,7] = 2.0*popt[2]*popt[1]*popt[0]*math.pi**2
  if check_error_bars:
# First print the error in the least-square adjustment
    print(i+i_min,np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt))**2)
    popt_minus=np.copy(popt)
    popt_plus =np.copy(popt)
# The prints the errors in the least-square adjustment when moving by +/- one error bar in each direction
# Should more or less double
    popt_plus[0]=popt[0]+math.sqrt(pcov[0,0])
    popt_minus[0]=popt[0]-math.sqrt(pcov[0,0])
    print('    +/- direction 0',np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt_plus))**2,np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt_minus))**2)
    popt_plus[0]=popt[0]
    popt_minus[0]=popt[0]
    popt_plus[1]=popt[1]+math.sqrt(pcov[1,1])
    popt_minus[1]=popt[1]-math.sqrt(pcov[1,1])
    print('    +/- direction 1',np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt_plus))**2,np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt_minus))**2)
    popt_plus[1]=popt[1]
    popt_minus[1]=popt[1]
    popt_plus[2]=popt[2]+math.sqrt(pcov[2,2])
    popt_minus[2]=popt[2]-math.sqrt(pcov[2,2])
    print('    +/- direction 2',np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt_plus))**2,np.linalg.norm(y[imin:imax]-lorentzian(x[imin:imax],*popt_minus))**2)
header_string = 'Result of Lorentzian fit\nHalf-width of the fitting interval = '+str(half_width)+'\n'\
               +'Column 1: index of file (not time)\n'\
               +'Column 2: Half-width of the Lorentzian\n'\
               +'Column 3: Uncertainty on the half-width\n'\
               +'Column 4: Center of the Lorentzian\n'\
               +'Column 5: Uncertainty on the center\n'\
               +'Column 6: Maximum of the Lorentzian\n'\
               +'Column 7: Uncertainty on the maximum\n'\
               +'Column 8: Total weight of the Lorentzian (should be close to 1)\n'

np.savetxt('result_fit.dat',result,header=header_string)