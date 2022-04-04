#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 20:32:27 2020

@author: delande
"""

import math
import numpy as np
import scipy.optimize

def lorentzian(x, center, half_width_squared, maximum):
  return maximum/((x-center)**2+half_width_squared)

i_min=1
i_max=59

result = np.zeros((i_max+2-i_min,6))
for i in range(0,i_max+2-i_min):
  result[i,0]=i+i_min
  if i<i_max+1-i_min:
    filename='density_momentum_intermediate_'+str(i+i_min)+'_radial.dat'
  else:
    filename='density_momentum_final_radial.dat'
  data=np.loadtxt(filename)
  x=data[:,0]
  y=data[:,1]
#  print(x)
#  print(y)
#  p0 = [1.57, 1, 0.005]
  popt, pcov = scipy.optimize.curve_fit(lorentzian, x, y)
  result[i,1] = math.sqrt(popt[1])
  result[i,2] = 0.5*math.sqrt(pcov[1,1]/popt[1])
  result[i,3] = popt[0]
  result[i,4] = popt[2]/popt[1]
  result[i,5] = 2.0*popt[2]*popt[0]*math.pi**2/math.sqrt(popt[1])
np.savetxt('result_fit.dat',result)